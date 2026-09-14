//! The adaptive split-operator GRAPE objective.
//!
//! Port of `kernel/obj_fun/grape_state_qoala.m` and
//! `kernel/obj_fun/grape_ugate_qoala.m`, which are the same algorithm with two
//! different fidelity functionals.
//!
//! # The method
//!
//! One time slice of the propagator is factorised into single-spin rotations
//! and interaction propagators,
//!
//! ```text
//! U_n  ~  B_{j_M} A_{i_M} ... B_{j_1} A_{i_1} B_{j_0}
//! ```
//!
//! where the `A` are Euler-Rodrigues rotations (closed form, no matrix
//! exponential) and the `B` are the small set of interaction propagators
//! precomputed by [`crate::splittings`].  The forward trajectory is stored at
//! every stage so that the backward pass can read off
//! `<chi| ... dA/dc ... |rho>` directly.
//!
//! # Adaptivity
//!
//! Early in an optimisation a crude propagator is good enough; near
//! convergence it is not.  Rather than fixing the splitting order and Trotter
//! number, QOALA climbs a ladder of `(trotter, order)` pairs until two
//! successive rungs agree to within a tolerance that itself shrinks with the
//! infidelity (or the gradient norm).  That is what makes the method cheap:
//! most iterations run at the bottom of the ladder.
//!
//! # Backward pass
//!
//! The MATLAB applies the interaction propagators in *forward* index order
//! during the backward sweep, and has a separate branch for splitting orders
//! below two.  That is exact for the symmetric splittings (orders 2, 4 and 6 -
//! every order the package uses by default), because their index sequences are
//! palindromes.  For the asymmetric orders 0, 1 and 3 it is not the adjoint of
//! the forward step.  This port always walks the sequence in reverse, which
//! agrees with the MATLAB everywhere the MATLAB is exact and gives the exact
//! gradient of the split propagator elsewhere.  See `DEVIATIONS.md`.

use super::{
    rotation_grid, scale_waveform, single_spin_dim, EvalOrder, FidelityKind, ObjectiveResult,
    TrajData,
};
use crate::config::{ControlSystem, DriftSystem};
use crate::error::{QoalaError, Result};
use crate::linalg::{norm_2_state, real_trace_adjoint, CDense, CMat, EPS};
use crate::rodrigues::{prepend_unit_row, propagator_from_elements, rodrigues_with_derivatives};
use crate::splittings::SplitSet;
use crate::time::Instant;
use crate::types::{AdaptMethod, AdaptScale, PropCache, PropMethod, StateSpace};
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::borrow::Cow;

/// Stored forward pass: everything the backward sweep and the adaptivity
/// comparison need.
struct ForwardPass {
    /// State or propagator after each time slice; `nsteps + 1` entries.
    fwd_traj: Vec<CDense>,
    /// State after each single-spin stage, indexed `[stage][step]`.
    fwd_split: Vec<Vec<CDense>>,
    /// Single-spin propagators, indexed `[split][step]`.
    p_n: Vec<Vec<CMat>>,
    /// Fidelity implied by this pass.
    fidelity: f64,
}

/// Evaluate the adaptive split-operator objective.
#[allow(clippy::too_many_arguments)]
pub fn objective(
    kind: FidelityKind,
    sys: &ControlSystem,
    drift: &DriftSystem,
    ctrls: &[Vec<Option<CMat>>],
    amps: &[f64],
    wf: &DMatrix<f64>,
    init: &CDense,
    targ: &CDense,
    order: EvalOrder,
) -> Result<ObjectiveResult> {
    if sys.space != StateSpace::Liouville {
        return Err(QoalaError::NotImplemented(
            "the split-operator objective is only coded for Liouville space".into(),
        ));
    }
    if drift.interaction.is_none() {
        return Err(QoalaError::NotImplemented(
            "the split-operator objective needs an interaction Hamiltonian; \
             use the auxiliary-matrix objective for a single spin"
                .into(),
        ));
    }
    let started = Instant::now();

    let mut result = match order {
        EvalOrder::Value => {
            let split = split_for_rung(sys, drift, drift.ladder_index().unwrap_or(0))?;
            let pass = forward_pass(kind, sys, drift, &split, ctrls, amps, wf, init, targ)?;
            ObjectiveResult {
                data: TrajData {
                    final_state: pass.fwd_traj.last().cloned(),
                    ..Default::default()
                },
                fidelity: pass.fidelity,
                grad: None,
            }
        }
        EvalOrder::Gradient if !drift.adapt => {
            let split = split_for_rung(sys, drift, drift.ladder_index().unwrap_or(0))?;
            gradient(kind, sys, drift, &split, ctrls, amps, wf, init, targ, None)?
        }
        EvalOrder::Gradient => adaptive_gradient(kind, sys, drift, ctrls, amps, wf, init, targ)?,
    };

    result.data.timer = Some(started.elapsed().as_secs_f64());
    Ok(result)
}

/// The splitting data for ladder rung `k`, recomputing it when the propagator
/// cache is not carried in memory.
fn split_for_rung<'a>(
    sys: &ControlSystem,
    drift: &'a DriftSystem,
    k: usize,
) -> Result<Cow<'a, SplitSet>> {
    if sys.prop_cache == PropCache::Carry && drift.splits.len() > 1 {
        return Ok(Cow::Borrowed(drift.split_at(k)?));
    }
    let want = drift.adaptset.get(k).copied();
    match want {
        Some((q, p)) if drift.splits.len() == 1 => {
            let have = &drift.splits[0];
            if have.trotter == q && have.order == p {
                Ok(Cow::Borrowed(have))
            } else {
                let interaction = drift
                    .interaction
                    .as_ref()
                    .ok_or_else(|| QoalaError::MissingField("interaction".into()))?
                    .at(0)?;
                Ok(Cow::Owned(SplitSet::build(
                    p,
                    q,
                    sys.uniform_dt(),
                    interaction,
                    PropMethod::Taylor,
                    sys.prop_zeroed,
                )?))
            }
        }
        _ => Ok(Cow::Borrowed(drift.split_at(k)?)),
    }
}

/// Forward propagation, storing every intermediate stage.
#[allow(clippy::too_many_arguments)]
fn forward_pass(
    kind: FidelityKind,
    sys: &ControlSystem,
    drift: &DriftSystem,
    split: &SplitSet,
    _ctrls: &[Vec<Option<CMat>>],
    amps: &[f64],
    wf: &DMatrix<f64>,
    init: &CDense,
    targ: &CDense,
) -> Result<ForwardPass> {
    let nsteps = wf.nrows();
    let scaled = scale_waveform(wf, amps)?;
    let grid = rotation_grid(sys, drift, &scaled, &split.coeffs.c_spn, split.trotter)?;

    let elements = crate::rodrigues::rodrigues(sys.space, sys.basis, &grid.axes, &grid.dt)?;
    let elements = prepend_unit_row(sys.basis, elements);

    let p_n = build_propagators(sys, split, &elements, nsteps)?;
    let (fwd_traj, fwd_split) = propagate_forward(sys, split, &p_n, init, nsteps)?;

    let last = fwd_traj
        .last()
        .ok_or_else(|| QoalaError::Numerical("empty forward trajectory".into()))?;
    let fidelity = kind.overlap(targ, last, sys.dim);

    Ok(ForwardPass {
        fwd_traj,
        fwd_split,
        p_n,
        fidelity,
    })
}

/// Assemble the single-spin propagators for every split stage and time slice.
fn build_propagators(
    sys: &ControlSystem,
    split: &SplitSet,
    elements: &DMatrix<Complex64>,
    nsteps: usize,
) -> Result<Vec<Vec<CMat>>> {
    let nsplits = split.coeffs.nsplits();
    let nspins = sys.nspins;
    let single_dim = single_spin_dim(&sys.prop_ind)?;
    let mut p_n = vec![Vec::with_capacity(nsteps); nsplits];
    let mut cols = vec![0usize; nspins];
    for n in 0..nsteps {
        for (i, row) in p_n.iter_mut().enumerate() {
            for (s, col) in cols.iter_mut().enumerate() {
                *col = (s * nsteps + n) * nsplits + i;
            }
            let p = propagator_from_elements(&sys.prop_ind, elements, &cols, single_dim)?
                .densify_if(sys.sparsity, sys.dim);
            row.push(p);
        }
    }
    Ok(p_n)
}

/// Apply an interaction propagator, treating `None` as the identity.
#[inline]
fn apply_opt(p: Option<&CMat>, x: &CDense) -> Result<CDense> {
    match p {
        None => Ok(x.clone()),
        Some(m) => m.apply(x),
    }
}

/// Walk the split sequence forward, storing every stage.
fn propagate_forward(
    _sys: &ControlSystem,
    split: &SplitSet,
    p_n: &[Vec<CMat>],
    init: &CDense,
    nsteps: usize,
) -> Result<(Vec<CDense>, Vec<Vec<CDense>>)> {
    let k = &split.coeffs;
    let (num_a, num_b) = (k.num_a(), k.num_b());

    let mut fwd_traj: Vec<CDense> = Vec::with_capacity(nsteps + 1);
    fwd_traj.push(init.clone());
    let mut fwd_split: Vec<Vec<CDense>> = vec![Vec::with_capacity(nsteps); num_a];

    for n in 0..nsteps {
        let mut stage = apply_opt(split.int_prop(k.ind_int[0])?, &fwd_traj[n])?;
        fwd_split[0].push(stage.clone());
        for i in 1..num_a {
            let spun = p_n[k.ind_spn[i - 1]][n].apply(&stage)?;
            stage = apply_opt(split.int_prop(k.ind_int[i])?, &spun)?;
            fwd_split[i].push(stage.clone());
        }
        let spun = p_n[k.ind_spn[num_a - 1]][n].apply(&stage)?;
        fwd_traj.push(apply_opt(split.int_prop(k.ind_int[num_b - 1])?, &spun)?);
    }
    Ok((fwd_traj, fwd_split))
}

/// Fidelity and gradient for a fixed splitting.
#[allow(clippy::too_many_arguments)]
fn gradient(
    kind: FidelityKind,
    sys: &ControlSystem,
    drift: &DriftSystem,
    split: &SplitSet,
    ctrls: &[Vec<Option<CMat>>],
    amps: &[f64],
    wf: &DMatrix<f64>,
    init: &CDense,
    targ: &CDense,
    cached: Option<ForwardPass>,
) -> Result<ObjectiveResult> {
    let (nsteps, kctrls) = (wf.nrows(), wf.ncols());
    let scaled = scale_waveform(wf, amps)?;
    let grid = rotation_grid(sys, drift, &scaled, &split.coeffs.c_spn, split.trotter)?;

    let (elements, dp) = rodrigues_with_derivatives(
        sys.space,
        sys.basis,
        ctrls,
        &grid.axes,
        &grid.dt,
        sys.gradops,
        &sys.spin_control,
    )?;
    if dp.len() < kctrls {
        return Err(QoalaError::Dimension(format!(
            "{} control derivatives for {kctrls} waveform channels",
            dp.len()
        )));
    }

    let pass = match cached {
        Some(p) => p,
        None => {
            let elements = prepend_unit_row(sys.basis, elements);
            let p_n = build_propagators(sys, split, &elements, nsteps)?;
            let (fwd_traj, fwd_split) = propagate_forward(sys, split, &p_n, init, nsteps)?;
            let last = fwd_traj
                .last()
                .ok_or_else(|| QoalaError::Numerical("empty forward trajectory".into()))?;
            let fidelity = kind.overlap(targ, last, sys.dim);
            ForwardPass {
                fwd_traj,
                fwd_split,
                p_n,
                fidelity,
            }
        }
    };

    let last = pass
        .fwd_traj
        .last()
        .ok_or_else(|| QoalaError::Numerical("empty forward trajectory".into()))?;
    let fidelity = kind.overlap(targ, last, sys.dim);

    // Backward sweep.  The split sequence is walked in reverse so that the
    // stored vector for stage i is exactly what the gradient term for that
    // stage needs.
    let k = &split.coeffs;
    let (num_a, num_b) = (k.num_a(), k.num_b());
    let nsplits = k.nsplits();
    let nrm = kind.norm(sys.dim);

    // Adjoints of the interaction propagators, formed once.
    let int_adj: Vec<Option<CMat>> = split
        .inter_prop
        .iter()
        .map(|o| o.as_ref().map(|m| m.adjoint()))
        .collect();
    let adj = |i: usize| -> Result<Option<&CMat>> {
        int_adj
            .get(i)
            .map(|o| o.as_ref())
            .ok_or_else(|| QoalaError::Dimension(format!("interaction propagator {i} missing")))
    };

    let mut grad = DMatrix::<f64>::zeros(nsteps, kctrls);
    let mut bwd = targ.clone();
    let mut stage_adjoints: Vec<CDense> = vec![CDense::zeros(0, 0); num_a];

    for n in (0..nsteps).rev() {
        let mut v = apply_opt(adj(k.ind_int[num_b - 1])?, &bwd)?;
        for j in (0..num_a).rev() {
            v = pass.p_n[k.ind_spn[j]][n].adjoint().apply(&v)?;
            stage_adjoints[j] = v.clone();
            v = apply_opt(adj(k.ind_int[j])?, &v)?;
        }
        bwd = v;

        for (ctrl, dpk) in dp.iter().enumerate().take(kctrls) {
            let mut acc = 0.0;
            for (i, adjoint) in stage_adjoints.iter().enumerate().take(num_a) {
                let dpn = dpk.matrix(n * nsplits + k.ind_spn[i])?;
                let rho_n = dpn.apply(&pass.fwd_split[i][n])?;
                acc += amps[ctrl] * real_trace_adjoint(adjoint, &rho_n) / nrm;
            }
            grad[(n, ctrl)] = acc;
        }
    }

    Ok(ObjectiveResult {
        data: TrajData {
            final_state: Some(last.clone()),
            ..Default::default()
        },
        fidelity,
        grad: Some(grad),
    })
}

/// Fidelity and gradient with the adaptivity ladder.
#[allow(clippy::too_many_arguments)]
fn adaptive_gradient(
    kind: FidelityKind,
    sys: &ControlSystem,
    drift: &DriftSystem,
    ctrls: &[Vec<Option<CMat>>],
    amps: &[f64],
    wf: &DMatrix<f64>,
    init: &CDense,
    targ: &CDense,
) -> Result<ObjectiveResult> {
    // Not enough iterations have passed since the last check: run at the
    // current rung and bump the counter.
    if drift.adapt_counter < drift.adapt_minit as f64 {
        let split = split_for_rung(sys, drift, drift.ladder_index().unwrap_or(0))?;
        let mut out = gradient(kind, sys, drift, &split, ctrls, amps, wf, init, targ, None)?;
        let mut updated = drift.clone();
        updated.adapt_counter = drift.adapt_counter + 1.0;
        updated.adapt = true;
        out.data.drift_sys = Some(updated);
        return Ok(out);
    }

    let mut f_adapt_count = 0usize;
    let mut k = drift.ladder_index().unwrap_or(0);

    // With the 'exact' method the reference is a full, unsplit propagation.
    let mut reference: Option<CDense> = None;
    if drift.adapt_method == AdaptMethod::Exact {
        let (data, _) = crate::waveform::waveform_fidelity(
            kind,
            sys,
            drift,
            &sys.operators,
            amps,
            wf,
            init,
            targ,
        )?;
        let mut rho0 = data
            .final_state
            .ok_or_else(|| QoalaError::Numerical("exact check returned no final state".into()))?;
        if kind.normalise_for_adaptivity() {
            let n = norm_2_state(&rho0);
            if n > 0.0 {
                rho0 /= Complex64::new(n, 0.0);
            }
        }
        reference = Some(rho0);
    }

    let mut split = split_for_rung(sys, drift, k)?;
    let mut pass = forward_pass(kind, sys, drift, &split, ctrls, amps, wf, init, targ)?;

    // The rung whose trajectories are currently stored.
    let mut stored_k = k;
    let mut stored: ForwardPass = ForwardPass {
        fwd_traj: pass.fwd_traj.clone(),
        fwd_split: pass.fwd_split.clone(),
        p_n: pass.p_n.clone(),
        fidelity: pass.fidelity,
    };

    let mut rho1 = normalised_final(&pass, kind)?;
    let mut error_f = 1.0;
    if let Some(r0) = &reference {
        error_f *= (1.0 - kind.overlap(&rho1, r0, sys.dim)).abs();
    }
    let mut tol_f = adaptivity_tolerance(sys, drift, pass.fidelity);

    while error_f > tol_f {
        stored_k = k;
        stored = ForwardPass {
            fwd_traj: pass.fwd_traj.clone(),
            fwd_split: pass.fwd_split.clone(),
            p_n: pass.p_n.clone(),
            fidelity: pass.fidelity,
        };

        k += 1;
        if k >= drift.adaptset.len() {
            break;
        }

        // The 'gain' method compares against the previous rung, not an exact
        // reference.
        let compare_against = match drift.adapt_method {
            AdaptMethod::Gain => rho1.clone(),
            AdaptMethod::Exact => reference
                .clone()
                .ok_or_else(|| QoalaError::Numerical("missing exact reference".into()))?,
        };

        split = split_for_rung(sys, drift, k)?;
        pass = forward_pass(kind, sys, drift, &split, ctrls, amps, wf, init, targ)?;
        rho1 = normalised_final(&pass, kind)?;
        f_adapt_count += 1;

        error_f = (1.0 - kind.overlap(&rho1, &compare_against, sys.dim)).abs();
        tol_f = adaptivity_tolerance(sys, drift, pass.fidelity);

        // The counter starts at infinity so that the very first gradient call
        // can climb as far as it needs; after that the ladder is walked at
        // most `adapt_interval` rungs per check.
        if drift.adapt_counter.is_finite() {
            let budget = match drift.adapt_method {
                AdaptMethod::Gain => drift.adapt_interval,
                AdaptMethod::Exact => drift.adapt_interval - 1.0,
            };
            if f_adapt_count as f64 > budget {
                break;
            }
        }
    }

    let mut updated = drift.clone();
    if drift.adapt_method != AdaptMethod::Exact {
        // Settle on the rung whose trajectories were kept.
        let (q, p) = drift.adaptset[stored_k];
        updated.trotter_number = q;
        updated.split_order = p;
        updated.adapt_counter = 1.0;
    }
    updated.adapt = true;

    let stored_split = split_for_rung(sys, drift, stored_k)?;
    let mut out = gradient(
        kind,
        sys,
        drift,
        &stored_split,
        ctrls,
        amps,
        wf,
        init,
        targ,
        Some(stored),
    )?;

    let grad_norm = out
        .grad
        .as_ref()
        .map(|g| g.iter().map(|v| v * v).sum::<f64>().sqrt());
    out.data.drift_sys = Some(updated);
    out.data.f_adapt_count = Some(f_adapt_count);
    out.data.grad_norm = grad_norm;
    Ok(out)
}

/// Final state, normalised when the fidelity functional calls for it.
fn normalised_final(pass: &ForwardPass, kind: FidelityKind) -> Result<CDense> {
    let mut rho = pass
        .fwd_traj
        .last()
        .cloned()
        .ok_or_else(|| QoalaError::Numerical("empty forward trajectory".into()))?;
    if kind.normalise_for_adaptivity() {
        let n = norm_2_state(&rho);
        if n > 0.0 {
            rho /= Complex64::new(n, 0.0);
        }
    }
    Ok(rho)
}

/// The adaptivity threshold, scaled as configured.
fn adaptivity_tolerance(sys: &ControlSystem, drift: &DriftSystem, fidelity_est: f64) -> f64 {
    let mut tol = drift.adapt_tols[0];
    if drift.adapt_scale.contains(&AdaptScale::Gradient) {
        if let Some(g) = sys.grad_norm {
            tol *= g.min(1.0);
        }
    }
    if drift.adapt_scale.contains(&AdaptScale::Infidelity) {
        tol *= (1.0 - fidelity_est.clamp(0.0, 1.0)).abs().max(EPS);
    }
    tol
}
