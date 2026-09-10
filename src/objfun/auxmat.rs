//! The exact auxiliary-matrix GRAPE objective.
//!
//! Port of `kernel/obj_fun/grape_state_auxmat.m` and
//! `kernel/obj_fun/grape_ugate_auxmat.m`.  This is the reference method QOALA
//! is measured against: it exponentiates the full Hamiltonian at every time
//! slice, and gets the directional derivative from the block matrix
//!
//! ```text
//! expm( -i dt [[L, C], [0, L]] )  =  [[ U, dU/dc ], [ 0, U ]]
//! ```
//!
//! so the top-right block is exactly the derivative of the propagator with
//! respect to the amplitude of control `C`.
//!
//! References: Auxiliary matrix method, <http://dx.doi.org/10.1063/1.4928978>;
//! Krylov variant, <http://dx.doi.org/10.5258/soton/t0003>.

use super::{scale_waveform, EvalOrder, FidelityKind, ObjectiveResult, TrajData};
use crate::config::{ControlSystem, DriftSystem};
use crate::error::{QoalaError, Result};
use crate::linalg::{real_trace_adjoint, CDense, CMat};
use crate::propagate::{propagate_state, propagator};
use crate::types::StateSpace;
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::time::Instant;

/// Evaluate the auxiliary-matrix objective.
#[allow(clippy::too_many_arguments)]
pub fn objective(
    kind: FidelityKind,
    sys: &ControlSystem,
    drift: &DriftSystem,
    ctrls: &[CMat],
    amps: &[f64],
    wf: &DMatrix<f64>,
    init: &CDense,
    targ: &CDense,
    order: EvalOrder,
) -> Result<ObjectiveResult> {
    let started = Instant::now();
    let out = match sys.space {
        StateSpace::Liouville => liouville(kind, sys, drift, ctrls, amps, wf, init, targ, order),
        StateSpace::Hilbert => hilbert(kind, sys, drift, ctrls, amps, wf, init, targ, order),
    };
    let mut out = out?;
    out.data.timer = Some(started.elapsed().as_secs_f64());
    Ok(out)
}

/// The drift Hamiltonian plus the current controls.
fn hamiltonian(
    drift: &DriftSystem,
    ctrls: &[CMat],
    scaled: &DMatrix<f64>,
    n: usize,
) -> Result<CMat> {
    let base = drift
        .drift
        .as_ref()
        .ok_or_else(|| {
            QoalaError::MissingField(
                "the auxiliary-matrix objective needs a full drift Hamiltonian".into(),
            )
        })?
        .at(n)?;
    let mut l = base.clone();
    for (k, c) in ctrls.iter().enumerate() {
        l = l.add(&c.scale(Complex64::new(scaled[(n, k)], 0.0)))?;
    }
    Ok(l)
}

/// Stack `[[L, C], [0, L]]`.
fn auxiliary_matrix(l: &CMat, c: &CMat) -> Result<CMat> {
    let dim = l.nrows();
    let ls = l.to_sparse();
    let cs = c.to_sparse();
    let mut triplets: Vec<(usize, usize, Complex64)> = Vec::new();
    for (r, col, v) in ls.triplet_iter() {
        triplets.push((r, col, *v));
        triplets.push((r + dim, col + dim, *v));
    }
    for (r, col, v) in cs.triplet_iter() {
        triplets.push((r, col + dim, *v));
    }
    CMat::from_triplets(2 * dim, 2 * dim, triplets)
}

/// Stack `[[0], [x]]`.
fn auxiliary_state(x: &CDense) -> CDense {
    let dim = x.nrows();
    let mut out = CDense::zeros(2 * dim, x.ncols());
    for c in 0..x.ncols() {
        for r in 0..dim {
            out[(r + dim, c)] = x[(r, c)];
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn liouville(
    kind: FidelityKind,
    sys: &ControlSystem,
    drift: &DriftSystem,
    ctrls: &[CMat],
    amps: &[f64],
    wf: &DMatrix<f64>,
    init: &CDense,
    targ: &CDense,
    order: EvalOrder,
) -> Result<ObjectiveResult> {
    let (nsteps, kctrls) = (wf.nrows(), wf.ncols());
    let scaled = scale_waveform(wf, amps)?;
    let dim = sys.dim;
    let nrm = kind.norm(dim);

    let mut fwd = init.clone();
    let want_grad = order == EvalOrder::Gradient;
    // Directional derivatives: one stored state per control and time slice.
    let mut dp_rho: Vec<Vec<CDense>> = if want_grad {
        vec![Vec::with_capacity(nsteps); kctrls]
    } else {
        Vec::new()
    };

    for n in 0..nsteps {
        let l = hamiltonian(drift, ctrls, &scaled, n)?;
        if want_grad {
            let aux_vec = auxiliary_state(&fwd);
            for (k, c) in ctrls.iter().enumerate() {
                let aux_mat = auxiliary_matrix(&l, c)?;
                let aux_out = propagate_state(
                    sys.space,
                    &aux_mat,
                    Complex64::new(sys.pulse_dt[n], 0.0),
                    sys.auxmat_method,
                    sys.prop_zeroed,
                    &aux_vec,
                )?;
                dp_rho[k].push(aux_out.rows(0, dim).into_owned());
            }
        }
        fwd = propagate_state(
            sys.space,
            &l,
            Complex64::new(sys.pulse_dt[n], 0.0),
            sys.step_method,
            sys.prop_zeroed,
            &fwd,
        )?;
    }

    let fidelity = real_trace_adjoint(targ, &fwd) / nrm;
    let mut result = ObjectiveResult {
        data: TrajData {
            final_state: Some(fwd),
            ..Default::default()
        },
        fidelity,
        grad: None,
    };
    if !want_grad {
        return Ok(result);
    }

    let mut grad = DMatrix::<f64>::zeros(nsteps, kctrls);
    let mut bwd = targ.clone();
    for n in (0..nsteps).rev() {
        for (k, amp) in amps.iter().enumerate().take(kctrls) {
            grad[(n, k)] = amp * real_trace_adjoint(&bwd, &dp_rho[k][n]) / nrm;
        }
        let l = hamiltonian(drift, ctrls, &scaled, n)?;
        bwd = propagate_state(
            sys.space,
            &l,
            Complex64::new(-sys.pulse_dt[n], 0.0),
            sys.step_method,
            sys.prop_zeroed,
            &bwd,
        )?;
    }
    result.grad = Some(grad);
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn hilbert(
    kind: FidelityKind,
    sys: &ControlSystem,
    drift: &DriftSystem,
    ctrls: &[CMat],
    amps: &[f64],
    wf: &DMatrix<f64>,
    init: &CDense,
    targ: &CDense,
    order: EvalOrder,
) -> Result<ObjectiveResult> {
    let (nsteps, kctrls) = (wf.nrows(), wf.ncols());
    let scaled = scale_waveform(wf, amps)?;
    let dim = sys.dim;
    let want_grad = order == EvalOrder::Gradient;
    let aux_tol = sys.prop_zeroed.min(1e-14);

    let mut fwd = init.clone();
    let mut p_n: Vec<CDense> = Vec::with_capacity(nsteps);
    let mut dp_rho: Vec<Vec<CDense>> = if want_grad {
        vec![Vec::with_capacity(nsteps); kctrls]
    } else {
        Vec::new()
    };

    for n in 0..nsteps {
        let l = hamiltonian(drift, ctrls, &scaled, n)?;
        if want_grad {
            let mut step_prop: Option<CDense> = None;
            for (k, c) in ctrls.iter().enumerate() {
                let aux_mat = auxiliary_matrix(&l, c)?;
                let full = propagator(
                    &aux_mat,
                    Complex64::new(sys.pulse_dt[n], 0.0),
                    sys.auxmat_method,
                    aux_tol,
                )?
                .to_dense();
                let u = full.view((0, 0), (dim, dim)).into_owned();
                let du = full.view((0, dim), (dim, dim)).into_owned();
                if step_prop.is_none() {
                    step_prop = Some(u.clone());
                }
                dp_rho[k].push((u.adjoint() * du) * &fwd);
            }
            let u = step_prop.ok_or_else(|| {
                QoalaError::BadValue("a Hilbert-space gradient needs at least one control".into())
            })?;
            fwd = &u * &fwd * u.adjoint();
            p_n.push(u);
        } else {
            fwd = propagate_state(
                sys.space,
                &l,
                Complex64::new(sys.pulse_dt[n], 0.0),
                sys.step_method,
                sys.prop_zeroed,
                &fwd,
            )?;
        }
    }

    let fidelity = real_trace_adjoint(targ, &fwd) / kind.norm(dim);
    let mut result = ObjectiveResult {
        data: TrajData {
            final_state: Some(fwd),
            ..Default::default()
        },
        fidelity,
        grad: None,
    };
    if !want_grad {
        return Ok(result);
    }

    let mut grad = DMatrix::<f64>::zeros(nsteps, kctrls);
    let mut bwd = targ.adjoint();
    for n in (0..nsteps).rev() {
        let u = &p_n[n];
        bwd = u.adjoint() * &bwd * u;
        for (k, amp) in amps.iter().enumerate().take(kctrls) {
            // 2 Re tr(chi dP_rho); the trace of a plain product, not an
            // adjoint one, because chi is already Hermitian.
            let mut acc = 0.0;
            let m = &dp_rho[k][n];
            for i in 0..dim {
                for j in 0..dim {
                    acc += (bwd[(i, j)] * m[(j, i)]).re;
                }
            }
            grad[(n, k)] = amp * 2.0 * acc / kind.norm(dim);
        }
    }
    result.grad = Some(grad);
    Ok(result)
}
