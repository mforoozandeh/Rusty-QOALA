//! High-level drivers: describe the problem, get an optimised pulse back.
//!
//! Ports `utilities/state2state_xy.m` and `utilities/universal_gate_xy.m`,
//! together with the `optimfun_grape_xy` cost function they share.  These wrap
//! the whole pipeline - build the control system, hand it to the optimiser -
//! for the common case of a chain of coupled two-level systems with an x and a
//! y control on each.

use crate::config::{
    optimconset, ControlOptions, ControlSystem, DriftOptions, PowerLevels, TimeDependent,
};
use crate::error::{QoalaError, Result};
use crate::linalg::{CDense, CMat};
use crate::objfun::{evaluate, Controls, EvalOrder, TrajData};
use crate::optim::newton::{fmaxnewton_with_progress, Optimisation};
use crate::optim::{CostFunction, NoProgress, ProgressSink};
use crate::penalty::{penalty, PenaltyOrder};
use crate::report::Reporter;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The GRAPE cost function used by both drivers.
///
/// It evaluates the chosen objective function and then every configured
/// penalty, returning them separately so the optimiser can report and combine
/// them itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct GrapeXyCost;

impl GrapeXyCost {
    fn parts<'a>(
        sys: &'a ControlSystem,
        objfun: ObjectiveFn,
    ) -> Result<(Controls<'a>, Vec<f64>, &'a CDense, &'a CDense)> {
        if sys.drift_sys.len() != 1 {
            return Err(QoalaError::NotImplemented(
                "drift ensembles with more than one member".into(),
            ));
        }
        let ctrls = if objfun.is_qoala() {
            Controls::Pauli(&sys.pauli_operators)
        } else {
            Controls::Composite(&sys.operators)
        };
        let amps = sys.power_row(0);
        let init = sys
            .initials
            .first()
            .ok_or_else(|| QoalaError::MissingField("initial state".into()))?;
        let targ = sys
            .targets
            .first()
            .ok_or_else(|| QoalaError::MissingField("target state".into()))?;
        Ok((ctrls, amps, init, targ))
    }
}

impl CostFunction for GrapeXyCost {
    fn value(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        objfun: ObjectiveFn,
    ) -> Result<(TrajData, Vec<f64>)> {
        let (ctrls, amps, init, targ) = Self::parts(sys, objfun)?;
        let drift = &sys.drift_sys[0];
        let res = evaluate(
            objfun,
            sys,
            drift,
            &ctrls,
            &amps,
            wf,
            init,
            targ,
            EvalOrder::Value,
        )?;

        let ctx = sys.penalty_context();
        let mut fidelities = Vec::with_capacity(1 + sys.penalties.len());
        fidelities.push(res.fidelity);
        for (n, p) in sys.penalties.iter().enumerate() {
            let out = penalty(
                *p,
                wf,
                &sys.l_bound,
                &sys.u_bound,
                sys.p_weights[n],
                &ctx,
                PenaltyOrder::Value,
            )?;
            fidelities.push(out.value);
        }
        Ok((res.data, fidelities))
    }

    fn value_grad(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        objfun: ObjectiveFn,
    ) -> Result<(TrajData, Vec<f64>, Vec<DMatrix<f64>>)> {
        let (ctrls, amps, init, targ) = Self::parts(sys, objfun)?;
        let drift = &sys.drift_sys[0];
        let res = evaluate(
            objfun,
            sys,
            drift,
            &ctrls,
            &amps,
            wf,
            init,
            targ,
            EvalOrder::Gradient,
        )?;
        let grad = res
            .grad
            .ok_or_else(|| QoalaError::Numerical("objective returned no gradient".into()))?;

        let ctx = sys.penalty_context();
        let mut fidelities = Vec::with_capacity(1 + sys.penalties.len());
        let mut grads = Vec::with_capacity(1 + sys.penalties.len());
        fidelities.push(res.fidelity);
        grads.push(grad);
        for (n, p) in sys.penalties.iter().enumerate() {
            let out = penalty(
                *p,
                wf,
                &sys.l_bound,
                &sys.u_bound,
                sys.p_weights[n],
                &ctx,
                PenaltyOrder::Gradient,
            )?;
            fidelities.push(out.value);
            grads.push(
                out.grad
                    .unwrap_or_else(|| DMatrix::zeros(wf.nrows(), wf.ncols())),
            );
        }
        Ok((res.data, fidelities, grads))
    }
}

/// A state-to-state transfer problem.
///
/// The counterpart of the name-value arguments `state2state_xy` accepts.
#[derive(Debug, Clone)]
pub struct StateTransfer {
    /// Resonance offset of each spin, in Hz.
    pub omega: Vec<f64>,
    /// Initial state, normalised internally.
    pub initial: CDense,
    /// Target state, normalised internally.
    pub target: CDense,
    /// Maximum control amplitude per control pair, in Hz.
    pub amplitudes: Vec<f64>,
    /// Pulse duration in seconds.
    pub duration: f64,
    /// Number of time slices.
    pub increments: usize,
    /// Interaction Hamiltonian.
    pub interaction: CMat,
    /// Single-spin Cartesian operators, `nspins x 3*npairs`.
    pub cartops: Vec<Vec<Option<CMat>>>,
    /// Starting waveform; random in `[-1, 1]` when absent.
    pub init_pulse: Option<DMatrix<f64>>,
    /// Seed for the random starting waveform.
    pub seed: Option<u64>,
    /// Overrides for anything else the parser accepts.
    pub tuning: Tuning,
}

/// A gate-synthesis problem.
///
/// The counterpart of the name-value arguments `universal_gate_xy` accepts.
#[derive(Debug, Clone)]
pub struct GateSynthesis {
    /// Resonance offset of each spin, in Hz.
    pub omega: Vec<f64>,
    /// Target effective propagator.
    pub target: CDense,
    /// Maximum control amplitude per control pair, in Hz.
    pub amplitudes: Vec<f64>,
    /// Pulse duration in seconds.
    pub duration: f64,
    /// Number of time slices.
    pub increments: usize,
    /// Interaction Hamiltonian.
    pub interaction: CMat,
    /// Single-spin Cartesian operators, `nspins x 3*npairs`.
    pub cartops: Vec<Vec<Option<CMat>>>,
    /// Starting waveform; random in `[-1, 1]` when absent.
    pub init_pulse: Option<DMatrix<f64>>,
    /// Seed for the random starting waveform.
    pub seed: Option<u64>,
    /// Overrides for anything else the parser accepts.
    pub tuning: Tuning,
}

/// Optional settings both drivers share, with the MATLAB defaults.
#[derive(Debug, Clone)]
pub struct Tuning {
    /// Penalty terms; default a single spillout-norm-square term.
    pub penalties: Vec<Penalty>,
    /// Step-norm termination threshold.
    pub tol_x: f64,
    /// Gradient-norm termination threshold.
    pub tol_g: f64,
    /// Iteration budget.
    pub max_iter: usize,
    /// Interaction propagator caching strategy.
    pub prop_cache: PropCache,
    /// Propagator element cutoff.
    pub prop_zeroed: f64,
    /// Splitting orders the adaptive step may choose from.
    pub splitset: Vec<usize>,
    /// Trotter numbers the adaptive step may choose from.
    pub trotterset: Option<Vec<usize>>,
    /// LBFGS history length.
    pub n_grads: usize,
    /// Independent fidelity check at every iteration.
    pub fidelity_chk: bool,
    /// Where the report goes.
    pub output: Reporter,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            penalties: vec![Penalty::Sns],
            tol_x: 1e-12,
            tol_g: 1e-12,
            max_iter: 100,
            prop_cache: PropCache::Carry,
            prop_zeroed: 1e-12,
            splitset: vec![2, 3, 4],
            trotterset: None,
            n_grads: 25,
            fidelity_chk: false,
            output: Reporter::stdout(),
        }
    }
}

/// Adaptive optimal control for state-to-state transfer across a chain of
/// coupled two-level systems.
///
/// ```
/// use num_complex::Complex64;
/// use qoala::drivers::{state2state_xy, StateTransfer, Tuning};
/// use qoala::report::Reporter;
/// use qoala::spinops;
///
/// // Two spins weakly coupled at 140 Hz, with an x/y control pair on each.
/// let interaction = spinops::zz_coupling(2, 0, 1)?
///     .scale(Complex64::new(2.0 * std::f64::consts::PI * 140.0, 0.0));
///
/// let optimised = state2state_xy(StateTransfer {
///     omega: vec![0.0, 0.0],                   // resonance offsets, Hz
///     initial: spinops::z_state(2, 0),         // z-magnetisation on spin 1
///     target: spinops::z_state(2, 1),          // ... moved onto spin 2
///     amplitudes: vec![1000.0, 1000.0],        // maximum control power, Hz
///     duration: 0.01,                          // seconds
///     increments: 50,                          // time slices
///     interaction,
///     cartops: spinops::one_pair_per_spin(2),
///     init_pulse: None,                        // random starting pulse
///     seed: Some(1),
///     tuning: Tuning { max_iter: 5, output: Reporter::silent(), ..Default::default() },
/// })?;
///
/// assert_eq!(optimised.waveform.shape(), (50, 4));
/// # Ok::<(), qoala::error::QoalaError>(())
/// ```
pub fn state2state_xy(spec: StateTransfer) -> Result<Optimisation> {
    state2state_xy_with_progress(spec, &mut NoProgress)
}

/// As [`state2state_xy`], reporting every iteration to `progress` and
/// stopping when it asks to.
pub fn state2state_xy_with_progress(
    spec: StateTransfer,
    progress: &mut dyn ProgressSink,
) -> Result<Optimisation> {
    let (mut sys, guess) = state_transfer_system(&spec)?;
    fmaxnewton_with_progress(&mut sys, &GrapeXyCost, &guess, progress)
}

/// The configured control system and starting waveform a
/// [`state2state_xy`] run would use, without running it.
///
/// Useful for driving the optimiser yourself - with a progress sink, a
/// different cost function, or a waveform carried over from an earlier run.
pub fn state_transfer_system(spec: &StateTransfer) -> Result<(ControlSystem, DMatrix<f64>)> {
    let npairs = pair_count(&spec.cartops)?;
    let nspins = spec.omega.len();
    check_shapes(&spec.cartops, nspins, npairs, spec.amplitudes.len())?;

    let init = normalise(&spec.initial);
    let targ = normalise(&spec.target);

    let mut opts = base_options(
        &spec.omega,
        &spec.amplitudes,
        spec.duration,
        spec.increments,
        &spec.interaction,
        &spec.cartops,
        &spec.tuning,
    )?;
    opts.optimcon_fun = Some(ObjectiveFn::StateQoala);
    opts.rho_init = Some(vec![init]);
    opts.rho_targ = Some(vec![targ]);

    let guess = spec
        .init_pulse
        .clone()
        .unwrap_or_else(|| random_pulse(spec.increments, 2 * npairs, spec.seed));

    Ok((optimconset(opts)?, guess))
}

/// Adaptive optimal control for gate synthesis across a chain of coupled
/// two-level systems.
///
/// ```
/// use num_complex::Complex64;
/// use qoala::drivers::{universal_gate_xy, GateSynthesis, Tuning};
/// use qoala::report::Reporter;
/// use qoala::spinops;
///
/// let interaction = spinops::zz_coupling(2, 0, 1)?
///     .scale(Complex64::new(2.0 * std::f64::consts::PI * 140.0, 0.0));
///
/// let optimised = universal_gate_xy(GateSynthesis {
///     omega: vec![0.0, 0.0],
///     target: spinops::swap_gate(2, 0, 1)?,    // SWAP the two spins
///     amplitudes: vec![1000.0, 1000.0],
///     duration: 0.012,
///     increments: 60,
///     interaction,
///     cartops: spinops::one_pair_per_spin(2),
///     init_pulse: None,
///     seed: Some(1),
///     tuning: Tuning { max_iter: 3, output: Reporter::silent(), ..Default::default() },
/// })?;
///
/// assert_eq!(optimised.waveform.shape(), (60, 4));
/// # Ok::<(), qoala::error::QoalaError>(())
/// ```
pub fn universal_gate_xy(spec: GateSynthesis) -> Result<Optimisation> {
    universal_gate_xy_with_progress(spec, &mut NoProgress)
}

/// As [`universal_gate_xy`], reporting every iteration to `progress` and
/// stopping when it asks to.
pub fn universal_gate_xy_with_progress(
    spec: GateSynthesis,
    progress: &mut dyn ProgressSink,
) -> Result<Optimisation> {
    let (mut sys, guess) = gate_synthesis_system(&spec)?;
    fmaxnewton_with_progress(&mut sys, &GrapeXyCost, &guess, progress)
}

/// The configured control system and starting waveform a
/// [`universal_gate_xy`] run would use, without running it.
pub fn gate_synthesis_system(spec: &GateSynthesis) -> Result<(ControlSystem, DMatrix<f64>)> {
    let npairs = pair_count(&spec.cartops)?;
    let nspins = spec.omega.len();
    check_shapes(&spec.cartops, nspins, npairs, spec.amplitudes.len())?;

    let mut opts = base_options(
        &spec.omega,
        &spec.amplitudes,
        spec.duration,
        spec.increments,
        &spec.interaction,
        &spec.cartops,
        &spec.tuning,
    )?;
    opts.optimcon_fun = Some(ObjectiveFn::UgateQoala);
    opts.prop_targ = Some(vec![spec.target.clone()]);
    opts.auxmat_method = Some(PropMethod::Taylor);

    let guess = spec
        .init_pulse
        .clone()
        .unwrap_or_else(|| random_pulse(spec.increments, 2 * npairs, spec.seed));

    Ok((optimconset(opts)?, guess))
}

/// Options common to both drivers.
#[allow(clippy::too_many_arguments)]
fn base_options(
    omega: &[f64],
    amplitudes: &[f64],
    duration: f64,
    increments: usize,
    interaction: &CMat,
    cartops: &[Vec<Option<CMat>>],
    tuning: &Tuning,
) -> Result<ControlOptions> {
    let two_pi = 2.0 * std::f64::consts::PI;
    let nspins = omega.len();
    let npairs = pair_count(cartops)?;
    let kctrls = 2 * npairs;

    // Composite control operators: the x operators of every spin a channel
    // drives, summed, then the y operators.
    let mut operators: Vec<Option<CMat>> = vec![None; kctrls];
    let mut spin_control = DMatrix::from_element(nspins, kctrls, false);
    let mut ctrl_axes = Vec::with_capacity(kctrls);
    let mut pwr_levels = Vec::with_capacity(kctrls);

    for k in 0..npairs {
        ctrl_axes.push(Axis::X);
        ctrl_axes.push(Axis::Y);
        let amp = *amplitudes.get(k).ok_or_else(|| {
            QoalaError::Dimension(format!("no amplitude given for control pair {}", k + 1))
        })?;
        pwr_levels.push(two_pi * amp);
        pwr_levels.push(two_pi * amp);

        for (m, row) in cartops.iter().enumerate().take(nspins) {
            for (axis, col) in [(0usize, 2 * k), (1usize, 2 * k + 1)] {
                if let Some(op) = row.get(3 * k + axis).and_then(|o| o.as_ref()) {
                    operators[col] = Some(match &operators[col] {
                        None => op.clone(),
                        Some(acc) => acc.add(op)?,
                    });
                    spin_control[(m, col)] = true;
                }
            }
        }
    }

    let operators: Vec<CMat> = operators
        .into_iter()
        .enumerate()
        .map(|(k, o)| {
            o.ok_or_else(|| {
                QoalaError::MissingField(format!("no operator for control channel {}", k + 1))
            })
        })
        .collect::<Result<_>>()?;

    let drift = DriftOptions {
        interaction: Some(TimeDependent::Constant(interaction.clone())),
        offset: Some(DVector::from_iterator(
            nspins,
            omega.iter().map(|w| two_pi * w),
        )),
        splitset: Some(tuning.splitset.clone()),
        trotterset: tuning.trotterset.clone(),
        ..Default::default()
    };

    Ok(ControlOptions {
        basis: Some(Basis::Sphten),
        space: Some(StateSpace::Liouville),
        nspins: Some(nspins),
        operators: Some(operators),
        pauli_operators: Some(cartops.to_vec()),
        spin_control: Some(spin_control),
        ctrl_axes: Some(ctrl_axes),
        pwr_levels: Some(PowerLevels::PerChannel(pwr_levels)),
        pulse_dur: Some(duration),
        pulse_nsteps: Some(increments),
        penalties: Some(tuning.penalties.clone()),
        prop_cache: Some(tuning.prop_cache),
        prop_zeroed: Some(tuning.prop_zeroed),
        method: Some(OptMethod::Lbfgs),
        n_grads: Some(tuning.n_grads),
        tol_x: Some(tuning.tol_x),
        tol_g: Some(tuning.tol_g),
        max_iter: Some(tuning.max_iter),
        drift_sys: vec![drift],
        output: Some(tuning.output.clone()),
        fidelity_chk: tuning.fidelity_chk.then_some(ObjectiveFn::WaveformFidelity),
        ..Default::default()
    })
}

/// Number of `(x, y)` control pairs implied by the Cartesian operator array.
pub fn pair_count(cartops: &[Vec<Option<CMat>>]) -> Result<usize> {
    let row = cartops
        .first()
        .ok_or_else(|| QoalaError::MissingField("cartops".into()))?;
    if row.len() % 3 != 0 {
        return Err(QoalaError::Dimension(
            "cartops needs three columns (x, y, z) per control pair".into(),
        ));
    }
    Ok(row.len() / 3)
}

fn check_shapes(
    cartops: &[Vec<Option<CMat>>],
    nspins: usize,
    npairs: usize,
    namps: usize,
) -> Result<()> {
    if cartops.len() != nspins {
        return Err(QoalaError::Dimension(format!(
            "cartops has {} rows but there are {nspins} spins",
            cartops.len()
        )));
    }
    for (m, row) in cartops.iter().enumerate() {
        if row.len() != 3 * npairs {
            return Err(QoalaError::Dimension(format!(
                "cartops row {m} has {} entries, expected {}",
                row.len(),
                3 * npairs
            )));
        }
    }
    if namps != npairs {
        return Err(QoalaError::Dimension(format!(
            "{namps} amplitudes for {npairs} control pairs"
        )));
    }
    Ok(())
}

/// Normalise a state, as both drivers do before optimising.
fn normalise(x: &CDense) -> CDense {
    let n = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if n == 0.0 {
        x.clone()
    } else {
        x.map(|z| z / Complex64::new(n, 0.0))
    }
}

/// A random starting waveform, uniform in `[-1, 1]`.
pub fn random_pulse(nsteps: usize, nchannels: usize, seed: Option<u64>) -> DMatrix<f64> {
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    DMatrix::from_fn(nsteps, nchannels, |_, _| rng.gen_range(-1.0..1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_pulses_are_reproducible_and_bounded() {
        let a = random_pulse(5, 4, Some(7));
        let b = random_pulse(5, 4, Some(7));
        assert_eq!(a, b);
        assert!(a.iter().all(|v| *v >= -1.0 && *v < 1.0));
        let c = random_pulse(5, 4, Some(8));
        assert_ne!(a, c);
    }

    #[test]
    fn pair_count_reads_the_operator_layout() {
        let ops: Vec<Vec<Option<CMat>>> = vec![vec![None; 6], vec![None; 6]];
        assert_eq!(pair_count(&ops).unwrap(), 2);
        let bad: Vec<Vec<Option<CMat>>> = vec![vec![None; 5]];
        assert!(pair_count(&bad).is_err());
    }

    #[test]
    fn normalise_gives_unit_length() {
        let x = CDense::from_column_slice(
            3,
            1,
            &[
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 4.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let n = normalise(&x);
        let len: f64 = n.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        assert!((len - 1.0).abs() < 1e-15);
    }
}
