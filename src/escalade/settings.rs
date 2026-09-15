//! The problem description, with the MATLAB defaults, and its resolution into
//! the settings a run works from.
//!
//! Port of `main/readDefaults.m` and `main/readSettings.m`.  The MATLAB takes
//! name-value pairs and an `inputParser`; here [`Escalade`] is a struct with
//! a [`Default`] holding the same defaults, and [`Escalade::resolve`] does the
//! parser's consistency checks and fills in what was left out.

use super::propagators::{spin_operator, Op2};
use crate::error::{QoalaError, Result};
use crate::{bad, missing};
use nalgebra::DMatrix;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Initial or target states for the spins.
#[derive(Debug, Clone, PartialEq)]
pub enum States {
    /// The same state for every spin: the MATLAB `singleinit` and
    /// `singletarget`.  Divided by `sqrt(2 nspins)` on resolution, so that a
    /// unit-norm operator gives a fidelity of at most one.
    Single(Op2),
    /// One state per spin, used as given: `fullinit` and `fulltarget`.  Their
    /// count sets the number of spins.
    PerSpin(Vec<Op2>),
}

/// Magnetisation along `(x, y, z)`, normalised to unit spectral norm.
///
/// `magnetisation(0.0, 0.0, 1.0)` is the default initial state and
/// `magnetisation(0.0, -1.0, 0.0)` the default target, as `readDefaults`
/// builds them.
pub fn magnetisation(x: f64, y: f64, z: f64) -> Op2 {
    let norm = (x * x + y * y + z * z).sqrt();
    if norm == 0.0 {
        return Op2::zeros();
    }
    spin_operator([2.0 * x / norm, 2.0 * y / norm, 2.0 * z / norm])
}

/// An ESCALADE problem: a band of uncoupled spins, one x/y pulse to act on
/// all of them.
///
/// Field names follow the MATLAB parameters.  Units are SI throughout, with
/// frequencies in Hz.
#[derive(Debug, Clone, PartialEq)]
pub struct Escalade {
    /// Number of spins spread across the bandwidth (`nspins`).  Ignored when
    /// `offsets`, or per-spin states, say otherwise.
    pub nspins: usize,
    /// Number of pulse points (`np_pulse`).  Ignored when `start` is given.
    pub np_pulse: usize,
    /// Pulse duration in seconds (`tau_p`).
    pub tau_p: f64,
    /// Nominal radio-frequency amplitude in Hz (`rf`).  More than one value
    /// optimises across that spread of fields, which is how a pulse is made
    /// insensitive to B1 inhomogeneity.
    pub rf: Vec<f64>,
    /// Relative weight of each field in `rf` (`rfweights`); equal when absent.
    pub rf_weights: Option<Vec<f64>>,
    /// Bandwidth in Hz the spins are spread over (`sw`).
    pub sw: f64,
    /// Explicit resonance offsets in Hz, overruling `sw` and `nspins`.  The
    /// MATLAB `Om` takes these in rad/s.
    pub offsets: Option<Vec<f64>>,
    /// Initial states.
    pub initial: States,
    /// Target states.
    pub target: States,
    /// Starting pulse, `np x 2` (`F0`, `G0`); random in `[0, 1)` when absent.
    pub start: Option<DMatrix<f64>>,
    /// Seed for the random starting pulse.
    pub seed: Option<u64>,
    /// Newton trust region with the analytic Hessian (`usehess`) rather than
    /// L-BFGS on the gradient alone.
    pub use_hessian: bool,
    /// Iteration budget (`maxiter`).
    pub max_iter: usize,
    /// Stop once the fidelity passes this (`targetfidelity`, which the MATLAB
    /// writes negated because `fmincon` minimises).
    pub target_fidelity: f64,
    /// Weight of the amplitude penalty keeping `sqrt(f^2 + g^2) <= 1`.
    ///
    /// The penalty is soft.  A heavier weight holds the amplitude closer to
    /// the limit and makes the problem stiffer; the default typically leaves
    /// less than one percent of overshoot while converging as fast as a
    /// lighter one.
    pub amplitude_weight: f64,
}

impl Default for Escalade {
    /// `readDefaults`.
    fn default() -> Self {
        Escalade {
            nspins: 51,
            np_pulse: 40,
            tau_p: 80e-6,
            rf: vec![20000.0],
            rf_weights: None,
            sw: 20000.0,
            offsets: None,
            initial: States::Single(magnetisation(0.0, 0.0, 1.0)),
            target: States::Single(magnetisation(0.0, -1.0, 0.0)),
            start: None,
            seed: None,
            use_hessian: false,
            max_iter: 1000,
            target_fidelity: 0.99,
            amplitude_weight: 100.0,
        }
    }
}

/// Everything a run needs, resolved and checked: the MATLAB `settings`
/// struct.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// Number of spins.
    pub nspins: usize,
    /// Number of pulse points.
    pub np_pulse: usize,
    /// Pulse duration in seconds.
    pub tau_p: f64,
    /// Slice width in seconds (`settings.t`).
    pub dt: f64,
    /// Resonance offset of each spin in rad/s (`settings.Om`).
    pub offsets: Vec<f64>,
    /// Radio-frequency amplitudes in Hz.
    pub rf: Vec<f64>,
    /// Weight of each amplitude, summing to one.
    pub rf_weights: Vec<f64>,
    /// Initial state of each spin (`fullinit`).
    pub initial: Vec<Op2>,
    /// Target state of each spin (`fulltarget`).
    pub target: Vec<Op2>,
    /// Starting pulse, `np_pulse x 2`.
    pub start: DMatrix<f64>,
    /// Use the analytic Hessian.
    pub use_hessian: bool,
    /// Iteration budget.
    pub max_iter: usize,
    /// Fidelity at which to stop.
    pub target_fidelity: f64,
    /// Amplitude penalty weight.
    pub amplitude_weight: f64,
}

impl Escalade {
    /// `readSettings`: check the description and fill in everything derived.
    pub fn resolve(&self) -> Result<Settings> {
        // Per-spin states and explicit offsets each dictate the spin count,
        // and have to agree when more than one does.
        let dictated: Vec<(&str, usize)> = [
            ("initial states", per_spin_len(&self.initial)),
            ("target states", per_spin_len(&self.target)),
            ("offsets", self.offsets.as_ref().map(Vec::len)),
        ]
        .into_iter()
        .filter_map(|(name, n)| n.map(|n| (name, n)))
        .collect();
        let nspins = match dictated.first() {
            Some(&(_, n)) => {
                if let Some((name, m)) = dictated.iter().find(|(_, m)| *m != n) {
                    return Err(bad!("{name} give {m} spins but {} give {n}", dictated[0].0));
                }
                n
            }
            None => self.nspins,
        };
        if nspins == 0 {
            return Err(bad!("at least one spin is needed"));
        }

        let initial = expand(&self.initial, nspins, "initial")?;
        let target = expand(&self.target, nspins, "target")?;

        let two_pi = 2.0 * std::f64::consts::PI;
        let offsets: Vec<f64> = match &self.offsets {
            Some(hz) => hz.iter().map(|w| two_pi * w).collect(),
            None => linspace(-self.sw / 2.0, self.sw / 2.0, nspins)
                .into_iter()
                .map(|w| two_pi * w)
                .collect(),
        };
        if offsets.iter().any(|w| !w.is_finite()) {
            return Err(bad!("offsets must be finite"));
        }

        if self.rf.is_empty() {
            return Err(missing!("at least one radio-frequency amplitude"));
        }
        if self.rf.iter().any(|a| !a.is_finite()) {
            return Err(bad!("radio-frequency amplitudes must be finite"));
        }
        let weights = match &self.rf_weights {
            None => vec![1.0; self.rf.len()],
            Some(w) if w.len() == self.rf.len() => w.clone(),
            Some(w) => {
                return Err(bad!(
                    "{} weights for {} radio-frequency amplitudes",
                    w.len(),
                    self.rf.len()
                ))
            }
        };
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total == 0.0 {
            return Err(bad!("radio-frequency weights must have a non-zero sum"));
        }
        let rf_weights = weights.iter().map(|w| w / total).collect();

        let start = match &self.start {
            Some(pulse) => {
                if pulse.ncols() != 2 || pulse.nrows() == 0 {
                    return Err(bad!(
                        "the starting pulse must be np x 2, got {}x{}",
                        pulse.nrows(),
                        pulse.ncols()
                    ));
                }
                if pulse.iter().any(|v| !v.is_finite()) {
                    return Err(bad!("the starting pulse must be finite"));
                }
                pulse.clone()
            }
            None => {
                if self.np_pulse == 0 {
                    return Err(bad!("the pulse needs at least one point"));
                }
                random_start(self.np_pulse, self.seed)
            }
        };
        let np_pulse = start.nrows();

        if !(self.tau_p > 0.0 && self.tau_p.is_finite()) {
            return Err(bad!("the pulse duration must be positive"));
        }
        if !(self.amplitude_weight >= 0.0 && self.amplitude_weight.is_finite()) {
            return Err(bad!("the amplitude penalty weight must be non-negative"));
        }

        Ok(Settings {
            nspins,
            np_pulse,
            tau_p: self.tau_p,
            dt: self.tau_p / np_pulse as f64,
            offsets,
            rf: self.rf.clone(),
            rf_weights,
            initial,
            target,
            start,
            use_hessian: self.use_hessian,
            max_iter: self.max_iter,
            target_fidelity: self.target_fidelity,
            amplitude_weight: self.amplitude_weight,
        })
    }
}

fn per_spin_len(states: &States) -> Option<usize> {
    match states {
        States::Single(_) => None,
        States::PerSpin(v) => Some(v.len()),
    }
}

fn expand(states: &States, nspins: usize, what: &str) -> Result<Vec<Op2>> {
    let out = match states {
        States::Single(op) => {
            let scale = num_complex::Complex64::new(1.0 / (2.0 * nspins as f64).sqrt(), 0.0);
            vec![op * scale; nspins]
        }
        States::PerSpin(ops) => ops.clone(),
    };
    // `readSettings` accepts only Hermitian 2x2 matrices.
    if out.iter().any(|op| (op.adjoint() - op).norm() >= 1e-10) {
        return Err(QoalaError::BadValue(format!(
            "{what} states must be Hermitian"
        )));
    }
    Ok(out)
}

/// MATLAB's `linspace`, including its answer for a single point.
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![b],
        _ => (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect(),
    }
}

/// `rand(np, 1)` for each quadrature, as `readSettings` draws them.
fn random_start(np: usize, seed: Option<u64>) -> DMatrix<f64> {
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    let mut pulse = DMatrix::zeros(np, 2);
    for c in 0..2 {
        for r in 0..np {
            pulse[(r, c)] = rng.gen::<f64>();
        }
    }
    pulse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_resolve_as_the_matlab_does() {
        let s = Escalade {
            seed: Some(1),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        assert_eq!((s.nspins, s.np_pulse), (51, 40));
        assert!((s.dt - 2e-6).abs() < 1e-18);
        let two_pi = 2.0 * std::f64::consts::PI;
        assert!((s.offsets[0] + two_pi * 10000.0).abs() < 1e-9);
        assert!((s.offsets[50] - two_pi * 10000.0).abs() < 1e-9);
        assert_eq!(s.rf_weights, vec![1.0]);
        assert!(s.start.iter().all(|v| (0.0..1.0).contains(v)));

        // A unit state spread over every spin gives a maximum fidelity of one.
        let total: f64 = s.initial.iter().map(|r| (r.adjoint() * r).trace().re).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn explicit_offsets_set_the_spin_count() {
        let s = Escalade {
            offsets: Some(vec![-100.0, 0.0, 100.0]),
            seed: Some(2),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        assert_eq!(s.nspins, 3);
        assert_eq!(s.initial.len(), 3);
    }

    #[test]
    fn a_starting_pulse_that_is_not_finite_is_refused() {
        let mut spec = Escalade {
            start: Some(DMatrix::from_element(4, 2, 0.1)),
            seed: Some(4),
            ..Default::default()
        };
        assert!(spec.resolve().is_ok());
        spec.start = Some(DMatrix::from_fn(
            4,
            2,
            |r, _| if r == 2 { f64::NAN } else { 0.1 },
        ));
        assert!(spec.resolve().is_err());
    }

    #[test]
    fn disagreeing_spin_counts_are_refused() {
        let spec = Escalade {
            offsets: Some(vec![0.0, 1.0]),
            target: States::PerSpin(vec![magnetisation(1.0, 0.0, 0.0); 3]),
            ..Default::default()
        };
        assert!(spec.resolve().is_err());
    }

    #[test]
    fn weights_are_normalised_and_checked() {
        let mut spec = Escalade {
            rf: vec![1.0, 2.0],
            rf_weights: Some(vec![1.0, 3.0]),
            seed: Some(3),
            ..Default::default()
        };
        assert_eq!(spec.resolve().unwrap().rf_weights, vec![0.25, 0.75]);
        spec.rf_weights = Some(vec![1.0]);
        assert!(spec.resolve().is_err());
    }

    #[test]
    fn non_hermitian_states_are_refused() {
        let mut op = magnetisation(0.0, 0.0, 1.0);
        op[(0, 1)] = num_complex::Complex64::new(0.3, 0.0);
        let spec = Escalade {
            initial: States::Single(op),
            ..Default::default()
        };
        assert!(spec.resolve().is_err());
    }

    #[test]
    fn the_starting_pulse_sets_the_point_count() {
        let s = Escalade {
            start: Some(DMatrix::from_element(7, 2, 0.1)),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        assert_eq!(s.np_pulse, 7);
        assert!((s.dt - 80e-6 / 7.0).abs() < 1e-18);
    }
}
