//! Shared fixtures for the integration tests.

#![allow(dead_code)]

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use qoala::config::{
    optimconset, ControlOptions, ControlSystem, DriftOptions, PowerLevels, TimeDependent,
};
use qoala::error::Result;
use qoala::linalg::{CDense, CMat};
use qoala::report::Reporter;
use qoala::spinops;
use qoala::types::*;

/// How a test system is set up.
pub struct Fixture {
    pub nspins: usize,
    pub coupling_hz: f64,
    pub offsets_hz: Vec<f64>,
    pub amplitude_hz: f64,
    pub duration: f64,
    pub nsteps: usize,
    pub split_order: usize,
    pub trotter: usize,
    pub splitset: Option<Vec<usize>>,
    pub gate: bool,
    pub objective: ObjectiveFn,
    pub penalties: Vec<Penalty>,
}

impl Default for Fixture {
    fn default() -> Self {
        Fixture {
            nspins: 2,
            coupling_hz: 140.0,
            offsets_hz: vec![0.0, 0.0],
            amplitude_hz: 1000.0,
            duration: 4e-3,
            nsteps: 6,
            split_order: 2,
            trotter: 1,
            splitset: None,
            gate: false,
            objective: ObjectiveFn::StateQoala,
            penalties: vec![Penalty::None],
        }
    }
}

impl Fixture {
    /// The interaction Hamiltonian for this fixture: a linear zz chain.
    pub fn interaction(&self) -> Result<CMat> {
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut h = spinops::zz_coupling(self.nspins, 0, 1)?
            .scale(Complex64::new(two_pi * self.coupling_hz, 0.0));
        for i in 1..self.nspins.saturating_sub(1) {
            h = h.add(
                &spinops::zz_coupling(self.nspins, i, i + 1)?
                    .scale(Complex64::new(two_pi * -160.0, 0.0)),
            )?;
        }
        Ok(h)
    }

    /// The full drift, interaction plus offsets, for the exact objective.
    pub fn drift(&self) -> Result<CMat> {
        let two_pi = 2.0 * std::f64::consts::PI;
        let ops = spinops::cartesian_operators(self.nspins);
        let mut h = self.interaction()?;
        for (s, w) in self.offsets_hz.iter().enumerate() {
            if *w != 0.0 {
                h = h.add(&ops[s].2.scale(Complex64::new(two_pi * w, 0.0)))?;
            }
        }
        Ok(h)
    }

    /// Composite control operators, x and y on each spin.
    pub fn controls(&self) -> Vec<CMat> {
        spinops::cartesian_operators(self.nspins)
            .into_iter()
            .flat_map(|(x, y, _)| [x, y])
            .collect()
    }

    /// Build the control system.
    pub fn build(&self) -> Result<ControlSystem> {
        let two_pi = 2.0 * std::f64::consts::PI;
        let dim = 4usize.pow(self.nspins as u32);
        let cartops = spinops::one_pair_per_spin(self.nspins);
        let kctrls = 2 * self.nspins;

        let mut spin_control = DMatrix::from_element(self.nspins, kctrls, false);
        for s in 0..self.nspins {
            spin_control[(s, 2 * s)] = true;
            spin_control[(s, 2 * s + 1)] = true;
        }

        let ctrl_axes: Vec<Axis> = (0..self.nspins).flat_map(|_| [Axis::X, Axis::Y]).collect();
        let pwr: Vec<f64> = vec![two_pi * self.amplitude_hz; kctrls];

        let drift = DriftOptions {
            interaction: Some(TimeDependent::Constant(self.interaction()?)),
            singlespin: None,
            drift: Some(TimeDependent::Constant(self.drift()?)),
            offset: Some(DVector::from_iterator(
                self.nspins,
                self.offsets_hz.iter().map(|w| two_pi * w),
            )),
            split_order: Some(self.split_order),
            trotter_number: Some(self.trotter),
            splitset: self.splitset.clone(),
            ..Default::default()
        };

        let mut opts = ControlOptions {
            basis: Some(Basis::Sphten),
            space: Some(StateSpace::Liouville),
            nspins: Some(self.nspins),
            operators: Some(self.controls()),
            pauli_operators: Some(cartops),
            spin_control: Some(spin_control),
            ctrl_axes: Some(ctrl_axes),
            pwr_levels: Some(PowerLevels::PerChannel(pwr)),
            pulse_dur: Some(self.duration),
            pulse_nsteps: Some(self.nsteps),
            penalties: Some(self.penalties.clone()),
            p_weights: Some(vec![1.0; self.penalties.len()]),
            prop_cache: Some(PropCache::Carry),
            prop_zeroed: Some(1e-14),
            optimcon_fun: Some(self.objective),
            method: Some(OptMethod::Lbfgs),
            max_iter: Some(20),
            output: Some(Reporter::silent()),
            drift_sys: vec![drift],
            ..Default::default()
        };

        if self.gate {
            opts.prop_targ = Some(vec![gate_target(dim)]);
        } else {
            opts.rho_init = Some(vec![normalise(&spinops::z_state(self.nspins, 0))]);
            opts.rho_targ = Some(vec![normalise(&spinops::z_state(
                self.nspins,
                self.nspins - 1,
            ))]);
        }
        optimconset(opts)
    }
}

/// A simple unitary target for gate tests: a cyclic permutation.
pub fn gate_target(dim: usize) -> CDense {
    let mut m = CDense::zeros(dim, dim);
    for i in 0..dim {
        m[((i + 1) % dim, i)] = Complex64::new(1.0, 0.0);
    }
    m
}

/// Normalise a state.
pub fn normalise(x: &CDense) -> CDense {
    let n = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if n == 0.0 {
        x.clone()
    } else {
        x.map(|z| z / Complex64::new(n, 0.0))
    }
}

/// A deterministic test waveform.
pub fn test_waveform(nsteps: usize, kctrls: usize) -> DMatrix<f64> {
    DMatrix::from_fn(nsteps, kctrls, |n, k| {
        let t = n as f64 / nsteps as f64;
        0.6 * ((3.0 + k as f64) * t + 0.4 * k as f64).sin()
    })
}
