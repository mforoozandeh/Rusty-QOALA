//! Independent fidelity check by direct propagation.
//!
//! Port of `utilities/waveform_fidelity.m`.  No splitting, no derivatives:
//! just push the initial state through the pulse one exact exponential at a
//! time and take the overlap.  QOALA uses it two ways - as the `fidelity_chk`
//! that reports the true fidelity next to the approximate one at every
//! iteration, and as the reference for the `exact` adaptivity method.

use crate::config::{ControlSystem, DriftSystem};
use crate::error::{QoalaError, Result};
use crate::linalg::{real_trace_adjoint, CDense, CMat};
use crate::objfun::{scale_waveform, FidelityKind, TrajData};
use crate::propagate::propagate_state;
use nalgebra::DMatrix;
use num_complex::Complex64;

/// Element cutoff used when propagating the check, matching the MATLAB's
/// hard-coded `prop_chop = 2*eps`.
const PROP_CHOP: f64 = 2.0 * f64::EPSILON;

/// Propagate a waveform exactly and return the fidelity.
#[allow(clippy::too_many_arguments)]
pub fn waveform_fidelity(
    kind: FidelityKind,
    sys: &ControlSystem,
    drift: &DriftSystem,
    ctrls: &[CMat],
    amps: &[f64],
    wf: &DMatrix<f64>,
    init: &CDense,
    targ: &CDense,
) -> Result<(TrajData, f64)> {
    let nsteps = wf.nrows();
    let scaled = scale_waveform(wf, amps)?;
    let drift_term = drift.drift.as_ref().ok_or_else(|| {
        QoalaError::MissingField(
            "waveform_fidelity needs a full drift Hamiltonian: supply interaction and/or singlespin"
                .into(),
        )
    })?;

    let mut fwd = init.clone();
    for n in 0..nsteps {
        let mut l = drift_term.at(n)?.clone();
        for (k, c) in ctrls.iter().enumerate() {
            l = l.add(&c.scale(Complex64::new(scaled[(n, k)], 0.0)))?;
        }
        fwd = propagate_state(
            sys.space,
            &l,
            Complex64::new(sys.pulse_dt[n], 0.0),
            sys.step_method,
            PROP_CHOP,
            &fwd,
        )?;
    }

    let fidelity = real_trace_adjoint(targ, &fwd) / kind.norm(sys.dim);
    Ok((
        TrajData {
            final_state: Some(fwd),
            ..Default::default()
        },
        fidelity,
    ))
}

/// Standalone fidelity check, without building a whole control system.
///
/// This is the shape the example scripts use `waveform_fidelity` in: a drift
/// Hamiltonian, a set of composite control operators, a waveform and a time
/// grid.  `amps` are the per-channel power levels in rad/s and multiply the
/// waveform, exactly as in the MATLAB.
#[allow(clippy::too_many_arguments)]
pub fn check_fidelity(
    space: crate::types::StateSpace,
    kind: FidelityKind,
    dim: usize,
    drift: &CMat,
    ctrls: &[CMat],
    amps: &[f64],
    wf: &DMatrix<f64>,
    dt: &nalgebra::DVector<f64>,
    method: crate::types::PropMethod,
    init: &CDense,
    targ: &CDense,
) -> Result<f64> {
    let nsteps = wf.nrows();
    if dt.len() != nsteps {
        return Err(QoalaError::Dimension(format!(
            "{} time steps for a waveform with {nsteps} slices",
            dt.len()
        )));
    }
    let scaled = scale_waveform(wf, amps)?;
    let mut fwd = init.clone();
    for n in 0..nsteps {
        let mut l = drift.clone();
        for (k, c) in ctrls.iter().enumerate() {
            l = l.add(&c.scale(Complex64::new(scaled[(n, k)], 0.0)))?;
        }
        fwd = propagate_state(
            space,
            &l,
            Complex64::new(dt[n], 0.0),
            method,
            PROP_CHOP,
            &fwd,
        )?;
    }
    Ok(real_trace_adjoint(targ, &fwd) / kind.norm(dim))
}
