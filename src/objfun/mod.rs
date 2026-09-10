//! GRAPE objective functions.
//!
//! Two families, matching `kernel/obj_fun/`:
//!
//! * [`qoala`] - the adaptive split-operator method.  Single-spin propagators
//!   come from the Euler-Rodrigues formula and the interaction from a fixed
//!   set of precomputed propagators, and the splitting order and Trotter
//!   number are chosen on the fly so that the estimate is only ever as
//!   accurate as the optimiser currently needs.
//! * [`auxmat`] - the exact auxiliary-matrix method, which exponentiates the
//!   full Hamiltonian at every time step and gets the directional derivative
//!   from the block matrix `[[L, C], [0, L]]`.
//!
//! Both return the fidelity and, on request, its gradient with respect to
//! every control amplitude at every time slice.

pub mod auxmat;
pub mod qoala;

use crate::config::{ControlSystem, DriftSystem};
use crate::error::{QoalaError, Result};
use crate::linalg::{real_trace_adjoint, CDense};
use crate::types::ObjectiveFn;
use nalgebra::{DMatrix, DVector};

/// How much of the objective to compute; replaces MATLAB's `nargout` switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvalOrder {
    /// Fidelity only.
    Value,
    /// Fidelity and gradient.
    Gradient,
}

/// Diagnostics an objective function hands back to the optimiser.
///
/// This is the MATLAB `data` struct, whose fields the optimiser probes with
/// `isfield`.
#[derive(Debug, Clone, Default)]
pub struct TrajData {
    /// Final state or effective propagator.
    pub final_state: Option<CDense>,
    /// Wall-clock seconds spent inside the objective function.
    pub timer: Option<f64>,
    /// Updated drift system, when the adaptive step changed it.
    pub drift_sys: Option<DriftSystem>,
    /// Extra fidelity evaluations spent walking the adaptivity ladder.
    pub f_adapt_count: Option<usize>,
    /// Two-norm of the gradient, fed back into the adaptivity tolerance.
    pub grad_norm: Option<f64>,
}

/// Result of one objective-function call.
#[derive(Debug, Clone)]
pub struct ObjectiveResult {
    /// Diagnostics.
    pub data: TrajData,
    /// Fidelity.
    pub fidelity: f64,
    /// Gradient, `nsteps x nchannels`, when one was requested.
    pub grad: Option<DMatrix<f64>>,
}

/// Fidelity normalisation: point-to-point overlaps are already normalised,
/// gate fidelities are divided by the space dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FidelityKind {
    /// `Re<target|rho(T)>`.
    State,
    /// `Re tr(target^dagger U(T)) / dim`.
    Gate,
}

impl FidelityKind {
    /// Divisor applied to the raw overlap.
    pub fn norm(self, dim: usize) -> f64 {
        match self {
            FidelityKind::State => 1.0,
            FidelityKind::Gate => dim as f64,
        }
    }
    /// Overlap of a target with a final state or propagator.
    pub fn overlap(self, target: &CDense, x: &CDense, dim: usize) -> f64 {
        real_trace_adjoint(target, x) / self.norm(dim)
    }
    /// Whether the adaptive step normalises the compared states.  The
    /// point-to-point objective divides by the state norm before comparing;
    /// the gate objective compares propagators directly.
    pub fn normalise_for_adaptivity(self) -> bool {
        self == FidelityKind::State
    }
}

/// Evaluate one of the objective functions.
///
/// `ctrls` are the control operators the objective expects: single-spin
/// Cartesian operators for the QOALA family, composite control operators for
/// the auxiliary-matrix family.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    which: ObjectiveFn,
    sys: &ControlSystem,
    drift: &DriftSystem,
    ctrls: &Controls,
    amps: &[f64],
    wf: &DMatrix<f64>,
    init: &CDense,
    targ: &CDense,
    order: EvalOrder,
) -> Result<ObjectiveResult> {
    match which {
        ObjectiveFn::StateQoala => qoala::objective(
            FidelityKind::State,
            sys,
            drift,
            ctrls.pauli()?,
            amps,
            wf,
            init,
            targ,
            order,
        ),
        ObjectiveFn::UgateQoala => qoala::objective(
            FidelityKind::Gate,
            sys,
            drift,
            ctrls.pauli()?,
            amps,
            wf,
            init,
            targ,
            order,
        ),
        ObjectiveFn::StateAuxmat => auxmat::objective(
            FidelityKind::State,
            sys,
            drift,
            ctrls.composite()?,
            amps,
            wf,
            init,
            targ,
            order,
        ),
        ObjectiveFn::UgateAuxmat => auxmat::objective(
            FidelityKind::Gate,
            sys,
            drift,
            ctrls.composite()?,
            amps,
            wf,
            init,
            targ,
            order,
        ),
        ObjectiveFn::WaveformFidelity => {
            let kind = match sys.ctrl_type {
                crate::types::CtrlType::UniversalRotation => FidelityKind::Gate,
                crate::types::CtrlType::PointToPoint => FidelityKind::State,
            };
            let (data, fidelity) = crate::waveform::waveform_fidelity(
                kind,
                sys,
                drift,
                ctrls.composite()?,
                amps,
                wf,
                init,
                targ,
            )?;
            if order == EvalOrder::Gradient {
                return Err(QoalaError::NotImplemented(
                    "waveform_fidelity has no gradient; it is a fidelity check only".into(),
                ));
            }
            Ok(ObjectiveResult {
                data,
                fidelity,
                grad: None,
            })
        }
    }
}

/// The two shapes of control-operator array the objective functions want.
#[derive(Debug, Clone)]
pub enum Controls<'a> {
    /// Composite-space control operators, one per channel.
    Composite(&'a [crate::linalg::CMat]),
    /// Single-spin Cartesian operators, `nspins x 3*npairs`.
    Pauli(&'a [Vec<Option<crate::linalg::CMat>>]),
}

impl<'a> Controls<'a> {
    fn composite(&self) -> Result<&'a [crate::linalg::CMat]> {
        match self {
            Controls::Composite(c) => Ok(c),
            Controls::Pauli(_) => Err(QoalaError::BadValue(
                "this objective function needs composite control operators".into(),
            )),
        }
    }
    fn pauli(&self) -> Result<&'a [Vec<Option<crate::linalg::CMat>>]> {
        match self {
            Controls::Pauli(p) => Ok(p),
            Controls::Composite(_) => Err(QoalaError::BadValue(
                "this objective function needs single-spin Cartesian operators".into(),
            )),
        }
    }
}

/// Scale a waveform by the per-channel power levels.
///
/// `wf` is `nsteps x nchannels` and `amps` has one entry per channel.
pub fn scale_waveform(wf: &DMatrix<f64>, amps: &[f64]) -> Result<DMatrix<f64>> {
    if amps.len() != wf.ncols() {
        return Err(QoalaError::Dimension(format!(
            "{} power levels for {} control channels",
            amps.len(),
            wf.ncols()
        )));
    }
    let mut out = wf.clone();
    for (k, a) in amps.iter().enumerate() {
        for r in 0..out.nrows() {
            out[(r, k)] *= a;
        }
    }
    Ok(out)
}

/// Build the rotation axes and time steps the Rodrigues formula needs.
///
/// The layout matches the MATLAB exactly: entries run spin-major, then time
/// slice, then splitting stage, so that entry `m` of the returned arrays
/// corresponds to `((spin * nsteps) + step) * nsplits + stage`.
pub struct RotationGrid {
    /// `M x 3` rotation axes in `(x, y, z)`.
    pub axes: DMatrix<f64>,
    /// `M` time steps.
    pub dt: DVector<f64>,
}

/// Assemble the rotation grid for a scaled waveform.
pub fn rotation_grid(
    sys: &ControlSystem,
    drift: &DriftSystem,
    scaled_wf: &DMatrix<f64>,
    c_spn: &[f64],
    trotter: usize,
) -> Result<RotationGrid> {
    let nsteps = scaled_wf.nrows();
    let nspins = sys.nspins;
    let nsplits = c_spn.len();
    let offsets = drift
        .offset
        .as_ref()
        .ok_or_else(|| QoalaError::MissingField("drift offsets".into()))?;
    if offsets.len() < nspins {
        return Err(QoalaError::Dimension(format!(
            "{} offsets for {nspins} spins",
            offsets.len()
        )));
    }

    let m = nsteps * nspins * nsplits;
    let mut axes = DMatrix::<f64>::zeros(m, 3);
    let mut dt = DVector::<f64>::zeros(m);

    for s in 0..nspins {
        // Columns of the waveform that drive this spin, in channel order.
        let cols: Vec<usize> = (0..sys.spin_control.ncols())
            .filter(|&c| sys.spin_control[(s, c)])
            .collect();
        let (cx, cy) = match cols.len() {
            0 => (None, None),
            2 => (Some(cols[0]), Some(cols[1])),
            n => {
                return Err(QoalaError::BadValue(format!(
                    "spin {s} is driven by {n} control channels; the split-operator objective expects an x/y pair"
                )))
            }
        };
        for n in 0..nsteps {
            let x = cx.map(|c| scaled_wf[(n, c)]).unwrap_or(0.0);
            let y = cy.map(|c| scaled_wf[(n, c)]).unwrap_or(0.0);
            let z = offsets[s];
            let base_dt = sys.pulse_dt[n] / trotter as f64;
            for (i, cs) in c_spn.iter().enumerate() {
                let idx = (s * nsteps + n) * nsplits + i;
                axes[(idx, 0)] = x;
                axes[(idx, 1)] = y;
                axes[(idx, 2)] = z;
                dt[idx] = base_dt * cs;
            }
        }
    }
    Ok(RotationGrid { axes, dt })
}

/// Dimension of a single spin's propagator block, inferred from the index
/// table.
pub fn single_spin_dim(prop_ind: &[(usize, usize)]) -> Result<usize> {
    prop_ind
        .iter()
        .map(|&(r, c)| r.max(c))
        .max()
        .map(|m| m + 1)
        .ok_or_else(|| QoalaError::BadValue("empty propagator index table".into()))
}
