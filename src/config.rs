//! Control-system configuration: options in, a parsed system out.
//!
//! Port of `kernel/optimconset.m`.  The MATLAB takes a struct of loosely typed
//! fields, strips them one by one, fills in defaults, prints a long
//! `WARNING`/`RESOLVE` commentary and hands back a `ctrl_sys` struct; anything
//! left over is reported as unrecognised.  Here [`ControlOptions`] is the input
//! and [`ControlSystem`] the output, so "unrecognised option" is a compile
//! error rather than a run-time warning, but the defaults, the resolutions and
//! the printed report all follow the original.

use crate::error::{QoalaError, Result};
use crate::linalg::{CDense, CMat};
use crate::penalty::PenaltyContext;
use crate::prop_index::propagator_ind;
use crate::report::{fmt_g, fmt_g_signed, int2str, pad, Reporter};
use crate::splittings::SplitSet;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::path::{Path, PathBuf};

/// A Hamiltonian term that may be constant or piecewise-constant in time.
#[derive(Debug, Clone)]
pub enum TimeDependent {
    /// The same operator at every time slice.
    Constant(CMat),
    /// One operator per time slice.
    PerStep(Vec<CMat>),
}

impl TimeDependent {
    /// The operator at time slice `n`.
    pub fn at(&self, n: usize) -> Result<&CMat> {
        match self {
            TimeDependent::Constant(m) => Ok(m),
            TimeDependent::PerStep(v) => v.get(n).ok_or_else(|| {
                QoalaError::Dimension(format!("no time-dependent operator for step {n}"))
            }),
        }
    }
    /// True when the term does not change with time.
    pub fn is_constant(&self) -> bool {
        matches!(self, TimeDependent::Constant(_))
    }
    /// Every stored operator.
    pub fn all(&self) -> Vec<&CMat> {
        match self {
            TimeDependent::Constant(m) => vec![m],
            TimeDependent::PerStep(v) => v.iter().collect(),
        }
    }
    /// True when every stored operator is numerically zero.
    pub fn is_zero(&self) -> bool {
        self.all().iter().all(|m| m.nnz() == 0)
    }
}

/// How the control power levels were supplied.
#[derive(Debug, Clone)]
pub enum PowerLevels {
    /// One value shared by every channel.
    Uniform(f64),
    /// One value per channel.
    PerChannel(Vec<f64>),
    /// A list of values per channel; every combination becomes an ensemble
    /// member, as MATLAB's cell-array form does.
    Ensemble(Vec<Vec<f64>>),
}

/// The approximation flag, which selects a cheaper propagator index table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approximation {
    /// Build the index table as if the spins were uncoupled.
    NoCoupling,
}

/// Drift terms as supplied by the user.
#[derive(Debug, Clone, Default)]
pub struct DriftOptions {
    /// Complete drift Hamiltonian, used by the exact objective functions.
    pub drift: Option<TimeDependent>,
    /// Interaction (spin-spin) part of the drift.
    pub interaction: Option<TimeDependent>,
    /// Single-spin part of the drift.
    pub singlespin: Option<CMat>,
    /// Resonance offsets in rad/s, one per spin; required by QOALA.
    pub offset: Option<DVector<f64>>,
    /// Splitting method, taking precedence over `split_order`.
    pub split_method: Option<SplitMethod>,
    /// Splitting order.
    pub split_order: Option<usize>,
    /// Trotter number.
    pub trotter_number: Option<usize>,
    /// Splitting orders the adaptive step may choose from.
    pub splitset: Option<Vec<usize>>,
    /// Trotter numbers the adaptive step may choose from.
    pub trotterset: Option<Vec<usize>>,
    /// Explicit `(trotter, order)` ladder, overridden by the two sets above.
    pub adaptset: Option<Vec<(usize, usize)>>,
    /// `[max, min]` fidelity error thresholds for adaptivity.
    pub adapt_tols: Option<[f64; 2]>,
    /// How the adaptive step judges convergence.
    pub adapt_method: Option<AdaptMethod>,
    /// What the adaptivity tolerance is scaled by.
    pub adapt_scale: Option<Vec<AdaptScale>>,
    /// Minimum iterations between adaptivity checks.
    pub adapt_minit: Option<usize>,
    /// Maximum ladder jumps within one adaptivity check.
    pub adapt_interval: Option<f64>,
}

/// A parsed drift system, one per ensemble member.
#[derive(Debug, Clone)]
pub struct DriftSystem {
    /// Complete drift Hamiltonian, when one was built.
    pub drift: Option<TimeDependent>,
    /// Interaction part; its presence is what enables operator splitting.
    pub interaction: Option<TimeDependent>,
    /// Resonance offsets in rad/s, one per spin.
    pub offset: Option<DVector<f64>>,
    /// Splitting data.  One entry per rung of `adaptset` when the propagator
    /// cache is carried, otherwise a single entry for the current rung.
    pub splits: Vec<SplitSet>,
    /// The `(trotter, order)` ladder the adaptive step walks.
    pub adaptset: Vec<(usize, usize)>,
    /// Current Trotter number.
    pub trotter_number: usize,
    /// Current splitting order.
    pub split_order: usize,
    /// Whether the adaptive step is active.
    pub adapt: bool,
    /// `[max, min]` fidelity error thresholds.
    pub adapt_tols: [f64; 2],
    /// How the adaptive step judges convergence.
    pub adapt_method: AdaptMethod,
    /// What the adaptivity tolerance is scaled by.
    pub adapt_scale: Vec<AdaptScale>,
    /// Minimum iterations between adaptivity checks.
    pub adapt_minit: usize,
    /// Maximum ladder jumps within one adaptivity check.
    pub adapt_interval: f64,
    /// Iterations since the last adaptivity check; starts at infinity so the
    /// very first gradient call is allowed to climb the ladder freely.
    pub adapt_counter: f64,
}

impl DriftSystem {
    /// Position of the current `(trotter, order)` pair on the ladder.
    pub fn ladder_index(&self) -> Option<usize> {
        self.adaptset
            .iter()
            .position(|&(q, p)| q == self.trotter_number && p == self.split_order)
    }

    /// The splitting data for ladder rung `k`.
    ///
    /// With a carried propagator cache every rung is precomputed and `k`
    /// selects one; otherwise there is a single stored set.
    pub fn split_at(&self, k: usize) -> Result<&SplitSet> {
        if self.splits.len() == 1 {
            Ok(&self.splits[0])
        } else {
            self.splits.get(k).ok_or_else(|| {
                QoalaError::Dimension(format!("no splitting data for ladder rung {k}"))
            })
        }
    }

    /// The splitting data for the current `(trotter, order)` pair.
    pub fn current_split(&self) -> Result<&SplitSet> {
        let k = self.ladder_index().unwrap_or(0);
        self.split_at(k)
    }
}

/// User-facing configuration, the counterpart of the MATLAB `ctrl_param`
/// struct.  Build it with [`ControlOptions::new`] and fill in what you need.
#[derive(Debug, Clone, Default)]
pub struct ControlOptions {
    // --- formalism -------------------------------------------------------
    /// Basis set; required.
    pub basis: Option<Basis>,
    /// State-space formalism; defaults to Liouville for spherical tensors.
    pub space: Option<StateSpace>,
    /// Number of spins.
    pub nspins: Option<usize>,
    /// Multiplicity of each spin; defaults to all two-level systems.
    pub mults: Option<Vec<usize>>,

    // --- numerics --------------------------------------------------------
    /// Propagator elements below this are discarded; default `1e-12`.
    pub prop_zeroed: Option<f64>,
    /// Density above which a propagator is stored densely; default `0.15`.
    pub sparsity: Option<f64>,
    /// Interaction-propagator caching strategy; default `carry`.
    pub prop_cache: Option<PropCache>,
    /// Objective function; required.
    pub optimcon_fun: Option<ObjectiveFn>,
    /// Fidelity functional; default `real`.
    pub fidelity: Option<Fidelity>,
    /// Independent fidelity check run once per iteration.
    pub fidelity_chk: Option<ObjectiveFn>,

    // --- states and operators -------------------------------------------
    /// Initial states, for point-to-point control.
    pub rho_init: Option<Vec<CDense>>,
    /// Target states, for point-to-point control.
    pub rho_targ: Option<Vec<CDense>>,
    /// Initial propagators; defaults to the identity.
    pub prop_init: Option<Vec<CDense>>,
    /// Target propagators, for universal-rotation control.
    pub prop_targ: Option<Vec<CDense>>,
    /// Control operators, one per channel; required.
    pub operators: Option<Vec<CMat>>,
    /// Rotation axis of each control channel.
    pub ctrl_axes: Option<Vec<Axis>>,
    /// Control power levels in rad/s; required.
    pub pwr_levels: Option<PowerLevels>,
    /// Single-spin Cartesian operators, `nspins x 3*npairs`, used by QOALA.
    pub pauli_operators: Option<Vec<Vec<Option<CMat>>>>,
    /// Which channels (columns) drive which spins (rows); required by QOALA.
    pub spin_control: Option<DMatrix<bool>>,
    /// Whether gradient operators are composite or single-spin.
    pub gradops: Option<GradOps>,
    /// Precomputed single-spin propagator index table.
    pub prop_ind: Option<Vec<(usize, usize)>>,
    /// Cheaper index table for uncoupled systems.
    pub approximation: Option<Approximation>,

    // --- penalties -------------------------------------------------------
    /// Penalty terms; default none.
    pub penalties: Option<Vec<Penalty>>,
    /// Penalty weights; default the number of time slices.
    pub p_weights: Option<Vec<f64>>,
    /// Waveform ceiling; default `+1`.
    pub u_bound: Option<Bound>,
    /// Waveform floor; default `-1`.
    pub l_bound: Option<Bound>,
    /// Pulse bandwidth in Hz, required by the `ADIAB` penalty.
    pub bandwidth: Option<f64>,
    /// Pulse symmetry flags.
    pub pulse_syms: Option<[i32; 2]>,

    // --- timing ----------------------------------------------------------
    /// Explicit time grid; takes precedence over duration and step count.
    pub pulse_dt: Option<DVector<f64>>,
    /// Total pulse duration in seconds.
    pub pulse_dur: Option<f64>,
    /// Number of time slices.
    pub pulse_nsteps: Option<usize>,

    // --- drift -----------------------------------------------------------
    /// One entry per ensemble member; at least one is required.
    pub drift_sys: Vec<DriftOptions>,

    // --- offsets ---------------------------------------------------------
    /// Offset ensembles in Hz.
    pub offsets: Option<Vec<Vec<f64>>>,
    /// Operators the offsets multiply.
    pub offset_operator: Option<Vec<CMat>>,

    // --- propagation -----------------------------------------------------
    /// Time-step propagation method; default `krylov`.
    pub step_method: Option<PropMethod>,
    /// Auxiliary-matrix propagation method; default `taylor`.
    pub auxmat_method: Option<PropMethod>,

    // --- optimiser -------------------------------------------------------
    /// Optimisation algorithm; default LBFGS.
    pub method: Option<OptMethod>,
    /// Iteration budget; default 100.
    pub max_iter: Option<usize>,
    /// Fidelity termination threshold; default infinite (never triggers).
    pub tol_f: Option<f64>,
    /// Step-norm termination threshold; default `1e-3`.
    pub tol_x: Option<f64>,
    /// Gradient-norm termination threshold; default `1e-6`.
    pub tol_g: Option<f64>,
    /// LBFGS history length; default 20.
    pub n_grads: Option<usize>,
    /// Line search sufficient-increase constant; default `1e-2`.
    pub ls_c1: Option<f64>,
    /// Line search curvature constant; default `0.9`.
    pub ls_c2: Option<f64>,
    /// Line search bracket expansion factor; default `3`.
    pub ls_tau1: Option<f64>,
    /// Line search left contraction; default `0.1`.
    pub ls_tau2: Option<f64>,
    /// Line search right contraction; default `0.5`.
    pub ls_tau3: Option<f64>,
    /// Maximum RFO regularisation iterations; default 2500.
    pub reg_max_iter: Option<usize>,
    /// RFO scaling factor; default `1`.
    pub reg_alpha: Option<f64>,
    /// RFO conditioning multiplier; default `0.9`.
    pub reg_phi: Option<f64>,
    /// RFO maximum condition number; default `1e4`.
    pub reg_max_cond: Option<f64>,

    // --- housekeeping ----------------------------------------------------
    /// Scratch directory for the propagator cache.
    pub scratchdir: Option<PathBuf>,
    /// Job identifier; one is generated when absent.
    pub job_id: Option<String>,
    /// Where the report goes; defaults to standard output.
    pub output: Option<Reporter>,
    /// Cavity decay rate in Hz.  Reported but unused, as in the MATLAB.
    pub cavity_decay_rate: Option<f64>,
    /// Interpolation points per slice for the cavity response.
    pub cavity_n_interp: Option<Vec<usize>>,
}

impl ControlOptions {
    /// Empty options; every field defaults or is required.
    pub fn new() -> Self {
        Self::default()
    }
}

/// A parsed, validated control system: the MATLAB `ctrl_sys`.
#[derive(Debug, Clone)]
pub struct ControlSystem {
    /// Report destination.
    pub output: Reporter,
    /// Unique job identifier.
    pub job_id: String,
    /// Scratch directory for cached propagators.
    pub scratchdir: PathBuf,

    /// Basis set.
    pub basis: Basis,
    /// State-space formalism.
    pub space: StateSpace,
    /// Number of spins.
    pub nspins: usize,
    /// Multiplicity per spin.
    pub mults: Vec<usize>,
    /// Composite-space dimension.
    pub dim: usize,

    /// Propagator element cutoff.
    pub prop_zeroed: f64,
    /// Dense-storage threshold.
    pub sparsity: f64,
    /// Interaction propagator caching strategy.
    pub prop_cache: PropCache,

    /// Objective function.
    pub optimcon_fun: ObjectiveFn,
    /// Independent fidelity check.
    pub fidelity_chk: Option<ObjectiveFn>,
    /// Point-to-point or universal rotation.
    pub ctrl_type: CtrlType,
    /// Fidelity functional.
    pub fidelity: Fidelity,

    /// Initial states or propagators.
    pub initials: Vec<CDense>,
    /// Target states or propagators.
    pub targets: Vec<CDense>,
    /// Control operators.
    pub operators: Vec<CMat>,
    /// `commute[(n,m)]` is true when control operators `n` and `m` commute.
    pub commute: DMatrix<bool>,
    /// Rotation axis of each control channel.
    pub ctrl_axes: Vec<Axis>,
    /// Power levels, one row per ensemble member and one column per channel.
    pub pwr_levels: DMatrix<f64>,
    /// Single-spin Cartesian operators for the QOALA gradient.
    pub pauli_operators: Vec<Vec<Option<CMat>>>,
    /// Which channels drive which spins.
    pub spin_control: DMatrix<bool>,
    /// Gradient-operator storage.
    pub gradops: GradOps,
    /// Single-spin propagator index table.
    pub prop_ind: Vec<(usize, usize)>,
    /// Composite propagator index tables, one per spin.
    pub prop_grad_ind: Vec<Vec<(usize, usize)>>,

    /// Penalty terms.
    pub penalties: Vec<Penalty>,
    /// Penalty weights.
    pub p_weights: Vec<f64>,
    /// Waveform ceiling.
    pub u_bound: Bound,
    /// Waveform floor.
    pub l_bound: Bound,
    /// Pulse bandwidth in Hz.
    pub bandwidth: Option<f64>,
    /// Pulse symmetry flags.
    pub pulse_syms: [i32; 2],

    /// Time grid.
    pub pulse_dt: DVector<f64>,
    /// Total duration in seconds.
    pub pulse_dur: f64,
    /// Number of time slices.
    pub pulse_nsteps: usize,

    /// Drift ensemble.
    pub drift_sys: Vec<DriftSystem>,

    /// Offset ensembles in Hz.
    pub offsets: Vec<Vec<f64>>,
    /// Operators the offsets multiply.
    pub offset_operator: Vec<CMat>,

    /// Time-step propagation method.
    pub step_method: PropMethod,
    /// Auxiliary-matrix propagation method.
    pub auxmat_method: PropMethod,

    /// Optimisation algorithm.
    pub method: OptMethod,
    /// Iteration budget.
    pub max_iter: usize,
    /// Fidelity termination threshold.
    pub tol_f: f64,
    /// Step-norm termination threshold.
    pub tol_x: f64,
    /// Gradient-norm termination threshold.
    pub tol_g: f64,
    /// LBFGS history length.
    pub n_grads: usize,
    /// Line search sufficient-increase constant.
    pub ls_c1: f64,
    /// Line search curvature constant.
    pub ls_c2: f64,
    /// Line search bracket expansion factor.
    pub ls_tau1: f64,
    /// Line search left contraction.
    pub ls_tau2: f64,
    /// Line search right contraction.
    pub ls_tau3: f64,
    /// Maximum RFO regularisation iterations.
    pub reg_max_iter: usize,
    /// RFO scaling factor.
    pub reg_alpha: f64,
    /// RFO conditioning multiplier.
    pub reg_phi: f64,
    /// RFO maximum condition number.
    pub reg_max_cond: f64,

    /// Norm of the most recent gradient; set by the optimiser and read by the
    /// adaptive step when `adapt_scale` includes `gradient`.
    pub grad_norm: Option<f64>,
    /// Interpolation points per slice, kept for the memory report.
    pub cavity_n_interp: Vec<usize>,
}

impl ControlSystem {
    /// Context the penalty functions need.
    pub fn penalty_context(&self) -> PenaltyContext {
        let mut ctx = PenaltyContext::new();
        ctx.pulse_syms = self.pulse_syms;
        ctx.bandwidth = self.bandwidth;
        ctx.pwr_levels = self.pwr_levels.row(0).iter().cloned().collect();
        ctx
    }

    /// Power levels of ensemble member `n`.
    pub fn power_row(&self, n: usize) -> Vec<f64> {
        self.pwr_levels
            .row(n.min(self.pwr_levels.nrows() - 1))
            .iter()
            .cloned()
            .collect()
    }

    /// Uniform time step, which the operator splittings assume.
    pub fn uniform_dt(&self) -> f64 {
        self.pulse_dur / self.pulse_nsteps as f64
    }

    /// Directory the propagator cache lives in for this job.
    pub fn cache_dir(&self) -> PathBuf {
        self.scratchdir.join(&self.job_id)
    }
}

/// Parse and validate options into a control system, printing the same report
/// the MATLAB parser does.
///
/// This is `optimconset`.
pub fn optimconset(opts: ControlOptions) -> Result<ControlSystem> {
    let out = opts.output.clone().unwrap_or_default();

    // --- job identity and scratch ---------------------------------------
    let (job_id, new_id) = match &opts.job_id {
        Some(id) => (id.clone(), false),
        None => (generate_job_id(), true),
    };
    let scratchdir = opts
        .scratchdir
        .clone()
        .unwrap_or_else(|| PathBuf::from("scratch"));

    out.banner();
    out.line(&format!("scratch location: {}", scratchdir.display()));
    if new_id {
        out.line(&format!("new job identifier: {job_id}"));
    } else {
        out.line(&format!("inherited job identifier: {job_id}"));
    }
    out.blank();

    // --- formalism -------------------------------------------------------
    let basis = opts
        .basis
        .ok_or_else(|| QoalaError::MissingField("basis".into()))?;
    out.field("System basis set", basis.as_str());

    let space = match (basis, opts.space) {
        (Basis::Sphten, Some(StateSpace::Hilbert)) => {
            out.warn_resolve(
                "space should be 'liouville' for basis = 'sphten'",
                "setting space to 'liouville'",
            );
            StateSpace::Liouville
        }
        (Basis::Sphten, _) => StateSpace::Liouville,
        (_, Some(s)) => s,
        (_, None) => return Err(QoalaError::MissingField("space".into())),
    };
    out.field("System state space formalism", space.as_str());

    let (nspins, mults) = match (opts.nspins, &opts.mults) {
        (Some(n), Some(m)) => {
            if m.len() != n {
                out.warn_resolve(
                    "length of mults does not equal nspins",
                    "setting nspins to the length of mults",
                );
            }
            (m.len(), m.clone())
        }
        (Some(n), None) => {
            out.warn_resolve(
                "mults not provided",
                "assuming all 2-level systems in mults",
            );
            (n, vec![2; n])
        }
        (None, Some(m)) => (m.len(), m.clone()),
        (None, None) => {
            return Err(QoalaError::MissingField(
                "either nspins or mults must be provided".into(),
            ))
        }
    };
    let mut unique_mults: Vec<usize> = mults.clone();
    unique_mults.sort_unstable();
    unique_mults.dedup();
    for m in &unique_mults {
        out.field(
            &format!("Number of {m}-level systems"),
            &int2str(mults.iter().filter(|x| *x == m).count() as i64),
        );
    }

    // --- numerics --------------------------------------------------------
    let prop_zeroed = opts.prop_zeroed.unwrap_or(1e-12);
    let sparsity = opts.sparsity.unwrap_or(0.15);
    out.field_padded("Remove propagator elements below", &fmt_g(prop_zeroed, 8));
    out.field_padded("Propagator sparsity", &fmt_g(sparsity, 8));

    let prop_cache = opts.prop_cache.unwrap_or(PropCache::Carry);
    out.field_padded(
        "Propagator precalculation access",
        match prop_cache {
            PropCache::Store => "stored",
            PropCache::Carry => "carry through",
            PropCache::Calc => "calculate",
        },
    );

    let optimcon_fun = opts
        .optimcon_fun
        .ok_or_else(|| QoalaError::MissingField("optimcon_fun".into()))?;
    out.field("Optimal control function", &format!("@{optimcon_fun}"));

    // --- control type, initial and target --------------------------------
    let (ctrl_type, initials, targets) = match (&opts.prop_targ, &opts.rho_init, &opts.rho_targ) {
        (Some(pt), _, _) => {
            let inits = match &opts.prop_init {
                Some(pi) => pi.clone(),
                None => pt
                    .iter()
                    .map(|m| CDense::identity(m.nrows(), m.ncols()))
                    .collect(),
            };
            (CtrlType::UniversalRotation, inits, pt.clone())
        }
        (None, Some(ri), Some(rt)) => {
            if ri.len() != rt.len() {
                return Err(QoalaError::BadValue(
                    "rho_targ must have the same number of states as rho_init".into(),
                ));
            }
            (CtrlType::PointToPoint, ri.clone(), rt.clone())
        }
        _ => {
            return Err(QoalaError::MissingField(
                "targets must be supplied in rho_targ or prop_targ".into(),
            ))
        }
    };
    out.field(
        "Control algorithm type",
        match ctrl_type {
            CtrlType::PointToPoint => "point-to-point",
            CtrlType::UniversalRotation => "universal rotation",
        },
    );

    let fidelity = opts.fidelity.unwrap_or(Fidelity::Real);
    match fidelity {
        Fidelity::Real => {
            out.field_padded("Fidelity measure, range [-1,+1]", "Re(<target|rho(T)>)")
        }
        Fidelity::Square => {
            out.field_padded("Fidelity measure, range [0,+1]", "|<target|rho(T)>|^2")
        }
    }
    if fidelity == Fidelity::Square {
        return Err(QoalaError::NotImplemented(
            "the 'square' fidelity functional (the shipped objective functions only implement 'real')"
                .into(),
        ));
    }

    if let Some(chk) = opts.fidelity_chk {
        out.field("fidelity check function", &format!("@{chk}"));
    }

    match ctrl_type {
        CtrlType::PointToPoint => {
            out.field(
                "Initial states per ensemble member",
                &int2str(initials.len() as i64),
            );
            out.field(
                "Target states per ensemble member",
                &int2str(targets.len() as i64),
            );
        }
        CtrlType::UniversalRotation => {
            out.field(
                "Initial propagators per ensemble member",
                &int2str(initials.len() as i64),
            );
            out.field(
                "Target propagators per ensemble member",
                &int2str(targets.len() as i64),
            );
        }
    }

    // --- control operators ----------------------------------------------
    let operators = opts
        .operators
        .clone()
        .ok_or_else(|| QoalaError::MissingField("operators".into()))?;
    if operators.is_empty() {
        return Err(QoalaError::BadValue("no control operators supplied".into()));
    }
    let dim = operators[0].nrows();
    out.field("Dimension of systems", &int2str(dim as i64));
    out.field(
        "Number of control operators",
        &int2str(operators.len() as i64),
    );

    let kctrls = operators.len();
    let mut commute = DMatrix::from_element(kctrls, kctrls, false);
    for n in 0..kctrls {
        for m in 0..kctrls {
            let ab = operators[n].matmul(&operators[m])?;
            let ba = operators[m].matmul(&operators[n])?;
            let comm = ab.add(&ba.scale(Complex64::new(-1.0, 0.0)))?;
            commute[(n, m)] = comm.norm_1() < 1e-6;
        }
    }
    let commuting_pairs = (commute.iter().filter(|b| **b).count() - kctrls) / 2;
    out.field(
        "Number of commuting control pairs",
        &int2str(commuting_pairs as i64),
    );

    let ctrl_axes = opts.ctrl_axes.clone().unwrap_or_default();
    for (k, a) in ctrl_axes.iter().enumerate() {
        out.field(
            &format!("Axis of rotation for control channel {}", k + 1),
            &format!("{}-axis", a.as_str()),
        );
    }

    // --- power levels ----------------------------------------------------
    let pwr_levels = build_power_levels(
        opts.pwr_levels
            .clone()
            .ok_or_else(|| QoalaError::MissingField("pwr_levels".into()))?,
        kctrls,
    )?;
    report_power_levels(&out, &pwr_levels);

    // --- timing ----------------------------------------------------------
    let (pulse_dt, pulse_dur, pulse_nsteps) = match &opts.pulse_dt {
        Some(dt) => {
            if dt.iter().any(|v| *v <= 0.0) {
                return Err(QoalaError::BadValue(
                    "pulse_dt must be positive real numbers".into(),
                ));
            }
            if opts.pulse_dur.is_some() && opts.pulse_nsteps.is_some() {
                out.warn_resolve(
                    "pulse_dur, pulse_nsteps and pulse_dt provided",
                    "pulse_dt takes precedence",
                );
            } else if opts.pulse_dur.is_some() {
                out.warn_resolve(
                    "pulse_dur and pulse_dt provided",
                    "pulse_dt takes precedence",
                );
            } else if opts.pulse_nsteps.is_some() {
                out.warn_resolve(
                    "pulse_nsteps and pulse_dt provided",
                    "pulse_dt takes precedence",
                );
            }
            (dt.clone(), dt.iter().sum::<f64>(), dt.len())
        }
        None => {
            let dur = opts
                .pulse_dur
                .ok_or_else(|| QoalaError::MissingField("pulse_dur or pulse_dt".into()))?;
            let n = opts
                .pulse_nsteps
                .ok_or_else(|| QoalaError::MissingField("pulse_nsteps or pulse_dt".into()))?;
            if dur <= 0.0 {
                return Err(QoalaError::BadValue("pulse_dur must be positive".into()));
            }
            if n == 0 {
                return Err(QoalaError::BadValue("pulse_nsteps must be positive".into()));
            }
            (DVector::from_element(n, dur / n as f64), dur, n)
        }
    };

    let uniform = pulse_dt
        .iter()
        .all(|v| (v - pulse_dt[0]).abs() <= 1e-15 * pulse_dt[0].abs());
    if uniform {
        out.field("Waveform slice duration grid", "uniform");
        out.field_si("Control sequence slice duration", pulse_dt[0], "s");
    } else {
        out.field("Waveform slice duration grid", "non-uniform");
        out.field_si(
            "Average control sequence slice duration",
            pulse_dt.iter().sum::<f64>() / pulse_dt.len() as f64,
            "s",
        );
        out.warn("non-uniform slice duration grid not fully coded for operator splitting");
    }
    out.field(
        "Number of slices in the control sequence",
        &int2str(pulse_nsteps as i64),
    );
    out.field_si("Total duration of the control sequence", pulse_dur, "s");

    // --- penalties -------------------------------------------------------
    let penalties = opts
        .penalties
        .clone()
        .unwrap_or_else(|| vec![Penalty::None]);
    let p_weights = match &opts.p_weights {
        Some(w) => {
            if w.len() != penalties.len() {
                return Err(QoalaError::BadValue(
                    "p_weights must have one entry per penalty".into(),
                ));
            }
            w.clone()
        }
        None => {
            if penalties.len() == 1 && penalties[0] == Penalty::None {
                vec![0.0]
            } else {
                vec![pulse_nsteps as f64; penalties.len()]
            }
        }
    };
    for (n, p) in penalties.iter().enumerate() {
        out.field(
            &format!(
                "Penalty function {}, weight {}",
                n + 1,
                fmt_g(p_weights[n], 9)
            ),
            p.as_str(),
        );
    }

    let u_bound = opts.u_bound.clone().unwrap_or(Bound::Scalar(1.0));
    let l_bound = opts.l_bound.clone().unwrap_or(Bound::Scalar(-1.0));
    if penalties.contains(&Penalty::Sns) {
        out.field(
            "Maximum SNS penalty ceiling, fraction of power level",
            &fmt_g_signed(u_bound.max(), 9),
        );
        out.field(
            "Minimum SNS penalty floor, fraction of power level",
            &fmt_g_signed(l_bound.min(), 9),
        );
    }
    if penalties.contains(&Penalty::Snsa) {
        out.field(
            "Maximum SNSA penalty ceiling, fraction of power level",
            &fmt_g_signed(u_bound.max(), 9),
        );
    }

    let bandwidth = if penalties.contains(&Penalty::Adiab) {
        let bw = match (opts.bandwidth, &opts.offsets) {
            (Some(b), _) => b,
            (None, Some(o)) if !o.is_empty() => {
                let flat: Vec<f64> = o.iter().flatten().cloned().collect();
                let hi = flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lo = flat.iter().cloned().fold(f64::INFINITY, f64::min);
                hi - lo
            }
            _ => {
                return Err(QoalaError::MissingField(
                    "a pulse bandwidth must be provided with the ADIAB penalty".into(),
                ))
            }
        };
        out.field(
            "(Non-)Adiabatic penalty function bandwidth, Hz",
            &fmt_g_signed(bw, 9),
        );
        Some(bw)
    } else {
        opts.bandwidth
    };

    // --- drift systems ---------------------------------------------------
    if opts.drift_sys.is_empty() {
        return Err(QoalaError::MissingField(
            "information on the drift terms must be supplied in drift_sys".into(),
        ));
    }
    let ensemble = opts.drift_sys.len();
    if ensemble > 1 {
        out.field("Ensemble size", &int2str(ensemble as i64));
    }

    let mut gradops = opts.gradops.unwrap_or(GradOps::Full);
    let mut drift_sys = Vec::with_capacity(ensemble);
    for (n, d) in opts.drift_sys.iter().enumerate() {
        let sys_str = if ensemble > 1 {
            format!("Ensemble member {{{}}}: ", n + 1)
        } else {
            String::new()
        };
        let parsed = parse_drift(
            d,
            &sys_str,
            &out,
            optimcon_fun,
            opts.fidelity_chk,
            space,
            dim,
            prop_zeroed,
            prop_cache,
            pulse_dur / pulse_nsteps as f64,
            &scratchdir.join(&job_id),
            &mut gradops,
        )?;
        drift_sys.push(parsed);
    }

    // --- cavity ----------------------------------------------------------
    let cavity_n_interp = opts
        .cavity_n_interp
        .clone()
        .unwrap_or_else(|| vec![1; pulse_nsteps]);
    if let Some(rate) = opts.cavity_decay_rate {
        out.field("Cavity decay rate, Hertz", &fmt_g(rate, 9));
        if rate != 0.0 {
            out.warn("a cavity response is configured but no objective function applies one");
        }
    }

    // --- propagator index tables ----------------------------------------
    let (prop_ind, prop_grad_ind) = if let Some(pi) = opts.prop_ind.clone() {
        out.field("Sparse propagator index store", "true");
        let g = vec![pi.clone()];
        (pi, g)
    } else if optimcon_fun.is_qoala() && opts.approximation == Some(Approximation::NoCoupling) {
        let table = uncoupled_index_table(space, basis, nspins)?;
        out.field("Sparse propagator index store", "true");
        let g = vec![table.clone()];
        (table, g)
    } else if optimcon_fun.is_qoala() {
        let single = propagator_ind(space, basis, 1)?.remove(0);
        let grad = propagator_ind(space, basis, nspins)?;
        out.field("Sparse propagator index store", "true");
        (single, grad)
    } else {
        out.field("Sparse propagator index store", "false");
        (Vec::new(), vec![Vec::new()])
    };

    // --- spin control and gradient operators -----------------------------
    let spin_control = match &opts.spin_control {
        Some(sc) => sc.clone(),
        None if optimcon_fun.is_qoala() => {
            return Err(QoalaError::MissingField(
                "spin_control: an array saying which control channels (columns) drive which spins (rows)"
                    .into(),
            ))
        }
        None => DMatrix::from_element(0, 0, false),
    };

    let pauli_operators = match &opts.pauli_operators {
        Some(p) => p.clone(),
        None if optimcon_fun.is_qoala() => {
            build_pauli_operators(space, basis, nspins, &unique_mults, &spin_control, gradops)?
        }
        None => Vec::new(),
    };

    // --- offsets ---------------------------------------------------------
    let offsets = opts.offsets.clone().unwrap_or_else(|| vec![vec![0.0]]);
    let offset_operator = opts
        .offset_operator
        .clone()
        .unwrap_or_else(|| vec![CMat::zeros(1, 1)]);
    if opts.offsets.is_some() && opts.offset_operator.is_some() {
        for (n, o) in offsets.iter().enumerate() {
            out.field(
                &format!("Number of systems in offset ensemble {}", n + 1),
                &int2str(o.len() as i64),
            );
            out.field(
                &format!("Max offset in the ensemble {}, Hz", n + 1),
                &fmt_g_signed(o.iter().cloned().fold(f64::NEG_INFINITY, f64::max), 9),
            );
            out.field(
                &format!("Min offset in the ensemble {}, Hz", n + 1),
                &fmt_g_signed(o.iter().cloned().fold(f64::INFINITY, f64::min), 9),
            );
        }
    }

    // --- symmetry and propagation methods --------------------------------
    let pulse_syms = match opts.pulse_syms {
        Some(s) if s[0].abs() + s[1].abs() == 2 => {
            out.field("Symmetric pulse constraint", "true");
            s
        }
        _ => [0, 0],
    };
    let step_method = opts.step_method.unwrap_or(PropMethod::Krylov);
    let auxmat_method = opts.auxmat_method.unwrap_or(PropMethod::Taylor);

    // --- optimiser settings ----------------------------------------------
    let method = opts.method.unwrap_or(OptMethod::Lbfgs);
    out.field_padded("Optimisation method", method.as_str());

    let max_iter = opts.max_iter.unwrap_or(100);
    out.field_padded("Maximum number of iterations", &int2str(max_iter as i64));

    let tol_f = opts.tol_f.unwrap_or(f64::INFINITY);
    out.field_padded("Termination tolerance on fidelity", &fmt_g(tol_f, 8));
    let tol_x = opts.tol_x.unwrap_or(1e-3);
    out.field_padded("Termination tolerance on |delta_x|", &fmt_g(tol_x, 8));
    let tol_g = opts.tol_g.unwrap_or(1e-6);
    out.field_padded("Termination tolerance on |grad(x)|", &fmt_g(tol_g, 8));

    let n_grads = if method == OptMethod::Lbfgs {
        let n = opts.n_grads.unwrap_or(20);
        out.field_padded("Number of gradients in LBFGS history", &int2str(n as i64));
        n
    } else {
        if opts.n_grads.is_some() {
            out.field_padded(
                &format!("WARNING: LBFGS history not used in {method}"),
                "(removed)",
            );
        }
        0
    };

    let ls_c1 = opts.ls_c1.unwrap_or(1e-2);
    if opts.ls_c1.is_some() {
        out.field_padded(
            "Linesearch: Sufficient decrease condition, c1",
            &fmt_g(ls_c1, 8),
        );
    }
    let ls_c2 = opts.ls_c2.unwrap_or(0.9);
    if opts.ls_c2.is_some() {
        out.field_padded(
            "Linesearch: Curvature condition on grad, c2",
            &fmt_g(ls_c2, 8),
        );
    }
    let ls_tau1 = opts.ls_tau1.unwrap_or(3.0);
    if opts.ls_tau1.is_some() {
        out.field_padded(
            "Linesearch: Bracket expansion factor, tau1",
            &fmt_g(ls_tau1, 8),
        );
    }
    let ls_tau2 = opts.ls_tau2.unwrap_or(0.1);
    if opts.ls_tau2.is_some() {
        out.field_padded(
            "Linesearch: Left section contraction, tau2",
            &fmt_g(ls_tau2, 8),
        );
    }
    let ls_tau3 = opts.ls_tau3.unwrap_or(0.5);
    if opts.ls_tau3.is_some() {
        out.field_padded(
            "Linesearch: Right section contraction, tau3",
            &fmt_g(ls_tau3, 8),
        );
    }

    let reg_max_iter = opts.reg_max_iter.unwrap_or(2500);
    out.field_padded(
        "Maximum Hessian regularisation iterations - RFO",
        &int2str(reg_max_iter as i64),
    );
    let reg_alpha = opts.reg_alpha.unwrap_or(1.0);
    out.field_padded("Regularisation scaling factor - RFO", &fmt_g(reg_alpha, 8));
    let reg_phi = opts.reg_phi.unwrap_or(0.9);
    out.field_padded(
        "Regularisation conditioning multiplier - RFO",
        &fmt_g(reg_phi, 8),
    );
    let reg_max_cond = opts.reg_max_cond.unwrap_or(1e4);
    out.field_padded("Maximum condition number - RFO", &fmt_g(reg_max_cond, 8));

    let sys = ControlSystem {
        output: out,
        job_id,
        scratchdir,
        basis,
        space,
        nspins,
        mults,
        dim,
        prop_zeroed,
        sparsity,
        prop_cache,
        optimcon_fun,
        fidelity_chk: opts.fidelity_chk,
        ctrl_type,
        fidelity,
        initials,
        targets,
        operators,
        commute,
        ctrl_axes,
        pwr_levels,
        pauli_operators,
        spin_control,
        gradops,
        prop_ind,
        prop_grad_ind,
        penalties,
        p_weights,
        u_bound,
        l_bound,
        bandwidth,
        pulse_syms,
        pulse_dt,
        pulse_dur,
        pulse_nsteps,
        drift_sys,
        offsets,
        offset_operator,
        step_method,
        auxmat_method,
        method,
        max_iter,
        tol_f,
        tol_x,
        tol_g,
        n_grads,
        ls_c1,
        ls_c2,
        ls_tau1,
        ls_tau2,
        ls_tau3,
        reg_max_iter,
        reg_alpha,
        reg_phi,
        reg_max_cond,
        grad_norm: None,
        cavity_n_interp,
    };
    report_memory(&sys);
    Ok(sys)
}

/// Expand the supplied power levels into an `nperms x kctrls` table.
fn build_power_levels(levels: PowerLevels, kctrls: usize) -> Result<DMatrix<f64>> {
    match levels {
        PowerLevels::Uniform(v) => {
            if v <= 0.0 {
                return Err(QoalaError::BadValue("power levels must be positive".into()));
            }
            Ok(DMatrix::from_element(1, kctrls, v))
        }
        PowerLevels::PerChannel(v) => {
            if v.len() != kctrls {
                return Err(QoalaError::Dimension(format!(
                    "expected {kctrls} power levels, got {}",
                    v.len()
                )));
            }
            Ok(DMatrix::from_row_slice(1, kctrls, &v))
        }
        PowerLevels::Ensemble(per_channel) => {
            if per_channel.len() != kctrls {
                return Err(QoalaError::Dimension(format!(
                    "a power level list is needed for each of the {kctrls} channels"
                )));
            }
            let nperms: usize = per_channel.iter().map(|v| v.len()).product();
            let mut table = DMatrix::<f64>::zeros(nperms, kctrls);
            for (k, values) in per_channel.iter().enumerate() {
                // Repeat each value in a Kronecker pattern, as the MATLAB does.
                let inner: usize = per_channel[k + 1..].iter().map(|v| v.len()).product();
                let outer: usize = per_channel[..k].iter().map(|v| v.len()).product();
                let mut row = 0usize;
                for _ in 0..outer {
                    for v in values {
                        for _ in 0..inner {
                            table[(row, k)] = *v;
                            row += 1;
                        }
                    }
                }
            }
            Ok(table)
        }
    }
}

fn report_power_levels(out: &Reporter, pwr: &DMatrix<f64>) {
    let two_pi = 2.0 * std::f64::consts::PI;
    if pwr.nrows() > 1 {
        out.field(
            "Total number of power level permutations",
            &int2str(pwr.nrows() as i64),
        );
    }
    for k in 0..pwr.ncols() {
        let col: Vec<f64> = pwr.column(k).iter().map(|v| v / two_pi).collect();
        let uniq: Vec<f64> = {
            let mut c = col.clone();
            c.sort_by(|a, b| a.partial_cmp(b).unwrap());
            c.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
            c
        };
        if uniq.len() > 1 {
            out.field(
                &format!("Unique power levels on control channel {}", k + 1),
                &int2str(uniq.len() as i64),
            );
            out.field_si(
                &format!("Minimum power multiplier for control channel {}", k + 1),
                uniq[0],
                "Hz",
            );
            out.field_si(
                &format!("Average power multiplier for control channel {}", k + 1),
                col.iter().sum::<f64>() / col.len() as f64,
                "Hz",
            );
            out.field_si(
                &format!("Maximum power multiplier for control channel {}", k + 1),
                uniq[uniq.len() - 1],
                "Hz",
            );
        } else {
            out.field_si(
                &format!("Power multiplier for control channel {}", k + 1),
                col[0],
                "Hz",
            );
        }
    }
}

/// Parse one ensemble member's drift terms and set up its operator splittings.
#[allow(clippy::too_many_arguments)]
fn parse_drift(
    d: &DriftOptions,
    sys_str: &str,
    out: &Reporter,
    optimcon_fun: ObjectiveFn,
    fidelity_chk: Option<ObjectiveFn>,
    space: StateSpace,
    dim: usize,
    prop_zeroed: f64,
    prop_cache: PropCache,
    dt: f64,
    cache_dir: &Path,
    gradops: &mut GradOps,
) -> Result<DriftSystem> {
    let two_pi = 2.0 * std::f64::consts::PI;
    let wants_check = fidelity_chk == Some(ObjectiveFn::WaveformFidelity);

    let mut offset = None;
    let mut interaction = None;
    let mut drift = None;

    if optimcon_fun.is_qoala() {
        let off = d.offset.clone().ok_or_else(|| {
            QoalaError::MissingField(format!("{sys_str}offset is required by @{optimcon_fun}"))
        })?;
        offset = Some(off);
        if let Some(i) = &d.interaction {
            interaction = Some(i.clone());
            out.line(&pad(
                &format!("{sys_str}offset and interaction will be used for @{optimcon_fun}"),
                60,
            ));
        } else {
            out.line(&pad(
                &format!("{sys_str}offset will be used for @{optimcon_fun}"),
                60,
            ));
        }
        if wants_check {
            drift = build_drift(d, dim, out, sys_str, "waveform_fidelity")?;
        }
    } else {
        drift = build_drift(d, dim, out, sys_str, optimcon_fun.as_str())?;
    }

    // Diagnostics on the assembled terms.
    if let Some(dr) = &drift {
        if dr.is_zero() {
            out.field(&format!("{sys_str}2-norm of drift"), "  0.000 Hz");
        } else if dr.is_constant() {
            out.field(&format!("{sys_str}Time-dependent drift"), "no");
            out.field_si(
                &format!("{sys_str}2-norm of drift"),
                dr.at(0)?.norm_2() / two_pi,
                "Hz",
            );
        } else {
            out.field(&format!("{sys_str}Time-dependent drift"), "yes");
            let mats = dr.all();
            let mean = mats.iter().map(|m| m.norm_2()).sum::<f64>() / mats.len() as f64;
            out.field_si(
                &format!("{sys_str}average 2-norm of drifts"),
                mean / two_pi,
                "Hz",
            );
        }
    }
    if let Some(o) = &offset {
        out.field(&format!("{sys_str}Time-dependent offset"), "no");
        out.field_si(
            &format!("{sys_str}2-norm of offset"),
            o.norm() / two_pi,
            "Hz",
        );
    }
    if let Some(i) = &interaction {
        if i.is_zero() {
            out.warn_resolve(
                &format!("{sys_str}interaction = 0 detected"),
                &format!("{sys_str}interaction removed from drift_sys"),
            );
            interaction = None;
        } else if i.is_constant() {
            out.field(&format!("{sys_str}Time-dependent interaction"), "no");
            out.field_si(
                &format!("{sys_str}2-norm of interaction"),
                i.at(0)?.norm_2() / two_pi,
                "Hz",
            );
        } else {
            out.field(&format!("{sys_str}Time-dependent interaction"), "yes");
            let mats = i.all();
            let mean = mats.iter().map(|m| m.norm_2()).sum::<f64>() / mats.len() as f64;
            out.field_si(
                &format!("{sys_str}average 2-norm of interactions"),
                mean / two_pi,
                "Hz",
            );
        }
    }

    // Without an interaction there is nothing to split.
    let Some(inter) = interaction.clone() else {
        if d.split_method.is_some()
            || d.split_order.is_some()
            || d.trotter_number.is_some()
            || d.splitset.is_some()
            || d.trotterset.is_some()
            || d.adaptset.is_some()
        {
            out.warn_resolve(
                &format!("{sys_str}No interaction present"),
                &format!("{sys_str}splitting options removed from drift_sys"),
            );
        }
        *gradops = GradOps::Sparse;
        return Ok(DriftSystem {
            drift,
            interaction: None,
            offset,
            splits: Vec::new(),
            adaptset: Vec::new(),
            trotter_number: 1,
            split_order: 0,
            adapt: false,
            adapt_tols: [f64::INFINITY, f64::NEG_INFINITY],
            adapt_method: AdaptMethod::Gain,
            adapt_scale: vec![AdaptScale::Infidelity],
            adapt_minit: 5,
            adapt_interval: 1.0,
            adapt_counter: f64::INFINITY,
        });
    };

    // Splitting order.
    let mut splitset = d.splitset.clone();
    let split_order = if let Some(m) = d.split_method {
        if d.split_order.is_some() {
            out.warn_resolve(
                &format!("{sys_str}split_order and split_method supplied"),
                &format!("{sys_str}split_method takes precedence"),
            );
        }
        m.order()
    } else if let Some(o) = d.split_order {
        o
    } else if let Some(s) = &splitset {
        *s.iter()
            .min()
            .ok_or_else(|| QoalaError::BadValue("splitset must not be empty".into()))?
    } else {
        2
    };
    let split_method = SplitMethod::from_order(split_order)?;

    // Trotter number.
    let mut trotterset = d.trotterset.clone();
    let trotter_number = if let Some(t) = d.trotter_number {
        t
    } else if let Some(t) = &trotterset {
        *t.iter()
            .min()
            .ok_or_else(|| QoalaError::BadValue("trotterset must not be empty".into()))?
    } else {
        1
    };

    // Adaptivity ladder.
    let default_tols = [0.5, 1e-8];
    let (adapt, adaptset, adapt_tols) = match (splitset.take(), trotterset.take()) {
        (Some(s), Some(t)) => {
            if d.adaptset.is_some() {
                out.warn_resolve(
                    &format!("{sys_str}adaptset, trotterset and splitset provided"),
                    &format!("{sys_str}trotterset and splitset take precedence"),
                );
            }
            let mut ladder = Vec::with_capacity(s.len() * t.len());
            for order in &s {
                for tr in &t {
                    ladder.push((*tr, *order));
                }
            }
            (true, ladder, d.adapt_tols.unwrap_or(default_tols))
        }
        (None, Some(t)) => {
            if d.adaptset.is_some() {
                out.warn_resolve(
                    &format!("{sys_str}adaptset and trotterset provided"),
                    &format!("{sys_str}trotterset takes precedence"),
                );
            }
            (
                true,
                t.iter().map(|tr| (*tr, split_order)).collect(),
                d.adapt_tols.unwrap_or(default_tols),
            )
        }
        (Some(s), None) => {
            if d.adaptset.is_some() {
                out.warn_resolve(
                    &format!("{sys_str}adaptset and splitset provided"),
                    &format!("{sys_str}splitset takes precedence"),
                );
            }
            (
                true,
                s.iter().map(|o| (trotter_number, *o)).collect(),
                d.adapt_tols.unwrap_or(default_tols),
            )
        }
        (None, None) => match &d.adaptset {
            Some(a) => (true, a.clone(), d.adapt_tols.unwrap_or(default_tols)),
            None => (
                false,
                vec![(trotter_number, split_order)],
                [f64::INFINITY, f64::NEG_INFINITY],
            ),
        },
    };

    let adapt_method = d.adapt_method.unwrap_or(AdaptMethod::Gain);
    let adapt_scale = d
        .adapt_scale
        .clone()
        .unwrap_or_else(|| vec![AdaptScale::Infidelity]);
    let adapt_minit = d.adapt_minit.unwrap_or(5);
    let adapt_interval = d.adapt_interval.unwrap_or(1.0);

    if adapt_method == AdaptMethod::Exact && drift.is_none() {
        return Err(QoalaError::MissingField(format!(
            "{sys_str}adapt_method 'exact' needs a full drift Hamiltonian: supply interaction and singlespin"
        )));
    }

    // Interaction propagators.  A time-dependent interaction cannot be
    // pre-split, because the propagators would differ at every slice.
    let inter_const = match &inter {
        TimeDependent::Constant(m) => m.clone(),
        TimeDependent::PerStep(_) => {
            return Err(QoalaError::NotImplemented(
                "operator splitting of a time-dependent interaction".into(),
            ))
        }
    };

    let splits = match prop_cache {
        PropCache::Carry => {
            let mut v = Vec::with_capacity(adaptset.len());
            for &(q, p) in &adaptset {
                v.push(SplitSet::build(
                    p,
                    q,
                    dt,
                    &inter_const,
                    PropMethod::Taylor,
                    prop_zeroed,
                )?);
            }
            v
        }
        PropCache::Store => {
            let mut current = None;
            for &(q, p) in &adaptset {
                let s = SplitSet::build_cached(
                    p,
                    q,
                    dt,
                    &inter_const,
                    PropMethod::Taylor,
                    prop_zeroed,
                    cache_dir,
                )?;
                if q == trotter_number && p == split_order {
                    current = Some(s);
                }
            }
            vec![current.unwrap_or(SplitSet::build_cached(
                split_order,
                trotter_number,
                dt,
                &inter_const,
                PropMethod::Taylor,
                prop_zeroed,
                cache_dir,
            )?)]
        }
        PropCache::Calc => vec![SplitSet::build(
            split_order,
            trotter_number,
            dt,
            &inter_const,
            PropMethod::Taylor,
            prop_zeroed,
        )?],
    };

    // Report the splitting configuration.
    let orders: Vec<usize> = adaptset.iter().map(|&(_, p)| p).collect();
    let trotters: Vec<usize> = adaptset.iter().map(|&(q, _)| q).collect();
    let multi_order = orders.iter().any(|o| *o != orders[0]);
    let multi_trotter = trotters.iter().any(|t| *t != trotters[0]);
    if adaptset.len() > 1 && adapt {
        if multi_order {
            out.field(
                &format!("{sys_str}Initial operator splitting method"),
                split_method.as_str(),
            );
            out.field(
                &format!("{sys_str}Initial operator splitting order"),
                &int2str(split_order as i64),
            );
            out.field(
                &format!("{sys_str}Minimum operator splitting order"),
                &int2str(*orders.iter().min().unwrap() as i64),
            );
            out.field(
                &format!("{sys_str}Maximum operator splitting order"),
                &int2str(*orders.iter().max().unwrap() as i64),
            );
        } else {
            out.field(
                &format!("{sys_str}Operator splitting method"),
                split_method.as_str(),
            );
            out.field(
                &format!("{sys_str}Operator splitting order"),
                &int2str(split_order as i64),
            );
        }
        if multi_trotter {
            out.field(
                &format!("{sys_str}Initial Trotter number"),
                &int2str(trotter_number as i64),
            );
            out.field(
                &format!("{sys_str}Minimum Trotter number"),
                &int2str(*trotters.iter().min().unwrap() as i64),
            );
            out.field(
                &format!("{sys_str}Maximum Trotter number"),
                &int2str(*trotters.iter().max().unwrap() as i64),
            );
        } else {
            out.field(
                &format!("{sys_str}Trotter number"),
                &int2str(trotter_number as i64),
            );
        }
        out.field(
            &format!("{sys_str}Minimum fidelity error threshold"),
            &fmt_g_signed(adapt_tols[1], 9),
        );
        out.field(
            &format!("{sys_str}Maximum fidelity error threshold"),
            &fmt_g_signed(adapt_tols[0], 9),
        );
    } else {
        out.field(
            &format!("{sys_str}Operator splitting method"),
            split_method.as_str(),
        );
        out.field(
            &format!("{sys_str}Operator splitting order"),
            &int2str(split_order as i64),
        );
        out.field(
            &format!("{sys_str}Trotter number"),
            &int2str(trotter_number as i64),
        );
        if adapt {
            out.field(
                &format!("{sys_str}Fidelity error threshold"),
                &fmt_g_signed(adapt_tols[0], 9),
            );
        }
    }
    if adapt && adapt_minit > 0 {
        out.field(
            &format!("{sys_str}Minimum iterations between adaptivity"),
            &int2str(adapt_minit as i64),
        );
    }
    if adapt && adapt_interval > 0.0 && adapt_interval.is_finite() {
        out.field(
            &format!("{sys_str}Maximum jumps within adaptivity interations"),
            &int2str(adapt_interval as i64),
        );
    }

    if split_order != 0 && *gradops == GradOps::Sparse {
        out.warn_resolve(
            &format!("{sys_str}gradops = 'sparse' and split_order > 0"),
            &format!("{sys_str}changing to gradops = 'full'"),
        );
        *gradops = GradOps::Full;
    }

    let _ = space;
    Ok(DriftSystem {
        drift,
        interaction,
        offset,
        splits,
        adaptset,
        trotter_number,
        split_order,
        adapt,
        adapt_tols,
        adapt_method,
        adapt_scale,
        adapt_minit,
        adapt_interval,
        adapt_counter: f64::INFINITY,
    })
}

/// Assemble the full drift Hamiltonian from its parts.
fn build_drift(
    d: &DriftOptions,
    dim: usize,
    out: &Reporter,
    sys_str: &str,
    consumer: &str,
) -> Result<Option<TimeDependent>> {
    if let Some(explicit) = &d.drift {
        return Ok(Some(explicit.clone()));
    }
    match (&d.interaction, &d.singlespin) {
        (Some(i), Some(s)) => {
            out.line(&pad(
                &format!("{sys_str}singlespin and interaction will be used for @{consumer}"),
                60,
            ));
            Ok(Some(match i {
                TimeDependent::Constant(m) => TimeDependent::Constant(m.add(s)?),
                TimeDependent::PerStep(v) => {
                    TimeDependent::PerStep(v.iter().map(|m| m.add(s)).collect::<Result<Vec<_>>>()?)
                }
            }))
        }
        (Some(i), None) => {
            out.line(&pad(
                &format!("{sys_str}interaction will be used for @{consumer}"),
                60,
            ));
            Ok(Some(i.clone()))
        }
        (None, Some(s)) => {
            out.line(&pad(
                &format!("{sys_str}singlespin will be used for @{consumer}"),
                60,
            ));
            Ok(Some(TimeDependent::Constant(s.clone())))
        }
        (None, None) => {
            out.warn_resolve(
                &format!("{sys_str}interaction and/or singlespin required for @{consumer}"),
                &format!("{sys_str}assuming interaction + singlespin = 0"),
            );
            Ok(Some(TimeDependent::Constant(CMat::zeros(dim, dim))))
        }
    }
}

/// The index table for a system treated as uncoupled.
fn uncoupled_index_table(
    space: StateSpace,
    basis: Basis,
    nspins: usize,
) -> Result<Vec<(usize, usize)>> {
    match (space, basis) {
        (StateSpace::Liouville, Basis::Sphten) => {
            let mut out = Vec::with_capacity(9 * nspins);
            for s in 0..nspins {
                for c in 0..3 {
                    for r in 0..3 {
                        out.push((3 * s + r, 3 * s + c));
                    }
                }
            }
            Ok(out)
        }
        (StateSpace::Hilbert, Basis::Zeeman) => {
            let mut out = Vec::with_capacity(4 * nspins);
            for s in 0..nspins {
                for c in 0..2 {
                    for r in 0..2 {
                        out.push((2 * s + r, 2 * s + c));
                    }
                }
            }
            Ok(out)
        }
        (StateSpace::Liouville, Basis::Zeeman) => Err(QoalaError::NotImplemented(
            "uncoupled propagator indices for the zeeman basis in a liouville space".into(),
        )),
        (StateSpace::Hilbert, Basis::Sphten) => Err(QoalaError::NotImplemented(
            "uncoupled propagator indices for the sphten basis in a hilbert space".into(),
        )),
    }
}

/// Build composite-space single-spin Cartesian operators for the QOALA
/// gradient when the user did not supply them.
fn build_pauli_operators(
    space: StateSpace,
    basis: Basis,
    nspins: usize,
    unique_mults: &[usize],
    spin_control: &DMatrix<bool>,
    gradops: GradOps,
) -> Result<Vec<Vec<Option<CMat>>>> {
    if unique_mults.iter().any(|m| *m != 2) {
        return Err(QoalaError::NotImplemented(
            "single-spin operators for anything other than spin-1/2".into(),
        ));
    }
    let (l1x, l1y, l1z) = single_spin_operators(space, basis)?;

    if gradops == GradOps::Sparse {
        return Ok(vec![vec![
            Some(l1x.clone()),
            Some(l1y.clone()),
            Some(l1z.clone()),
        ]]);
    }

    let kctrls = spin_control.ncols();
    if kctrls == 0 {
        return Err(QoalaError::MissingField("spin_control".into()));
    }
    let npairs = kctrls / 2;
    let mut ops: Vec<Vec<Option<CMat>>> = vec![vec![None; 3 * npairs]; nspins];
    for k in 0..npairs {
        for n in 0..nspins {
            if !spin_control[(n, 2 * k + 1)] {
                continue;
            }
            // A mask that places the single-spin operator on spin n inside the
            // composite space; the MATLAB indexes it in reverse spin order.
            let mut mask = CDense::zeros(nspins, nspins);
            mask[(nspins - n - 1, nspins - n - 1)] = Complex64::new(1.0, 0.0);
            let mask = CMat::Dense(mask).into_sparse();
            let embed = |op: &CMat| -> CMat {
                let k = mask.kron(op);
                match basis {
                    // The unit state sits in front of the rank-1 block.
                    Basis::Sphten => block_diag_zero_first(&k),
                    Basis::Zeeman => k,
                }
            };
            ops[n][3 * k] = Some(embed(&l1x));
            ops[n][3 * k + 1] = Some(embed(&l1y));
            ops[n][3 * k + 2] = Some(embed(&l1z));
        }
    }
    Ok(ops)
}

/// `blkdiag(0, m)`.
fn block_diag_zero_first(m: &CMat) -> CMat {
    let sp = m.to_sparse();
    let n = sp.nrows();
    CMat::from_triplets(
        n + 1,
        sp.ncols() + 1,
        sp.triplet_iter().map(|(r, c, v)| (r + 1, c + 1, *v)),
    )
    .unwrap_or_else(|_| CMat::zeros(n + 1, sp.ncols() + 1))
}

/// Single-spin Cartesian operators in the requested formalism.
#[rustfmt::skip]
pub fn single_spin_operators(space: StateSpace, basis: Basis) -> Result<(CMat, CMat, CMat)> {
    let c = |re: f64, im: f64| Complex64::new(re, im);
    match (space, basis) {
        (StateSpace::Liouville, Basis::Sphten) => {
            let s = 1.0 / std::f64::consts::SQRT_2;
            let lx = CDense::from_row_slice(
                3,
                3,
                &[c(0.0, 0.0), c(s, 0.0), c(0.0, 0.0),
                  c(s, 0.0), c(0.0, 0.0), c(s, 0.0),
                  c(0.0, 0.0), c(s, 0.0), c(0.0, 0.0)],
            );
            let ly = CDense::from_row_slice(
                3,
                3,
                &[c(0.0, 0.0), c(0.0, -s), c(0.0, 0.0),
                  c(0.0, s), c(0.0, 0.0), c(0.0, -s),
                  c(0.0, 0.0), c(0.0, s), c(0.0, 0.0)],
            );
            let lz = CDense::from_row_slice(
                3,
                3,
                &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
                  c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
                  c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)],
            );
            Ok((
                CMat::Dense(lx).into_sparse(),
                CMat::Dense(ly).into_sparse(),
                CMat::Dense(lz).into_sparse(),
            ))
        }
        (StateSpace::Liouville, Basis::Zeeman) => {
            let h = 0.5;
            let lx = CDense::from_row_slice(
                4,
                4,
                &[c(0.0, 0.0), c(h, 0.0), c(-h, 0.0), c(0.0, 0.0),
                  c(h, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-h, 0.0),
                  c(-h, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(h, 0.0),
                  c(0.0, 0.0), c(-h, 0.0), c(h, 0.0), c(0.0, 0.0)],
            );
            let ly = CDense::from_row_slice(
                4,
                4,
                &[c(0.0, 0.0), c(0.0, -h), c(0.0, -h), c(0.0, 0.0),
                  c(0.0, h), c(0.0, 0.0), c(0.0, 0.0), c(0.0, -h),
                  c(0.0, h), c(0.0, 0.0), c(0.0, 0.0), c(0.0, -h),
                  c(0.0, 0.0), c(0.0, h), c(0.0, h), c(0.0, 0.0)],
            );
            let lz = CDense::from_row_slice(
                4,
                4,
                &[c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
                  c(0.0, 0.0), c(-1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
                  c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0),
                  c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
            );
            Ok((
                CMat::Dense(lx).into_sparse(),
                CMat::Dense(ly).into_sparse(),
                CMat::Dense(lz).into_sparse(),
            ))
        }
        (StateSpace::Hilbert, Basis::Zeeman) => {
            let h = 0.5;
            let lx = CDense::from_row_slice(2, 2, &[c(0.0, 0.0), c(h, 0.0), c(h, 0.0), c(0.0, 0.0)]);
            let ly = CDense::from_row_slice(2, 2, &[c(0.0, 0.0), c(0.0, -h), c(0.0, h), c(0.0, 0.0)]);
            let lz = CDense::from_row_slice(2, 2, &[c(h, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-h, 0.0)]);
            Ok((
                CMat::Dense(lx).into_sparse(),
                CMat::Dense(ly).into_sparse(),
                CMat::Dense(lz).into_sparse(),
            ))
        }
        (StateSpace::Hilbert, Basis::Sphten) => Err(QoalaError::NotImplemented(
            "single spin operators for the sphten basis in a hilbert space formalism".into(),
        )),
    }
}

/// Approximate memory footprint, as reported by the MATLAB parser.
fn report_memory(sys: &ControlSystem) {
    let out = &sys.output;
    let interp: usize = sys.cavity_n_interp.iter().sum();
    out.line(&pad("Calculating approximate memory footprint...", 60));

    let mem_traj = ((1 + interp) * 8 + 16 * sys.dim * (1 + interp)) as f64;
    report_bytes(out, "... trajectory array", mem_traj);
    let mut mem_req = mem_traj;

    if sys.optimcon_fun.is_qoala() {
        let (m1, m2) = match sys.drift_sys.first() {
            Some(d) => match d.splits.first() {
                Some(s) => (s.coeffs.nsplits(), s.coeffs.num_a()),
                None => (1, 1),
            },
            None => (1, 1),
        };
        let mem_elements = (9 * 16 * m1 * sys.nspins * interp) as f64;
        report_bytes(out, "... propagator elements array", mem_elements);
        mem_req += mem_elements;

        let mem_split = ((m2 * interp) * 8 + m2 * 16 * sys.dim * interp) as f64;
        report_bytes(out, "... split trajectory array", mem_split);
        mem_req += mem_split;

        let dense = 10f64.powi(sys.nspins as i32) / (4f64.powi(sys.nspins as i32)).powi(2) > 0.15;
        let mem_prop = if dense {
            let m = (m1 * 16 * sys.dim * sys.dim * interp) as f64;
            report_bytes(out, &format!("... {interp} propagators (full matrices)"), m);
            m
        } else {
            let m = (m1 * (8 * (1 + sys.dim) + 24 * (10 * sys.nspins)) * interp) as f64;
            report_bytes(
                out,
                &format!("... {interp} propagators (sparse matrices, no coupling)"),
                m,
            );
            m
        };
        mem_req += mem_prop;

        report_bytes(
            out,
            &format!("=== Total memory footprint of @{}", sys.optimcon_fun),
            mem_req,
        );
    }
}

fn report_bytes(out: &Reporter, label: &str, bytes: f64) {
    let (v, p) = crate::report::find_prefix(bytes, 1024.0);
    out.field(
        label,
        &format!(
            "{} {}B",
            crate::report::pad(&crate::report::fmt_f(v, 3), 7),
            p
        ),
    );
}

/// A short, unique-enough job identifier.
fn generate_job_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    // A cheap 128-bit mix, rendered as hex; the MATLAB used an MD5 of the
    // clock and pid for the same purpose.
    let mut h: u128 = nanos ^ (pid << 64) ^ 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    h ^= h >> 61;
    format!("{h:032x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_level_permutations_match_the_kronecker_pattern() {
        let table = build_power_levels(
            PowerLevels::Ensemble(vec![vec![1.0, 2.0], vec![10.0, 20.0, 30.0]]),
            2,
        )
        .unwrap();
        assert_eq!(table.nrows(), 6);
        assert_eq!(table.ncols(), 2);
        let rows: Vec<(f64, f64)> = (0..6).map(|r| (table[(r, 0)], table[(r, 1)])).collect();
        assert_eq!(
            rows,
            vec![
                (1.0, 10.0),
                (1.0, 20.0),
                (1.0, 30.0),
                (2.0, 10.0),
                (2.0, 20.0),
                (2.0, 30.0)
            ]
        );
    }

    #[test]
    fn uniform_power_broadcasts_over_channels() {
        let t = build_power_levels(PowerLevels::Uniform(6.5), 4).unwrap();
        assert_eq!(t.shape(), (1, 4));
        assert!(t.iter().all(|v| *v == 6.5));
    }

    #[test]
    fn single_spin_operators_obey_the_su2_algebra() {
        // [Lx, Ly] = i Lz in Hilbert space.
        let (lx, ly, lz) = single_spin_operators(StateSpace::Hilbert, Basis::Zeeman).unwrap();
        let comm = lx
            .matmul(&ly)
            .unwrap()
            .add(&ly.matmul(&lx).unwrap().scale(Complex64::new(-1.0, 0.0)))
            .unwrap();
        let want = lz.scale(Complex64::new(0.0, 1.0));
        assert!((comm.to_dense() - want.to_dense()).norm() < 1e-14);
    }

    #[test]
    fn liouville_single_spin_operators_obey_the_su2_algebra() {
        for basis in [Basis::Sphten, Basis::Zeeman] {
            let (lx, ly, lz) = single_spin_operators(StateSpace::Liouville, basis).unwrap();
            let comm = lx
                .matmul(&ly)
                .unwrap()
                .add(&ly.matmul(&lx).unwrap().scale(Complex64::new(-1.0, 0.0)))
                .unwrap();
            let want = lz.scale(Complex64::new(0.0, 1.0));
            assert!(
                (comm.to_dense() - want.to_dense()).norm() < 1e-13,
                "basis {basis:?} fails [Lx,Ly] = i Lz"
            );
        }
    }

    #[test]
    fn job_ids_are_distinct() {
        let a = generate_job_id();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = generate_job_id();
        assert_ne!(a, b);
        assert_eq!(a.len(), 32);
    }
}
