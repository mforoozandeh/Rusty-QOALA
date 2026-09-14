//! Running an optimisation: `main/ESCALADE.m`.
//!
//! # The optimiser
//!
//! The MATLAB hands the objective to `fmincon`, with box bounds of `[-1, 1]`
//! on each quadrature: the trust-region-reflective algorithm when the Hessian
//! is used, the default otherwise.  `fmincon` cannot be ported, so the
//! optimiser here is [argmin](https://argmin-rs.org)'s:
//!
//! * with the Hessian, a Newton trust region with a Steihaug-CG subproblem;
//! * without it, L-BFGS with a More-Thuente line search.
//!
//! Neither takes bounds.  The amplitude limit is instead the `SNSA` spillout
//! penalty from [`crate::penalty`] on `sqrt(f^2 + g^2)`: zero while every
//! point lies within the unit circle, quadratic outside it.  That is also the
//! physical limit on the field.  The MATLAB's box constrains each quadrature
//! separately, which lets the corners reach `sqrt(2)` times the nominal field.
//! Being a penalty, it can leave a small overshoot; [`Optimised`] reports the
//! pulse's largest amplitude so that it is visible.
//!
//! # Stopping
//!
//! As the MATLAB's `OutputFcn`, a run stops once the fidelity, less the
//! penalty, reaches the target.  Otherwise it stops when the gradient norm
//! falls below `1e-10` (`OptimalityTolerance`), when L-BFGS changes the cost by
//! less than `1e-10` (`FunctionTolerance`), when the objective has not moved
//! for [`MAX_STALLED`] iterations, when the iteration budget runs out, or when
//! the [`ProgressSink`] asks it to.

use std::collections::HashMap;
use std::sync::Mutex;

use argmin::core::{
    CostFunction, Error as ArgminError, Executor, Gradient, Hessian, IterState, OptimizationResult,
    Problem, Solver, State, TerminationReason, TerminationStatus, KV,
};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::trustregion::{Steihaug, TrustRegion};
use nalgebra::{DMatrix, DVector};

use super::objective::gradhess;
use super::profile::max_amplitude;
use super::settings::{Escalade, Settings};
use crate::error::{QoalaError, Result};
use crate::optim::{Counters, IterationReport, NoProgress, ObjectiveRequest, ProgressSink};
use crate::penalty::{penalty, PenaltyContext, PenaltyOrder};
use crate::time::Instant;
use crate::types::{Bound, ExitFlag, Penalty};

/// Gradient norm below which a run has converged.
const TOL_GRAD: f64 = 1e-10;
/// Cost change below which L-BFGS has converged.
const TOL_COST: f64 = 1e-10;
/// Iterations without any decrease in cost before a run gives up.
///
/// A trust region that keeps rejecting its step shrinks by a factor of four
/// each time, so thirty rejections leave it far below any meaningful step.
pub const MAX_STALLED: usize = 30;
/// L-BFGS history length, as QOALA's default.
const LBFGS_MEMORY: usize = 25;

const STALLED: &str = "no further progress";

/// What an optimisation produced.
#[derive(Debug, Clone)]
pub struct Optimised {
    /// The optimised pulse, `np x 2`: x and y amplitudes as fractions of the
    /// nominal field.
    pub pulse: DMatrix<f64>,
    /// Weighted fidelity over every spin and field, at most one.
    pub fidelity: f64,
    /// The amplitude penalty at the optimised pulse.
    pub penalty: f64,
    /// The largest `sqrt(f^2 + g^2)` in the pulse.
    pub max_amplitude: f64,
    /// Why the optimiser stopped.
    pub exitflag: ExitFlag,
    /// Iterations and evaluation counts.
    pub counters: Counters,
    /// Wall-clock seconds.
    pub elapsed_s: f64,
}

/// Optimise a pulse for `spec`.
///
/// ```
/// use qoala::escalade::{escalade, Escalade};
///
/// let optimised = escalade(&Escalade {
///     nspins: 11,
///     np_pulse: 20,
///     tau_p: 100e-6,
///     rf: vec![17000.0],
///     sw: 10000.0,
///     max_iter: 5,
///     seed: Some(1),
///     ..Default::default()
/// })?;
/// assert_eq!(optimised.pulse.shape(), (20, 2));
/// # Ok::<(), qoala::error::QoalaError>(())
/// ```
pub fn escalade(spec: &Escalade) -> Result<Optimised> {
    escalade_with_progress(spec, &mut NoProgress)
}

/// As [`escalade`], reporting every iteration to `progress` and stopping when
/// it asks to.
///
/// Reports reuse QOALA's [`IterationReport`]; the splitting order and Trotter
/// number, which ESCALADE does not have, are zero.
pub fn escalade_with_progress(
    spec: &Escalade,
    progress: &mut dyn ProgressSink,
) -> Result<Optimised> {
    optimise(&spec.resolve()?, progress)
}

/// Optimise from already resolved settings.
pub fn optimise(settings: &Settings, progress: &mut dyn ProgressSink) -> Result<Optimised> {
    let wall = Instant::now();
    let x0 = DVector::from_column_slice(settings.start.as_slice());
    let max_iters = settings.max_iter as u64;
    let monitor = |sink| Monitored {
        sink,
        wall,
        target_fidelity: settings.target_fidelity,
        last_total: f64::NEG_INFINITY,
        stalled: 0,
    };

    let outcome = if settings.use_hessian {
        let solver = Wrapped {
            inner: TrustRegion::new(Steihaug::new()),
            monitor: monitor(progress),
        };
        Executor::new(Cost::new(settings), solver)
            .configure(|state| state.param(x0).max_iters(max_iters))
            .run()
            .map(outcome)
    } else {
        let lbfgs = LBFGS::new(MoreThuenteLineSearch::new(), LBFGS_MEMORY)
            .with_tolerance_grad(TOL_GRAD)
            .and_then(|s| s.with_tolerance_cost(TOL_COST))
            .map_err(numerical)?;
        let solver = Wrapped {
            inner: lbfgs,
            monitor: monitor(progress),
        };
        Executor::new(Cost::new(settings), solver)
            .configure(|state| state.param(x0).max_iters(max_iters))
            .run()
            .map(outcome)
    }
    .map_err(numerical)?;

    let pulse = DMatrix::from_column_slice(settings.np_pulse, 2, outcome.param.as_slice());
    let finals = Cost::new(settings)
        .eval(&outcome.param, ObjectiveRequest::Value)
        .map_err(numerical)?;
    let count = |key: &str| outcome.counts.get(key).copied().unwrap_or(0) as usize;

    Ok(Optimised {
        max_amplitude: max_amplitude(&pulse),
        pulse,
        fidelity: finals.fidelity,
        penalty: finals.penalty,
        exitflag: exit_flag(outcome.reason.as_ref()),
        counters: Counters {
            iter: outcome.iterations as usize,
            fx: count("cost_count"),
            gfx: count("gradient_count"),
            hfx: count("hessian_count"),
            rfo: 0,
        },
        elapsed_s: wall.elapsed().as_secs_f64(),
    })
}

fn numerical(e: ArgminError) -> QoalaError {
    QoalaError::Numerical(e.to_string())
}

fn exit_flag(reason: Option<&TerminationReason>) -> ExitFlag {
    match reason {
        Some(TerminationReason::TargetCostReached) => ExitFlag::FidelityTolerance,
        Some(TerminationReason::Interrupt) => ExitFlag::Cancelled,
        Some(TerminationReason::SolverConverged) => ExitFlag::GradientTolerance,
        Some(TerminationReason::SolverExit(why)) if why == STALLED => ExitFlag::StepTolerance,
        Some(TerminationReason::SolverExit(_)) => ExitFlag::LineSearchFailed,
        _ => ExitFlag::MaxIterations,
    }
}

// ----------------------------------------------------------------- cost ----

/// One evaluation, split the way the report wants it.
#[derive(Debug, Clone)]
struct Parts {
    fidelity: f64,
    penalty: f64,
    cost: f64,
    grad: Option<DVector<f64>>,
    hess: Option<DMatrix<f64>>,
}

/// The cost argmin minimises: minus the fidelity, plus the amplitude penalty.
///
/// argmin asks for the cost, gradient and Hessian through separate calls,
/// each of which would redo the whole propagation, so the most recent
/// evaluation is kept and reused while the parameters are unchanged.
struct Cost<'a> {
    settings: &'a Settings,
    last: Mutex<Option<(DVector<f64>, ObjectiveRequest, Parts)>>,
}

fn rank(order: ObjectiveRequest) -> u8 {
    match order {
        ObjectiveRequest::Value => 0,
        ObjectiveRequest::Gradient => 1,
        ObjectiveRequest::Hessian => 2,
    }
}

impl<'a> Cost<'a> {
    fn new(settings: &'a Settings) -> Self {
        Cost {
            settings,
            last: Mutex::new(None),
        }
    }

    fn eval(
        &self,
        x: &DVector<f64>,
        want: ObjectiveRequest,
    ) -> std::result::Result<Parts, ArgminError> {
        let mut last = self.last.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((at, order, parts)) = last.as_ref() {
            if at == x && rank(*order) >= rank(want) {
                return Ok(parts.clone());
            }
        }

        // A trust region wants the Hessian wherever it wants the gradient,
        // and the Hessian evaluation produces both.
        let order = if want == ObjectiveRequest::Gradient && self.settings.use_hessian {
            ObjectiveRequest::Hessian
        } else {
            want
        };

        let s = self.settings;
        let pulse = DMatrix::from_column_slice(s.np_pulse, 2, x.as_slice());
        let fid = gradhess(s, &pulse, order)?;
        let pen = penalty(
            Penalty::Snsa,
            &pulse,
            &Bound::Scalar(-1.0),
            &Bound::Scalar(1.0),
            s.amplitude_weight,
            &PenaltyContext::new(),
            match order {
                ObjectiveRequest::Value => PenaltyOrder::Value,
                ObjectiveRequest::Gradient => PenaltyOrder::Gradient,
                ObjectiveRequest::Hessian => PenaltyOrder::Hessian,
            },
        )?;

        let parts = Parts {
            fidelity: -fid.value,
            penalty: pen.value,
            cost: fid.value + pen.value,
            grad: fid
                .grad
                .zip(pen.grad)
                .map(|(g, p)| g + DVector::from_column_slice(p.as_slice())),
            hess: fid.hess.zip(pen.hess).map(|(h, p)| h + p),
        };
        *last = Some((x.clone(), order, parts.clone()));
        Ok(parts)
    }
}

impl CostFunction for Cost<'_> {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> std::result::Result<f64, ArgminError> {
        Ok(self.eval(x, ObjectiveRequest::Value)?.cost)
    }
}

impl Gradient for Cost<'_> {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(&self, x: &Self::Param) -> std::result::Result<DVector<f64>, ArgminError> {
        self.eval(x, ObjectiveRequest::Gradient)?
            .grad
            .ok_or_else(|| ArgminError::msg("the objective returned no gradient"))
    }
}

impl Hessian for Cost<'_> {
    type Param = DVector<f64>;
    type Hessian = DMatrix<f64>;

    fn hessian(&self, x: &Self::Param) -> std::result::Result<DMatrix<f64>, ArgminError> {
        self.eval(x, ObjectiveRequest::Hessian)?
            .hess
            .ok_or_else(|| ArgminError::msg("the objective returned no Hessian"))
    }
}

// -------------------------------------------------------------- monitor ----

type Iterate<H> = IterState<DVector<f64>, DVector<f64>, (), H, (), f64>;

/// Everything the executor result is needed for, taken out of it.
struct Outcome {
    param: DVector<f64>,
    reason: Option<TerminationReason>,
    counts: HashMap<&'static str, u64>,
    iterations: u64,
}

fn outcome<S, H: Clone>(mut result: OptimizationResult<Cost<'_>, S, Iterate<H>>) -> Outcome {
    let state = &mut result.state;
    let reason = state.get_termination_reason().cloned();
    let iterations = state.get_iter();
    let param = state
        .take_best_param()
        .or_else(|| state.take_param())
        .unwrap_or_default();
    Outcome {
        param,
        reason,
        counts: result.problem.counts,
        iterations,
    }
}

/// The part of the wrapper that watches: reports, cancellation, the target
/// fidelity and stalling.
struct Monitored<'s> {
    sink: &'s mut dyn ProgressSink,
    wall: Instant,
    target_fidelity: f64,
    last_total: f64,
    stalled: usize,
}

impl Monitored<'_> {
    fn report<H: Clone>(
        &mut self,
        problem: &Problem<Cost<'_>>,
        state: &Iterate<H>,
        iteration: usize,
    ) -> std::result::Result<(), ArgminError> {
        let (Some(cost), Some(x)) = (problem.problem.as_ref(), state.get_param()) else {
            return Ok(());
        };
        // Usually the evaluation the solver has just made, so a cache hit.
        let parts = cost.eval(x, ObjectiveRequest::Value)?;
        let count = |key: &str| problem.counts.get(key).copied().unwrap_or(0) as usize;
        self.last_total = parts.fidelity - parts.penalty;
        self.sink.on_iteration(&IterationReport {
            iteration,
            fidelity: parts.fidelity,
            penalty: -parts.penalty,
            total: self.last_total,
            fidelity_check: parts.fidelity,
            gradient_norm: state.get_gradient().map(|g| g.norm()).unwrap_or(0.0),
            alpha: None,
            split_order: 0,
            trotter_number: 0,
            elapsed_s: self.wall.elapsed().as_secs_f64(),
            counters: Counters {
                iter: iteration,
                fx: count("cost_count"),
                gfx: count("gradient_count"),
                hfx: count("hessian_count"),
                rfo: 0,
            },
        });
        Ok(())
    }
}

/// An argmin solver with a [`Monitored`] around it.
struct Wrapped<'s, S> {
    inner: S,
    monitor: Monitored<'s>,
}

impl<'a, S, H> Solver<Cost<'a>, Iterate<H>> for Wrapped<'_, S>
where
    S: Solver<Cost<'a>, Iterate<H>>,
    H: Clone,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn init(
        &mut self,
        problem: &mut Problem<Cost<'a>>,
        state: Iterate<H>,
    ) -> std::result::Result<(Iterate<H>, Option<KV>), ArgminError> {
        let (state, kv) = self.inner.init(problem, state)?;
        self.monitor.report(problem, &state, 0)?;
        Ok((state, kv))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<Cost<'a>>,
        state: Iterate<H>,
    ) -> std::result::Result<(Iterate<H>, Option<KV>), ArgminError> {
        let iteration = state.get_iter() as usize + 1;
        let before = state.get_cost();
        // A solver that fails mid-iteration consumes the state; keep a copy
        // so the run ends on the last good point instead of losing it.
        let fallback = state.clone();
        let (state, kv) = match self.inner.next_iter(problem, state) {
            Ok(step) => step,
            Err(e) => {
                return Ok((
                    fallback.terminate_with(TerminationReason::SolverExit(e.to_string())),
                    None,
                ))
            }
        };

        let after = state.get_cost();
        if after < before - TOL_COST * before.abs().max(1.0) {
            self.monitor.stalled = 0;
        } else {
            self.monitor.stalled += 1;
        }
        self.monitor.report(problem, &state, iteration)?;
        Ok((state, kv))
    }

    fn terminate(&mut self, state: &Iterate<H>) -> TerminationStatus {
        let m = &self.monitor;
        if m.sink.should_cancel() {
            return TerminationStatus::Terminated(TerminationReason::Interrupt);
        }
        if m.last_total >= m.target_fidelity {
            return TerminationStatus::Terminated(TerminationReason::TargetCostReached);
        }
        if state.get_gradient().is_some_and(|g| g.norm() < TOL_GRAD) {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        if m.stalled >= MAX_STALLED {
            return TerminationStatus::Terminated(TerminationReason::SolverExit(STALLED.into()));
        }
        self.inner.terminate(state)
    }
}
