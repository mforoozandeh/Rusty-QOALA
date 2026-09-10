//! Optimisation: the cost-function interface, the diagnostics record and the
//! objective wrapper.
//!
//! Ports `kernel/objective.m` plus the bookkeeping `fmaxnewton.m` keeps in its
//! `data` struct.

pub mod linesearch;
pub mod newton;

use crate::config::ControlSystem;
use crate::error::{QoalaError, Result};
use crate::objfun::TrajData;
use crate::types::ObjectiveFn;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

/// What a value-plus-gradient call returns: diagnostics, the fidelity and its
/// penalty terms, and one gradient per term.
pub type ValueGrad = (TrajData, Vec<f64>, Vec<DMatrix<f64>>);

/// As [`ValueGrad`], with a Hessian per term as well.
pub type ValueGradHess = (TrajData, Vec<f64>, Vec<DMatrix<f64>>, Vec<DMatrix<f64>>);

/// Evaluation counters, the MATLAB `data.count` struct.
#[derive(Debug, Clone, Copy, Default)]
pub struct Counters {
    /// Optimiser iterations.
    pub iter: usize,
    /// Objective-function evaluations.
    pub fx: usize,
    /// Gradient evaluations.
    pub gfx: usize,
    /// Hessian evaluations.
    pub hfx: usize,
    /// Rational-function-optimisation regularisation steps.
    pub rfo: usize,
}

/// Diagnostics carried through an optimisation, the MATLAB `data` struct.
#[derive(Debug, Clone)]
pub struct OptData {
    /// Evaluation counters.
    pub count: Counters,
    /// Shape of the waveform, so the flat parameter vector can be folded back.
    pub x_shape: (usize, usize),
    /// Per-iteration record: iteration index, fidelity, then one column per
    /// penalty term.
    pub fx_store: DMatrix<f64>,
    /// Per-iteration independent fidelity check.
    pub fx_chk_store: DVector<f64>,
    /// Per-iteration timings: index, cumulative wall clock, cumulative time in
    /// the objective, cumulative time reported from inside the objective.
    pub timer: DMatrix<f64>,
    /// Fidelity and penalty terms from the most recent evaluation.
    pub fx_sep_pen: Vec<f64>,
    /// Independent fidelity check from the most recent evaluation.
    pub fx_chk: f64,
    /// Diagnostics from the most recent objective call.
    pub traj_data: TrajData,
    /// Name of the algorithm used, filled in by the footer.
    pub algorithm: String,
}

impl OptData {
    /// A fresh record sized for `max_iter` iterations and `npenalties` terms.
    pub fn new(max_iter: usize, npenalties: usize, x_shape: (usize, usize)) -> Self {
        OptData {
            count: Counters::default(),
            x_shape,
            fx_store: DMatrix::from_element(max_iter + 1, 2 + npenalties, f64::NAN),
            fx_chk_store: DVector::from_element(max_iter + 1, f64::NAN),
            timer: DMatrix::from_element(max_iter + 1, 4, f64::NAN),
            fx_sep_pen: vec![0.0; 1 + npenalties],
            fx_chk: f64::NAN,
            traj_data: TrajData::default(),
            algorithm: String::new(),
        }
    }

    /// Fidelity trajectory, one entry per iteration (`NaN` before the run
    /// reaches that iteration).
    pub fn fidelity_history(&self) -> Vec<f64> {
        (0..self.fx_store.nrows())
            .map(|r| self.fx_store[(r, 1)])
            .collect()
    }

    /// Cumulative wall-clock time per iteration.
    pub fn wall_clock_history(&self) -> Vec<f64> {
        (0..self.timer.nrows())
            .map(|r| self.timer[(r, 1)])
            .collect()
    }

    /// Cumulative time spent inside the objective function per iteration.
    pub fn inner_time_history(&self) -> Vec<f64> {
        (0..self.timer.nrows())
            .map(|r| self.timer[(r, 3)])
            .collect()
    }

    /// Index of the most recent iteration row.
    fn current_row(&self) -> usize {
        (0..self.timer.nrows())
            .rev()
            .find(|&r| !self.timer[(r, 0)].is_nan())
            .unwrap_or(0)
    }

    /// Accumulate a timing into column `col` of the current row, matching the
    /// MATLAB's "set on first touch, add afterwards, carry the previous
    /// iteration's total" rule.
    fn accumulate_timer(&mut self, col: usize, seconds: f64) {
        let row = self.current_row();
        if self.timer[(row, col)].is_nan() {
            self.timer[(row, col)] = if row == 0 {
                seconds
            } else {
                self.timer[(row - 1, col)] + seconds
            };
        } else {
            self.timer[(row, col)] += seconds;
        }
    }
}

/// The cost function handed to the optimiser.
///
/// It returns the objective fidelity followed by one entry per penalty term,
/// exactly like the MATLAB `optimfun_grape_xy` closure, so the optimiser can
/// report them separately and combine them itself.
pub trait CostFunction {
    /// Fidelity and penalty terms.
    fn value(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        objfun: ObjectiveFn,
    ) -> Result<(TrajData, Vec<f64>)>;

    /// Fidelity, penalty terms, and the gradient of each with respect to the
    /// waveform.
    fn value_grad(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        objfun: ObjectiveFn,
    ) -> Result<ValueGrad>;

    /// Fidelity, penalty terms, gradients and Hessians.
    ///
    /// The shipped objective functions do not provide second derivatives, so
    /// the default implementation declines.
    fn value_grad_hess(
        &self,
        _wf: &DMatrix<f64>,
        _sys: &ControlSystem,
        _objfun: ObjectiveFn,
    ) -> Result<ValueGradHess> {
        Err(QoalaError::NotImplemented(
            "this objective function does not provide a Hessian".into(),
        ))
    }
}

/// What an objective call produced.
pub struct ObjectiveEval {
    /// Combined objective: fidelity minus the penalties.
    pub fx: f64,
    /// Combined gradient, flattened column-major like the parameter vector.
    pub grad: Option<DVector<f64>>,
    /// Combined Hessian.
    pub hess: Option<DMatrix<f64>>,
}

/// How much of the objective to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveRequest {
    /// Value only.
    Value,
    /// Value and gradient.
    Gradient,
    /// Value, gradient and Hessian.
    Hessian,
}

/// Call the objective function and fold the result into the diagnostics.
///
/// This is `kernel/objective.m`: it counts the call, records the separate
/// fidelity and penalty terms, combines them into `fidelity - sum(penalties)`,
/// updates the timers, propagates any drift-system changes the adaptive step
/// made back into the control system, and runs the independent fidelity check.
pub fn objective(
    x: &DVector<f64>,
    cost: &dyn CostFunction,
    data: &mut OptData,
    sys: &mut ControlSystem,
    request: ObjectiveRequest,
) -> Result<ObjectiveEval> {
    let started = Instant::now();
    let (rows, cols) = data.x_shape;
    if x.len() != rows * cols {
        return Err(QoalaError::Dimension(format!(
            "parameter vector has {} entries but the waveform is {rows}x{cols}",
            x.len()
        )));
    }
    let wf = DMatrix::from_column_slice(rows, cols, x.as_slice());

    let (traj, fidelities, grads, hesses) = match request {
        ObjectiveRequest::Value => {
            let (t, f) = cost.value(&wf, sys, sys.optimcon_fun)?;
            data.count.fx += 1;
            (t, f, None, None)
        }
        ObjectiveRequest::Gradient => {
            let (t, f, g) = cost.value_grad(&wf, sys, sys.optimcon_fun)?;
            data.count.fx += 1;
            data.count.gfx += 1;
            (t, f, Some(g), None)
        }
        ObjectiveRequest::Hessian => {
            let (t, f, g, h) = cost.value_grad_hess(&wf, sys, sys.optimcon_fun)?;
            data.count.fx += 1;
            data.count.gfx += 1;
            data.count.hfx += 1;
            (t, f, Some(g), Some(h))
        }
    };

    data.fx_sep_pen = fidelities.clone();
    let fx = fidelities[0] - fidelities[1..].iter().sum::<f64>();

    let grad = grads.map(|g| combine(&g, rows, cols));
    let hess = hesses.map(|h| {
        let mut acc = h[0].clone();
        for extra in &h[1..] {
            acc -= extra;
        }
        acc
    });

    // Timers.
    data.accumulate_timer(2, started.elapsed().as_secs_f64());
    if let Some(inner) = traj.timer {
        data.accumulate_timer(3, inner);
    }

    // The adaptive step may have moved the splitting parameters on.
    if let Some(updated) = &traj.drift_sys {
        if let Some(slot) = sys.drift_sys.first_mut() {
            *slot = updated.clone();
        }
    }
    if let Some(n) = traj.f_adapt_count {
        data.count.fx += n;
    }
    if let Some(g) = traj.grad_norm {
        sys.grad_norm = Some(g);
    }

    // Independent fidelity check.
    data.fx_chk = match sys.fidelity_chk {
        Some(chk) => {
            let (_, f) = cost.value(&wf, sys, chk)?;
            f[0]
        }
        None => fidelities[0],
    };

    data.traj_data = traj;
    Ok(ObjectiveEval { fx, grad, hess })
}

/// Combine the objective gradient with the penalty gradients and flatten.
fn combine(grads: &[DMatrix<f64>], rows: usize, cols: usize) -> DVector<f64> {
    let mut acc = grads[0].clone();
    for extra in &grads[1..] {
        acc -= extra;
    }
    let mut out = DVector::zeros(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            out[c * rows + r] = acc[(r, c)];
        }
    }
    out
}

/// Fold a flat parameter vector back into a waveform.
pub fn to_waveform(x: &DVector<f64>, shape: (usize, usize)) -> DMatrix<f64> {
    DMatrix::from_column_slice(shape.0, shape.1, x.as_slice())
}

/// Flatten a waveform into a parameter vector.
pub fn to_vector(wf: &DMatrix<f64>) -> DVector<f64> {
    let mut out = DVector::zeros(wf.nrows() * wf.ncols());
    for c in 0..wf.ncols() {
        for r in 0..wf.nrows() {
            out[c * wf.nrows() + r] = wf[(r, c)];
        }
    }
    out
}
