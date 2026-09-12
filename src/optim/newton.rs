//! The optimiser: LBFGS and Newton-Raphson maximisation.
//!
//! Port of `kernel/fmaxnewton.m`, derived from D. Kroon's `fminlbfgs` and the
//! Spinach implementation.  Everything here climbs: the objective is a
//! fidelity to be maximised, so the LBFGS direction is an ascent direction and
//! the Newton step uses the negated Hessian.

use super::linesearch::fmaxlinesearch;
use super::progress::{IterationReport, NoProgress, ProgressSink};
use super::{objective, to_vector, CostFunction, ObjectiveRequest, OptData};
use crate::config::ControlSystem;
use crate::error::Result;
use crate::linalg::{cond_2_symmetric, is_positive_definite};
use crate::report::{fmt_e, int2str, pad, Reporter};
use crate::types::{ExitFlag, OptMethod};
use nalgebra::{DMatrix, DVector};

/// What an optimisation produced.
#[derive(Debug, Clone)]
pub struct Optimisation {
    /// The optimised waveform, `nsteps x nchannels`.
    pub waveform: DMatrix<f64>,
    /// Counters, per-iteration fidelities and timings.
    pub data: OptData,
    /// Why the optimiser stopped.
    pub exitflag: ExitFlag,
}

/// Maximise `cost` over the waveform, starting from `guess`.
///
/// Equivalent to [`fmaxnewton_with_progress`] with a sink that ignores every
/// report and never cancels.
pub fn fmaxnewton(
    sys: &mut ControlSystem,
    cost: &dyn CostFunction,
    guess: &DMatrix<f64>,
) -> Result<Optimisation> {
    fmaxnewton_with_progress(sys, cost, guess, &mut NoProgress)
}

/// Maximise `cost` over the waveform, reporting each iteration to `progress`
/// and stopping early when it asks to.
///
/// The console table is unaffected: it is itself a [`ProgressSink`] driven
/// from the same rows, so `progress` sees exactly what the table prints.
///
/// The sink is a parameter rather than a field of [`ControlSystem`] because
/// that struct is `Clone` and a boxed trait object is not.
pub fn fmaxnewton_with_progress(
    sys: &mut ControlSystem,
    cost: &dyn CostFunction,
    guess: &DMatrix<f64>,
    progress: &mut dyn ProgressSink,
) -> Result<Optimisation> {
    if guess.iter().all(|v| *v == 0.0) {
        sys.output
            .line("[ fmaxnewton                                        ]  WARNING: bad things happen when the initial guess is zeros.");
    }

    let mut table = TableSink::for_system(sys);
    let npen = sys.penalties.len();
    let mut data = OptData::new(sys.max_iter, npen, (guess.nrows(), guess.ncols()));
    let mut x = to_vector(guess);

    table.header();
    let wall = crate::time::Instant::now();

    let mut exitflag = ExitFlag::MaxIterations;
    let mut last_iteration = 0usize;
    let mut cancelled = false;

    if sys.max_iter == 0 {
        data.timer[(0, 0)] = 0.0;
        let eval = objective(&x, cost, &mut data, sys, ObjectiveRequest::Value)?;
        store_row(&mut data, 0, 0);
        data.timer[(0, 1)] = wall.elapsed().as_secs_f64();
        let elapsed = wall.elapsed().as_secs_f64();
        emit_iteration(
            &mut table, progress, sys, &data, eval.fx, None, None, elapsed,
        );
    } else {
        // LBFGS history, newest first.
        let mut dx_hist: Vec<DVector<f64>> = Vec::new();
        let mut dg_hist: Vec<DVector<f64>> = Vec::new();
        let mut old_x = x.clone();
        let mut old_g = DVector::zeros(x.len());
        let mut g = DVector::zeros(x.len());
        let mut fx = f64::NEG_INFINITY;

        for n in 1..=sys.max_iter {
            if progress.should_cancel() {
                exitflag = ExitFlag::Cancelled;
                cancelled = true;
                break;
            }
            last_iteration = n;
            if n == 1 {
                data.timer[(0, 0)] = 0.0;
            } else {
                data.timer[(n, 0)] = n as f64;
            }

            let dir: DVector<f64> = match sys.method {
                OptMethod::Lbfgs => {
                    if n == 1 {
                        let eval = objective(&x, cost, &mut data, sys, ObjectiveRequest::Gradient)?;
                        fx = eval.fx;
                        g = eval.grad.expect("gradient requested");
                        store_row(&mut data, 0, 0);
                        data.timer[(0, 1)] = wall.elapsed().as_secs_f64();
                        old_x = x.clone();
                        old_g = g.clone();
                        let elapsed = wall.elapsed().as_secs_f64();
                        emit_iteration(
                            &mut table,
                            progress,
                            sys,
                            &data,
                            fx,
                            Some(&g),
                            None,
                            elapsed,
                        );
                        data.timer[(1, 0)] = 1.0;
                        g.clone()
                    } else {
                        dx_hist.insert(0, &x - &old_x);
                        old_x = x.clone();
                        dg_hist.insert(0, &g - &old_g);
                        old_g = g.clone();
                        dx_hist.truncate(sys.n_grads);
                        dg_hist.truncate(sys.n_grads);
                        lbfgs_direction(&dx_hist, &dg_hist, &g)
                    }
                }
                OptMethod::NewtonRaphson | OptMethod::GaussNewton => {
                    let eval = objective(&x, cost, &mut data, sys, ObjectiveRequest::Hessian)?;
                    fx = eval.fx;
                    g = eval.grad.expect("gradient requested");
                    let h = eval.hess.expect("Hessian requested");
                    if n == 1 {
                        store_row(&mut data, 0, 0);
                        data.timer[(0, 1)] = wall.elapsed().as_secs_f64();
                        let elapsed = wall.elapsed().as_secs_f64();
                        emit_iteration(
                            &mut table,
                            progress,
                            sys,
                            &data,
                            fx,
                            Some(&g),
                            None,
                            elapsed,
                        );
                        data.timer[(1, 0)] = 1.0;
                    }
                    // Symmetrise, then regularise the negated Hessian so the
                    // Newton step is well conditioned.
                    let h = (&h + h.transpose()) * 0.5;
                    let reg = hessian_regularise(sys, &mut data, &(-h), &g);
                    reg.lu().solve(&g).unwrap_or_else(|| g.clone())
                }
            };

            data.count.iter += 1;

            let ls = fmaxlinesearch(sys, cost, &mut data, &dir, &x, fx, &g)?;

            store_row(&mut data, n, n);
            data.timer[(n, 1)] = wall.elapsed().as_secs_f64();

            let alpha = ls.alpha;
            if let Some(v) = ls.fx {
                fx = v;
            }
            if let Some(gr) = ls.grad.clone() {
                g = gr;
            }
            let elapsed = wall.elapsed().as_secs_f64();
            emit_iteration(
                &mut table,
                progress,
                sys,
                &data,
                fx,
                ls.grad.as_ref(),
                alpha,
                elapsed,
            );

            let Some(alpha) = alpha else {
                // No acceptable point: keep the current waveform and stop.
                exitflag = ExitFlag::LineSearchFailed;
                break;
            };
            x += &dir * alpha;

            let mut flag = ls.exitflag;
            if (&dir * alpha).abs().sum() < sys.tol_x {
                flag = Some(ExitFlag::StepTolerance);
            } else if ls.grad.is_none() || g.norm() < sys.tol_g {
                flag = Some(ExitFlag::GradientTolerance);
            } else if fx > sys.tol_f {
                flag = Some(ExitFlag::FidelityTolerance);
            }
            if let Some(f) = flag {
                exitflag = f;
                break;
            }
        }
    }

    // A cancelled run keeps its own exit reason even if it happened to be
    // asked to stop on the last permitted iteration.
    if !cancelled && last_iteration == sys.max_iter {
        exitflag = ExitFlag::MaxIterations;
    }

    data.algorithm = match sys.method {
        OptMethod::Lbfgs => "LBFGS quasi-Newton method".into(),
        OptMethod::NewtonRaphson => "Newton-Raphson method".into(),
        OptMethod::GaussNewton => "Gauss-Newton method".into(),
    };
    table.footer(&data, exitflag);

    let waveform = super::to_waveform(&x, data.x_shape);
    Ok(Optimisation {
        waveform,
        data,
        exitflag,
    })
}

/// Record the fidelity and check values for iteration `n` in row `row`.
fn store_row(data: &mut OptData, row: usize, n: usize) {
    if row >= data.fx_store.nrows() {
        return;
    }
    data.fx_store[(row, 0)] = n as f64;
    for (k, v) in data.fx_sep_pen.iter().enumerate() {
        if k + 1 < data.fx_store.ncols() {
            data.fx_store[(row, k + 1)] = *v;
        }
    }
    data.fx_chk_store[row] = data.fx_chk;
}

/// LBFGS two-loop recursion, returning an ascent direction.
///
/// The histories are newest-first, as in the MATLAB.
fn lbfgs_direction(
    dx_hist: &[DVector<f64>],
    dg_hist: &[DVector<f64>],
    g: &DVector<f64>,
) -> DVector<f64> {
    let n = dx_hist.len();
    if n == 0 {
        return g.clone();
    }
    let mut alpha = vec![0.0; n];
    let mut rho = vec![0.0; n];
    let mut q = g.clone();

    for i in 0..n {
        let denom = dg_hist[i].dot(&dx_hist[i]);
        rho[i] = if denom == 0.0 { 0.0 } else { 1.0 / denom };
        alpha[i] = rho[i] * dx_hist[i].dot(&q);
        q -= &dg_hist[i] * alpha[i];
    }

    // Scaling of the initial inverse Hessian.
    let denom = dg_hist[0].dot(&dg_hist[0]);
    let scale = if denom == 0.0 {
        1.0
    } else {
        dg_hist[0].dot(&dx_hist[0]) / denom
    };
    let mut dir = q * scale;

    for i in (0..n).rev() {
        let beta = rho[i] * dg_hist[i].dot(&dir);
        dir += &dx_hist[i] * (alpha[i] - beta);
    }
    -dir
}

/// Rational-function-optimisation regularisation of the Hessian.
///
/// Port of the `hessianreg` subfunction: shift the eigenvalues of an augmented
/// matrix until the Hessian is positive definite and well conditioned.
fn hessian_regularise(
    sys: &ControlSystem,
    data: &mut OptData,
    h_in: &DMatrix<f64>,
    g: &DVector<f64>,
) -> DMatrix<f64> {
    let mut h = h_in.clone();
    let mut alpha = sys.reg_alpha;

    if is_positive_definite(&h) && cond_2_symmetric(&h) < sys.reg_max_cond {
        return h;
    }

    let n = h.nrows();
    for _ in 0..sys.reg_max_iter {
        // Augmented Hessian [[a^2 H, a g], [a g', 0]].
        let mut aug = DMatrix::<f64>::zeros(n + 1, n + 1);
        for c in 0..n {
            for r in 0..n {
                aug[(r, c)] = alpha * alpha * h[(r, c)];
            }
        }
        for r in 0..n {
            aug[(r, n)] = alpha * g[r];
            aug[(n, r)] = alpha * g[r];
        }

        let sigma = aug
            .clone()
            .symmetric_eigenvalues()
            .iter()
            .cloned()
            .fold(0.0f64, f64::min);
        for i in 0..=n {
            aug[(i, i)] -= sigma;
        }

        h = aug.view((0, 0), (n, n)).into_owned() / (alpha * alpha);
        alpha *= sys.reg_phi;
        data.count.rfo += 1;

        if cond_2_symmetric(&h) < sys.reg_max_cond {
            break;
        }
    }

    (&h + h.transpose()) * 0.5
}

/// Which optional columns the iteration table carries.
struct ReportLayout {
    check: bool,
    penalties: bool,
    adaptive: bool,
}

impl ReportLayout {
    fn for_system(sys: &ControlSystem) -> Self {
        ReportLayout {
            check: sys.fidelity_chk.is_some(),
            penalties: !(sys.penalties.len() == 1
                && sys.penalties[0] == crate::types::Penalty::None),
            adaptive: sys.drift_sys.first().map(|d| d.adapt).unwrap_or(false),
        }
    }
    fn rule(&self, ch: char) -> String {
        let width = match (self.adaptive, self.check, self.penalties) {
            (false, true, true) => 102,
            (false, false, false) => 62,
            (false, false, true) => 88,
            (false, true, false) => 78,
            (true, true, true) => 112,
            (true, false, false) => 72,
            (true, false, true) => 98,
            (true, true, false) => 88,
        };
        std::iter::repeat_n(ch, width).collect()
    }
    fn columns(&self) -> String {
        let counters = if self.adaptive {
            "Iter  #f   #g   #H   #R   #T   #O    "
        } else {
            "Iter  #f   #g   #H   #R    "
        };
        let middle = match (self.check, self.penalties) {
            (true, true) => "fidelity      check         penalise      total       ",
            (false, false) => "fidelity      ",
            (false, true) => "fidelity      penalise      total       ",
            (true, false) => "fidelity      check           ",
        };
        format!("{counters}{middle}alpha      |grad|    ")
    }
}

/// Build the structured row for the current optimiser state.
fn make_report(
    sys: &ControlSystem,
    data: &OptData,
    fx: f64,
    grad: Option<&DVector<f64>>,
    alpha: Option<f64>,
    elapsed_s: f64,
) -> IterationReport {
    // The adaptive step keeps the current splitting parameters on the drift
    // system; without adaptivity they are whatever the parser resolved.
    let (trotter_number, split_order) = sys
        .drift_sys
        .first()
        .map(|d| (d.trotter_number, d.split_order))
        .unwrap_or((1, 2));

    IterationReport {
        iteration: data.count.iter,
        fidelity: data.fx_sep_pen[0],
        penalty: -data.fx_sep_pen[1..].iter().sum::<f64>(),
        total: fx,
        fidelity_check: data.fx_chk,
        gradient_norm: grad.map(|g| g.norm()).unwrap_or(0.0),
        alpha,
        split_order,
        trotter_number,
        elapsed_s,
        counters: data.count,
    }
}

/// Build one row and hand it to the console table and the caller's sink.
#[allow(clippy::too_many_arguments)]
fn emit_iteration(
    table: &mut TableSink,
    progress: &mut dyn ProgressSink,
    sys: &ControlSystem,
    data: &OptData,
    fx: f64,
    grad: Option<&DVector<f64>>,
    alpha: Option<f64>,
    elapsed_s: f64,
) {
    let report = make_report(sys, data, fx, grad, alpha, elapsed_s);
    table.on_iteration(&report);
    progress.on_iteration(&report);
}

/// The MATLAB iteration table, as one [`ProgressSink`] among others.
///
/// It owns everything it needs to format a row, so the optimiser drives it
/// through the same interface as any caller-supplied sink.
pub struct TableSink {
    output: Reporter,
    prefix: String,
    layout: ReportLayout,
}

impl TableSink {
    /// A table laid out for this control system, writing to its reporter.
    pub fn for_system(sys: &ControlSystem) -> Self {
        TableSink {
            output: sys.output.clone(),
            prefix: format!("fmaxnewton@{}", sys.optimcon_fun),
            layout: ReportLayout::for_system(sys),
        }
    }

    fn emit(&self, text: &str) {
        self.output
            .line(&format!("[ {} ]  {}", pad(&self.prefix, 50), text));
    }

    /// The rule-columns-rule banner above the table.
    pub fn header(&self) {
        self.emit(&self.layout.rule('='));
        self.emit(&self.layout.columns());
        self.emit(&self.layout.rule('-'));
    }

    /// The summary block below the table.
    pub fn footer(&self, data: &OptData, exitflag: ExitFlag) {
        let r = &self.layout;
        self.emit(&r.rule('-'));
        self.emit(&format!("    Algorithm Used     : {}", data.algorithm));
        self.emit(&format!("    Exit message       : {}", exitflag.message()));
        self.emit(&format!(
            "    Iterations         : {}",
            int2str(data.count.iter as i64)
        ));
        self.emit(&format!(
            "    Function Count     : {}",
            int2str(data.count.fx as i64)
        ));
        self.emit(&format!(
            "    Gradient Count     : {}",
            int2str(data.count.gfx as i64)
        ));
        self.emit(&format!(
            "    Hessian Count      : {}",
            int2str(data.count.hfx as i64)
        ));
        self.emit(&r.rule('='));
    }
}

impl ProgressSink for TableSink {
    fn on_iteration(&mut self, rep: &IterationReport) {
        let r = &self.layout;
        let (fid, pens, chk, fx) = (rep.fidelity, rep.penalty, rep.fidelity_check, rep.total);
        let c = &rep.counters;

        let mut row = String::new();
        row.push_str(&pad(&int2str(c.iter as i64), 6));
        row.push_str(&pad(&int2str(c.fx as i64), 5));
        row.push_str(&pad(&int2str(c.gfx as i64), 5));
        row.push_str(&pad(&int2str(c.hfx as i64), 5));
        row.push_str(&pad(&int2str(c.rfo as i64), 5));
        if r.adaptive {
            row.push_str(&pad(&int2str(rep.trotter_number as i64), 5));
            row.push_str(&pad(&int2str(rep.split_order as i64), 5));
        }

        row.push_str(&pad(&format!("{fid:+.8}"), 11));
        match (r.check, r.penalties) {
            (true, true) => {
                row.push_str("  ");
                row.push_str(&pad(&format!("({chk:+11.8})"), 11));
                row.push_str("  ");
                row.push_str(&pad(&format!("{pens:+.6}"), 11));
                row.push_str("   ");
                row.push_str(&pad(&format!("{fx:+.6}"), 11));
                row.push_str("  ");
            }
            (false, false) => row.push_str("    "),
            (false, true) => {
                row.push_str("   ");
                row.push_str(&pad(&format!("{pens:+.6}"), 11));
                row.push_str("   ");
                row.push_str(&pad(&format!("{fx:+.6}"), 11));
                row.push_str("  ");
            }
            (true, false) => {
                row.push_str("   ");
                row.push_str(&pad(&format!("({chk:+11.8})"), 11));
                row.push_str("    ");
            }
        }

        row.push_str(&pad(&rep.alpha.map(|a| fmt_e(a, 2)).unwrap_or_default(), 9));
        row.push_str("  ");
        row.push_str(&pad(&fmt_e(rep.gradient_norm, 4), 10));

        self.emit(&row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The two-loop recursion must reproduce the exact inverse Hessian for a
    /// quadratic once it has seen enough curvature pairs.
    #[test]
    fn lbfgs_direction_on_a_quadratic() {
        // Maximise f(x) = -0.5 x' A x with A = diag(1, 4): the ascent
        // direction from the exact inverse Hessian is -A^{-1} g.
        let a = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 4.0]));
        let grad = |x: &DVector<f64>| -(&a * x);

        let x0 = DVector::from_row_slice(&[1.0, 1.0]);
        let x1 = DVector::from_row_slice(&[1.3, 0.8]);
        let dx = &x1 - &x0;
        let dg = grad(&x1) - grad(&x0);

        let g = grad(&x1);
        let dir = lbfgs_direction(&[dx], &[dg], &g);
        // The direction must be an ascent direction.
        assert!(
            dir.dot(&g) > 0.0,
            "direction {dir:?} is not an ascent direction"
        );
    }

    #[test]
    fn lbfgs_with_no_history_returns_the_gradient() {
        let g = DVector::from_row_slice(&[1.0, -2.0]);
        let dir = lbfgs_direction(&[], &[], &g);
        assert_relative_eq!((dir - g).amax(), 0.0, epsilon = 0.0);
    }
}
