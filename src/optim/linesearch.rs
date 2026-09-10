//! Wolfe-condition line search.
//!
//! Port of `kernel/fmaxlinesearch.m`, itself derived from D. Kroon's
//! `fminlbfgs`.  The search maximises, so the "sufficient decrease" condition
//! becomes a sufficient *increase* and an acceptable direction is one along
//! which the objective rises.
//!
//! Two phases: bracketing widens the step until it straddles an acceptable
//! point, then sectioning shrinks the bracket by cubic interpolation.
//!
//! When any drift system is due an adaptivity check the search is skipped
//! entirely and a unit step is taken - re-evaluating the objective at several
//! trial steps while the splitting parameters are still moving would compare
//! points computed to different accuracies.

use super::{objective, CostFunction, ObjectiveRequest, OptData};
use crate::config::ControlSystem;
use crate::error::Result;
use crate::types::{ExitFlag, OptMethod};
use nalgebra::DVector;

/// Outcome of a line search.
#[derive(Debug, Clone)]
pub struct LineSearch {
    /// Accepted step length, absent when no acceptable point was found.
    pub alpha: Option<f64>,
    /// Objective value at the accepted point.
    pub fx: Option<f64>,
    /// Gradient at the accepted point.
    pub grad: Option<DVector<f64>>,
    /// Set when the search failed.
    pub exitflag: Option<ExitFlag>,
}

/// One end of a bracket.
#[derive(Debug, Clone)]
struct Bracket {
    alpha: f64,
    fx: f64,
    grad: DVector<f64>,
}

/// Find an acceptable step along `dir` from `x0`.
#[allow(clippy::too_many_arguments)]
pub fn fmaxlinesearch(
    sys: &mut ControlSystem,
    cost: &dyn CostFunction,
    data: &mut OptData,
    dir: &DVector<f64>,
    x0: &DVector<f64>,
    fx0: f64,
    gfx0: &DVector<f64>,
) -> Result<LineSearch> {
    // Opening step: LBFGS has no curvature information on its first
    // iteration, so it takes a deliberately cautious one.
    let mut alpha = if sys.method == OptMethod::Lbfgs && data.count.iter == 1 {
        (1.0 / gfx0.amax()).min(5.0)
    } else {
        1.0
    };

    // Is any drift system due an adaptivity check this iteration?
    let adapt_due = data.count.iter != 1
        && sys
            .drift_sys
            .iter()
            .any(|d| d.adapt && d.adapt_counter >= d.adapt_minit as f64);

    if adapt_due {
        alpha = 1.0;
        let eval = objective(
            &(x0 + dir * alpha),
            cost,
            data,
            sys,
            ObjectiveRequest::Gradient,
        )?;
        return Ok(LineSearch {
            alpha: Some(alpha),
            fx: Some(eval.fx),
            grad: eval.grad,
            exitflag: None,
        });
    }

    let eval = objective(
        &(x0 + dir * alpha),
        cost,
        data,
        sys,
        ObjectiveRequest::Gradient,
    )?;
    let fx2 = eval.fx;
    let gfx2 = eval
        .grad
        .expect("gradient requested from the objective function");

    // The trial evaluations below must not move the splitting parameters:
    // every point in the search has to be measured on the same footing.
    let saved: Vec<bool> = sys.drift_sys.iter().map(|d| d.adapt).collect();
    for d in sys.drift_sys.iter_mut() {
        d.adapt = false;
    }

    let outcome = (|| -> Result<LineSearch> {
        match bracketing(sys, cost, data, dir, x0, fx0, gfx0, alpha, fx2, gfx2)? {
            Bracketing::Accepted(ls) => Ok(ls),
            Bracketing::Bracketed(a, b) => sectioning(sys, cost, data, dir, x0, fx0, gfx0, a, b),
        }
    })();

    for (d, was) in sys.drift_sys.iter_mut().zip(saved) {
        d.adapt = was;
    }
    outcome
}

enum Bracketing {
    /// An acceptable point was found while bracketing.
    Accepted(LineSearch),
    /// A bracket `[A, B]` was located.
    Bracketed(Bracket, Bracket),
}

#[allow(clippy::too_many_arguments, clippy::neg_cmp_op_on_partial_ord)]
fn bracketing(
    sys: &mut ControlSystem,
    cost: &dyn CostFunction,
    data: &mut OptData,
    dir: &DVector<f64>,
    x0: &DVector<f64>,
    fx0: f64,
    gfx0: &DVector<f64>,
    alpha: f64,
    mut fx2: f64,
    mut gfx2: DVector<f64>,
) -> Result<Bracketing> {
    let d0 = gfx0.dot(dir);
    let mut fx1 = fx0;
    let mut gfx1 = gfx0.clone();
    let mut alpha1 = 0.0;
    let mut alpha2 = alpha;

    loop {
        // Sufficient increase fails, or the objective did not rise at all:
        // written as a negated comparison so that a NaN objective counts as
        // "did not rise" and brackets rather than being accepted.
        // the acceptable point lies between the last two trials.
        //
        // The MATLAB tests the Armijo condition with the *initial* step length
        // rather than the one being trialled, which is a stale variable; the
        // step actually under test is used here.  See `DEVIATIONS.md`.
        if !armijo(alpha2, fx0, fx2, d0, sys.ls_c1) || !(fx2 > fx0) {
            return Ok(Bracketing::Bracketed(
                Bracket {
                    alpha: alpha1,
                    fx: fx1,
                    grad: gfx1,
                },
                Bracket {
                    alpha: alpha2,
                    fx: fx2,
                    grad: gfx2,
                },
            ));
        }

        // Strong Wolfe curvature condition: this point will do.
        if gfx2.dot(dir).abs() <= sys.ls_c2 * d0.abs() {
            return Ok(Bracketing::Accepted(LineSearch {
                alpha: Some(alpha2),
                fx: Some(fx2),
                grad: Some(gfx2),
                exitflag: None,
            }));
        }

        // The objective has started falling again: bracket backwards.
        if gfx2.dot(dir) <= 0.0 {
            return Ok(Bracketing::Bracketed(
                Bracket {
                    alpha: alpha2,
                    fx: fx2,
                    grad: gfx2,
                },
                Bracket {
                    alpha: alpha1,
                    fx: fx1,
                    grad: gfx1,
                },
            ));
        }

        let end_a = 2.0 * alpha2 - alpha1;
        let end_b = alpha2 + sys.ls_tau1 * (alpha2 - alpha1);
        let alpha_new = cubic_interpolation(
            end_a,
            end_b,
            alpha1,
            alpha2,
            fx1,
            gfx1.dot(dir),
            fx2,
            gfx2.dot(dir),
        );

        alpha1 = alpha2;
        alpha2 = alpha_new;
        fx1 = fx2;
        gfx1 = gfx2;

        let eval = objective(
            &(x0 + dir * alpha2),
            cost,
            data,
            sys,
            ObjectiveRequest::Gradient,
        )?;
        fx2 = eval.fx;
        gfx2 = eval.grad.expect("gradient requested");
    }
}

#[allow(clippy::too_many_arguments, clippy::neg_cmp_op_on_partial_ord)]
fn sectioning(
    sys: &mut ControlSystem,
    cost: &dyn CostFunction,
    data: &mut OptData,
    dir: &DVector<f64>,
    x0: &DVector<f64>,
    fx0: f64,
    gfx0: &DVector<f64>,
    mut a: Bracket,
    mut b: Bracket,
) -> Result<LineSearch> {
    let d0 = gfx0.dot(dir);

    loop {
        let end_a = a.alpha + sys.ls_tau2.min(sys.ls_c2) * (b.alpha - a.alpha);
        let end_b = b.alpha - sys.ls_tau3 * (b.alpha - a.alpha);
        let alpha = cubic_interpolation(
            end_a,
            end_b,
            a.alpha,
            b.alpha,
            a.fx,
            a.grad.dot(dir),
            b.fx,
            b.grad.dot(dir),
        );

        if ((alpha - a.alpha) * a.grad.dot(dir)).abs() <= eps_at(fx0.abs().max(1.0)) {
            return Ok(failed());
        }

        let eval = objective(
            &(x0 + dir * alpha),
            cost,
            data,
            sys,
            ObjectiveRequest::Gradient,
        )?;
        let fx1 = eval.fx;
        let gfx1 = eval.grad.expect("gradient requested");

        let previous_a = a.clone();

        if !armijo(alpha, fx0, fx1, d0, sys.ls_c1) || !(fx1 > a.fx) {
            b = Bracket {
                alpha,
                fx: fx1,
                grad: gfx1,
            };
        } else {
            if gfx1.dot(dir).abs() <= sys.ls_c2 * d0.abs() {
                return Ok(LineSearch {
                    alpha: Some(alpha),
                    fx: Some(fx1),
                    grad: Some(gfx1),
                    exitflag: None,
                });
            }
            let slope = gfx1.dot(dir);
            a = Bracket {
                alpha,
                fx: fx1,
                grad: gfx1,
            };
            if (a.alpha - b.alpha) * slope >= 0.0 {
                b = previous_a;
            }
        }

        if (b.alpha - a.alpha).abs() < f64::EPSILON {
            return Ok(failed());
        }
    }
}

fn failed() -> LineSearch {
    LineSearch {
        alpha: None,
        fx: None,
        grad: None,
        exitflag: Some(ExitFlag::LineSearchFailed),
    }
}

/// Sufficient-increase (Armijo) condition for a maximisation.
#[inline]
fn armijo(alpha: f64, fx0: f64, fx1: f64, d0: f64, c1: f64) -> bool {
    fx1 >= fx0 + c1 * alpha * d0
}

/// Maximiser of the cubic through two points with known values and slopes,
/// restricted to `[end_a, end_b]`.
///
/// The search is climbing, so this picks the highest point in the bracket.
/// Port of the `cubic_interpolation` subfunction.
#[allow(clippy::too_many_arguments)]
pub fn cubic_interpolation(
    end_a: f64,
    end_b: f64,
    alpha_a: f64,
    alpha_b: f64,
    f_a: f64,
    deriv_a: f64,
    f_b: f64,
    deriv_b: f64,
) -> f64 {
    let span = alpha_b - alpha_a;
    let c1 = -2.0 * (f_b - f_a) + (deriv_a + deriv_b) * span;
    let c2 = 3.0 * (f_b - f_a) - (2.0 * deriv_a + deriv_b) * span;
    let c3 = span * deriv_a;
    let c4 = f_a;

    if span == 0.0 || !span.is_finite() {
        return alpha_a;
    }

    // Work in the normalised variable z = (alpha - alpha_a)/span.
    let lo = ((end_a - alpha_a) / span).min((end_b - alpha_a) / span);
    let hi = ((end_a - alpha_a) / span).max((end_b - alpha_a) / span);

    let mut points = vec![lo, hi];
    for root in quadratic_roots(3.0 * c1, 2.0 * c2, c3) {
        if root >= lo && root <= hi {
            points.push(root);
        }
    }

    let poly = |z: f64| c1 * z * z * z + c2 * z * z + c3 * z + c4;
    let mut best = points[0];
    let mut best_value = poly(points[0]);
    for z in &points[1..] {
        let v = poly(*z);
        if v > best_value {
            best_value = v;
            best = *z;
        }
    }
    alpha_a + best * span
}

/// Real roots of `a x^2 + b x + c`, degenerating gracefully as MATLAB's
/// `roots` does when leading coefficients vanish.
fn quadratic_roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a == 0.0 {
        if b == 0.0 {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return Vec::new();
    }
    let s = disc.sqrt();
    vec![(-b + s) / (2.0 * a), (-b - s) / (2.0 * a)]
}

/// MATLAB's `eps(x)`: the spacing of floating-point numbers at `x`.
pub fn eps_at(x: f64) -> f64 {
    if x == 0.0 {
        return f64::MIN_POSITIVE;
    }
    let next = f64::from_bits(x.abs().to_bits() + 1);
    next - x.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn eps_at_one_matches_machine_epsilon() {
        assert_relative_eq!(eps_at(1.0), f64::EPSILON, epsilon = 0.0);
        assert!(eps_at(1e6) > eps_at(1.0));
    }

    #[test]
    fn cubic_interpolation_finds_the_interior_maximum() {
        // f(z) = -(z - 0.4)^2 + 1 on [0, 1], sampled at the ends.
        let f = |z: f64| -(z - 0.4) * (z - 0.4) + 1.0;
        let df = |z: f64| -2.0 * (z - 0.4);
        let alpha = cubic_interpolation(0.0, 1.0, 0.0, 1.0, f(0.0), df(0.0), f(1.0), df(1.0));
        assert_relative_eq!(alpha, 0.4, epsilon = 1e-9);
    }

    #[test]
    fn cubic_interpolation_clamps_to_the_bracket() {
        // A monotone rise: the best point in [0, 1] is the right end.
        let f = |z: f64| z;
        let df = |_z: f64| 1.0;
        let alpha = cubic_interpolation(0.0, 1.0, 0.0, 1.0, f(0.0), df(0.0), f(1.0), df(1.0));
        assert_relative_eq!(alpha, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn quadratic_roots_degenerate_gracefully() {
        assert_eq!(quadratic_roots(0.0, 0.0, 1.0).len(), 0);
        assert_relative_eq!(quadratic_roots(0.0, 2.0, -4.0)[0], 2.0, epsilon = 1e-14);
        let r = quadratic_roots(1.0, 0.0, -4.0);
        assert_eq!(r.len(), 2);
        assert!(r.contains(&2.0) && r.contains(&-2.0));
        assert_eq!(quadratic_roots(1.0, 0.0, 4.0).len(), 0);
    }
}
