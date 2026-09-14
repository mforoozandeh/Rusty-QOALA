//! The ESCALADE objective, its gradient and its Hessian.
//!
//! Port of `main/gradhess_vectorized_B1_parallel.m`.  `gradhess_vectorized.m`
//! is the same computation for a single field with unit weight, so it needs
//! no separate port: a one-element `rf` gives it exactly.
//!
//! For each field `r` and spin `p`, with `rho = U r0 U^dagger`,
//!
//! ```text
//! value  = -sum w_r Re tr(rT^dagger rho)
//! grad_n = -sum 2 w_r Im tr(L_n rho rT^dagger)
//! ```
//!
//! where `L_n` is slice `n`'s derivative operator carried to the end of the
//! pulse (see [`super::propagators::trajectory`]).  The Hessian follows the
//! MATLAB's arrangement: the upper triangle and the diagonal of each of the
//! four control blocks are computed and the lower triangle is mirrored.
//!
//! The sign is the MATLAB's: `fmincon` minimises, so a perfect transfer has a
//! value of minus one.
//!
//! The MATLAB `parfor` over fields becomes a plain loop.  Each spin is a
//! handful of 2x2 products per slice, and `wasm32` has no threads.

use super::propagators::{slice, trajectory, Op2, Slice};
use super::settings::Settings;
use crate::error::{QoalaError, Result};
use crate::optim::ObjectiveRequest;
use nalgebra::{DMatrix, DVector};

/// What one evaluation produced.
#[derive(Debug, Clone)]
pub struct Evaluation {
    /// The MATLAB `Fidelity`: minus the weighted fidelity.
    pub value: f64,
    /// Gradient over the parameters `[f_1..f_N, g_1..g_N]`.
    pub grad: Option<DVector<f64>>,
    /// Hessian over the same parameters.
    pub hess: Option<DMatrix<f64>>,
}

/// Evaluate the objective for `pulse`, `np_pulse x 2`, to the order asked.
pub fn gradhess(
    settings: &Settings,
    pulse: &DMatrix<f64>,
    request: ObjectiveRequest,
) -> Result<Evaluation> {
    let n = settings.np_pulse;
    if pulse.shape() != (n, 2) {
        return Err(QoalaError::Dimension(format!(
            "the pulse is {}x{} but the settings want {n}x2",
            pulse.nrows(),
            pulse.ncols()
        )));
    }
    let want_grad = request != ObjectiveRequest::Value;
    let want_hess = request == ObjectiveRequest::Hessian;

    let mut value = 0.0;
    let mut grad = want_grad.then(|| DVector::<f64>::zeros(2 * n));
    let mut hess = want_hess.then(|| DMatrix::<f64>::zeros(2 * n, 2 * n));

    let two_pi = 2.0 * std::f64::consts::PI;
    let mut slices: Vec<Slice> = Vec::with_capacity(n);

    for (&rf, &w) in settings.rf.iter().zip(&settings.rf_weights) {
        let omega1 = two_pi * rf;
        for p in 0..settings.nspins {
            slices.clear();
            slices.extend((0..n).map(|k| {
                slice(
                    settings.dt,
                    omega1,
                    pulse[(k, 0)],
                    pulse[(k, 1)],
                    settings.offsets[p],
                    request,
                )
            }));
            let traj = trajectory(&slices);

            let ut = traj.total;
            let r0 = settings.initial[p];
            let rtc = settings.target[p].adjoint();
            let rt = ut * r0 * ut.adjoint();
            let rr = rt * rtc;

            value -= w * (rtc * rt).trace().re;

            if let Some(gr) = grad.as_mut() {
                for (k, [llf, llg]) in traj.first.iter().enumerate() {
                    gr[k] -= 2.0 * w * (llf * rr).trace().im;
                    gr[n + k] -= 2.0 * w * (llg * rr).trace().im;
                }
            }

            if let Some(h) = hess.as_mut() {
                hessian_terms(h, &traj.first, &traj.second, &rt, &rr, &rtc, w);
            }
        }
    }

    Ok(Evaluation { value, grad, hess })
}

/// Add one spin's contribution to the Hessian.
fn hessian_terms(
    h: &mut DMatrix<f64>,
    first: &[[Op2; 2]],
    second: &[[Op2; 4]],
    rt: &Op2,
    rr: &Op2,
    rtc: &Op2,
    w: f64,
) {
    let n = first.len();
    let term = |a: &Op2, b: &Op2| -2.0 * w * (a * b).trace().re;

    for m in 0..n {
        let [lmf, lmg] = &first[m];
        // Two products per m rather than per (n, m): tr(L_n rho [L_m, rT]).
        let prod_mf = rt * lmf.adjoint() * rtc - rr * lmf;
        let prod_mg = rt * lmg.adjoint() * rtc - rr * lmg;

        for (k, [lnf, lng]) in first.iter().enumerate().take(m) {
            let ff = term(lnf, &prod_mf);
            let fg = term(lnf, &prod_mg);
            let gf = term(lng, &prod_mf);
            let gg = term(lng, &prod_mg);
            h[(k, m)] += ff;
            h[(m, k)] += ff;
            h[(k, n + m)] += fg;
            h[(n + m, k)] += fg;
            h[(n + k, m)] += gf;
            h[(m, n + k)] += gf;
            h[(n + k, n + m)] += gg;
            h[(n + m, n + k)] += gg;
        }

        let [dff, dfg, dgf, dgg] = &second[m];
        h[(m, m)] += term(lmf, &prod_mf) + 2.0 * w * (dff * rr).trace().re;
        h[(m, n + m)] += term(lmf, &prod_mg) + 2.0 * w * (dfg * rr).trace().re;
        h[(n + m, m)] += term(lmg, &prod_mf) + 2.0 * w * (dgf * rr).trace().re;
        h[(n + m, n + m)] += term(lmg, &prod_mg) + 2.0 * w * (dgg * rr).trace().re;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::escalade::settings::{magnetisation, Escalade, States};

    /// A small problem with two fields of unequal weight, spins either side
    /// of resonance including one exactly on it, and a pulse that is not
    /// special anywhere.
    fn small() -> (Settings, DMatrix<f64>) {
        let spec = Escalade {
            nspins: 4,
            np_pulse: 6,
            tau_p: 100e-6,
            rf: vec![15000.0, 19000.0],
            rf_weights: Some(vec![1.0, 2.0]),
            offsets: Some(vec![-8000.0, 0.0, 3000.0, 9500.0]),
            initial: States::Single(magnetisation(0.0, 0.0, 1.0)),
            target: States::Single(magnetisation(0.0, -1.0, 0.0)),
            seed: Some(11),
            ..Default::default()
        };
        let s = spec.resolve().unwrap();
        let pulse = DMatrix::from_fn(6, 2, |r, c| 0.8 * ((1.3 + c as f64) * r as f64 + 0.4).sin());
        (s, pulse)
    }

    fn value(s: &Settings, p: &DMatrix<f64>) -> f64 {
        gradhess(s, p, ObjectiveRequest::Value).unwrap().value
    }

    fn nudge(p: &DMatrix<f64>, i: usize, h: f64) -> DMatrix<f64> {
        let mut q = p.clone();
        let n = p.nrows();
        q[(i % n, i / n)] += h;
        q
    }

    #[test]
    fn the_gradient_matches_finite_differences() {
        let (s, p) = small();
        let g = gradhess(&s, &p, ObjectiveRequest::Gradient)
            .unwrap()
            .grad
            .unwrap();
        let h = 1e-6;
        for i in 0..g.len() {
            let fd = (value(&s, &nudge(&p, i, h)) - value(&s, &nudge(&p, i, -h))) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-7, "parameter {i}: {} vs {fd}", g[i]);
        }
    }

    #[test]
    fn the_hessian_matches_finite_differences_of_the_gradient() {
        let (s, p) = small();
        let eval = gradhess(&s, &p, ObjectiveRequest::Hessian).unwrap();
        let hess = eval.hess.unwrap();
        let grad = |q: &DMatrix<f64>| {
            gradhess(&s, q, ObjectiveRequest::Gradient)
                .unwrap()
                .grad
                .unwrap()
        };
        let h = 1e-5;
        let dim = hess.nrows();
        for j in 0..dim {
            let col = (grad(&nudge(&p, j, h)) - grad(&nudge(&p, j, -h))) / (2.0 * h);
            for i in 0..dim {
                assert!(
                    (hess[(i, j)] - col[i]).abs() < 1e-6,
                    "entry ({i}, {j}): {} vs {}",
                    hess[(i, j)],
                    col[i]
                );
            }
        }
        // The same value and gradient come back whatever order is asked for.
        let lower = gradhess(&s, &p, ObjectiveRequest::Gradient).unwrap();
        assert_eq!(eval.value, lower.value);
        assert_eq!(eval.grad.unwrap(), lower.grad.unwrap());
    }

    /// A pulse that does nothing leaves z-magnetisation where it is, which
    /// has no overlap with -y: the value is zero.  A hard 90-degree pulse on
    /// resonance, for a single spin, turns z onto -y exactly.
    #[test]
    fn known_pulses_give_known_values() {
        let (s, _) = small();
        assert!(value(&s, &DMatrix::zeros(6, 2)).abs() < 1e-12);

        // A single on-resonance spin, 90 degrees about +x: z goes to -y.
        let rf = 10000.0;
        let n = 5;
        let tau = 0.25 / rf; // omega1 tau = pi/2
        let one = Escalade {
            offsets: Some(vec![0.0]),
            rf: vec![rf],
            tau_p: tau,
            start: Some(DMatrix::from_fn(
                n,
                2,
                |_, c| if c == 0 { 1.0 } else { 0.0 },
            )),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let v = value(&one, &one.start);
        assert!((v + 1.0).abs() < 1e-12, "value {v}");
    }
}
