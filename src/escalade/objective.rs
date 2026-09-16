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
//! Where the MATLAB's `parfor` runs the fields in parallel and each field's
//! spins in one vectorised block, here every (field, spin) pair is a work
//! item.  The items, field-major, are cut into contiguous pieces; each piece
//! sums its items in order, and with the `parallel` feature each piece is a
//! rayon task.  Each work item is still a time-ordered product over the pulse;
//! it is the work items that are independent.
//!
//! Every piece carries its own accumulator, which for a Hessian is a dense
//! `2N x 2N` matrix, so long pulses are cut into fewer pieces, keeping the
//! accumulators within 256 MB together.  The grouping depends on the number
//! of fields, spins and pulse points and on the order asked for, never on the
//! thread count, so the result does not either (see the crate documentation).
//! Past a few hundred points a Hessian evaluation is grouped differently from
//! a value or gradient evaluation, and the value it returns can differ from
//! theirs in the last digit.

use super::propagators::{slice, trajectory, Op2, Slice};
use super::settings::Settings;
use crate::error::{QoalaError, Result};
use crate::optim::ObjectiveRequest;
use crate::parallel;
use nalgebra::{DMatrix, DVector};

/// Most pieces one evaluation is cut into: enough to spread the work items
/// evenly over any ordinary machine.
const MAX_PIECES: usize = 64;

/// Most memory, in bytes, that the accumulators of one evaluation's pieces may
/// take together.
///
/// A Hessian accumulator is dense: 8 MB at 500 pulse points, 32 MB at 1000,
/// 800 MB at 5000.  Under this budget a Hessian is cut into 64 pieces up to
/// about 360 points, 8 at 1000, and one - no more than a sequential loop
/// needs - from about 2050.  Few pieces cost little speed at that size: the
/// threads are by then contending for memory bandwidth rather than for work.
/// On a 12-core machine a 1000-point Hessian over 561 work items (51 spins
/// x 11 fields) took 0.75 s in 8 pieces and 0.71 s in 32, which needed 780 MB
/// against 270 MB.
const ACCUMULATOR_BUDGET: usize = 256 << 20;

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
    let nspins = settings.nspins;
    Ok(parallel::fold(
        settings.rf.len() * nspins,
        pieces(n, request),
        || Evaluation::zeros(n, request),
        |acc, task| add_spin(acc, settings, pulse, request, task / nspins, task % nspins),
        Evaluation::absorb,
    ))
}

/// How many pieces an evaluation of `request` over `n` pulse points is cut
/// into: as many as [`ACCUMULATOR_BUDGET`] allows, from one to
/// [`MAX_PIECES`].
fn pieces(n: usize, request: ObjectiveRequest) -> usize {
    let linear = 1usize.saturating_add(2usize.saturating_mul(n));
    let entries = match request {
        ObjectiveRequest::Value => 1,
        ObjectiveRequest::Gradient => linear,
        ObjectiveRequest::Hessian => {
            linear.saturating_add(4usize.saturating_mul(n).saturating_mul(n))
        }
    };
    let bytes = entries.saturating_mul(std::mem::size_of::<f64>());
    (ACCUMULATOR_BUDGET / bytes).clamp(1, MAX_PIECES)
}

impl Evaluation {
    /// Nothing yet, with room for what `request` asks for.
    fn zeros(n: usize, request: ObjectiveRequest) -> Self {
        Evaluation {
            value: 0.0,
            grad: (request != ObjectiveRequest::Value).then(|| DVector::zeros(2 * n)),
            hess: (request == ObjectiveRequest::Hessian).then(|| DMatrix::zeros(2 * n, 2 * n)),
        }
    }

    /// Add `other`, which was started with the same request.
    fn absorb(&mut self, other: Evaluation) {
        self.value += other.value;
        if let (Some(g), Some(o)) = (self.grad.as_mut(), other.grad) {
            *g += o;
        }
        if let (Some(h), Some(o)) = (self.hess.as_mut(), other.hess) {
            *h += o;
        }
    }
}

/// Add spin `p`'s contribution at field `r`.
fn add_spin(
    acc: &mut Evaluation,
    settings: &Settings,
    pulse: &DMatrix<f64>,
    request: ObjectiveRequest,
    r: usize,
    p: usize,
) {
    let n = settings.np_pulse;
    let omega1 = 2.0 * std::f64::consts::PI * settings.rf[r];
    let w = settings.rf_weights[r];
    let slices: Vec<Slice> = (0..n)
        .map(|k| {
            slice(
                settings.dt,
                omega1,
                pulse[(k, 0)],
                pulse[(k, 1)],
                settings.offsets[p],
                request,
            )
        })
        .collect();
    let traj = trajectory(&slices);

    let ut = traj.total;
    let r0 = settings.initial[p];
    let rtc = settings.target[p].adjoint();
    let rt = ut * r0 * ut.adjoint();
    let rr = rt * rtc;

    acc.value -= w * (rtc * rt).trace().re;

    if let Some(gr) = acc.grad.as_mut() {
        for (k, [llf, llg]) in traj.first.iter().enumerate() {
            gr[k] -= 2.0 * w * (llf * rr).trace().im;
            gr[n + k] -= 2.0 * w * (llg * rr).trace().im;
        }
    }

    if let Some(h) = acc.hess.as_mut() {
        hessian_terms(h, &traj.first, &traj.second, &rt, &rr, &rtc, w);
    }
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
        // The same value and gradient come back whatever order is asked for:
        // a pulse this short is cut into the same pieces at every order.
        let lower = gradhess(&s, &p, ObjectiveRequest::Gradient).unwrap();
        assert_eq!(eval.value, lower.value);
        assert_eq!(eval.grad.unwrap(), lower.grad.unwrap());
    }

    /// Long pulses are cut into fewer pieces, keeping their accumulators
    /// within budget; short ones, and anything without a Hessian, get them all.
    #[test]
    fn the_piece_count_keeps_to_the_memory_budget() {
        use ObjectiveRequest::{Gradient, Hessian, Value};
        assert_eq!(pieces(50, Hessian), MAX_PIECES);
        assert_eq!(pieces(1000, Hessian), 8);
        assert_eq!(pieces(361, Hessian), MAX_PIECES);
        assert_eq!(pieces(362, Hessian), MAX_PIECES - 1);
        assert_eq!(pieces(2047, Hessian), 2);
        assert_eq!(pieces(2048, Hessian), 1);
        assert_eq!(pieces(5000, Hessian), 1);
        assert_eq!(pieces(usize::MAX, Hessian), 1);
        assert_eq!(pieces(usize::MAX, Gradient), 1);
        for n in [1, 50, 5000] {
            assert_eq!(pieces(n, Value), MAX_PIECES);
            assert_eq!(pieces(n, Gradient), MAX_PIECES);
        }
        for n in [100, 361, 362, 500, 700, 1000, 2047, 2048] {
            let p = pieces(n, Hessian);
            let bytes = 8 * (1 + 2 * n + 4 * n * n);
            assert!(
                p == 1 || p * bytes <= ACCUMULATOR_BUDGET,
                "{n} points, {p} pieces"
            );
        }
    }

    /// Spreading the work items over threads changes nothing, to the last bit.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    #[test]
    fn the_thread_count_does_not_change_the_evaluation() {
        let (s, p) = small();
        let on = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| gradhess(&s, &p, ObjectiveRequest::Hessian).unwrap())
        };
        let one = on(1);
        for threads in [2, 5] {
            let many = on(threads);
            assert_eq!(one.value.to_bits(), many.value.to_bits());
            assert_eq!(one.grad, many.grad);
            assert_eq!(one.hess, many.hess);
        }
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
