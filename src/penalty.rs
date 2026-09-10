//! Waveform penalty terms and their derivatives.
//!
//! Port of `kernel/penalty_fun.m`.  Penalties are subtracted from the fidelity
//! by the optimiser, so a penalty returns a non-negative number that grows as
//! the waveform misbehaves.
//!
//! # Waveform layout
//!
//! Throughout this crate a waveform is `nsteps x nchannels`: a row is one time
//! slice, a column is one control channel.  That is the layout the QOALA
//! objective functions and the optimiser use.
//!
//! The MATLAB `penalty_fun` is inconsistent about this.  `SNS` and `SNSM` are
//! written for `nsteps x nchannels`, but `NS`, `DNS` and `ADIAB` normalise and
//! differentiate along the other axis, i.e. they were written for Spinach's
//! transposed `nchannels x nsteps` convention.  Since QOALA only ever hands
//! them `nsteps x nchannels`, `DNS` in the MATLAB smooths *across channels*
//! rather than across time, which is not what a smoothing penalty is for.
//! Every penalty here works in the `nsteps x nchannels` layout and does what
//! its documentation says: `DNS` smooths along time, and `NS` normalises by
//! the number of time slices.  See `DEVIATIONS.md`.

use crate::error::{QoalaError, Result};
use crate::linalg::{
    cartesian2polar, cartesian2spherical, fdmat_second_derivative_5pt, polar2cartesian,
    spherical2cartesian,
};
use crate::types::{Bound, Penalty};
use nalgebra::DMatrix;

/// How much of the penalty to compute, replacing MATLAB's `nargout` switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyOrder {
    /// Value only.
    Value,
    /// Value and gradient.
    Gradient,
    /// Value, gradient and Hessian.
    Hessian,
}

/// Result of evaluating a penalty term.
#[derive(Debug, Clone)]
pub struct PenaltyOutput {
    /// The penalty value, already multiplied by its weight.
    pub value: f64,
    /// Gradient with respect to the waveform, shaped like the waveform.
    pub grad: Option<DMatrix<f64>>,
    /// Hessian with respect to the column-major flattened waveform.
    pub hess: Option<DMatrix<f64>>,
}

/// Extra information some penalties need from the control system.
#[derive(Debug, Clone, Default)]
pub struct PenaltyContext {
    /// Pulse symmetry flags, `[x_sym, y_sym]`, each `+1`, `-1` or `0`.
    pub pulse_syms: [i32; 2],
    /// Pulse bandwidth in Hz, needed by `ADIAB`.
    pub bandwidth: Option<f64>,
    /// Power levels per channel in rad/s, needed by `ADIAB`.
    pub pwr_levels: Vec<f64>,
    /// Ceiling used by `SNSM`, which the MATLAB hard-codes to 0.2.
    pub snsm_ceiling: f64,
    /// Floor used by `SNSM`, which the MATLAB hard-codes to 0.
    pub snsm_floor: f64,
}

impl PenaltyContext {
    /// Context with the MATLAB defaults.
    pub fn new() -> Self {
        PenaltyContext {
            pulse_syms: [0, 0],
            bandwidth: None,
            pwr_levels: Vec::new(),
            snsm_ceiling: 0.2,
            snsm_floor: 0.0,
        }
    }
}

/// Evaluate a penalty term.
///
/// * `wf` - waveform, `nsteps x nchannels`.
/// * `lower`, `upper` - floor and ceiling for the spillout penalties.
/// * `weight` - multiplies the value, gradient and Hessian.
pub fn penalty(
    kind: Penalty,
    wf: &DMatrix<f64>,
    lower: &Bound,
    upper: &Bound,
    weight: f64,
    ctx: &PenaltyContext,
    order: PenaltyOrder,
) -> Result<PenaltyOutput> {
    let (nsteps, nchan) = (wf.nrows(), wf.ncols());
    let want_grad = order != PenaltyOrder::Value;
    let want_hess = order == PenaltyOrder::Hessian;

    let mut value = 0.0;
    let mut grad = want_grad.then(|| DMatrix::<f64>::zeros(nsteps, nchan));
    let mut hess = want_hess.then(|| DMatrix::<f64>::zeros(nsteps * nchan, nsteps * nchan));

    match kind {
        Penalty::None => {}

        Penalty::Ns => {
            // Norm square of the waveform, averaged over time slices.
            value = wf.iter().map(|v| v * v).sum::<f64>() / nsteps as f64;
            if let Some(g) = grad.as_mut() {
                *g = wf.map(|v| 2.0 * v / nsteps as f64);
            }
            if let Some(h) = hess.as_mut() {
                for i in 0..nsteps * nchan {
                    h[(i, i)] = 2.0 / nsteps as f64;
                }
            }
        }

        Penalty::Dns => {
            // Norm square of the second time derivative: a smoothing penalty.
            let d = fdmat_second_derivative_5pt(nsteps)?;
            let dwf = &d * wf;
            value = dwf.iter().map(|v| v * v).sum::<f64>() / nsteps as f64;
            if let Some(g) = grad.as_mut() {
                *g = (d.transpose() * &dwf).map(|v| 2.0 * v / nsteps as f64);
            }
            if let Some(h) = hess.as_mut() {
                let dtd = d.transpose() * &d;
                for k in 0..nchan {
                    for j in 0..nsteps {
                        for i in 0..nsteps {
                            h[(k * nsteps + i, k * nsteps + j)] = 2.0 * dtd[(i, j)] / nsteps as f64;
                        }
                    }
                }
            }
        }

        Penalty::Sns => {
            // Spillout norm square: only excursions past the bounds cost
            // anything, so the waveform is free to roam inside them.
            for c in 0..nchan {
                for r in 0..nsteps {
                    let w = wf[(r, c)];
                    let ceil = upper.at(r, c);
                    let floor = lower.at(r, c);
                    let mut spill = 0.0;
                    let mut curv = 0.0;
                    if w > ceil {
                        spill += w - ceil;
                        curv += 2.0;
                    }
                    if w < floor {
                        spill += w - floor;
                        curv += 2.0;
                    }
                    value += spill * spill;
                    if let Some(g) = grad.as_mut() {
                        g[(r, c)] = 2.0 * spill / nsteps as f64;
                    }
                    if let Some(h) = hess.as_mut() {
                        let idx = c * nsteps + r;
                        h[(idx, idx)] = curv / nsteps as f64;
                    }
                }
            }
            value /= nsteps as f64;
        }

        Penalty::Snsa => {
            // Spillout on the polar amplitude of each (x, y) channel pair, so
            // the constraint is on pulse power rather than on each quadrature.
            if nchan % 2 != 0 {
                return Err(QoalaError::BadValue(
                    "the SNSA penalty needs an even number of control channels (x/y pairs)".into(),
                ));
            }
            for pair in 0..nchan / 2 {
                let (cx, cy) = (2 * pair, 2 * pair + 1);
                for r in 0..nsteps {
                    let (amp, phi) = cartesian2polar(wf[(r, cx)], wf[(r, cy)]);
                    let ceil = upper.at(r, cx);
                    if amp <= ceil {
                        continue;
                    }
                    let spill = amp - ceil;
                    value += spill * spill;
                    if let Some(g) = grad.as_mut() {
                        // d(spill^2)/d(x,y) = 2*spill * (cos phi, sin phi)
                        let (gx, gy) = polar2cartesian(2.0 * spill / nsteps as f64, phi);
                        g[(r, cx)] = gx;
                        g[(r, cy)] = gy;
                    }
                    if let Some(h) = hess.as_mut() {
                        // Exact Hessian of (|w| - c)^2 outside the ceiling.
                        let (cp, sp) = (phi.cos(), phi.sin());
                        let radial = 2.0 / nsteps as f64;
                        let tangential = 2.0 * spill / (amp * nsteps as f64);
                        let block = [
                            radial * cp * cp + tangential * sp * sp,
                            (radial - tangential) * cp * sp,
                            (radial - tangential) * cp * sp,
                            radial * sp * sp + tangential * cp * cp,
                        ];
                        let ix = cx * nsteps + r;
                        let iy = cy * nsteps + r;
                        h[(ix, ix)] = block[0];
                        h[(ix, iy)] = block[1];
                        h[(iy, ix)] = block[2];
                        h[(iy, iy)] = block[3];
                    }
                }
            }
            value /= nsteps as f64;
        }

        Penalty::Snsm => {
            // Spillout on the summed magnitude of each channel over the whole
            // pulse.  The MATLAB hard-codes the bounds; they are configurable
            // here but default to the same values.
            let ceil = ctx.snsm_ceiling;
            let floor = ctx.snsm_floor;
            for c in 0..nchan {
                let total: f64 = (0..nsteps).map(|r| wf[(r, c)].abs()).sum();
                let mut spill = 0.0;
                if total > ceil {
                    spill += total - ceil;
                }
                if total < floor {
                    spill += total - floor;
                }
                value += spill * spill;
                if let Some(g) = grad.as_mut() {
                    let d = 2.0 * spill / nchan as f64;
                    for r in 0..nsteps {
                        g[(r, c)] = d * wf[(r, c)].signum();
                    }
                }
                if let Some(h) = hess.as_mut() {
                    if spill != 0.0 {
                        for rj in 0..nsteps {
                            for ri in 0..nsteps {
                                h[(c * nsteps + ri, c * nsteps + rj)] =
                                    2.0 * wf[(ri, c)].signum() * wf[(rj, c)].signum()
                                        / nchan as f64;
                            }
                        }
                    }
                }
            }
            value /= nchan as f64;
        }

        Penalty::Adiab => {
            let out = adiabaticity_penalty(wf, ctx, want_grad)?;
            value = out.0;
            if let Some(g) = grad.as_mut() {
                *g = out.1;
            }
            if want_hess {
                return Err(QoalaError::NotImplemented(
                    "Hessian of the ADIAB penalty".into(),
                ));
            }
        }
    }

    Ok(PenaltyOutput {
        value: weight * value,
        grad: grad.map(|g| g.map(|v| weight * v)),
        hess: hess.map(|h| h.map(|v| weight * v)),
    })
}

/// Roughness of the effective-field polar angle, averaged over power levels.
///
/// This is the `ADIAB` branch of `penalty_fun`.  The MATLAB version cannot run
/// as shipped: it calls `penaltyox`, `cartesian2spherical` and
/// `spherical2cartesian`, none of which are in the package.  The reconstruction
/// here follows the documented intent - build the effective field from the two
/// quadratures and a spread of resonance offsets, take the polar angle, and
/// penalise its second derivative along the pulse, transforming the gradient
/// back to Cartesian coordinates.
fn adiabaticity_penalty(
    wf: &DMatrix<f64>,
    ctx: &PenaltyContext,
    want_grad: bool,
) -> Result<(f64, DMatrix<f64>)> {
    let bandwidth = ctx
        .bandwidth
        .ok_or_else(|| QoalaError::MissingField("a pulse bandwidth is needed by ADIAB".into()))?;
    if wf.ncols() < 2 {
        return Err(QoalaError::BadValue(
            "the ADIAB penalty needs at least an x and a y channel".into(),
        ));
    }
    if ctx.pwr_levels.is_empty() {
        return Err(QoalaError::MissingField(
            "power levels are needed by ADIAB".into(),
        ));
    }

    // Optional symmetry extension: mirror the waveform with the given parity.
    let symmetric = ctx.pulse_syms[0].abs() == 1 && ctx.pulse_syms[1].abs() == 1;
    let nsteps = wf.nrows();
    let n = if symmetric { 2 * nsteps } else { nsteps };
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for r in 0..nsteps {
        x[r] = wf[(r, 0)];
        y[r] = wf[(r, 1)];
    }
    if symmetric {
        for r in 0..nsteps {
            x[nsteps + r] = ctx.pulse_syms[0] as f64 * wf[(nsteps - 1 - r, 0)];
            y[nsteps + r] = ctx.pulse_syms[1] as f64 * wf[(nsteps - 1 - r, 1)];
        }
    }

    // Offsets spread across the bandwidth.
    let offsets: Vec<f64> = (0..n)
        .map(|i| {
            let f = if n > 1 {
                -0.5 + i as f64 / (n - 1) as f64
            } else {
                0.0
            };
            2.0 * std::f64::consts::PI * bandwidth * f
        })
        .collect();

    let d = fdmat_second_derivative_5pt(n)?;
    let mut total = 0.0;
    let mut grad_ext = DMatrix::<f64>::zeros(n, 2);

    for &pwr in &ctx.pwr_levels {
        // Effective field, normalised to a unit rotation axis.
        let mut theta = nalgebra::DVector::<f64>::zeros(n);
        let mut spherical = Vec::with_capacity(n);
        for i in 0..n {
            let (ex, ey, ez) = (pwr * x[i], pwr * y[i], offsets[i]);
            let norm = (ex * ex + ey * ey + ez * ez).sqrt();
            let (ux, uy, uz) = if norm == 0.0 {
                (0.0, 0.0, 0.0)
            } else {
                (ex / norm, ey / norm, ez / norm)
            };
            let (r, th, ph) = cartesian2spherical(ux, uy, uz);
            theta[i] = th;
            spherical.push((r, th, ph, norm));
        }

        // Smoothing penalty on the polar angle.
        let dtheta = &d * &theta;
        total += dtheta.iter().map(|v| v * v).sum::<f64>() / n as f64;

        if want_grad {
            let dpen_dtheta = (d.transpose() * &dtheta).map(|v| 2.0 * v / n as f64);
            for i in 0..n {
                let (r, th, ph, norm) = spherical[i];
                let (_, _, _, dx, dy, _) = spherical2cartesian(r, th, ph, 0.0, dpen_dtheta[i], 0.0);
                // Chain rule back through the normalisation and the power level.
                let scale = if norm == 0.0 { 0.0 } else { pwr / norm };
                grad_ext[(i, 0)] += dx * scale;
                grad_ext[(i, 1)] += dy * scale;
            }
        }
    }

    let npwr = ctx.pwr_levels.len() as f64;
    total /= npwr;
    grad_ext /= npwr;

    // Fold the mirrored half back onto the optimised half.
    let mut grad = DMatrix::<f64>::zeros(nsteps, wf.ncols());
    for r in 0..nsteps {
        grad[(r, 0)] = grad_ext[(r, 0)];
        grad[(r, 1)] = grad_ext[(r, 1)];
    }
    if symmetric {
        for r in 0..nsteps {
            grad[(r, 0)] += ctx.pulse_syms[0] as f64 * grad_ext[(2 * nsteps - 1 - r, 0)];
            grad[(r, 1)] += ctx.pulse_syms[1] as f64 * grad_ext[(2 * nsteps - 1 - r, 1)];
        }
    }
    Ok((total, grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn wf() -> DMatrix<f64> {
        DMatrix::from_row_slice(
            6,
            2,
            &[
                0.2, -0.4, 1.4, 0.1, -1.9, 0.6, 0.3, 1.2, -0.1, -1.5, 0.7, 0.05,
            ],
        )
    }

    fn numeric_grad(
        kind: Penalty,
        w: &DMatrix<f64>,
        lo: &Bound,
        hi: &Bound,
        ctx: &PenaltyContext,
    ) -> DMatrix<f64> {
        let h = 1e-7;
        let mut g = DMatrix::zeros(w.nrows(), w.ncols());
        for c in 0..w.ncols() {
            for r in 0..w.nrows() {
                let mut wp = w.clone();
                let mut wm = w.clone();
                wp[(r, c)] += h;
                wm[(r, c)] -= h;
                let fp = penalty(kind, &wp, lo, hi, 1.0, ctx, PenaltyOrder::Value)
                    .unwrap()
                    .value;
                let fm = penalty(kind, &wm, lo, hi, 1.0, ctx, PenaltyOrder::Value)
                    .unwrap()
                    .value;
                g[(r, c)] = (fp - fm) / (2.0 * h);
            }
        }
        g
    }

    #[test]
    fn gradients_match_finite_differences() {
        let w = wf();
        let lo = Bound::Scalar(-1.0);
        let hi = Bound::Scalar(1.0);
        let ctx = PenaltyContext::new();
        for kind in [Penalty::Ns, Penalty::Dns, Penalty::Sns, Penalty::Snsa] {
            let analytic = penalty(kind, &w, &lo, &hi, 1.0, &ctx, PenaltyOrder::Gradient)
                .unwrap()
                .grad
                .unwrap();
            let numeric = numeric_grad(kind, &w, &lo, &hi, &ctx);
            assert_relative_eq!((analytic - numeric).amax(), 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn hessians_match_finite_differences_of_the_gradient() {
        let w = wf();
        let lo = Bound::Scalar(-1.0);
        let hi = Bound::Scalar(1.0);
        let ctx = PenaltyContext::new();
        let h = 1e-6;
        for kind in [Penalty::Ns, Penalty::Dns, Penalty::Sns] {
            let analytic = penalty(kind, &w, &lo, &hi, 1.0, &ctx, PenaltyOrder::Hessian)
                .unwrap()
                .hess
                .unwrap();
            let n = w.nrows() * w.ncols();
            let mut numeric = DMatrix::<f64>::zeros(n, n);
            for j in 0..n {
                let (r, c) = (j % w.nrows(), j / w.nrows());
                let mut wp = w.clone();
                let mut wm = w.clone();
                wp[(r, c)] += h;
                wm[(r, c)] -= h;
                let gp = penalty(kind, &wp, &lo, &hi, 1.0, &ctx, PenaltyOrder::Gradient)
                    .unwrap()
                    .grad
                    .unwrap();
                let gm = penalty(kind, &wm, &lo, &hi, 1.0, &ctx, PenaltyOrder::Gradient)
                    .unwrap()
                    .grad
                    .unwrap();
                let col = (gp - gm).map(|v| v / (2.0 * h));
                for i in 0..n {
                    numeric[(i, j)] = col[(i % w.nrows(), i / w.nrows())];
                }
            }
            assert_relative_eq!((analytic - numeric).amax(), 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn spillout_is_zero_inside_the_bounds() {
        let w = DMatrix::from_row_slice(3, 2, &[0.1, -0.2, 0.5, 0.4, -0.9, 0.0]);
        let out = penalty(
            Penalty::Sns,
            &w,
            &Bound::Scalar(-1.0),
            &Bound::Scalar(1.0),
            7.0,
            &PenaltyContext::new(),
            PenaltyOrder::Gradient,
        )
        .unwrap();
        assert_eq!(out.value, 0.0);
        assert_eq!(out.grad.unwrap().amax(), 0.0);
    }

    #[test]
    fn spillout_counts_only_the_excursion() {
        // One element 0.5 above the ceiling, in a 4-step waveform.
        let w = DMatrix::from_row_slice(4, 1, &[0.0, 1.5, 0.0, 0.0]);
        let out = penalty(
            Penalty::Sns,
            &w,
            &Bound::Scalar(-1.0),
            &Bound::Scalar(1.0),
            1.0,
            &PenaltyContext::new(),
            PenaltyOrder::Value,
        )
        .unwrap();
        assert_relative_eq!(out.value, 0.25 / 4.0, epsilon = 1e-15);
    }

    #[test]
    fn weight_scales_everything() {
        let w = wf();
        let (lo, hi) = (Bound::Scalar(-1.0), Bound::Scalar(1.0));
        let ctx = PenaltyContext::new();
        let a = penalty(Penalty::Sns, &w, &lo, &hi, 1.0, &ctx, PenaltyOrder::Hessian).unwrap();
        let b = penalty(Penalty::Sns, &w, &lo, &hi, 3.5, &ctx, PenaltyOrder::Hessian).unwrap();
        assert_relative_eq!(b.value, 3.5 * a.value, epsilon = 1e-14);
        assert_relative_eq!(
            (b.grad.unwrap() - a.grad.unwrap().map(|v| 3.5 * v)).amax(),
            0.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            (b.hess.unwrap() - a.hess.unwrap().map(|v| 3.5 * v)).amax(),
            0.0,
            epsilon = 1e-14
        );
    }

    #[test]
    fn smoothing_penalty_prefers_smooth_waveforms() {
        let n = 20;
        let smooth = DMatrix::from_fn(n, 1, |i, _| (i as f64 / n as f64).sin());
        let rough = DMatrix::from_fn(n, 1, |i, _| if i % 2 == 0 { 1.0 } else { -1.0 });
        let ctx = PenaltyContext::new();
        let (lo, hi) = (Bound::Scalar(-1.0), Bound::Scalar(1.0));
        let s = penalty(
            Penalty::Dns,
            &smooth,
            &lo,
            &hi,
            1.0,
            &ctx,
            PenaltyOrder::Value,
        )
        .unwrap()
        .value;
        let r = penalty(
            Penalty::Dns,
            &rough,
            &lo,
            &hi,
            1.0,
            &ctx,
            PenaltyOrder::Value,
        )
        .unwrap()
        .value;
        assert!(s < r, "smooth {s} should cost less than rough {r}");
    }

    #[test]
    fn adiabaticity_penalty_runs_and_has_a_consistent_gradient() {
        let n = 16;
        let w = DMatrix::from_fn(n, 2, |i, j| {
            let t = i as f64 / n as f64;
            if j == 0 {
                (2.0 * t).cos() * 0.6
            } else {
                (2.0 * t).sin() * 0.6
            }
        });
        let mut ctx = PenaltyContext::new();
        ctx.bandwidth = Some(5000.0);
        ctx.pwr_levels = vec![2.0 * std::f64::consts::PI * 1000.0];
        let out = penalty(
            Penalty::Adiab,
            &w,
            &Bound::Scalar(-1.0),
            &Bound::Scalar(1.0),
            1.0,
            &ctx,
            PenaltyOrder::Gradient,
        )
        .unwrap();
        assert!(out.value.is_finite() && out.value >= 0.0);
        let g = out.grad.unwrap();
        assert!(g.iter().all(|v| v.is_finite()));
    }
}
