//! Single-spin propagators, their control derivatives, and the derivatives
//! carried to the end of the pulse.
//!
//! Port of `main/derivatives_vectorized.m` and `main/LL_vectorized.m`.
//!
//! One slice rotates a spin-1/2 about its effective field:
//!
//! ```text
//! h = (omega1 f, omega1 g, Omega),    s = dt h,    U = exp(-i s . sigma)
//! ```
//!
//! with `sigma` the spin-1/2 operators, half the Pauli matrices.  `U` has a
//! closed form, and so do its derivatives.  Writing `X_a = i U^dagger dU/da`,
//! which is Hermitian,
//!
//! ```text
//! X_f = dt omega1 D_1l sigma_l,    X_g = dt omega1 D_2l sigma_l,
//! D = I + c1 S + c2 S^2,    S_jl = eps_jlk s_k,
//! c1 = (cos|s| - 1)/|s|^2,    c2 = (|s| - sin|s|)/|s|^3
//! ```
//!
//! which is the `D` matrix of [`crate::rodrigues`] in the Hilbert-space form
//! ESCALADE works in.  The second derivatives differentiate `D` once more.
//!
//! # Zero field
//!
//! The MATLAB adds `eps` to every offset so that `|s|` is never exactly zero,
//! but `c1`, `c2` and their derivatives still lose every significant digit to
//! cancellation as `|s|` goes to zero.  Here they switch to their Taylor
//! series below `|s| = 0.1`, which is exact to rounding on both sides of the
//! switch, and the propagator uses `sin(|s|/2)/|s|` directly, which is finite
//! at zero.  No offset shift is needed.

use crate::optim::ObjectiveRequest;
use nalgebra::Matrix2;
use num_complex::Complex64;

/// A 2x2 complex matrix: a single-spin operator, state or propagator.
pub type Op2 = Matrix2<Complex64>;

/// Below this `|s|` the derivative coefficients come from their series.
const SERIES_BELOW: f64 = 0.1;

/// `v[0] sigma_x + v[1] sigma_y + v[2] sigma_z`, spin-1/2 operators.
pub fn spin_operator(v: [f64; 3]) -> Op2 {
    let c = Complex64::new;
    Op2::new(
        c(0.5 * v[2], 0.0),
        c(0.5 * v[0], -0.5 * v[1]),
        c(0.5 * v[0], 0.5 * v[1]),
        c(-0.5 * v[2], 0.0),
    )
}

/// One slice of the pulse for one spin.
#[derive(Debug, Clone, Copy)]
pub struct Slice {
    /// The propagator.
    pub u: Op2,
    /// `[X_f, X_g]`, the MATLAB `DF` and `DG`.
    pub first: Option<[Op2; 2]>,
    /// `[dX_f/df, dX_f/dg, dX_g/df, dX_g/dg]`, the MATLAB `DFF`, `DFG`, `DGF`
    /// and `DGG`.
    pub second: Option<[Op2; 4]>,
}

/// The propagator of one slice and, on request, its derivatives with respect
/// to the two control amplitudes.
///
/// `omega1` is the nominal field in rad/s, `f` and `g` the dimensionless x
/// and y amplitudes, `offset` the resonance offset in rad/s.
pub fn slice(
    dt: f64,
    omega1: f64,
    f: f64,
    g: f64,
    offset: f64,
    request: ObjectiveRequest,
) -> Slice {
    let s = [dt * omega1 * f, dt * omega1 * g, dt * offset];
    let ns = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();

    let half_sinc = if ns == 0.0 {
        0.5
    } else {
        (0.5 * ns).sin() / ns
    };
    let a = Complex64::new((0.5 * ns).cos(), -s[2] * half_sinc);
    let b = Complex64::new(-s[1] * half_sinc, -s[0] * half_sinc);
    let c = Complex64::new(s[1] * half_sinc, -s[0] * half_sinc);
    let u = Op2::new(a, b, c, a.conj());

    if request == ObjectiveRequest::Value {
        return Slice {
            u,
            first: None,
            second: None,
        };
    }

    let (c1, c2, dc1, dc2) = coefficients(ns);

    // First two rows of S and S^2; the third is never needed.
    let sm = [[0.0, s[2], -s[1]], [-s[2], 0.0, s[0]]];
    let s2m = [
        [-s[1] * s[1] - s[2] * s[2], s[0] * s[1], s[0] * s[2]],
        [s[0] * s[1], -s[0] * s[0] - s[2] * s[2], s[1] * s[2]],
    ];

    let scale = dt * omega1;
    let d_row = |j: usize| -> [f64; 3] {
        std::array::from_fn(|l| scale * (f64::from(j == l) + c1 * sm[j][l] + c2 * s2m[j][l]))
    };
    let first = Some([spin_operator(d_row(0)), spin_operator(d_row(1))]);

    if request == ObjectiveRequest::Gradient {
        return Slice {
            u,
            first,
            second: None,
        };
    }

    // dD/ds_1 and dD/ds_2: the chain rule through |s|, then the anticommutator
    // of S with dS/ds_k, then c1 dS/ds_k.
    let s_anti_f = [[0.0, s[1], s[2]], [s[1], -2.0 * s[0], 0.0]];
    let s_anti_g = [[-2.0 * s[1], s[0], 0.0], [s[0], 0.0, s[2]]];
    let mut dd_f = [[0.0; 3]; 2];
    let mut dd_g = [[0.0; 3]; 2];
    for j in 0..2 {
        for l in 0..3 {
            dd_f[j][l] = dc1 * s[0] * sm[j][l] + dc2 * s[0] * s2m[j][l] + c2 * s_anti_f[j][l];
            dd_g[j][l] = dc1 * s[1] * sm[j][l] + dc2 * s[1] * s2m[j][l] + c2 * s_anti_g[j][l];
        }
    }
    dd_f[1][2] += c1;
    dd_g[0][2] -= c1;

    let scale2 = scale * scale;
    let op = |row: [f64; 3]| spin_operator(row.map(|v| scale2 * v));
    Slice {
        u,
        first,
        second: Some([op(dd_f[0]), op(dd_g[0]), op(dd_f[1]), op(dd_g[1])]),
    }
}

/// `(c1, c2, c1'/x, c2'/x)` at `x = |s|`.
///
/// The derivatives come divided by `x` because they are only ever used
/// multiplied by `s_k / x`, and the quotient is what stays finite at zero.
fn coefficients(x: f64) -> (f64, f64, f64, f64) {
    if x < SERIES_BELOW {
        let x2 = x * x;
        (
            -0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 / 40320.0)),
            1.0 / 6.0 + x2 * (-1.0 / 120.0 + x2 * (1.0 / 5040.0 - x2 / 362_880.0)),
            1.0 / 12.0 + x2 * (-1.0 / 180.0 + x2 * (1.0 / 6720.0 - x2 / 453_600.0)),
            -1.0 / 60.0 + x2 * (1.0 / 1260.0 + x2 * (-1.0 / 60480.0 + x2 / 4_989_600.0)),
        )
    } else {
        let (cos, sin) = (x.cos(), x.sin());
        let x2 = x * x;
        let x3 = x2 * x;
        let dc1 = -2.0 * (cos - 1.0) / x3 - sin / x2;
        let dc2 = (1.0 - cos) / x3 - 3.0 * (x - sin) / (x2 * x2);
        ((cos - 1.0) / x2, (x - sin) / x3, dc1 / x, dc2 / x)
    }
}

/// One spin's journey through the whole pulse.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// `U_N ... U_1`, the MATLAB `UT`.
    pub total: Op2,
    /// Per slice, `[L X_f L^dagger, L X_g L^dagger]` with `L = U_N ... U_n`:
    /// the MATLAB `LLF` and `LLG`.  Empty unless derivatives were requested.
    pub first: Vec<[Op2; 2]>,
    /// Per slice, `i L dX L^dagger` for the four second derivatives: the
    /// MATLAB `DDFF`, `DDFG`, `DDGF` and `DDGG`.  Empty unless requested.
    pub second: Vec<[Op2; 4]>,
}

/// Multiply the slices together, carrying each slice's derivatives to the end
/// of the pulse as the product grows backwards from the last slice.
pub fn trajectory(slices: &[Slice]) -> Trajectory {
    let n = slices.len();
    let has_first = slices.first().is_some_and(|s| s.first.is_some());
    let has_second = slices.first().is_some_and(|s| s.second.is_some());
    let mut first = vec![[Op2::zeros(); 2]; if has_first { n } else { 0 }];
    let mut second = vec![[Op2::zeros(); 4]; if has_second { n } else { 0 }];

    let i = Complex64::new(0.0, 1.0);
    let mut total = Op2::identity();
    for (k, sl) in slices.iter().enumerate().rev() {
        total *= sl.u;
        let adj = total.adjoint();
        if let Some(d) = sl.first {
            first[k] = d.map(|x| total * x * adj);
        }
        if let Some(d) = sl.second {
            second[k] = d.map(|x| total * x * adj * i);
        }
    }
    Trajectory {
        total,
        first,
        second,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::CDense;
    use crate::propagate::expm_pade;

    fn dense(op: &Op2) -> CDense {
        CDense::from_iterator(2, 2, op.iter().cloned())
    }

    fn close(a: &Op2, b: &Op2) -> f64 {
        (a - b).norm()
    }

    const DT: f64 = 2e-6;
    const W1: f64 = 2.0 * std::f64::consts::PI * 17000.0;

    /// Points on both sides of the series switch, near zero field, and well
    /// away from it.
    fn points() -> Vec<(f64, f64, f64)> {
        vec![
            (0.3, -0.7, 2.0 * std::f64::consts::PI * 4000.0),
            (-0.9, 0.2, -2.0 * std::f64::consts::PI * 9000.0),
            (0.0, 0.0, 0.0),
            (1e-4, -2e-4, 30.0),
            // |s| just below and just above 0.1.
            (0.46, 0.0, 0.0),
            (0.48, 0.0, 0.0),
            (1.3, 1.1, 2.0 * std::f64::consts::PI * 20000.0),
        ]
    }

    #[test]
    fn the_propagator_is_the_matrix_exponential() {
        for (f, g, om) in points() {
            let got = slice(DT, W1, f, g, om, ObjectiveRequest::Value).u;
            let h = spin_operator([W1 * f, W1 * g, om]);
            let want = expm_pade(&dense(&h).map(|z| z * Complex64::new(0.0, -DT))).unwrap();
            assert!((dense(&got) - want).norm() < 1e-13, "{f} {g} {om}");
        }
    }

    #[test]
    fn first_derivatives_match_finite_differences() {
        let h = 1e-6;
        let i = Complex64::new(0.0, 1.0);
        for (f, g, om) in points() {
            let sl = slice(DT, W1, f, g, om, ObjectiveRequest::Gradient);
            let [xf, xg] = sl.first.unwrap();
            let u = |f, g| slice(DT, W1, f, g, om, ObjectiveRequest::Value).u;
            let du_f = (u(f + h, g) - u(f - h, g)) / Complex64::new(2.0 * h, 0.0);
            let du_g = (u(f, g + h) - u(f, g - h)) / Complex64::new(2.0 * h, 0.0);
            let adj = sl.u.adjoint();
            assert!(close(&xf, &(adj * du_f * i)) < 1e-8, "X_f at {f} {g} {om}");
            assert!(close(&xg, &(adj * du_g * i)) < 1e-8, "X_g at {f} {g} {om}");
        }
    }

    #[test]
    fn second_derivatives_match_finite_differences() {
        let h = 1e-5;
        for (f, g, om) in points() {
            let sl = slice(DT, W1, f, g, om, ObjectiveRequest::Hessian);
            let x = |f, g| {
                slice(DT, W1, f, g, om, ObjectiveRequest::Gradient)
                    .first
                    .unwrap()
            };
            let two_h = Complex64::new(2.0 * h, 0.0);
            let (pf, mf) = (x(f + h, g), x(f - h, g));
            let (pg, mg) = (x(f, g + h), x(f, g - h));
            let want = [
                (pf[0] - mf[0]) / two_h,
                (pg[0] - mg[0]) / two_h,
                (pf[1] - mf[1]) / two_h,
                (pg[1] - mg[1]) / two_h,
            ];
            let got = sl.second.unwrap();
            for k in 0..4 {
                let scale = want[k].norm().max(1e-3);
                assert!(
                    close(&got[k], &want[k]) / scale < 1e-6,
                    "second derivative {k} at {f} {g} {om}: {} vs {}",
                    got[k],
                    want[k]
                );
            }
        }
    }

    #[test]
    fn the_coefficients_are_continuous_across_the_series_switch() {
        let below = coefficients(SERIES_BELOW * (1.0 - 1e-12));
        let above = coefficients(SERIES_BELOW * (1.0 + 1e-12));
        for (a, b) in [
            (below.0, above.0),
            (below.1, above.1),
            (below.2, above.2),
            (below.3, above.3),
        ] {
            assert!((a - b).abs() < 1e-11, "{a} vs {b}");
        }
    }

    #[test]
    fn the_trajectory_is_the_ordered_product() {
        let slices: Vec<Slice> = [(0.2, 0.5), (-0.4, 0.1), (0.9, -0.8)]
            .iter()
            .map(|&(f, g)| slice(DT, W1, f, g, 5000.0, ObjectiveRequest::Gradient))
            .collect();
        let t = trajectory(&slices);
        let want = slices[2].u * slices[1].u * slices[0].u;
        assert!(close(&t.total, &want) < 1e-14);
        // The first slice's derivative is carried through all three.
        let l = want;
        let carried = l * slices[0].first.unwrap()[0] * l.adjoint();
        assert!(close(&t.first[0][0], &carried) < 1e-14);
        assert!(t.second.is_empty());
    }
}
