//! Matrix exponentials and single-step propagation.
//!
//! Port of `kernel/propagate.m`, which computes `expm(-1i*L*timestep)` by one
//! of three routes and optionally applies it to a state.  The time step is
//! complex here because the higher-order operator splittings (order 3 in
//! particular) use complex coefficients.

use crate::error::{QoalaError, Result};
use crate::linalg::{norm_2_dense, norm_2_state, CDense, CMat, C1, CI, EPS};
use crate::types::{PropMethod, StateSpace};
use num_complex::Complex64;

/// Hard cap on series length, so a pathological input reports a failure
/// instead of spinning forever the way the MATLAB `while` loops can.
const MAX_SERIES_TERMS: usize = 10_000;

/// `expm(-1i*L*timestep)` as an operator.
///
/// Mirrors `propagate(space, L, timestep, method, nonzero_tol)` called with
/// five arguments.  [`PropMethod::Krylov`] is rejected because it only makes
/// sense applied to a state.
pub fn propagator(
    l: &CMat,
    timestep: Complex64,
    method: PropMethod,
    nonzero_tol: f64,
) -> Result<CMat> {
    let a = l.scale(-CI * timestep);
    match method {
        PropMethod::Pade => {
            let p = expm_pade(&a.to_dense())?;
            Ok(if l.is_dense() {
                CMat::Dense(p)
            } else {
                CMat::Dense(p).into_sparse()
            })
        }
        PropMethod::Taylor => expm_taylor(&a, nonzero_tol),
        PropMethod::Krylov => Err(QoalaError::BadValue(
            "the krylov method needs a state to propagate".into(),
        )),
    }
}

/// `expm(-1i*L*timestep)` applied to a state.
///
/// Mirrors `propagate(space, L, timestep, method, nonzero_tol, rho)`: in
/// Liouville space the propagator multiplies the state from the left, in
/// Hilbert space it conjugates it.
pub fn propagate_state(
    space: StateSpace,
    l: &CMat,
    timestep: Complex64,
    method: PropMethod,
    nonzero_tol: f64,
    rho: &CDense,
) -> Result<CDense> {
    match method {
        PropMethod::Krylov => krylov_step(l, timestep, rho),
        _ => {
            let p = propagator(l, timestep, method, nonzero_tol)?;
            let out = p.apply(rho)?;
            Ok(match space {
                StateSpace::Liouville => out,
                StateSpace::Hilbert => {
                    let pd = p.to_dense();
                    out * pd.adjoint()
                }
            })
        }
    }
}

/// Truncated Taylor series with scaling, squaring and element chopping.
///
/// This is `propagate(...,'taylor',...)`: the norm sets the number of
/// squarings, every intermediate is quantised to `nonzero_tol`, and the series
/// stops when a term chops to nothing.
pub fn expm_taylor(a_in: &CMat, nonzero_tol: f64) -> Result<CMat> {
    let mat_norm = a_in.norm_2();
    let n_squarings = if mat_norm > 0.0 {
        mat_norm.log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let scaling_factor = 2f64.powi(n_squarings as i32);

    let mut a = if scaling_factor > 1.0 {
        a_in.scale(Complex64::new(1.0 / scaling_factor, 0.0))
    } else {
        a_in.clone()
    };
    a = a.chop(nonzero_tol);

    let dim = a.nrows();
    let mut p = CMat::identity(dim);
    let mut next_term = CMat::identity(dim);
    let mut n = 1usize;

    while next_term.nnz() > 0 {
        if n > MAX_SERIES_TERMS {
            return Err(QoalaError::Numerical(
                "Taylor series for expm did not terminate".into(),
            ));
        }
        next_term = a
            .matmul(&next_term)?
            .scale(Complex64::new(1.0 / n as f64, 0.0))
            .chop(nonzero_tol);
        p = p.add(&next_term)?;
        n += 1;
    }

    p = p.chop(nonzero_tol);
    for _ in 0..n_squarings {
        p = p.matmul(&p)?.chop(nonzero_tol);
    }
    Ok(p)
}

/// Krylov-style propagation of a state, as in `propagate(...,'krylov',...,rho)`.
///
/// The time step is broken into `ceil(||L||_2 |dt| / 2)` sub-steps and each
/// sub-step is summed as a series until every element falls below `eps`.
pub fn krylov_step(l: &CMat, timestep: Complex64, rho: &CDense) -> Result<CDense> {
    let mut rho = rho.clone();
    let norm_mat = l.norm_2() * timestep.norm();
    let nsteps = (norm_mat / 2.0).ceil().max(0.0) as usize;

    let scaling = norm_2_state(&rho).max(1.0);
    rho /= Complex64::new(scaling, 0.0);

    for _ in 0..nsteps {
        let mut next_term = rho.clone();
        let mut k = 1usize;
        loop {
            if next_term.iter().all(|z| z.norm() <= EPS) {
                break;
            }
            if k > MAX_SERIES_TERMS {
                return Err(QoalaError::Numerical(
                    "Krylov series did not terminate".into(),
                ));
            }
            let coeff = -CI * timestep / Complex64::new((k * nsteps) as f64, 0.0);
            next_term = l.apply(&next_term)? * coeff;
            rho += &next_term;
            k += 1;
        }
    }

    Ok(rho * Complex64::new(scaling, 0.0))
}

/// Scaling-and-squaring Pade approximant of `expm`, the algorithm behind
/// MATLAB's `expm` (Higham, SIAM J. Matrix Anal. Appl. 26 (2005) 1179).
pub fn expm_pade(a: &CDense) -> Result<CDense> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(QoalaError::Dimension("expm needs a square matrix".into()));
    }
    if n == 0 {
        return Ok(a.clone());
    }

    let norm_1 = |m: &CDense| -> f64 {
        (0..m.ncols())
            .map(|c| (0..m.nrows()).map(|r| m[(r, c)].norm()).sum::<f64>())
            .fold(0.0, f64::max)
    };

    let a1 = norm_1(a);
    if !a1.is_finite() {
        return Err(QoalaError::Numerical("expm input is not finite".into()));
    }

    const THETA: [f64; 5] = [
        1.495_585_217_958_292e-2,
        2.539_398_330_063_23e-1,
        9.504_178_996_162_932e-1,
        2.097_847_961_257_068e0,
        5.371_920_351_148_152e0,
    ];

    for (i, m) in [3usize, 5, 7, 9].iter().enumerate() {
        if a1 <= THETA[i] {
            return pade_small(a, *m);
        }
    }

    let s = if a1 > THETA[4] {
        (a1 / THETA[4]).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let scaled = a.map(|z| z / 2f64.powi(s as i32));
    let mut f = pade13(&scaled)?;
    for _ in 0..s {
        f = &f * &f;
    }
    Ok(f)
}

/// Pade approximants of order 3, 5, 7 and 9.
fn pade_small(a: &CDense, m: usize) -> Result<CDense> {
    let b: &[f64] = match m {
        3 => &[120.0, 60.0, 12.0, 1.0],
        5 => &[30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0],
        7 => &[
            17_297_280.0,
            8_648_640.0,
            1_995_840.0,
            277_200.0,
            25_200.0,
            1512.0,
            56.0,
            1.0,
        ],
        9 => &[
            17_643_225_600.0,
            8_821_612_800.0,
            2_075_673_600.0,
            302_702_400.0,
            30_270_240.0,
            2_162_160.0,
            110_880.0,
            3960.0,
            90.0,
            1.0,
        ],
        _ => return Err(QoalaError::BadValue(format!("no Pade table for order {m}"))),
    };

    let n = a.nrows();
    let ident = CDense::identity(n, n);
    let a2 = a * a;

    // Even powers A^0, A^2, A^4, ... up to A^(m-1).
    let mut powers = vec![ident.clone()];
    while powers.len() * 2 <= m {
        let last = powers.last().unwrap().clone();
        powers.push(&last * &a2);
    }

    let mut u = CDense::zeros(n, n);
    let mut v = CDense::zeros(n, n);
    for (k, p) in powers.iter().enumerate() {
        if 2 * k + 1 < b.len() {
            u += p.map(|z| z * b[2 * k + 1]);
        }
        if 2 * k < b.len() {
            v += p.map(|z| z * b[2 * k]);
        }
    }
    let u = a * u;
    solve_pade(&u, &v)
}

/// The order-13 Pade approximant with Higham's economical evaluation scheme.
fn pade13(a: &CDense) -> Result<CDense> {
    const B: [f64; 14] = [
        64_764_752_532_480_000.0,
        32_382_376_266_240_000.0,
        7_771_770_303_897_600.0,
        1_187_353_796_428_800.0,
        129_060_195_264_000.0,
        10_559_470_521_600.0,
        670_442_572_800.0,
        33_522_128_640.0,
        1_323_241_920.0,
        40_840_800.0,
        960_960.0,
        16_380.0,
        182.0,
        1.0,
    ];
    let n = a.nrows();
    let ident = CDense::identity(n, n);
    let a2 = a * a;
    let a4 = &a2 * &a2;
    let a6 = &a2 * &a4;

    let inner_u = a6.map(|z| z * B[13]) + a4.map(|z| z * B[11]) + a2.map(|z| z * B[9]);
    let u = a
        * (&a6 * &inner_u
            + a6.map(|z| z * B[7])
            + a4.map(|z| z * B[5])
            + a2.map(|z| z * B[3])
            + ident.map(|z| z * B[1]));

    let inner_v = a6.map(|z| z * B[12]) + a4.map(|z| z * B[10]) + a2.map(|z| z * B[8]);
    let v = &a6 * &inner_v
        + a6.map(|z| z * B[6])
        + a4.map(|z| z * B[4])
        + a2.map(|z| z * B[2])
        + ident.map(|z| z * B[0]);

    solve_pade(&u, &v)
}

/// Solve `(V - U) F = (V + U)` for the Pade quotient.
fn solve_pade(u: &CDense, v: &CDense) -> Result<CDense> {
    let lhs = v - u;
    let rhs = v + u;
    lhs.lu()
        .solve(&rhs)
        .ok_or_else(|| QoalaError::Numerical("singular Pade denominator in expm".into()))
}

/// `expm` of a matrix that is already `-i L dt`, chosen by method.  Small
/// convenience used by tests and by callers that already have the generator.
pub fn expm(a: &CMat, method: PropMethod, nonzero_tol: f64) -> Result<CMat> {
    match method {
        PropMethod::Pade => Ok(CMat::Dense(expm_pade(&a.to_dense())?)),
        PropMethod::Taylor => expm_taylor(a, nonzero_tol),
        PropMethod::Krylov => Err(QoalaError::BadValue(
            "krylov needs a state; use krylov_step".into(),
        )),
    }
}

/// Spectral norm helper re-exported for callers that hold a dense matrix.
pub fn spectral_norm(a: &CDense) -> f64 {
    norm_2_dense(a)
}

/// Identity of the same size, as a dense matrix.
pub fn dense_identity(n: usize) -> CDense {
    CDense::from_diagonal_element(n, n, C1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::CMat;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// exp(-i * theta * sigma_z / 2) has a closed form we can check against.
    #[test]
    #[rustfmt::skip]
    fn pade_matches_analytic_pauli_rotation() {
        let theta = 0.7_f64;
        let sz = CDense::from_row_slice(2, 2, &[c(0.5, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-0.5, 0.0)]);
        let a = sz.map(|z| z * c(0.0, -theta));
        let p = expm_pade(&a).unwrap();
        assert_relative_eq!(p[(0, 0)].re, (theta / 2.0).cos(), epsilon = 1e-13);
        assert_relative_eq!(p[(0, 0)].im, -(theta / 2.0).sin(), epsilon = 1e-13);
        assert_relative_eq!(p[(1, 1)].re, (theta / 2.0).cos(), epsilon = 1e-13);
        assert_relative_eq!(p[(1, 1)].im, (theta / 2.0).sin(), epsilon = 1e-13);
    }

    #[test]
    #[rustfmt::skip]
    fn pade_handles_large_norms_via_squaring() {
        // A big non-normal matrix, where scaling and squaring actually matters.
        let a = CDense::from_row_slice(
            2,
            2,
            &[c(0.0, 0.0), c(30.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        );
        // Nilpotent: exp(A) = I + A exactly.
        let p = expm_pade(&a).unwrap();
        assert_relative_eq!(p[(0, 1)].re, 30.0, epsilon = 1e-9);
        assert_relative_eq!(p[(0, 0)].re, 1.0, epsilon = 1e-12);
    }

    #[test]
    #[rustfmt::skip]
    fn taylor_and_pade_agree() {
        let l = CMat::Dense(CDense::from_row_slice(
            3,
            3,
            &[
                c(0.3, 0.0), c(0.1, -0.2), c(0.0, 0.0),
                c(0.1, 0.2), c(-0.4, 0.0), c(0.05, 0.0),
                c(0.0, 0.0), c(0.05, 0.0), c(0.9, 0.0),
            ],
        ));
        let dt = Complex64::new(0.37, 0.0);
        let p_taylor = propagator(&l, dt, PropMethod::Taylor, 1e-14).unwrap().to_dense();
        let p_pade = propagator(&l, dt, PropMethod::Pade, 1e-14).unwrap().to_dense();
        assert_relative_eq!((p_taylor - p_pade).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    #[rustfmt::skip]
    fn krylov_agrees_with_pade_applied_to_a_state() {
        let l = CMat::Dense(CDense::from_row_slice(
            3,
            3,
            &[
                c(1.0, 0.0), c(0.4, -0.1), c(0.0, 0.0),
                c(0.4, 0.1), c(-2.0, 0.0), c(0.7, 0.0),
                c(0.0, 0.0), c(0.7, 0.0), c(3.0, 0.0),
            ],
        ));
        let rho = CDense::from_column_slice(3, 1, &[c(0.6, 0.0), c(0.0, 0.0), c(0.8, 0.0)]);
        let dt = Complex64::new(0.9, 0.0);
        let a = krylov_step(&l, dt, &rho).unwrap();
        let b = propagate_state(StateSpace::Liouville, &l, dt, PropMethod::Pade, 1e-14, &rho).unwrap();
        assert_relative_eq!((a - b).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    #[rustfmt::skip]
    fn complex_time_steps_are_supported() {
        // Order-3 splittings use complex coefficients; check against Pade.
        let l = CMat::Dense(CDense::from_row_slice(
            2,
            2,
            &[c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        ));
        let dt = Complex64::new(0.2, 0.15);
        let p_t = propagator(&l, dt, PropMethod::Taylor, 1e-15).unwrap().to_dense();
        let p_p = propagator(&l, dt, PropMethod::Pade, 1e-15).unwrap().to_dense();
        assert_relative_eq!((p_t - p_p).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    #[rustfmt::skip]
    fn hilbert_space_propagation_conjugates() {
        let l = CMat::Dense(CDense::from_row_slice(
            2,
            2,
            &[c(0.0, 0.0), c(0.5, 0.0), c(0.5, 0.0), c(0.0, 0.0)],
        ));
        let rho = CDense::from_row_slice(2, 2, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
        let dt = Complex64::new(0.4, 0.0);
        let out = propagate_state(StateSpace::Hilbert, &l, dt, PropMethod::Pade, 1e-14, &rho).unwrap();
        // A conjugated projector stays a projector: trace 1 and rho^2 = rho.
        assert_relative_eq!(out.trace().re, 1.0, epsilon = 1e-12);
        assert_relative_eq!((&out * &out - &out).norm(), 0.0, epsilon = 1e-12);
    }
}
