//! Spin-1/2 operators in the spherical tensor Liouville basis.
//!
//! The QOALA example scripts build every operator from the same handful of
//! 4x4 single-spin matrices and a lot of Kronecker products.  Those building
//! blocks live here so that a system can be described in a line or two rather
//! than a page of literals.
//!
//! The single-spin Liouville space is four-dimensional, spanned by the unit
//! state and the three rank-1 spherical tensors, and `Jx`, `Jy`, `Jz` are the
//! commutation superoperators of the Cartesian spin operators in that space.

use crate::error::Result;
use crate::linalg::{CDense, CMat};
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Single-spin Liouville-space identity.
pub fn identity() -> CMat {
    CMat::identity(4)
}

/// Single-spin `Jx` commutation superoperator.
#[rustfmt::skip]
pub fn jx() -> CMat {
    let s = 1.0 / std::f64::consts::SQRT_2;
    dense4(&[
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, s, 0.0,
        0.0, s, 0.0, s,
        0.0, 0.0, s, 0.0,
    ])
}

/// Single-spin `Jy` commutation superoperator.
#[rustfmt::skip]
pub fn jy() -> CMat {
    let s = 1.0 / std::f64::consts::SQRT_2;
    CMat::Dense(CDense::from_row_slice(
        4,
        4,
        &[
            c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
            c(0.0, 0.0), c(0.0, 0.0), c(0.0, -s), c(0.0, 0.0),
            c(0.0, 0.0), c(0.0, s), c(0.0, 0.0), c(0.0, -s),
            c(0.0, 0.0), c(0.0, 0.0), c(0.0, s), c(0.0, 0.0),
        ],
    ))
    .into_sparse()
}

/// Single-spin `Jz` commutation superoperator.
#[rustfmt::skip]
pub fn jz() -> CMat {
    dense4(&[
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, -1.0,
    ])
}

fn dense4(v: &[f64; 16]) -> CMat {
    CMat::Dense(CDense::from_row_slice(
        4,
        4,
        &v.iter().map(|x| c(*x, 0.0)).collect::<Vec<_>>(),
    ))
    .into_sparse()
}

/// The unit state of one spin.
pub fn unit_state() -> CDense {
    CDense::from_column_slice(4, 1, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)])
}

/// The z-magnetisation state of one spin, normalised as the examples do.
pub fn z_state_single() -> CDense {
    CDense::from_column_slice(4, 1, &[c(0.0, 0.0), c(0.0, 0.0), c(0.5, 0.0), c(0.0, 0.0)])
}

/// `unit * rho_z' + rho_z * unit'`: the object that turns a `Jz` commutation
/// superoperator on one spin into the `zz` product operator across two.
pub fn zz_projector() -> CMat {
    let u = unit_state();
    let z = z_state_single();
    CMat::Dense(&u * z.adjoint() + &z * u.adjoint()).into_sparse()
}

/// The raising-operator basis state, `T(1,+1)`.
pub fn p_state() -> CDense {
    let s = -1.0 / std::f64::consts::SQRT_2;
    CDense::from_column_slice(4, 1, &[c(0.0, 0.0), c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0)])
}

/// The lowering-operator basis state, `T(1,-1)`.
pub fn m_state() -> CDense {
    let s = 1.0 / std::f64::consts::SQRT_2;
    CDense::from_column_slice(4, 1, &[c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)])
}

/// The x-magnetisation state of one spin.
pub fn x_state_single() -> CDense {
    (p_state() + m_state()).map(|z| z / Complex64::new(2.0, 0.0))
}

/// The y-magnetisation state of one spin.
pub fn y_state_single() -> CDense {
    (p_state() - m_state()).map(|z| z / Complex64::new(0.0, 2.0))
}

/// Which Cartesian direction a product operator acts along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductAxis {
    /// `xx` product operator.
    X,
    /// `yy` product operator.
    Y,
    /// `zz` product operator.
    Z,
}

impl ProductAxis {
    fn operator(self) -> CMat {
        match self {
            ProductAxis::X => jx(),
            ProductAxis::Y => jy(),
            ProductAxis::Z => jz(),
        }
    }
    fn projector(self) -> CMat {
        let u = unit_state();
        let r = match self {
            ProductAxis::X => x_state_single(),
            ProductAxis::Y => y_state_single(),
            ProductAxis::Z => z_state_single(),
        };
        CMat::Dense(&u * r.adjoint() + &r * u.adjoint()).into_sparse()
    }
}

/// Kronecker product of a list of operators, left to right.
pub fn kron_all(ops: &[CMat]) -> CMat {
    let mut out = ops[0].clone();
    for op in &ops[1..] {
        out = out.kron(op);
    }
    out
}

/// Kronecker product of a list of state vectors, left to right.
pub fn kron_states(states: &[CDense]) -> CDense {
    let mut out = states[0].clone();
    for s in &states[1..] {
        out = out.kronecker(s);
    }
    out
}

/// Embed a single-spin operator into an `nspins` composite space.
pub fn embed(nspins: usize, spin: usize, op: &CMat) -> CMat {
    let ops: Vec<CMat> = (0..nspins)
        .map(|s| if s == spin { op.clone() } else { identity() })
        .collect();
    kron_all(&ops)
}

/// Composite-space Cartesian operators `(Lx, Ly, Lz)` for every spin.
pub fn cartesian_operators(nspins: usize) -> Vec<(CMat, CMat, CMat)> {
    (0..nspins)
        .map(|s| {
            (
                embed(nspins, s, &jx()),
                embed(nspins, s, &jy()),
                embed(nspins, s, &jz()),
            )
        })
        .collect()
}

/// A bilinear product-operator interaction superoperator between two spins.
///
/// `product_coupling(n, i, j, Z)` is the `zz` term used by the weak-coupling
/// examples; summing all three axes gives the isotropic (strong-coupling)
/// term.  The normalisation is such that a coupling of `J` Hz contributes
/// `2*pi*J*L`.
pub fn product_coupling(nspins: usize, i: usize, j: usize, axis: ProductAxis) -> Result<CMat> {
    let p = axis.projector();
    let o = axis.operator();
    let term = |a: usize, first: &CMat, b: usize, second: &CMat| -> CMat {
        let ops: Vec<CMat> = (0..nspins)
            .map(|s| {
                if s == a {
                    first.clone()
                } else if s == b {
                    second.clone()
                } else {
                    identity()
                }
            })
            .collect();
        kron_all(&ops)
    };
    term(i, &o, j, &p).add(&term(i, &p, j, &o))
}

/// The weak-coupling (`zz`) interaction superoperator between two spins.
pub fn zz_coupling(nspins: usize, i: usize, j: usize) -> Result<CMat> {
    product_coupling(nspins, i, j, ProductAxis::Z)
}

/// The isotropic (strong-coupling) interaction superoperator: `xx + yy + zz`.
pub fn isotropic_coupling(nspins: usize, i: usize, j: usize) -> Result<CMat> {
    product_coupling(nspins, i, j, ProductAxis::X)?
        .add(&product_coupling(nspins, i, j, ProductAxis::Y)?)?
        .add(&product_coupling(nspins, i, j, ProductAxis::Z)?)
}

/// The z-magnetisation state of one spin inside an `nspins` composite space.
pub fn z_state(nspins: usize, spin: usize) -> CDense {
    let states: Vec<CDense> = (0..nspins)
        .map(|s| {
            if s == spin {
                z_state_single()
            } else {
                unit_state()
            }
        })
        .collect();
    kron_states(&states)
}

/// The SWAP propagator between two spins, as a Liouville-space superoperator.
///
/// Built from Linden's 1999 decomposition (eq. 14), which is how every QOALA
/// example script constructs its gate target: three `zz` evolutions
/// interleaved with hard pulses.
pub fn swap_gate(nspins: usize, a: usize, b: usize) -> Result<CDense> {
    use crate::propagate::expm_pade;
    let pi = std::f64::consts::PI;
    let ops = cartesian_operators(nspins);
    let (xa, ya, za) = (&ops[a].0, &ops[a].1, &ops[a].2);
    let (xb, yb, zb) = (&ops[b].0, &ops[b].1, &ops[b].2);
    let two_lzz = zz_coupling(nspins, a, b)?.scale(Complex64::new(2.0, 0.0));

    let rot = |g: &CMat, angle: f64| -> Result<CDense> {
        expm_pade(&g.to_dense().map(|z| z * Complex64::new(0.0, -angle)))
    };

    // Applied innermost first, exactly as the nested expm calls in the
    // MATLAB scripts are evaluated.
    let mut u = rot(ya, pi / 2.0)?;
    u = rot(&two_lzz, pi / 2.0)? * u;
    u = rot(&xa.add(xb)?, pi / 2.0)? * u;
    u = rot(&two_lzz, pi / 2.0)? * u;
    u = rot(&ya.add(yb)?.scale(Complex64::new(-1.0, 0.0)), pi / 2.0)? * u;
    u = rot(&two_lzz, pi / 2.0)? * u;
    u = rot(&xa.scale(Complex64::new(-1.0, 0.0)), pi / 2.0)? * u;
    u = rot(&za.add(zb)?, 3.0 * pi / 2.0)? * u;

    // Clean up, as the scripts do.
    Ok(CMat::Dense(u).chop(1e-9).to_dense())
}

/// The `nspins`-spin Cartesian operator array in the layout the drivers want:
/// one row per spin, `(Lx, Ly, Lz)` per control pair, with `None` where a
/// channel does not drive that spin.
///
/// This is the "one control pair per spin" arrangement used by most of the
/// examples.
pub fn one_pair_per_spin(nspins: usize) -> Vec<Vec<Option<CMat>>> {
    let ops = cartesian_operators(nspins);
    (0..nspins)
        .map(|s| {
            let mut row: Vec<Option<CMat>> = vec![None; 3 * nspins];
            row[3 * s] = Some(ops[s].0.clone());
            row[3 * s + 1] = Some(ops[s].1.clone());
            row[3 * s + 2] = Some(ops[s].2.clone());
            row
        })
        .collect()
}

/// A single control pair driving every spin at once.
pub fn one_pair_all_spins(nspins: usize) -> Vec<Vec<Option<CMat>>> {
    let ops = cartesian_operators(nspins);
    ops.into_iter()
        .map(|(x, y, z)| vec![Some(x), Some(y), Some(z)])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::CMat;
    use approx::assert_relative_eq;

    #[test]
    fn single_spin_operators_close_the_algebra() {
        let (x, y, z) = (jx(), jy(), jz());
        let comm = x
            .matmul(&y)
            .unwrap()
            .add(&y.matmul(&x).unwrap().scale(Complex64::new(-1.0, 0.0)))
            .unwrap();
        let want = z.scale(Complex64::new(0.0, 1.0));
        assert_relative_eq!(
            (comm.to_dense() - want.to_dense()).norm(),
            0.0,
            epsilon = 1e-14
        );
    }

    /// The zz coupling superoperator must reproduce the literal interaction
    /// matrix written out in the two-spin example scripts.
    #[test]
    #[rustfmt::skip]
    fn zz_coupling_matches_the_example_matrix() {
        let lzz = zz_coupling(2, 0, 1).unwrap();
        let h = lzz.scale(Complex64::new(2.0 * std::f64::consts::PI * 140.0, 0.0));

        // pi*140 times the matrix written out in z2z_2spin_1.m, as
        // (row, col, value) with one-based indices from the script.
        let entries = [
            (2, 10, 1.0),
            (4, 12, -1.0),
            (5, 7, 1.0),
            (7, 5, 1.0),
            (10, 2, 1.0),
            (12, 4, -1.0),
            (13, 15, -1.0),
            (15, 13, -1.0),
        ];
        let scale = std::f64::consts::PI * 140.0;
        let want = CMat::from_triplets(
            16,
            16,
            entries
                .iter()
                .map(|&(r, c, v)| (r - 1, c - 1, Complex64::new(scale * v, 0.0))),
        )
        .unwrap();
        assert_relative_eq!(
            (h.to_dense() - want.to_dense()).norm(),
            0.0,
            epsilon = 1e-10
        );
    }

    /// The isotropic coupling superoperator must reproduce the literal
    /// strong-coupling matrix written out in z2z_2spin_2.m.
    #[test]
    #[rustfmt::skip]
    fn isotropic_coupling_matches_the_example_matrix() {
        let l = isotropic_coupling(2, 0, 1).unwrap();
        let h = l.scale(Complex64::new(2.0 * std::f64::consts::PI * 20.0, 0.0));

        let entries: [(usize, usize, f64); 24] = [
            (2, 7, -1.0), (2, 10, 1.0),
            (3, 8, -1.0), (3, 14, 1.0),
            (4, 12, -1.0), (4, 15, 1.0),
            (5, 7, 1.0), (5, 10, -1.0),
            (7, 2, -1.0), (7, 5, 1.0),
            (8, 3, -1.0), (8, 9, 1.0),
            (9, 8, 1.0), (9, 14, -1.0),
            (10, 2, 1.0), (10, 5, -1.0),
            (12, 4, -1.0), (12, 13, 1.0),
            (13, 12, 1.0), (13, 15, -1.0),
            (14, 3, 1.0), (14, 9, -1.0),
            (15, 4, 1.0), (15, 13, -1.0),
        ];
        let scale = std::f64::consts::PI * 20.0;
        let want = CMat::from_triplets(
            16,
            16,
            entries
                .iter()
                .map(|&(r, c, v)| (r - 1, c - 1, Complex64::new(scale * v, 0.0))),
        )
        .unwrap();
        assert_relative_eq!(
            (h.to_dense() - want.to_dense()).norm(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn z_states_land_where_the_examples_put_them() {
        // Two spins: z on spin 1 is element 9 (one-based), z on spin 2 is 3.
        let s1 = z_state(2, 0);
        let s2 = z_state(2, 1);
        assert_relative_eq!(s1[(8, 0)].re, 0.5, epsilon = 1e-15);
        assert_eq!(s1.iter().filter(|z| z.norm() > 0.0).count(), 1);
        assert_relative_eq!(s2[(2, 0)].re, 0.5, epsilon = 1e-15);
        assert_eq!(s2.iter().filter(|z| z.norm() > 0.0).count(), 1);
    }

    /// The constructed SWAP must actually swap: it has to carry
    /// z-magnetisation from one spin onto the other.
    #[test]
    fn swap_gate_exchanges_the_two_spins() {
        for (nspins, a, b) in [(2usize, 0usize, 1usize), (3, 0, 2)] {
            let u = swap_gate(nspins, a, b).unwrap();
            let from = z_state(nspins, a);
            let to = z_state(nspins, b);
            let moved = &u * &from;
            assert_relative_eq!((moved - &to).norm(), 0.0, epsilon = 1e-8);
            // And back again.
            let moved_back = &u * &to;
            assert_relative_eq!((moved_back - &from).norm(), 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn embedding_preserves_the_algebra_per_spin() {
        let ops = cartesian_operators(3);
        // Operators on different spins commute.
        let a = &ops[0].0;
        let b = &ops[2].1;
        let comm = a
            .matmul(b)
            .unwrap()
            .add(&b.matmul(a).unwrap().scale(Complex64::new(-1.0, 0.0)))
            .unwrap();
        assert_relative_eq!(comm.to_dense().norm(), 0.0, epsilon = 1e-14);
    }
}
