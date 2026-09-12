//! Hilbert-space gates and their Liouville-space superoperators.
//!
//! [`crate::spinops::swap_gate`] builds its target by composing Liouville
//! rotations, which is how the MATLAB example scripts do it and which does not
//! generalise past SWAP.  This module goes the other way: it takes an
//! arbitrary Hilbert-space unitary `U` and returns the superoperator for
//! `rho -> U rho U^dagger` in the same normalised product spherical-tensor
//! basis everything else in the crate uses, so any gate that can be written as
//! a matrix can be an optimisation target.
//!
//! ## The basis
//!
//! One spin-1/2 spans a four-dimensional Liouville space.  The ordering and
//! normalisation are fixed by the existing [`crate::spinops::jx`],
//! [`crate::spinops::jy`] and [`crate::spinops::jz`] commutation
//! superoperators, and work out as
//!
//! | index | tensor    | matrix          |
//! |-------|-----------|-----------------|
//! | 0     | `T(0,0)`  | `E / sqrt(2)`   |
//! | 1     | `T(1,+1)` | `-I+`           |
//! | 2     | `T(1,0)`  | `sqrt(2) Iz`    |
//! | 3     | `T(1,-1)` | `I-`            |
//!
//! which satisfies `Tr(B_i^dag B_j) = delta_ij`.  The composite basis is the
//! Kronecker product of those, spin 0 leftmost, exactly as
//! [`crate::spinops::kron_all`] and [`crate::spinops::kron_states`] order
//! theirs.  `superoperator_from_unitary(SWAP, 2)` reproducing
//! `swap_gate(2, 0, 1)` is the test that pins all of it.

use crate::error::{QoalaError, Result};
use crate::linalg::{CDense, CMat, C0, C1};
use num_complex::Complex64;

/// Largest register this module will build a superoperator for.
///
/// The superoperator is `4^n x 4^n`; six spins is already a 4096x4096 dense
/// complex matrix, a quarter of a gigabyte, and nothing in the crate can use
/// one.
const MAX_SPINS: usize = 6;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// The four normalised single-spin spherical tensors, in the crate's order.
fn single_spin_basis() -> [CDense; 4] {
    let s = 1.0 / std::f64::consts::SQRT_2;
    [
        // T(0,0): the identity, normalised.
        CDense::from_row_slice(2, 2, &[c(s, 0.0), C0, C0, c(s, 0.0)]),
        // T(1,+1): -I+.
        CDense::from_row_slice(2, 2, &[C0, c(-1.0, 0.0), C0, C0]),
        // T(1,0): sqrt(2) Iz.
        CDense::from_row_slice(2, 2, &[c(s, 0.0), C0, C0, c(-s, 0.0)]),
        // T(1,-1): I-.
        CDense::from_row_slice(2, 2, &[C0, C0, C1, C0]),
    ]
}

/// The normalised product spherical-tensor basis for `nspins` spin-1/2.
///
/// Returns `4^nspins` matrices of size `2^nspins`, ordered so that the
/// composite index is the base-4 number whose most significant digit is spin
/// 0 - the same ordering [`crate::spinops::kron_states`] produces.
pub fn sphten_basis(nspins: usize) -> Result<Vec<CDense>> {
    if nspins == 0 || nspins > MAX_SPINS {
        return Err(QoalaError::BadValue(format!(
            "spherical tensor basis needs between 1 and {MAX_SPINS} spins, got {nspins}"
        )));
    }
    let single = single_spin_basis();
    let count = 4usize.pow(nspins as u32);
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        // Digits of `index` in base 4, most significant first, are the
        // single-spin tensors of spins 0, 1, ... in order.
        let mut m = single[(index >> (2 * (nspins - 1))) & 3].clone();
        for s in 1..nspins {
            let digit = (index >> (2 * (nspins - 1 - s))) & 3;
            m = m.kronecker(&single[digit]);
        }
        out.push(m);
    }
    Ok(out)
}

/// Superoperator for `rho -> U rho U^dagger`, in the sphten Liouville basis.
///
/// `u` is a `2^nspins` square Hilbert-space unitary.  The result is
/// `4^nspins` square and can be used directly as a `universal_gate_xy` target
/// once converted with [`CMat::to_dense`].
///
/// A global phase on `u` makes no difference, which is what lets gates
/// assembled from rotations be compared with gates written down as matrices.
///
/// The result is exact: rounding noise is left in place rather than chopped,
/// because a legitimately small entry and a noise entry are indistinguishable
/// from in here.  Call [`CMat::chop`] on it for a sparse target, as
/// [`Gate::superoperator`] does.
pub fn superoperator_from_unitary(u: &CDense, nspins: usize) -> Result<CMat> {
    let basis = sphten_basis(nspins)?;
    let dim = 1usize << nspins;
    if u.nrows() != dim || u.ncols() != dim {
        return Err(QoalaError::Dimension(format!(
            "a {nspins}-spin unitary must be {dim}x{dim}, got {}x{}",
            u.nrows(),
            u.ncols()
        )));
    }

    let udag = u.adjoint();
    let n = basis.len();
    let mut out = CDense::zeros(n, n);
    for (j, bj) in basis.iter().enumerate() {
        // Conjugate the basis element once, then project it onto the basis:
        // S[i][j] = Tr(B_i^dag U B_j U^dag), and the trace of a product with
        // a dagger is the elementwise Hilbert-Schmidt sum.
        let conjugated = u * bj * &udag;
        for (i, bi) in basis.iter().enumerate() {
            out[(i, j)] = bi
                .iter()
                .zip(conjugated.iter())
                .map(|(x, y)| x.conj() * y)
                .sum();
        }
    }
    Ok(CMat::Dense(out))
}

/// Coefficients of a Hilbert-space operator in the sphten Liouville basis.
///
/// The inverse of reading a Liouville state vector as an operator, and the
/// way to check a superoperator against a density matrix computed by hand.
pub fn coefficients(rho: &CDense, nspins: usize) -> Result<CDense> {
    let basis = sphten_basis(nspins)?;
    let dim = 1usize << nspins;
    if rho.nrows() != dim || rho.ncols() != dim {
        return Err(QoalaError::Dimension(format!(
            "a {nspins}-spin operator must be {dim}x{dim}, got {}x{}",
            rho.nrows(),
            rho.ncols()
        )));
    }
    let mut out = CDense::zeros(basis.len(), 1);
    for (i, bi) in basis.iter().enumerate() {
        out[(i, 0)] = bi.iter().zip(rho.iter()).map(|(x, y)| x.conj() * y).sum();
    }
    Ok(out)
}

// --------------------------------------------------------------- gates -----

/// Pauli `X`, the Hilbert-space bit flip.
#[rustfmt::skip]
pub fn pauli_x() -> CDense {
    CDense::from_row_slice(2, 2, &[
        C0, C1,
        C1, C0,
    ])
}

/// Pauli `Y`.
#[rustfmt::skip]
pub fn pauli_y() -> CDense {
    CDense::from_row_slice(2, 2, &[
        C0, c(0.0, -1.0),
        c(0.0, 1.0), C0,
    ])
}

/// Pauli `Z`.
#[rustfmt::skip]
pub fn pauli_z() -> CDense {
    CDense::from_row_slice(2, 2, &[
        C1, C0,
        C0, c(-1.0, 0.0),
    ])
}

/// The Hadamard gate.
#[rustfmt::skip]
pub fn hadamard() -> CDense {
    let s = 1.0 / std::f64::consts::SQRT_2;
    CDense::from_row_slice(2, 2, &[
        c(s, 0.0), c(s, 0.0),
        c(s, 0.0), c(-s, 0.0),
    ])
}

/// Rotation by `angle` radians about the unit axis `(nx, ny, nz)`:
/// `exp(-i angle (n . I))` with `I` the spin-1/2 operators, i.e.
/// `cos(angle/2) E - i sin(angle/2) (n . sigma)`.
///
/// The axis is normalised internally; a zero axis is an error.
pub fn axis_rotation(nx: f64, ny: f64, nz: f64, angle: f64) -> Result<CDense> {
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len == 0.0 {
        return Err(QoalaError::BadValue(
            "rotation axis must not be the zero vector".into(),
        ));
    }
    let (ux, uy, uz) = (nx / len, ny / len, nz / len);
    let (ch, sh) = ((angle / 2.0).cos(), (angle / 2.0).sin());
    let mut out = CDense::zeros(2, 2);
    out[(0, 0)] = c(ch, 0.0);
    out[(1, 1)] = c(ch, 0.0);
    let f = c(0.0, -sh);
    // -i sin(angle/2) (ux X + uy Y + uz Z).
    out[(0, 1)] += f * c(ux, 0.0);
    out[(1, 0)] += f * c(ux, 0.0);
    out[(0, 1)] += f * c(0.0, -uy);
    out[(1, 0)] += f * c(0.0, uy);
    out[(0, 0)] += f * c(uz, 0.0);
    out[(1, 1)] += f * c(-uz, 0.0);
    Ok(out)
}

/// `SWAP`: exchange the two qubits.
#[rustfmt::skip]
pub fn swap() -> CDense {
    CDense::from_row_slice(4, 4, &[
        C1, C0, C0, C0,
        C0, C0, C1, C0,
        C0, C1, C0, C0,
        C0, C0, C0, C1,
    ])
}

/// `iSWAP`: exchange the two qubits, picking up a factor of `i`.
#[rustfmt::skip]
pub fn iswap() -> CDense {
    let i = c(0.0, 1.0);
    CDense::from_row_slice(4, 4, &[
        C1, C0, C0, C0,
        C0, C0, i,  C0,
        C0, i,  C0, C0,
        C0, C0, C0, C1,
    ])
}

/// `sqrt(SWAP)`: the square root of [`swap`] whose square is SWAP exactly.
#[rustfmt::skip]
pub fn sqrt_swap() -> CDense {
    let p = c(0.5, 0.5);
    let m = c(0.5, -0.5);
    CDense::from_row_slice(4, 4, &[
        C1, C0, C0, C0,
        C0, p,  m,  C0,
        C0, m,  p,  C0,
        C0, C0, C0, C1,
    ])
}

/// `CNOT`: flip the second qubit when the first is `|1>`.
#[rustfmt::skip]
pub fn cnot() -> CDense {
    CDense::from_row_slice(4, 4, &[
        C1, C0, C0, C0,
        C0, C1, C0, C0,
        C0, C0, C0, C1,
        C0, C0, C1, C0,
    ])
}

/// `CZ`: a sign on `|11>`.
#[rustfmt::skip]
pub fn cz() -> CDense {
    CDense::from_row_slice(4, 4, &[
        C1, C0, C0, C0,
        C0, C1, C0, C0,
        C0, C0, C1, C0,
        C0, C0, C0, c(-1.0, 0.0),
    ])
}

// ------------------------------------------------------------ embedding ----

/// Bit of composite index `k` belonging to qubit `q` of `nspins`.
///
/// Qubit 0 is the most significant bit, matching the Kronecker order used
/// throughout the crate.
#[inline]
fn bit(k: usize, q: usize, nspins: usize) -> usize {
    (k >> (nspins - 1 - q)) & 1
}

/// Set qubit `q`'s bit of `k` to `v`.
#[inline]
fn with_bit(k: usize, q: usize, nspins: usize, v: usize) -> usize {
    let mask = 1usize << (nspins - 1 - q);
    if v == 1 {
        k | mask
    } else {
        k & !mask
    }
}

/// Embed a single-qubit gate on qubit `q` of an `nspins` register.
pub fn embed_one_qubit(nspins: usize, q: usize, g: &CDense) -> Result<CDense> {
    if nspins == 0 || nspins > MAX_SPINS {
        return Err(QoalaError::BadValue(format!(
            "embedding needs between 1 and {MAX_SPINS} spins, got {nspins}"
        )));
    }
    if q >= nspins {
        return Err(QoalaError::BadValue(format!(
            "qubit {q} is outside a {nspins}-spin register"
        )));
    }
    if g.nrows() != 2 || g.ncols() != 2 {
        return Err(QoalaError::Dimension(format!(
            "a single-qubit gate must be 2x2, got {}x{}",
            g.nrows(),
            g.ncols()
        )));
    }
    let eye = CDense::identity(2, 2);
    let mut out = if q == 0 { g.clone() } else { eye.clone() };
    for s in 1..nspins {
        out = out.kronecker(if s == q { g } else { &eye });
    }
    Ok(out)
}

/// Embed a two-qubit gate on qubits `(a, b)` of an `nspins` register.
///
/// `a` is the gate's first qubit and `b` its second, so
/// `embed_two_qubit(n, a, b, cnot())` puts the control on `a`.  The two need
/// not be adjacent and `b` may be less than `a`.
pub fn embed_two_qubit(nspins: usize, a: usize, b: usize, g: &CDense) -> Result<CDense> {
    if !(2..=MAX_SPINS).contains(&nspins) {
        return Err(QoalaError::BadValue(format!(
            "embedding a two-qubit gate needs between 2 and {MAX_SPINS} spins, got {nspins}"
        )));
    }
    if a >= nspins || b >= nspins {
        return Err(QoalaError::BadValue(format!(
            "qubits ({a}, {b}) are outside a {nspins}-spin register"
        )));
    }
    if a == b {
        return Err(QoalaError::BadValue(format!(
            "a two-qubit gate needs two distinct qubits, got ({a}, {b})"
        )));
    }
    if g.nrows() != 4 || g.ncols() != 4 {
        return Err(QoalaError::Dimension(format!(
            "a two-qubit gate must be 4x4, got {}x{}",
            g.nrows(),
            g.ncols()
        )));
    }

    let dim = 1usize << nspins;
    let mut out = CDense::zeros(dim, dim);
    for col in 0..dim {
        let (ia, ib) = (bit(col, a, nspins), bit(col, b, nspins));
        let gcol = 2 * ia + ib;
        for grow in 0..4 {
            let v = g[(grow, gcol)];
            if v == C0 {
                continue;
            }
            let (oa, ob) = (grow >> 1, grow & 1);
            let row = with_bit(with_bit(col, a, nspins, oa), b, nspins, ob);
            out[(row, col)] = v;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- menu -----

/// A gate from the shipped library, for a user interface to offer as a list.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gate {
    /// Exchange two qubits.
    Swap,
    /// Exchange two qubits with a factor of `i`.
    ISwap,
    /// The square root of [`Gate::Swap`].
    SqrtSwap,
    /// Controlled NOT; the first qubit is the control.
    Cnot,
    /// Controlled phase.
    Cz,
    /// Pauli X on one qubit.
    X,
    /// Pauli Y on one qubit.
    Y,
    /// Pauli Z on one qubit.
    Z,
    /// Hadamard on one qubit.
    H,
    /// Rotation of one qubit about an arbitrary axis.
    Rotation {
        /// x component of the rotation axis.
        nx: f64,
        /// y component of the rotation axis.
        ny: f64,
        /// z component of the rotation axis.
        nz: f64,
        /// Rotation angle in radians.
        angle: f64,
    },
}

/// Every gate that takes no parameters, for populating a menu.
pub const GATE_MENU: [Gate; 9] = [
    Gate::Swap,
    Gate::ISwap,
    Gate::SqrtSwap,
    Gate::Cnot,
    Gate::Cz,
    Gate::X,
    Gate::Y,
    Gate::Z,
    Gate::H,
];

impl Gate {
    /// How many qubits the gate acts on: 1 or 2.
    pub fn qubits(&self) -> usize {
        match self {
            Gate::Swap | Gate::ISwap | Gate::SqrtSwap | Gate::Cnot | Gate::Cz => 2,
            Gate::X | Gate::Y | Gate::Z | Gate::H | Gate::Rotation { .. } => 1,
        }
    }

    /// Short label for a menu.
    pub fn name(&self) -> &'static str {
        match self {
            Gate::Swap => "SWAP",
            Gate::ISwap => "iSWAP",
            Gate::SqrtSwap => "sqrt(SWAP)",
            Gate::Cnot => "CNOT",
            Gate::Cz => "CZ",
            Gate::X => "X",
            Gate::Y => "Y",
            Gate::Z => "Z",
            Gate::H => "H",
            Gate::Rotation { .. } => "rotation",
        }
    }

    /// The gate's own Hilbert-space matrix, 2x2 or 4x4.
    pub fn hilbert(&self) -> Result<CDense> {
        Ok(match self {
            Gate::Swap => swap(),
            Gate::ISwap => iswap(),
            Gate::SqrtSwap => sqrt_swap(),
            Gate::Cnot => cnot(),
            Gate::Cz => cz(),
            Gate::X => pauli_x(),
            Gate::Y => pauli_y(),
            Gate::Z => pauli_z(),
            Gate::H => hadamard(),
            Gate::Rotation { nx, ny, nz, angle } => axis_rotation(*nx, *ny, *nz, *angle)?,
        })
    }

    /// The gate embedded on `targets` of an `nspins` register, as a Liouville
    /// superoperator ready to be a `universal_gate_xy` target.
    ///
    /// `targets` must have [`Gate::qubits`] entries.  Rounding noise is
    /// chopped at 1e-12 and the result stored sparsely, matching what
    /// [`crate::spinops::swap_gate`] does to its own output; use
    /// [`superoperator_from_unitary`] directly for an unchopped one.
    pub fn superoperator(&self, nspins: usize, targets: &[usize]) -> Result<CMat> {
        let want = self.qubits();
        if targets.len() != want {
            return Err(QoalaError::BadValue(format!(
                "{} acts on {want} qubit(s), but {} were given",
                self.name(),
                targets.len()
            )));
        }
        let g = self.hilbert()?;
        let embedded = if want == 1 {
            embed_one_qubit(nspins, targets[0], &g)?
        } else {
            embed_two_qubit(nspins, targets[0], targets[1], &g)?
        };
        Ok(superoperator_from_unitary(&embedded, nspins)?
            .chop(1e-12)
            .into_sparse())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spinops;
    use approx::assert_relative_eq;

    fn ket(nspins: usize, bits: &[usize]) -> CDense {
        let dim = 1usize << nspins;
        let mut index = 0usize;
        for (q, b) in bits.iter().enumerate() {
            index = with_bit(index, q, nspins, *b);
        }
        let mut v = CDense::zeros(dim, 1);
        v[(index, 0)] = C1;
        v
    }

    /// `|bra><ket|` as a density-matrix-shaped operator.
    fn outer(nspins: usize, left: &[usize], right: &[usize]) -> CDense {
        ket(nspins, left) * ket(nspins, right).adjoint()
    }

    /// Apply a gate's superoperator to an operator's coefficient vector.
    fn push(s: &CMat, rho: &CDense, nspins: usize) -> CDense {
        s.to_dense() * coefficients(rho, nspins).unwrap()
    }

    /// The basis must be orthonormal, or every trace projection below is
    /// measuring the wrong thing.
    #[test]
    fn the_product_basis_is_orthonormal() {
        for nspins in 1..=3 {
            let basis = sphten_basis(nspins).unwrap();
            assert_eq!(basis.len(), 4usize.pow(nspins as u32));
            for (i, bi) in basis.iter().enumerate() {
                for (j, bj) in basis.iter().enumerate() {
                    let ip: Complex64 = bi.iter().zip(bj.iter()).map(|(x, y)| x.conj() * y).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert_relative_eq!(ip.re, want, epsilon = 1e-14);
                    assert_relative_eq!(ip.im, 0.0, epsilon = 1e-14);
                }
            }
        }
    }

    /// The convention test named in the plan: the new path must reproduce the
    /// independently-verified `spinops::swap_gate`, which is built by
    /// composing Liouville rotations rather than from a matrix.
    #[test]
    fn superoperator_from_swap_matches_the_rotation_built_gate() {
        let want = spinops::swap_gate(2, 0, 1).unwrap();
        let got = superoperator_from_unitary(&swap(), 2).unwrap().to_dense();
        assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-10);
    }

    /// And with the two qubits embedded in a larger, non-adjacent register.
    #[test]
    fn superoperator_from_embedded_swap_matches_the_rotation_built_gate() {
        let want = spinops::swap_gate(3, 0, 2).unwrap();
        let embedded = embed_two_qubit(3, 0, 2, &swap()).unwrap();
        let got = superoperator_from_unitary(&embedded, 3).unwrap().to_dense();
        assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-10);
    }

    /// Conjugation by a unitary preserves the trace, which in this basis
    /// means the top row of the superoperator is `(1, 0, 0, ...)`.  It is
    /// also unital, so the first column is too.
    #[test]
    fn every_gate_is_trace_preserving_and_unital() {
        let cases: Vec<(&str, CDense, usize)> = vec![
            ("SWAP", swap(), 2),
            ("iSWAP", iswap(), 2),
            ("sqrt(SWAP)", sqrt_swap(), 2),
            ("CNOT", cnot(), 2),
            ("CZ", cz(), 2),
            ("X", pauli_x(), 1),
            ("Y", pauli_y(), 1),
            ("Z", pauli_z(), 1),
            ("H", hadamard(), 1),
            (
                "Rz(pi/2)",
                axis_rotation(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2).unwrap(),
                1,
            ),
        ];
        for (name, u, nspins) in cases {
            let s = superoperator_from_unitary(&u, nspins).unwrap().to_dense();
            for j in 0..s.ncols() {
                let want = if j == 0 { 1.0 } else { 0.0 };
                assert_relative_eq!(s[(0, j)].re, want, epsilon = 1e-12);
                assert_relative_eq!(s[(0, j)].im, 0.0, epsilon = 1e-12);
                assert_relative_eq!(s[(j, 0)].re, want, epsilon = 1e-12);
                assert_relative_eq!(s[(j, 0)].im, 0.0, epsilon = 1e-12);
            }
            // Conjugation by a unitary is an isometry of the Hilbert-Schmidt
            // inner product, so the superoperator is unitary too.
            let eye = CDense::identity(s.nrows(), s.nrows());
            assert_relative_eq!(
                (s.adjoint() * &s - eye).norm(),
                0.0,
                epsilon = 1e-12,
                max_relative = 1e-12
            );
            let _ = name;
        }
    }

    /// CNOT must flip the target when the control is up and leave it alone
    /// when the control is down.
    #[test]
    fn cnot_flips_the_target_when_the_control_is_up() {
        let s = superoperator_from_unitary(&cnot(), 2).unwrap();
        // |10><10| -> |11><11|
        let got = push(&s, &outer(2, &[1, 0], &[1, 0]), 2);
        let want = coefficients(&outer(2, &[1, 1], &[1, 1]), 2).unwrap();
        assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-12);
        // |01><01| is untouched.
        let got = push(&s, &outer(2, &[0, 1], &[0, 1]), 2);
        let want = coefficients(&outer(2, &[0, 1], &[0, 1]), 2).unwrap();
        assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-12);
    }

    /// The control may be any qubit of a bigger register.
    #[test]
    fn an_embedded_cnot_controls_from_the_qubit_it_was_given() {
        let u = embed_two_qubit(3, 2, 0, &cnot()).unwrap();
        let s = superoperator_from_unitary(&u, 3).unwrap();
        // Control is qubit 2, target qubit 0: |0 0 1> -> |1 0 1>.
        let got = push(&s, &outer(3, &[0, 0, 1], &[0, 0, 1]), 3);
        let want = coefficients(&outer(3, &[1, 0, 1], &[1, 0, 1]), 3).unwrap();
        assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-12);
    }

    /// iSWAP exchanges the qubits and multiplies by `i`.  A population is
    /// blind to that phase, so the test uses a coherence, where it survives.
    #[test]
    fn iswap_exchanges_with_the_expected_phase() {
        let s = superoperator_from_unitary(&iswap(), 2).unwrap();
        // |01><11| -> (i|10>)(<11|) = i |10><11|.
        let got = push(&s, &outer(2, &[0, 1], &[1, 1]), 2);
        let want = coefficients(
            &outer(2, &[1, 0], &[1, 1]).map(|z| z * Complex64::new(0.0, 1.0)),
            2,
        )
        .unwrap();
        assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-12);
    }

    /// The square root of SWAP squares to SWAP, and its superoperator
    /// squares to SWAP's.
    #[test]
    fn sqrt_swap_squares_to_swap() {
        let s2 = &sqrt_swap() * &sqrt_swap();
        assert_relative_eq!((s2 - swap()).norm(), 0.0, epsilon = 1e-14);

        let a = superoperator_from_unitary(&sqrt_swap(), 2)
            .unwrap()
            .to_dense();
        let b = superoperator_from_unitary(&swap(), 2).unwrap().to_dense();
        assert_relative_eq!((&a * &a - b).norm(), 0.0, epsilon = 1e-12);
    }

    /// CZ puts a sign on `|11>`, visible in a coherence with `|01>`.
    #[test]
    fn cz_signs_the_doubly_excited_coherence() {
        let s = superoperator_from_unitary(&cz(), 2).unwrap();
        let got = push(&s, &outer(2, &[1, 1], &[0, 1]), 2);
        let want = coefficients(
            &outer(2, &[1, 1], &[0, 1]).map(|z| z * Complex64::new(-1.0, 0.0)),
            2,
        )
        .unwrap();
        assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-12);
    }

    /// Hadamard exchanges z and x magnetisation; the Paulis invert the two
    /// axes they are not.
    #[test]
    fn single_qubit_gates_move_magnetisation_where_they_should() {
        let ix = pauli_x().map(|z| z * Complex64::new(0.5, 0.0));
        let iy = pauli_y().map(|z| z * Complex64::new(0.5, 0.0));
        let iz = pauli_z().map(|z| z * Complex64::new(0.5, 0.0));

        let check = |u: &CDense, from: &CDense, to: &CDense, what: &str| {
            let s = superoperator_from_unitary(u, 1).unwrap();
            let got = push(&s, from, 1);
            let want = coefficients(to, 1).unwrap();
            assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-12);
            let _ = what;
        };

        let minus = |m: &CDense| m.map(|z| -z);

        check(&hadamard(), &iz, &ix, "H: Iz -> Ix");
        check(&hadamard(), &ix, &iz, "H: Ix -> Iz");
        check(&hadamard(), &iy, &minus(&iy), "H: Iy -> -Iy");
        check(&pauli_x(), &iz, &minus(&iz), "X: Iz -> -Iz");
        check(&pauli_x(), &iy, &minus(&iy), "X: Iy -> -Iy");
        check(&pauli_z(), &ix, &minus(&ix), "Z: Ix -> -Ix");
        check(&pauli_y(), &ix, &minus(&ix), "Y: Ix -> -Ix");
    }

    /// A z rotation by `theta` takes Ix to `Ix cos(theta) + Iy sin(theta)`.
    #[test]
    fn an_axis_rotation_rotates_by_the_angle_it_was_given() {
        let ix = pauli_x().map(|z| z * Complex64::new(0.5, 0.0));
        let iy = pauli_y().map(|z| z * Complex64::new(0.5, 0.0));
        for theta in [0.3_f64, std::f64::consts::FRAC_PI_2, 2.1] {
            let u = axis_rotation(0.0, 0.0, 1.0, theta).unwrap();
            let s = superoperator_from_unitary(&u, 1).unwrap();
            let got = push(&s, &ix, 1);
            let mixed = ix.map(|z| z * Complex64::new(theta.cos(), 0.0))
                + iy.map(|z| z * Complex64::new(theta.sin(), 0.0));
            let want = coefficients(&mixed, 1).unwrap();
            assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-12);
        }
    }

    /// The menu path must agree with building the matrix by hand.
    #[test]
    fn the_menu_agrees_with_the_matrices() {
        let want = superoperator_from_unitary(&embed_two_qubit(3, 1, 2, &cnot()).unwrap(), 3)
            .unwrap()
            .to_dense();
        let got = Gate::Cnot.superoperator(3, &[1, 2]).unwrap().to_dense();
        // The menu path chops at 1e-12; the primitive does not.
        assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-10);

        for gate in GATE_MENU {
            let targets: Vec<usize> = (0..gate.qubits()).collect();
            assert!(gate.superoperator(2, &targets).is_ok(), "{}", gate.name());
        }
        // Wrong arity is refused rather than guessed at.
        assert!(Gate::Cnot.superoperator(3, &[0]).is_err());
        assert!(Gate::H.superoperator(3, &[0, 1]).is_err());
        assert!(Gate::Swap.superoperator(3, &[1, 1]).is_err());
    }
}
