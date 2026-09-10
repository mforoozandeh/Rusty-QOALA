//! Dense/sparse complex matrix plumbing.
//!
//! MATLAB switches between full and sparse storage silently; QOALA leans on
//! that heavily (`sparse`, `full`, `nnz`, `speye`, `spalloc`, and the
//! `tols.dense_matrix` heuristic).  [`CMat`] makes the same choice explicit:
//! it is either dense or CSR, every operation knows how to handle both, and
//! [`CMat::densify_if`] reproduces the "more than `sparsity*dim^2` non-zeros
//! means go dense" rule used throughout the objective functions.

use crate::error::{QoalaError, Result};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use num_complex::Complex64;

/// Dense complex matrix, the storage used for states and trajectories.
pub type CDense = DMatrix<Complex64>;
/// Dense complex vector.
pub type CVector = DVector<Complex64>;

/// Complex zero.
pub const C0: Complex64 = Complex64::new(0.0, 0.0);
/// Complex one.
pub const C1: Complex64 = Complex64::new(1.0, 0.0);
/// Imaginary unit.
pub const CI: Complex64 = Complex64::new(0.0, 1.0);

/// MATLAB's `eps`: the double-precision unit roundoff.
pub const EPS: f64 = f64::EPSILON;

/// A complex matrix that is either dense or compressed sparse row.
#[derive(Debug, Clone)]
pub enum CMat {
    /// Column-major dense storage.
    Dense(CDense),
    /// Compressed sparse row storage.
    Sparse(CsrMatrix<Complex64>),
}

impl CMat {
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        match self {
            CMat::Dense(m) => m.nrows(),
            CMat::Sparse(m) => m.nrows(),
        }
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        match self {
            CMat::Dense(m) => m.ncols(),
            CMat::Sparse(m) => m.ncols(),
        }
    }

    /// Number of structurally-or-numerically non-zero entries, matching
    /// MATLAB's `nnz` (which counts numerical non-zeros).
    pub fn nnz(&self) -> usize {
        match self {
            CMat::Dense(m) => m.iter().filter(|z| !z.is_zero()).count(),
            CMat::Sparse(m) => m.values().iter().filter(|z| !z.is_zero()).count(),
        }
    }

    /// True when stored densely.
    pub fn is_dense(&self) -> bool {
        matches!(self, CMat::Dense(_))
    }

    /// Square identity in sparse storage (MATLAB `speye`).
    pub fn identity(n: usize) -> Self {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, C1);
        }
        CMat::Sparse(CsrMatrix::from(&coo))
    }

    /// All-zero sparse matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        CMat::Sparse(CsrMatrix::zeros(rows, cols))
    }

    /// Build from `(row, col, value)` triplets, summing duplicates exactly as
    /// MATLAB's `sparse(i,j,v)` does.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: impl IntoIterator<Item = (usize, usize, Complex64)>,
    ) -> Result<Self> {
        let mut coo = CooMatrix::new(rows, cols);
        for (r, c, v) in triplets {
            if r >= rows || c >= cols {
                return Err(QoalaError::Dimension(format!(
                    "triplet ({r},{c}) outside {rows}x{cols}"
                )));
            }
            coo.push(r, c, v);
        }
        Ok(CMat::Sparse(CsrMatrix::from(&coo)))
    }

    /// Copy into dense storage (MATLAB `full`).
    pub fn to_dense(&self) -> CDense {
        match self {
            CMat::Dense(m) => m.clone(),
            CMat::Sparse(m) => {
                let mut d = CDense::zeros(m.nrows(), m.ncols());
                for (r, c, v) in m.triplet_iter() {
                    d[(r, c)] += *v;
                }
                d
            }
        }
    }

    /// Copy into sparse storage (MATLAB `sparse`), dropping exact zeros.
    pub fn to_sparse(&self) -> CsrMatrix<Complex64> {
        match self {
            CMat::Sparse(m) => m.clone(),
            CMat::Dense(m) => {
                let mut coo = CooMatrix::new(m.nrows(), m.ncols());
                for c in 0..m.ncols() {
                    for r in 0..m.nrows() {
                        let v = m[(r, c)];
                        if !v.is_zero() {
                            coo.push(r, c, v);
                        }
                    }
                }
                CsrMatrix::from(&coo)
            }
        }
    }

    /// Force dense storage in place.
    pub fn into_dense(self) -> Self {
        match self {
            CMat::Dense(_) => self,
            CMat::Sparse(_) => CMat::Dense(self.to_dense()),
        }
    }

    /// Force sparse storage in place.
    pub fn into_sparse(self) -> Self {
        match self {
            CMat::Sparse(_) => self,
            CMat::Dense(_) => CMat::Sparse(self.to_sparse()),
        }
    }

    /// The `nnz(P) > sparsity*dim*dim  ->  full(P)` heuristic that QOALA
    /// applies to every propagator it forms.
    pub fn densify_if(self, sparsity: f64, dim: usize) -> Self {
        if !self.is_dense() && (self.nnz() as f64) > sparsity * (dim as f64) * (dim as f64) {
            self.into_dense()
        } else {
            self
        }
    }

    /// Conjugate transpose (MATLAB `'`).
    pub fn adjoint(&self) -> CMat {
        match self {
            CMat::Dense(m) => CMat::Dense(m.adjoint()),
            CMat::Sparse(m) => {
                let mut coo = CooMatrix::new(m.ncols(), m.nrows());
                for (r, c, v) in m.triplet_iter() {
                    coo.push(c, r, v.conj());
                }
                CMat::Sparse(CsrMatrix::from(&coo))
            }
        }
    }

    /// Unconjugated transpose (MATLAB `.'`).
    pub fn transpose(&self) -> CMat {
        match self {
            CMat::Dense(m) => CMat::Dense(m.transpose()),
            CMat::Sparse(m) => {
                let mut coo = CooMatrix::new(m.ncols(), m.nrows());
                for (r, c, v) in m.triplet_iter() {
                    coo.push(c, r, *v);
                }
                CMat::Sparse(CsrMatrix::from(&coo))
            }
        }
    }

    /// Multiply every entry by a scalar.
    pub fn scale(&self, s: Complex64) -> CMat {
        match self {
            CMat::Dense(m) => CMat::Dense(m.map(|z| z * s)),
            CMat::Sparse(m) => {
                let mut out = m.clone();
                for v in out.values_mut() {
                    *v *= s;
                }
                CMat::Sparse(out)
            }
        }
    }

    /// Matrix sum; the result is dense if either operand is dense.
    pub fn add(&self, other: &CMat) -> Result<CMat> {
        if self.nrows() != other.nrows() || self.ncols() != other.ncols() {
            return Err(QoalaError::Dimension(format!(
                "cannot add {}x{} to {}x{}",
                self.nrows(),
                self.ncols(),
                other.nrows(),
                other.ncols()
            )));
        }
        Ok(match (self, other) {
            (CMat::Sparse(a), CMat::Sparse(b)) => CMat::Sparse(a + b),
            _ => CMat::Dense(self.to_dense() + other.to_dense()),
        })
    }

    /// Matrix product.  Sparse times sparse stays sparse; anything else goes
    /// dense, which is what MATLAB does too once one operand is full.
    pub fn matmul(&self, other: &CMat) -> Result<CMat> {
        if self.ncols() != other.nrows() {
            return Err(QoalaError::Dimension(format!(
                "cannot multiply {}x{} by {}x{}",
                self.nrows(),
                self.ncols(),
                other.nrows(),
                other.ncols()
            )));
        }
        Ok(match (self, other) {
            (CMat::Sparse(a), CMat::Sparse(b)) => CMat::Sparse(a * b),
            (CMat::Sparse(a), CMat::Dense(b)) => CMat::Dense(spmm_dense(a, b)),
            (CMat::Dense(a), CMat::Sparse(b)) => {
                CMat::Dense(spmm_dense(&b.transpose(), &a.transpose()).transpose())
            }
            (CMat::Dense(a), CMat::Dense(b)) => CMat::Dense(a * b),
        })
    }

    /// Apply this operator to a dense state (vector or matrix).
    pub fn apply(&self, x: &CDense) -> Result<CDense> {
        if self.ncols() != x.nrows() {
            return Err(QoalaError::Dimension(format!(
                "cannot apply {}x{} operator to {}x{} state",
                self.nrows(),
                self.ncols(),
                x.nrows(),
                x.ncols()
            )));
        }
        Ok(match self {
            CMat::Dense(m) => m * x,
            CMat::Sparse(m) => spmm_dense(m, x),
        })
    }

    /// Kronecker product, preserving sparsity when both operands are sparse.
    pub fn kron(&self, other: &CMat) -> CMat {
        match (self, other) {
            (CMat::Sparse(a), CMat::Sparse(b)) => {
                let (ar, ac) = (a.nrows(), a.ncols());
                let (br, bc) = (b.nrows(), b.ncols());
                let mut coo = CooMatrix::new(ar * br, ac * bc);
                for (i, j, va) in a.triplet_iter() {
                    if va.is_zero() {
                        continue;
                    }
                    for (k, l, vb) in b.triplet_iter() {
                        if vb.is_zero() {
                            continue;
                        }
                        coo.push(i * br + k, j * bc + l, va * vb);
                    }
                }
                CMat::Sparse(CsrMatrix::from(&coo))
            }
            _ => CMat::Dense(kron_dense(&self.to_dense(), &other.to_dense())),
        }
    }

    /// `tol*round(A/tol)` applied elementwise: quantise and drop anything
    /// below half a tolerance.  This is QOALA's `nonzero_tol` cleanup.
    pub fn chop(&self, tol: f64) -> CMat {
        if tol <= 0.0 {
            return self.clone();
        }
        match self {
            CMat::Dense(m) => CMat::Dense(m.map(|z| chop_scalar(z, tol))),
            CMat::Sparse(m) => {
                let mut coo = CooMatrix::new(m.nrows(), m.ncols());
                for (r, c, v) in m.triplet_iter() {
                    let z = chop_scalar(*v, tol);
                    if !z.is_zero() {
                        coo.push(r, c, z);
                    }
                }
                CMat::Sparse(CsrMatrix::from(&coo))
            }
        }
    }

    /// Maximum absolute column sum (MATLAB `norm(A,1)`).
    pub fn norm_1(&self) -> f64 {
        let mut sums = vec![0.0f64; self.ncols()];
        match self {
            CMat::Dense(m) => {
                for c in 0..m.ncols() {
                    for r in 0..m.nrows() {
                        sums[c] += m[(r, c)].norm();
                    }
                }
            }
            CMat::Sparse(m) => {
                for (_, c, v) in m.triplet_iter() {
                    sums[c] += v.norm();
                }
            }
        }
        sums.into_iter().fold(0.0, f64::max)
    }

    /// Largest singular value (MATLAB `norm(full(A),2)`).
    pub fn norm_2(&self) -> f64 {
        norm_2_dense(&self.to_dense())
    }

    /// Frobenius norm.
    pub fn norm_fro(&self) -> f64 {
        let s: f64 = match self {
            CMat::Dense(m) => m.iter().map(|z| z.norm_sqr()).sum(),
            CMat::Sparse(m) => m.values().iter().map(|z| z.norm_sqr()).sum(),
        };
        s.sqrt()
    }

    /// Trace of a square matrix.
    pub fn trace(&self) -> Complex64 {
        match self {
            CMat::Dense(m) => (0..m.nrows().min(m.ncols())).map(|i| m[(i, i)]).sum(),
            CMat::Sparse(m) => m
                .triplet_iter()
                .filter(|(r, c, _)| r == c)
                .map(|(_, _, v)| *v)
                .sum(),
        }
    }
}

impl From<CDense> for CMat {
    fn from(m: CDense) -> Self {
        CMat::Dense(m)
    }
}

impl From<CsrMatrix<Complex64>> for CMat {
    fn from(m: CsrMatrix<Complex64>) -> Self {
        CMat::Sparse(m)
    }
}

/// Sparse-times-dense product.
fn spmm_dense(a: &CsrMatrix<Complex64>, b: &CDense) -> CDense {
    let mut out = CDense::zeros(a.nrows(), b.ncols());
    for (r, c, v) in a.triplet_iter() {
        if v.is_zero() {
            continue;
        }
        for k in 0..b.ncols() {
            out[(r, k)] += *v * b[(c, k)];
        }
    }
    out
}

/// Dense Kronecker product (nalgebra spells it `kronecker`, kept here so the
/// call sites read like the MATLAB).
pub fn kron_dense(a: &CDense, b: &CDense) -> CDense {
    a.kronecker(b)
}

/// Elementwise `tol*round(z/tol)` with real and imaginary parts treated
/// separately, as MATLAB's `round` does for complex numbers.
#[inline]
pub fn chop_scalar(z: Complex64, tol: f64) -> Complex64 {
    Complex64::new((z.re / tol).round() * tol, (z.im / tol).round() * tol)
}

/// Largest singular value of a dense complex matrix.
///
/// Computed as `sqrt(lambda_max(A^H A))` by symmetric eigendecomposition of
/// the (real-symmetric-equivalent) Gram matrix, which is numerically steadier
/// here than a complex SVD and matches `norm(A,2)` to full precision.
pub fn norm_2_dense(a: &CDense) -> f64 {
    if a.nrows() == 0 || a.ncols() == 0 {
        return 0.0;
    }
    // Build the Hermitian Gram matrix G = A^H A, then realify it as
    //     [[Re G, -Im G], [Im G, Re G]]
    // whose eigenvalues are those of G, each twice.
    let g = a.adjoint() * a;
    let n = g.nrows();
    let mut real = DMatrix::<f64>::zeros(2 * n, 2 * n);
    for c in 0..n {
        for r in 0..n {
            let z = g[(r, c)];
            real[(r, c)] = z.re;
            real[(r, c + n)] = -z.im;
            real[(r + n, c)] = z.im;
            real[(r + n, c + n)] = z.re;
        }
    }
    // Symmetrise against round-off before the eigensolver.
    let sym = (&real + real.transpose()) * 0.5;
    let lambda_max = sym
        .symmetric_eigenvalues()
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);
    lambda_max.max(0.0).sqrt()
}

/// 2-norm of a dense complex vector-like matrix, following MATLAB's rule that
/// `norm(x,2)` is the Euclidean norm for vectors and the spectral norm for
/// matrices.
pub fn norm_2_state(x: &CDense) -> f64 {
    if x.nrows() == 1 || x.ncols() == 1 {
        x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
    } else {
        norm_2_dense(x)
    }
}

/// `real(trace(a' * b))` without forming the product.
pub fn real_trace_adjoint(a: &CDense, b: &CDense) -> f64 {
    debug_assert_eq!(a.nrows(), b.nrows());
    debug_assert_eq!(a.ncols(), b.ncols());
    let mut acc = 0.0;
    for c in 0..a.ncols() {
        for r in 0..a.nrows() {
            let x = a[(r, c)];
            let y = b[(r, c)];
            acc += x.re * y.re + x.im * y.im;
        }
    }
    acc
}

/// Condition number in the 2-norm of a real symmetric matrix.
pub fn cond_2_symmetric(a: &DMatrix<f64>) -> f64 {
    let ev = a.clone().symmetric_eigenvalues();
    let mut hi = 0.0f64;
    let mut lo = f64::INFINITY;
    for v in ev.iter() {
        let m = v.abs();
        if m > hi {
            hi = m;
        }
        if m < lo {
            lo = m;
        }
    }
    if lo == 0.0 {
        f64::INFINITY
    } else {
        hi / lo
    }
}

/// True when a real symmetric matrix is positive definite, i.e. when MATLAB's
/// `[~,p] = chol(H)` returns `p == 0`.
pub fn is_positive_definite(a: &DMatrix<f64>) -> bool {
    nalgebra::Cholesky::new(a.clone()).is_some()
}

/// Five-point finite-difference matrix, the subset of Spinach's `fdmat` that
/// QOALA's `DNS` penalty needs: `npoints` grid points, 5-point stencil,
/// second derivative, with "wall" (one-sided at the edges) boundaries.
pub fn fdmat_second_derivative_5pt(npoints: usize) -> Result<DMatrix<f64>> {
    if npoints < 5 {
        return Err(QoalaError::BadValue(format!(
            "a five-point second-derivative stencil needs at least 5 points, got {npoints}"
        )));
    }
    let n = npoints;
    let mut d = DMatrix::<f64>::zeros(n, n);
    // Interior: standard centred five-point second derivative.
    let interior = [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0];
    for r in 2..n - 2 {
        for (k, w) in interior.iter().enumerate() {
            d[(r, r + k - 2)] = *w;
        }
    }
    // Edges: one-sided five-point second derivative ("wall" boundary).
    let forward = [
        35.0 / 12.0,
        -26.0 / 3.0,
        19.0 / 2.0,
        -14.0 / 3.0,
        11.0 / 12.0,
    ];
    let offset = [11.0 / 12.0, -5.0 / 3.0, 1.0 / 2.0, 1.0 / 3.0, -1.0 / 12.0];
    for (k, w) in forward.iter().enumerate() {
        d[(0, k)] = *w;
        d[(n - 1, n - 1 - k)] = *w;
    }
    for (k, w) in offset.iter().enumerate() {
        d[(1, k)] = *w;
        d[(n - 2, n - 1 - k)] = *w;
    }
    Ok(d)
}

/// Cartesian to polar for a pair of waveform rows, as used by the `SNSA`
/// penalty.
#[inline]
pub fn cartesian2polar(x: f64, y: f64) -> (f64, f64) {
    ((x * x + y * y).sqrt(), y.atan2(x))
}

/// Polar to Cartesian, the inverse of [`cartesian2polar`].
#[inline]
pub fn polar2cartesian(r: f64, phi: f64) -> (f64, f64) {
    (r * phi.cos(), r * phi.sin())
}

/// Cartesian to spherical, returning `(r, theta, phi)` with `theta` the polar
/// angle from `+z`.
#[inline]
pub fn cartesian2spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();
    let theta = if r == 0.0 { 0.0 } else { (z / r).acos() };
    (r, theta, y.atan2(x))
}

/// Spherical to Cartesian together with the chain rule for a differential
/// `(dr, dtheta, dphi)`, mirroring the six-output MATLAB helper.
pub fn spherical2cartesian(
    r: f64,
    theta: f64,
    phi: f64,
    dr: f64,
    dtheta: f64,
    dphi: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    let x = r * st * cp;
    let y = r * st * sp;
    let z = r * ct;
    let dx = dr * st * cp + r * ct * cp * dtheta - r * st * sp * dphi;
    let dy = dr * st * sp + r * ct * sp * dtheta + r * st * cp * dphi;
    let dz = dr * ct - r * st * dtheta;
    (x, y, z, dx, dy, dz)
}

trait IsZero {
    fn is_zero(&self) -> bool;
}
impl IsZero for Complex64 {
    #[inline]
    fn is_zero(&self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn sparse_and_dense_products_agree() {
        let a = CMat::from_triplets(3, 3, [(0, 1, c(1.0, 2.0)), (2, 0, c(-1.0, 0.5))]).unwrap();
        let b = CMat::from_triplets(3, 3, [(1, 2, c(0.0, 1.0)), (0, 0, c(2.0, 0.0))]).unwrap();
        let sp = a.matmul(&b).unwrap().to_dense();
        let de = CMat::Dense(a.to_dense())
            .matmul(&CMat::Dense(b.to_dense()))
            .unwrap()
            .to_dense();
        assert_relative_eq!((sp - de).norm(), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn kron_matches_dense_definition() {
        let a = CMat::from_triplets(2, 2, [(0, 0, c(1.0, 0.0)), (1, 1, c(0.0, 2.0))]).unwrap();
        let b = CMat::from_triplets(2, 2, [(0, 1, c(3.0, 0.0)), (1, 0, c(0.0, -1.0))]).unwrap();
        let sp = a.kron(&b).to_dense();
        let de = kron_dense(&a.to_dense(), &b.to_dense());
        assert_relative_eq!((sp - de).norm(), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn norm_2_matches_known_singular_values() {
        // diag(3, 4) has spectral norm 4.
        let m = CMat::from_triplets(2, 2, [(0, 0, c(3.0, 0.0)), (1, 1, c(0.0, 4.0))]).unwrap();
        assert_relative_eq!(m.norm_2(), 4.0, epsilon = 1e-12);
        // A 2x2 all-ones matrix has singular values 2 and 0.
        let ones = CMat::Dense(CDense::from_element(2, 2, C1));
        assert_relative_eq!(ones.norm_2(), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn chop_quantises_and_drops() {
        let m = CMat::Dense(CDense::from_row_slice(
            1,
            2,
            &[c(1e-15, 0.0), c(1.0, 1e-15)],
        ));
        let chopped = m.chop(1e-12).to_dense();
        assert_eq!(chopped[(0, 0)], C0);
        assert_eq!(chopped[(0, 1)], C1);
    }

    #[test]
    fn fdmat_annihilates_linear_functions() {
        // A second-derivative operator must return zero on a straight line.
        let n = 12;
        let d = fdmat_second_derivative_5pt(n).unwrap();
        let x = DVector::<f64>::from_iterator(n, (0..n).map(|i| 2.0 * i as f64 + 1.0));
        let dd = &d * &x;
        assert_relative_eq!(dd.amax(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn fdmat_second_derivative_of_a_parabola_is_constant() {
        let n = 12;
        let d = fdmat_second_derivative_5pt(n).unwrap();
        let x = DVector::<f64>::from_iterator(n, (0..n).map(|i| (i as f64).powi(2)));
        let dd = &d * &x;
        for v in dd.iter() {
            assert_relative_eq!(*v, 2.0, epsilon = 1e-9);
        }
    }
}
