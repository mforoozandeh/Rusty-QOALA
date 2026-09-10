//! Euler-Rodrigues propagator elements and their control derivatives.
//!
//! Port of `utilities/rodrigues.m`.  A single spin-1/2 rotation about an
//! arbitrary axis has a closed form, so QOALA never exponentiates the control
//! part of the Hamiltonian: it evaluates the SU(2) parameters
//!
//! ```text
//! alpha = cos(delta/2) - i n_z sin(delta/2)
//! beta  = (n_y - i n_x) sin(delta/2)
//! ```
//!
//! and reads the propagator elements straight off them.  The derivatives with
//! respect to the control amplitudes come from
//!
//! ```text
//! U^dagger dU/dv_j = -i ( dt I + c1 S + c2 S^2 )_{jl} L_l
//! ```
//!
//! where `S_{jl} = eps_{jlk} v_k`, `c1 = (cos(delta)-1)/r^2` and
//! `c2 = (dt - sin(delta)/r)/r^2`.  That is exactly the `D` matrix built in
//! the MATLAB.
//!
//! # Storage
//!
//! The MATLAB materialises `dP{k}` as an `nsteps x dim^2` (sparse) array by
//! multiplying `D` into a stacked `vec(L)` matrix.  For a four-spin system
//! that array is 65536 columns wide.  Here the same object is kept factored -
//! the `D` block plus the operators - and the `dim x dim` derivative for a
//! given time point is formed on demand by [`ControlDerivative::matrix`].
//! The arithmetic is identical; only the memory traffic differs.

use crate::error::{QoalaError, Result};
use crate::linalg::{CDense, CMat, CI};
use crate::types::{Basis, GradOps, StateSpace};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

/// One control channel's propagator derivative, held in factored form.
///
/// Each term contributes `-i (d[m,0] Lx + d[m,1] Ly + d[m,2] Lz)` at time
/// point `m`; a control that drives several spins at once has one term per
/// spin, matching the stacked `sigma`/`D_stack` construction in the MATLAB.
#[derive(Debug, Clone)]
pub struct ControlDerivative {
    /// `(Lx, Ly, Lz, D block)` per contributing spin.  The `D` block has one
    /// row per time point and three columns.
    pub terms: Vec<DerivativeTerm>,
    /// Number of time points (`nsteps * nsplits`).
    pub npoints: usize,
    /// Composite-space dimension.
    pub dim: usize,
}

/// A single spin's contribution to a control derivative.
#[derive(Debug, Clone)]
pub struct DerivativeTerm {
    /// Cartesian control operators for this spin in the composite space.
    pub lx: CMat,
    /// See [`DerivativeTerm::lx`].
    pub ly: CMat,
    /// See [`DerivativeTerm::lx`].
    pub lz: CMat,
    /// `npoints x 3` slice of the Rodrigues `D` matrix for this spin and this
    /// control axis.
    pub d: DMatrix<f64>,
}

impl ControlDerivative {
    /// The `dim x dim` derivative matrix at time point `m`.
    ///
    /// Equivalent to `reshape(dP{k}(:,m), dim, dim)` in the MATLAB.
    pub fn matrix(&self, m: usize) -> Result<CMat> {
        if m >= self.npoints {
            return Err(QoalaError::Dimension(format!(
                "derivative time point {m} out of range (npoints = {})",
                self.npoints
            )));
        }
        let mut acc: Option<CMat> = None;
        for t in &self.terms {
            let contribution =
                t.lx.scale(-CI * t.d[(m, 0)])
                    .add(&t.ly.scale(-CI * t.d[(m, 1)]))?
                    .add(&t.lz.scale(-CI * t.d[(m, 2)]))?;
            acc = Some(match acc {
                None => contribution,
                Some(a) => a.add(&contribution)?,
            });
        }
        acc.ok_or_else(|| QoalaError::BadValue("control derivative has no terms".into()))
    }
}

/// Number of stored propagator elements per time point for a given formalism.
pub fn elements_per_point(space: StateSpace, basis: Basis) -> Result<usize> {
    Ok(match (space, basis) {
        (StateSpace::Liouville, Basis::Sphten) => 9,
        (StateSpace::Liouville, Basis::Zeeman) => 16,
        (StateSpace::Hilbert, Basis::Zeeman) => 4,
        (StateSpace::Hilbert, Basis::Sphten) => {
            return Err(QoalaError::NotImplemented(
                "spherical tensors in Hilbert space".into(),
            ))
        }
    })
}

/// Propagator elements only (the two-output MATLAB call).
///
/// `rot_axis` is `M x 3` in `(x, y, z)` order and `dt` has `M` entries; the
/// result is `nelem x M`, i.e. already transposed relative to the MATLAB so
/// that a time point is a column.
pub fn rodrigues(
    space: StateSpace,
    basis: Basis,
    rot_axis: &DMatrix<f64>,
    dt: &DVector<f64>,
) -> Result<DMatrix<Complex64>> {
    let m = dt.len();
    if rot_axis.nrows() != m || rot_axis.ncols() != 3 {
        return Err(QoalaError::Dimension(format!(
            "rot_axis must be {m}x3, got {}x{}",
            rot_axis.nrows(),
            rot_axis.ncols()
        )));
    }
    let nelem = elements_per_point(space, basis)?;
    let mut p = DMatrix::<Complex64>::zeros(nelem, m);

    for i in 0..m {
        let (x, y, z) = (rot_axis[(i, 0)], rot_axis[(i, 1)], rot_axis[(i, 2)]);
        let r = (x * x + y * y + z * z).sqrt();
        let delta = dt[i] * r;
        let (nx, ny, nz) = if r == 0.0 {
            (0.0, 0.0, 0.0)
        } else {
            (x / r, y / r, z / r)
        };
        let sind = (delta / 2.0).sin();
        let alpha = Complex64::new((delta / 2.0).cos(), -nz * sind);
        let beta = Complex64::new(ny * sind, -nx * sind);
        write_elements(space, basis, alpha, beta, &mut p, i);
    }
    Ok(p)
}

/// Propagator elements together with the control derivatives (the three-output
/// MATLAB call).
///
/// `rot_ops` is `nspins x 3*npairs`, laid out as `(Lx, Ly, Lz)` triples per
/// control pair exactly like the MATLAB `pauli_operators` cell.
/// `spin_control` is `nspins x kctrls` and says which control channel drives
/// which spin.
#[allow(clippy::too_many_arguments)]
pub fn rodrigues_with_derivatives(
    space: StateSpace,
    basis: Basis,
    rot_ops: &[Vec<Option<CMat>>],
    rot_axis: &DMatrix<f64>,
    dt: &DVector<f64>,
    gradops: GradOps,
    spin_control: &DMatrix<bool>,
) -> Result<(DMatrix<Complex64>, Vec<ControlDerivative>)> {
    let p = rodrigues(space, basis, rot_axis, dt)?;

    let nspins = rot_ops.len();
    if nspins == 0 {
        return Err(QoalaError::BadValue("no control operators supplied".into()));
    }
    let ncols = rot_ops[0].len();
    if ncols % 3 != 0 {
        return Err(QoalaError::Dimension(
            "control operator array must have 3 columns (x, y, z) per control pair".into(),
        ));
    }
    let npairs = ncols / 3;
    let kctrls = 2 * npairs;
    let m = dt.len();
    if m % nspins != 0 {
        return Err(QoalaError::Dimension(format!(
            "time-point count {m} is not a multiple of the spin count {nspins}"
        )));
    }
    let npoints = m / nspins;

    // The D matrix: one row per time point, six columns holding
    // [D11 D12 D13 D21 D22 D23] - the x-derivative row and the y-derivative
    // row of dt*I + c1*S + c2*S^2.
    let mut d = DMatrix::<f64>::zeros(m, 6);
    for i in 0..m {
        let (x, y, z) = (rot_axis[(i, 0)], rot_axis[(i, 1)], rot_axis[(i, 2)]);
        let dti = dt[i];
        let mut r = (x * x + y * y + z * z).sqrt();
        if r == 0.0 {
            r = f64::EPSILON; // MATLAB: r(r==0)=eps
        }
        let delta = dti * r;
        let r2 = r * r;
        let c1 = (delta.cos() - 1.0) / r2;
        let c2 = (dti - delta.sin() / r) / r2;

        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (x2, y2, z2) = (x * x, y * y, z * z);

        // Row 1 (d/dx), then row 2 (d/dy).
        d[(i, 0)] = dti + c1 * 0.0 + c2 * (-y2 - z2);
        d[(i, 1)] = c1 * z + c2 * xy;
        d[(i, 2)] = c1 * (-y) + c2 * xz;
        d[(i, 3)] = c1 * (-z) + c2 * xy;
        d[(i, 4)] = dti + c1 * 0.0 + c2 * (-x2 - z2);
        d[(i, 5)] = c1 * x + c2 * yz;
    }

    let dim = first_operator_dim(rot_ops)?;
    let mut dp: Vec<ControlDerivative> = Vec::with_capacity(kctrls);
    // MATLAB keeps a shrinking list of spins still to be assigned a control.
    let mut spin_pool: Vec<usize> = (0..nspins).collect();

    for k in 0..npairs {
        let multi = spin_control.nrows() > 0
            && (0..spin_control.nrows())
                .filter(|&s| spin_control[(s, 2 * k)])
                .count()
                > 1;

        let mut x_terms: Vec<DerivativeTerm> = Vec::new();
        let mut y_terms: Vec<DerivativeTerm> = Vec::new();

        if multi {
            if gradops != GradOps::Full {
                return Err(QoalaError::NotImplemented(
                    "a control channel spanning several spins requires gradops = 'full'".into(),
                ));
            }
            let contributing: Vec<usize> = spin_pool
                .iter()
                .cloned()
                .filter(|&s| spin_control[(s, 2 * k)])
                .collect();
            for s in &contributing {
                let (lx, ly, lz) = triple(rot_ops, *s, k)?;
                let block = d.rows(s * npoints, npoints).into_owned();
                x_terms.push(DerivativeTerm {
                    lx: lx.clone(),
                    ly: ly.clone(),
                    lz: lz.clone(),
                    d: block.columns(0, 3).into_owned(),
                });
                y_terms.push(DerivativeTerm {
                    lx,
                    ly,
                    lz,
                    d: block.columns(3, 3).into_owned(),
                });
            }
            spin_pool.retain(|s| !contributing.contains(s));
        } else {
            let s = *spin_pool
                .first()
                .ok_or_else(|| QoalaError::BadValue("ran out of spins for control pairs".into()))?;
            let (lx, ly, lz) = triple(rot_ops, s, k)?;
            // The D rows belonging to this spin.  The MATLAB indexed this
            // block by the control-pair number rather than the spin number,
            // which is the same thing whenever control pair k drives spin k -
            // the layout every QOALA configuration uses.  Indexing by the spin
            // keeps it correct for any ordering.
            let block = d.rows(s * npoints, npoints).into_owned();
            x_terms.push(DerivativeTerm {
                lx: lx.clone(),
                ly: ly.clone(),
                lz: lz.clone(),
                d: block.columns(0, 3).into_owned(),
            });
            y_terms.push(DerivativeTerm {
                lx,
                ly,
                lz,
                d: block.columns(3, 3).into_owned(),
            });
            spin_pool.remove(0);
        }

        dp.push(ControlDerivative {
            terms: x_terms,
            npoints,
            dim,
        });
        dp.push(ControlDerivative {
            terms: y_terms,
            npoints,
            dim,
        });
    }

    Ok((p, dp))
}

/// The second derivative is not available.
///
/// The MATLAB `rodrigues` has a Hessian branch, but it references variables
/// (`S1`, and a `sigma` that has gone out of scope) that do not exist at that
/// point, and it is flagged in the source as not general for multi-spin
/// systems.  Calling it would error in MATLAB too, so the port declines
/// explicitly rather than pretending to implement it.
pub fn rodrigues_hessian_unavailable() -> QoalaError {
    QoalaError::NotImplemented(
        "second propagator derivatives (the MATLAB rodrigues Hessian branch is incomplete)".into(),
    )
}

fn triple(rot_ops: &[Vec<Option<CMat>>], spin: usize, pair: usize) -> Result<(CMat, CMat, CMat)> {
    let row = rot_ops
        .get(spin)
        .ok_or_else(|| QoalaError::Dimension(format!("no control operators for spin {spin}")))?;
    let get = |c: usize| -> Result<CMat> {
        row.get(c).and_then(|o| o.clone()).ok_or_else(|| {
            QoalaError::MissingField(format!("control operator at spin {spin}, column {c}"))
        })
    };
    Ok((get(3 * pair)?, get(3 * pair + 1)?, get(3 * pair + 2)?))
}

fn first_operator_dim(rot_ops: &[Vec<Option<CMat>>]) -> Result<usize> {
    rot_ops
        .iter()
        .flat_map(|row| row.iter().flatten())
        .map(|op| op.nrows())
        .next()
        .ok_or_else(|| QoalaError::BadValue("control operator array contains no operators".into()))
}

/// Fill column `i` of `p` with the propagator elements for this `(alpha, beta)`.
///
/// The element order is the column-major fill of the propagator block, so it
/// lines up with the index tables from [`crate::prop_index`].
fn write_elements(
    space: StateSpace,
    basis: Basis,
    alpha: Complex64,
    beta: Complex64,
    p: &mut DMatrix<Complex64>,
    i: usize,
) {
    let root2 = std::f64::consts::SQRT_2;
    match (space, basis) {
        (StateSpace::Liouville, Basis::Sphten) => {
            let alpha2 = alpha * alpha;
            let beta2 = beta * beta;
            let ab = alpha * beta * root2;
            let acb = alpha.conj() * beta * root2;
            let diag = alpha.norm_sqr() - beta.norm_sqr();
            let col = [
                alpha2,
                ab,
                beta2,
                -acb.conj(),
                Complex64::new(diag, 0.0),
                acb,
                beta2.conj(),
                -ab.conj(),
                alpha2.conj(),
            ];
            for (r, v) in col.iter().enumerate() {
                p[(r, i)] = *v;
            }
        }
        (StateSpace::Liouville, Basis::Zeeman) => {
            let alpha2 = alpha * alpha;
            let beta2 = beta * beta;
            let ab = alpha * beta;
            let acb = alpha.conj() * beta;
            let abc = alpha * beta.conj();
            let aca = alpha.conj() * alpha;
            let bcb = beta.conj() * beta;
            let col = [
                aca,
                acb,
                abc,
                bcb,
                -ab.conj(),
                alpha2.conj(),
                -beta2.conj(),
                ab.conj(),
                -ab,
                -beta2,
                alpha2,
                ab,
                bcb,
                -acb,
                -abc,
                aca,
            ];
            for (r, v) in col.iter().enumerate() {
                p[(r, i)] = *v;
            }
        }
        (StateSpace::Hilbert, Basis::Zeeman) => {
            let col = [alpha, beta, -beta.conj(), alpha.conj()];
            for (r, v) in col.iter().enumerate() {
                p[(r, i)] = *v;
            }
        }
        (StateSpace::Hilbert, Basis::Sphten) => unreachable!("rejected by elements_per_point"),
    }
}

/// Assemble a single-spin propagator matrix from its stored elements and an
/// index table, then Kronecker the spins together.
///
/// This is the `ind2propagator`/`ind2propagatorPROP` pair from the MATLAB
/// objective functions, with the `'prod'` method (the only one the shipped
/// objectives use).
///
/// * `prop_ind` - `(row, col)` pairs for one spin, including the unit-state
///   entry when the basis is spherical tensors.
/// * `elements` - the `(nelem + unit) x M` element array, i.e. the output of
///   [`rodrigues`] with the unit row prepended.
/// * `columns` - one column index per spin.
pub fn propagator_from_elements(
    prop_ind: &[(usize, usize)],
    elements: &DMatrix<Complex64>,
    columns: &[usize],
    single_dim: usize,
) -> Result<CMat> {
    if prop_ind.len() != elements.nrows() {
        return Err(QoalaError::Dimension(format!(
            "propagator index table has {} rows but the element array has {}",
            prop_ind.len(),
            elements.nrows()
        )));
    }
    let mut out: Option<CMat> = None;
    for &col in columns {
        if col >= elements.ncols() {
            return Err(QoalaError::Dimension(format!(
                "propagator element column {col} out of range"
            )));
        }
        let single = CMat::from_triplets(
            single_dim,
            single_dim,
            prop_ind
                .iter()
                .enumerate()
                .map(|(r, &(i, j))| (i, j, elements[(r, col)])),
        )?;
        out = Some(match out {
            None => single,
            Some(acc) => acc.kron(&single),
        });
    }
    out.ok_or_else(|| QoalaError::BadValue("no spins to build a propagator from".into()))
}

/// Prepend the unit-state row that the spherical tensor basis needs.
///
/// In the sphten basis the identity acts as `1` on the unit state, which is
/// not part of the rank-1 Wigner block, so the MATLAB glues a row of ones onto
/// the element array before indexing it.
pub fn prepend_unit_row(basis: Basis, elements: DMatrix<Complex64>) -> DMatrix<Complex64> {
    match basis {
        Basis::Sphten => {
            let (nr, nc) = (elements.nrows(), elements.ncols());
            let mut out = DMatrix::<Complex64>::zeros(nr + 1, nc);
            for c in 0..nc {
                out[(0, c)] = Complex64::new(1.0, 0.0);
                for r in 0..nr {
                    out[(r + 1, c)] = elements[(r, c)];
                }
            }
            out
        }
        Basis::Zeeman => elements,
    }
}

/// Dense helper used by the tests: the single-spin propagator implied by the
/// stored elements at column `i`.
pub fn single_spin_propagator(
    prop_ind: &[(usize, usize)],
    elements: &DMatrix<Complex64>,
    col: usize,
    dim: usize,
) -> Result<CDense> {
    Ok(propagator_from_elements(prop_ind, elements, &[col], dim)?.to_dense())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prop_index::propagator_ind;
    use crate::propagate::expm_pade;
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    /// Single spin-1/2 Cartesian operators in the spherical tensor Liouville
    /// basis, exactly as written in the QOALA example scripts.
    #[rustfmt::skip]
    fn sphten_single_spin() -> (CDense, CDense, CDense) {
        let r2 = std::f64::consts::SQRT_2;
        let z = Complex64::new(0.0, 0.0);
        let jx = CDense::from_row_slice(
            4,
            4,
            &[
                z, z, z, z,
                z, z, Complex64::new(1.0 / r2, 0.0), z,
                z, Complex64::new(1.0 / r2, 0.0), z, Complex64::new(1.0 / r2, 0.0),
                z, z, Complex64::new(1.0 / r2, 0.0), z,
            ],
        );
        let jy = CDense::from_row_slice(
            4,
            4,
            &[
                z, z, z, z,
                z, z, Complex64::new(0.0, -1.0 / r2), z,
                z, Complex64::new(0.0, 1.0 / r2), z, Complex64::new(0.0, -1.0 / r2),
                z, z, Complex64::new(0.0, 1.0 / r2), z,
            ],
        );
        let jz = CDense::from_row_slice(
            4,
            4,
            &[
                z, z, z, z,
                z, Complex64::new(1.0, 0.0), z, z,
                z, z, z, z,
                z, z, z, Complex64::new(-1.0, 0.0),
            ],
        );
        (jx, jy, jz)
    }

    #[test]
    fn rodrigues_reproduces_the_matrix_exponential() {
        let (jx, jy, jz) = sphten_single_spin();
        let ind = propagator_ind(StateSpace::Liouville, Basis::Sphten, 1).unwrap();
        let table = &ind[0];

        for &(x, y, z, dt) in &[
            (1.0, 0.0, 0.0, 0.3_f64),
            (0.0, 1.0, 0.0, 0.9),
            (0.3, -0.7, 1.4, 0.21),
            (-2.0, 0.5, 0.0, 1.7),
        ] {
            let axis = DMatrix::from_row_slice(1, 3, &[x, y, z]);
            let dtv = DVector::from_row_slice(&[dt]);
            let elems = rodrigues(StateSpace::Liouville, Basis::Sphten, &axis, &dtv).unwrap();
            let elems = prepend_unit_row(Basis::Sphten, elems);
            let got = single_spin_propagator(table, &elems, 0, 4).unwrap();

            let gen = (jx.map(|v| v * x) + jy.map(|v| v * y) + jz.map(|v| v * z))
                .map(|v| v * Complex64::new(0.0, -dt));
            let want = expm_pade(&gen).unwrap();
            assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn zero_field_gives_the_identity() {
        let axis = DMatrix::from_row_slice(1, 3, &[0.0, 0.0, 0.0]);
        let dtv = DVector::from_row_slice(&[0.5]);
        let elems = rodrigues(StateSpace::Liouville, Basis::Sphten, &axis, &dtv).unwrap();
        let elems = prepend_unit_row(Basis::Sphten, elems);
        let ind = propagator_ind(StateSpace::Liouville, Basis::Sphten, 1).unwrap();
        let got = single_spin_propagator(&ind[0], &elems, 0, 4).unwrap();
        let want = CDense::identity(4, 4);
        assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-14);
    }

    #[test]
    #[rustfmt::skip]
    fn hilbert_elements_are_the_su2_matrix() {
        let axis = DMatrix::from_row_slice(1, 3, &[0.4, -0.2, 1.1]);
        let dtv = DVector::from_row_slice(&[0.63]);
        let elems = rodrigues(StateSpace::Hilbert, Basis::Zeeman, &axis, &dtv).unwrap();
        let ind = propagator_ind(StateSpace::Hilbert, Basis::Zeeman, 1).unwrap();
        let got = single_spin_propagator(&ind[0], &elems, 0, 2).unwrap();

        let z = Complex64::new(0.0, 0.0);
        let half = Complex64::new(0.5, 0.0);
        let sx = CDense::from_row_slice(2, 2, &[z, half, half, z]);
        let sy = CDense::from_row_slice(
            2,
            2,
            &[z, Complex64::new(0.0, -0.5), Complex64::new(0.0, 0.5), z],
        );
        let sz = CDense::from_row_slice(2, 2, &[half, z, z, -half]);
        let gen = (sx.map(|v| v * 0.4) + sy.map(|v| v * -0.2) + sz.map(|v| v * 1.1))
            .map(|v| v * Complex64::new(0.0, -0.63));
        let want = expm_pade(&gen).unwrap();
        assert_relative_eq!((&got - &want).norm(), 0.0, epsilon = 1e-13);
        // Unitarity, for good measure.
        assert_relative_eq!(
            (got.adjoint() * &got - CDense::identity(2, 2)).norm(),
            0.0,
            epsilon = 1e-13
        );
    }

    #[test]
    fn derivative_matches_finite_differences() {
        // Check U^dagger dU/dx against a central difference of the propagator.
        let (jx, jy, jz) = sphten_single_spin();
        let ops = vec![vec![
            Some(CMat::Dense(jx.clone())),
            Some(CMat::Dense(jy.clone())),
            Some(CMat::Dense(jz.clone())),
        ]];
        let spin_control = DMatrix::from_row_slice(1, 2, &[true, true]);
        let (x, y, z, dt) = (0.8_f64, -0.35_f64, 0.5_f64, 0.44_f64);

        let axis = DMatrix::from_row_slice(1, 3, &[x, y, z]);
        let dtv = DVector::from_row_slice(&[dt]);
        let (_, dp) = rodrigues_with_derivatives(
            StateSpace::Liouville,
            Basis::Sphten,
            &ops,
            &axis,
            &dtv,
            GradOps::Full,
            &spin_control,
        )
        .unwrap();

        let prop = |xx: f64, yy: f64, zz: f64| {
            let gen = (jx.map(|v| v * xx) + jy.map(|v| v * yy) + jz.map(|v| v * zz))
                .map(|v| v * Complex64::new(0.0, -dt));
            expm_pade(&gen).unwrap()
        };
        let u = prop(x, y, z);
        let h = 1e-6;

        for (axis_idx, du_num) in [
            (
                0usize,
                (prop(x + h, y, z) - prop(x - h, y, z)).map(|v| v / (2.0 * h)),
            ),
            (
                1usize,
                (prop(x, y + h, z) - prop(x, y - h, z)).map(|v| v / (2.0 * h)),
            ),
        ] {
            let want = u.adjoint() * du_num;
            let got = dp[axis_idx].matrix(0).unwrap().to_dense();
            assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-6);
        }
    }
}
