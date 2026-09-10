//! Sparse index tables for propagator elements.
//!
//! Port of `utilities/propagator_ind.m`.  [`crate::rodrigues`] produces a flat
//! list of propagator elements per time point; these tables say where each of
//! those elements lands in the (single-spin or composite) propagator matrix.
//!
//! Indices are returned zero-based, unlike the one-based MATLAB tables.

use crate::error::{QoalaError, Result};
use crate::types::{Basis, StateSpace};

/// Kronecker product of two integer row vectors, as MATLAB's `kron` defines it.
fn kron_i(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for &x in a {
        for &y in b {
            out.push(x * y);
        }
    }
    out
}

fn zip_pairs(rows: &[i64], cols: &[i64]) -> Vec<(usize, usize)> {
    rows.iter()
        .zip(cols.iter())
        .map(|(&r, &c)| ((r - 1) as usize, (c - 1) as usize))
        .collect()
}

/// Index tables, one per spin.
///
/// For `n_spins == 1` the single table maps the stored elements onto the
/// single-spin propagator block; that is the table the objective functions use
/// to rebuild propagators via Kronecker products.  For larger `n_spins` the
/// tables address the composite space directly.
pub fn propagator_ind(
    space: StateSpace,
    basis: Basis,
    n_spins: usize,
) -> Result<Vec<Vec<(usize, usize)>>> {
    if n_spins == 0 {
        return Err(QoalaError::BadValue("need at least one spin".into()));
    }
    match (space, basis) {
        (StateSpace::Liouville, Basis::Sphten) => sphten_liouville(n_spins),
        (StateSpace::Liouville, Basis::Zeeman) => zeeman_liouville(n_spins),
        (StateSpace::Hilbert, Basis::Zeeman) => zeeman_hilbert(n_spins),
        (StateSpace::Hilbert, Basis::Sphten) => Err(QoalaError::NotImplemented(
            "propagator_ind for the sphten basis in a hilbert space formalism".into(),
        )),
    }
}

/// Spherical tensors in Liouville space: the general construction.
fn sphten_liouville(n_spins: usize) -> Result<Vec<Vec<(usize, usize)>>> {
    // Element ordering inside one rank-1 Wigner block, and the unit-state
    // padding for the spins that are not being addressed.
    let n_row = kron_i(&[1, 1, 1], &[1, 2, 3]);
    let n_col = kron_i(&[1, 2, 3], &[1, 1, 1]);
    let i_dim = kron_i(&[1, 1, 1], &[1, 1, 1]);
    let ident = [1i64, 1, 1, 1];
    let n_vec = [0i64, 1, 2, 3];

    let mut out = Vec::with_capacity(n_spins);
    for n in 1..=n_spins {
        // MATLAB: N_mat = E([n_spins:-1:n_spins-n+2, 1:n_spins-n+1], :)
        let mut order: Vec<usize> = ((n_spins - n + 2)..=n_spins).rev().collect();
        order.extend(1..=(n_spins - n + 1));

        let mut rows_acc: Vec<i64> = Vec::new();
        let mut cols_acc: Vec<i64> = Vec::new();

        for (m1, &row_of_e) in order.iter().enumerate() {
            let mut kron_rows: Vec<i64> = vec![1];
            let mut kron_cols: Vec<i64> = vec![1];
            for m2 in 1..=n_spins {
                // E(row_of_e, m2) is 1 exactly when m2 == row_of_e.
                let on = m2 == row_of_e;
                let (row_cell, col_cell): (&[i64], &[i64]) = match (on, m2 == 1) {
                    (true, true) => (&n_row, &n_col),
                    (true, false) => (&n_vec, &n_vec),
                    (false, true) => (&i_dim, &i_dim),
                    (false, false) => (&ident, &ident),
                };
                kron_rows = kron_i(row_cell, &kron_rows);
                kron_cols = kron_i(col_cell, &kron_cols);
            }
            let scale = 4i64.pow((n_spins - (m1 + 1)) as u32);
            if rows_acc.is_empty() {
                rows_acc = vec![1; kron_rows.len()];
                cols_acc = vec![1; kron_cols.len()];
            }
            for (a, b) in rows_acc.iter_mut().zip(kron_rows.iter()) {
                *a += b * scale;
            }
            for (a, b) in cols_acc.iter_mut().zip(kron_cols.iter()) {
                *a += b * scale;
            }
        }
        out.push(zip_pairs(&rows_acc, &cols_acc));
    }

    if n_spins == 1 {
        // The unit state is not part of the rank-1 block, so it gets its own
        // leading entry at (1,1).
        let mut first = vec![(0usize, 0usize)];
        first.extend(out[0].iter().cloned());
        out[0] = first;
    }
    Ok(out)
}

/// Zeeman basis in Liouville space; the MATLAB tabulates one to four spins.
fn zeeman_liouville(n_spins: usize) -> Result<Vec<Vec<(usize, usize)>>> {
    let i = [1i64, 1, 1, 1];
    let n = [0i64, 1, 2, 3];
    let f = [1i64, 2, 3, 4];
    let n_row = kron_i(&[1, 1, 1], &[1, 2, 3]);
    let n_col = kron_i(&[1, 2, 3], &[1, 1, 1]);
    let i_dim = kron_i(&[1, 1, 1], &[1, 1, 1]);

    let add = |a: Vec<i64>, b: Vec<i64>| -> Vec<i64> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    };
    let mul = |a: Vec<i64>, s: i64| -> Vec<i64> { a.into_iter().map(|x| x * s).collect() };

    match n_spins {
        1 => {
            let rows = kron_i(&i, &f);
            let cols = kron_i(&f, &i);
            Ok(vec![zip_pairs(&rows, &cols)])
        }
        2 => {
            let mut out = Vec::new();
            let rows = add(
                mul(kron_i(&i, &kron_i(&i, &n)), 4),
                kron_i(&f, &kron_i(&i, &i)),
            );
            let cols = add(
                mul(kron_i(&i, &kron_i(&n, &i)), 4),
                kron_i(&f, &kron_i(&i, &i)),
            );
            out.push(zip_pairs(&rows, &cols));

            let rows = add(
                kron_i(&i, &kron_i(&i, &f)),
                mul(kron_i(&n, &kron_i(&i, &i)), 4),
            );
            let cols = add(
                kron_i(&i, &kron_i(&f, &i)),
                mul(kron_i(&n, &kron_i(&i, &i)), 4),
            );
            out.push(zip_pairs(&rows, &cols));
            Ok(out)
        }
        3 => {
            let mut out = Vec::new();
            for powers in [[2i64, 0, 1], [1, 0, 2], [0, 1, 2]] {
                let rows = add(
                    add(
                        mul(kron_i(&i, &kron_i(&i, &n_row)), 4i64.pow(powers[0] as u32)),
                        mul(kron_i(&i, &kron_i(&n, &i_dim)), 4i64.pow(powers[1] as u32)),
                    ),
                    mul(kron_i(&n, &kron_i(&i, &i_dim)), 4i64.pow(powers[2] as u32)),
                );
                let cols = add(
                    add(
                        mul(kron_i(&i, &kron_i(&i, &n_col)), 4i64.pow(powers[0] as u32)),
                        mul(kron_i(&i, &kron_i(&n, &i_dim)), 4i64.pow(powers[1] as u32)),
                    ),
                    mul(kron_i(&n, &kron_i(&i, &i_dim)), 4i64.pow(powers[2] as u32)),
                );
                out.push(zip_pairs(&rows, &cols));
            }
            Ok(out)
        }
        4 => {
            let mut out = Vec::new();
            for powers in [[3i64, 1, 2, 0], [2, 1, 3, 0], [1, 0, 2, 3], [0, 1, 2, 3]] {
                let rows = add(
                    add(
                        mul(
                            kron_i(&i, &kron_i(&i, &kron_i(&i, &n_row))),
                            4i64.pow(powers[0] as u32),
                        ),
                        mul(
                            kron_i(&i, &kron_i(&i, &kron_i(&n, &i_dim))),
                            4i64.pow(powers[1] as u32),
                        ),
                    ),
                    add(
                        mul(
                            kron_i(&i, &kron_i(&n, &kron_i(&i, &i_dim))),
                            4i64.pow(powers[2] as u32),
                        ),
                        mul(
                            kron_i(&n, &kron_i(&i, &kron_i(&i, &i_dim))),
                            4i64.pow(powers[3] as u32),
                        ),
                    ),
                );
                let cols = add(
                    add(
                        mul(
                            kron_i(&i, &kron_i(&i, &kron_i(&i, &n_col))),
                            4i64.pow(powers[0] as u32),
                        ),
                        mul(
                            kron_i(&i, &kron_i(&i, &kron_i(&n, &i_dim))),
                            4i64.pow(powers[1] as u32),
                        ),
                    ),
                    add(
                        mul(
                            kron_i(&i, &kron_i(&n, &kron_i(&i, &i_dim))),
                            4i64.pow(powers[2] as u32),
                        ),
                        mul(
                            kron_i(&n, &kron_i(&i, &kron_i(&i, &i_dim))),
                            4i64.pow(powers[3] as u32),
                        ),
                    ),
                );
                out.push(zip_pairs(&rows, &cols));
            }
            Ok(out)
        }
        _ => Err(QoalaError::NotImplemented(format!(
            "zeeman-liouville propagator indices for {n_spins} spins"
        ))),
    }
}

/// Zeeman basis in Hilbert space; the MATLAB tabulates one and two spins.
fn zeeman_hilbert(n_spins: usize) -> Result<Vec<Vec<(usize, usize)>>> {
    let one = [1i64, 1];
    let f = [1i64, 2];
    let n = [0i64, 1];
    let add = |a: Vec<i64>, b: Vec<i64>| -> Vec<i64> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    };
    let mul = |a: Vec<i64>, s: i64| -> Vec<i64> { a.into_iter().map(|x| x * s).collect() };

    match n_spins {
        1 => {
            let rows = kron_i(&one, &f);
            let cols = kron_i(&f, &one);
            Ok(vec![zip_pairs(&rows, &cols)])
        }
        2 => {
            let mut out = Vec::new();
            let rows = add(
                mul(kron_i(&one, &kron_i(&one, &n)), 2),
                kron_i(&f, &kron_i(&one, &one)),
            );
            let cols = add(
                kron_i(&f, &kron_i(&one, &one)),
                mul(kron_i(&one, &kron_i(&n, &one)), 2),
            );
            out.push(zip_pairs(&rows, &cols));

            let rows = add(
                kron_i(&one, &kron_i(&one, &f)),
                mul(kron_i(&n, &kron_i(&one, &one)), 2),
            );
            let cols = add(
                kron_i(&one, &kron_i(&f, &one)),
                mul(kron_i(&n, &kron_i(&one, &one)), 2),
            );
            out.push(zip_pairs(&rows, &cols));
            Ok(out)
        }
        _ => Err(QoalaError::NotImplemented(format!(
            "zeeman-hilbert propagator indices for {n_spins} spins"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_spin_sphten_table_has_the_unit_state_first() {
        let t = propagator_ind(StateSpace::Liouville, Basis::Sphten, 1).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].len(), 10);
        assert_eq!(t[0][0], (0, 0));
        // The remaining nine entries fill rows/cols 1..3 column-major.
        let expected = [
            (1, 1),
            (2, 1),
            (3, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (1, 3),
            (2, 3),
            (3, 3),
        ];
        assert_eq!(&t[0][1..], &expected[..]);
    }

    #[test]
    fn single_spin_zeeman_tables_are_column_major_fills() {
        let t = propagator_ind(StateSpace::Liouville, Basis::Zeeman, 1).unwrap();
        assert_eq!(t[0].len(), 16);
        for (k, &(r, c)) in t[0].iter().enumerate() {
            assert_eq!(r, k % 4);
            assert_eq!(c, k / 4);
        }
        let h = propagator_ind(StateSpace::Hilbert, Basis::Zeeman, 1).unwrap();
        assert_eq!(h[0].len(), 4);
        for (k, &(r, c)) in h[0].iter().enumerate() {
            assert_eq!(r, k % 2);
            assert_eq!(c, k / 2);
        }
    }

    #[test]
    fn multi_spin_sphten_tables_stay_inside_the_composite_space() {
        for nspins in 1..=4 {
            let t = propagator_ind(StateSpace::Liouville, Basis::Sphten, nspins).unwrap();
            assert_eq!(t.len(), nspins);
            let dim = 4usize.pow(nspins as u32);
            for table in &t {
                for &(r, c) in table {
                    assert!(r < dim, "row {r} outside dim {dim}");
                    assert!(c < dim, "col {c} outside dim {dim}");
                }
            }
        }
    }

    #[test]
    fn unsupported_combinations_are_rejected() {
        assert!(propagator_ind(StateSpace::Hilbert, Basis::Sphten, 1).is_err());
        assert!(propagator_ind(StateSpace::Liouville, Basis::Zeeman, 5).is_err());
        assert!(propagator_ind(StateSpace::Hilbert, Basis::Zeeman, 3).is_err());
    }
}
