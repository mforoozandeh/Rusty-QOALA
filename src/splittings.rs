//! Operator splittings for the interaction Hamiltonian.
//!
//! Port of `utilities/operator_splittings.m`.  QOALA never exponentiates the
//! full Hamiltonian: it splits it into a single-spin part (handled in closed
//! form by [`crate::rodrigues`]) and an interaction part (handled by a fixed
//! set of small propagators computed once here), then interleaves them.
//!
//! Supported orders are 0 (interaction ignored), 1 (Lie-Trotter), 2 (Strang),
//! 3 and 4 (Blanes) and 6 (Omelyan).  The order-3 scheme uses complex
//! coefficients, which is why every propagator time step in this crate is a
//! [`Complex64`].
//!
//! References:
//! * Blanes, *Applying splitting methods with complex coefficients to the
//!   numerical integration of unitary problems* (2021).
//! * Blanes, *Splitting methods with complex coefficients for some classes of
//!   evolution equations*.
//! * Blanes, *Splitting and composition methods with embedded error
//!   estimators* (2019).
//! * Castella, *Splitting methods with complex times for parabolic equations*
//!   (2009).

use crate::error::{QoalaError, Result};
use crate::linalg::CMat;
use crate::propagate::propagator;
use crate::types::PropMethod;
use num_complex::Complex64;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// Splitting constants and the interleaving order for one `(order, trotter)`
/// pair.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitCoefficients {
    /// Time-step multipliers for the single-spin propagators.
    pub c_spn: Vec<f64>,
    /// Time-step multipliers for the interaction propagators; complex for
    /// order 3.
    pub c_cpl: Vec<Complex64>,
    /// Zero-based order in which the single-spin propagators are applied.
    pub ind_spn: Vec<usize>,
    /// Zero-based order in which the interaction propagators are applied.
    pub ind_int: Vec<usize>,
}

impl SplitCoefficients {
    /// Number of distinct single-spin propagators per time step.
    pub fn nsplits(&self) -> usize {
        self.c_spn.len()
    }
    /// Number of single-spin stages per time step.
    pub fn num_a(&self) -> usize {
        self.ind_spn.len()
    }
    /// Number of interaction stages per time step.
    pub fn num_b(&self) -> usize {
        self.ind_int.len()
    }
}

/// The raw table for one splitting order: single-spin constants, coupling
/// constants, the opening index sequences, the sequences repeated per Trotter
/// step, and the closing interaction index.
type CoefficientTable = (
    Vec<f64>,
    Vec<Complex64>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    usize,
);

/// Build the splitting constants and index sequences.
pub fn split_coefficients(order: usize, trotter: usize) -> Result<SplitCoefficients> {
    if trotter == 0 {
        return Err(QoalaError::BadValue(
            "Trotter number must be at least 1".into(),
        ));
    }
    let table: CoefficientTable = match order {
        0 => (
            vec![1.0],
            vec![Complex64::new(0.0, 0.0)],
            vec![1],
            vec![1],
            vec![1],
            vec![1],
            1,
        ),
        // Lie-Trotter.  The MATLAB repeats single-spin index 2 and
        // interaction index 3 inside the Trotter loop, but this order defines
        // only one single-spin constant and two coupling constants, so those
        // indices do not exist and a Trotter number above one cannot run.
        // The repeats below are the intended Lie-Trotter sequence
        // (e^{-iA dt'} e^{-iB dt'} repeated), which is unchanged for the
        // trotter = 1 case the MATLAB actually reaches.
        1 => (
            vec![1.0],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![1],
            vec![1],
            vec![1],
            vec![2],
            2,
        ),
        2 => (
            vec![1.0],
            vec![Complex64::new(0.5, 0.0)],
            vec![1],
            vec![1],
            vec![1],
            vec![2],
            1,
        ),
        3 => {
            let s = (59.0f64 / 2.0).sqrt();
            (
                vec![3.0 / 10.0, 2.0 / 5.0],
                vec![
                    Complex64::new(13.0 / 126.0, s / 63.0),
                    Complex64::new(25.0 / 63.0, -5.0 * s / 126.0),
                    Complex64::new(25.0 / 63.0, 5.0 * s / 126.0),
                    Complex64::new(13.0 / 126.0, -s / 63.0),
                ],
                vec![1, 2, 1],
                vec![1, 2, 3],
                vec![1, 2, 1],
                vec![5, 2, 3],
                4,
            )
        }
        4 => {
            let a1 = 0.209_515_106_613_362;
            let a2 = -0.143_851_773_179_818;
            let b1 = 0.079_203_696_431_195_7;
            let b2 = 0.353_172_906_049_774;
            let b3 = -0.042_065_080_357_719_5;
            (
                vec![a1, a2, 0.5 - (a1 + a2)],
                vec![
                    Complex64::new(b1, 0.0),
                    Complex64::new(b2, 0.0),
                    Complex64::new(b3, 0.0),
                    Complex64::new(1.0 - 2.0 * (b1 + b2 + b3), 0.0),
                ],
                vec![1, 2, 3, 3, 2, 1],
                vec![1, 2, 3, 4, 3, 2],
                vec![1, 2, 3, 3, 2, 1],
                vec![5, 2, 3, 4, 3, 2],
                1,
            )
        }
        6 => {
            let a1 = 0.148_816_447_901_042;
            let a2 = -0.132_385_865_767_784;
            let a3 = 0.067_307_604_692_185;
            let a4 = 0.432_666_402_578_175;
            let b1 = 0.050_262_764_400_392_2;
            let b2 = 0.413_514_300_428_344;
            let b3 = 0.045_079_889_794_397_7;
            let b4 = -0.188_054_853_819_569;
            let b5 = 0.541_960_678_450_780;
            (
                vec![a1, a2, a3, a4, 0.5 - (a1 + a2 + a3 + a4)],
                vec![
                    Complex64::new(b1, 0.0),
                    Complex64::new(b2, 0.0),
                    Complex64::new(b3, 0.0),
                    Complex64::new(b4, 0.0),
                    Complex64::new(b5, 0.0),
                    Complex64::new(1.0 - 2.0 * (b1 + b2 + b3 + b4 + b5), 0.0),
                ],
                vec![1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
                vec![1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
                vec![1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
                vec![7, 2, 3, 4, 5, 6, 5, 4, 3, 2],
                1,
            )
        }
        n => {
            return Err(QoalaError::BadValue(format!(
                "splitting order {n} is not coded"
            )))
        }
    };

    let (c_spn, c_cpl, mut i_spn, mut i_int, repeat_spn, repeat_int, tail_int) = table;
    for _ in 1..trotter {
        i_spn.extend(repeat_spn.iter().cloned());
        i_int.extend(repeat_int.iter().cloned());
    }
    i_int.push(tail_int);

    Ok(SplitCoefficients {
        c_spn,
        c_cpl,
        ind_spn: i_spn.into_iter().map(|i| i - 1).collect(),
        ind_int: i_int.into_iter().map(|i| i - 1).collect(),
    })
}

/// Interaction propagators for one `(order, trotter)` pair.
///
/// An entry of `None` stands for the identity, which is how the MATLAB stores
/// the trivial propagators of orders 0 and 1 (`P{1} = 1`, "don't even use
/// matrices").  When the Trotter number is greater than one an extra
/// propagator is appended: the merged first-and-last half step that joins
/// consecutive Trotter repeats.
pub fn interaction_propagators(
    coeffs: &SplitCoefficients,
    order: usize,
    trotter: usize,
    dt: f64,
    interaction: &CMat,
    method: PropMethod,
    nonzero_tol: f64,
) -> Result<Vec<Option<CMat>>> {
    let dt_trt = dt / trotter as f64;

    if order == 0 {
        return Ok(vec![None]);
    }
    if order == 1 {
        return Ok(vec![
            None,
            Some(
                propagator(interaction, coeffs.c_cpl[1] * dt_trt, method, nonzero_tol)?
                    .into_sparse(),
            ),
        ]);
    }

    let mut out: Vec<Option<CMat>> = Vec::with_capacity(coeffs.c_cpl.len() + 1);
    for c in &coeffs.c_cpl {
        out.push(Some(
            propagator(interaction, c * dt_trt, method, nonzero_tol)?.into_sparse(),
        ));
    }
    if trotter != 1 {
        let first = coeffs.c_cpl[coeffs.ind_int[0]];
        let last = coeffs.c_cpl[*coeffs.ind_int.last().unwrap()];
        let a = propagator(interaction, first * dt_trt, method, nonzero_tol)?;
        let b = propagator(interaction, last * dt_trt, method, nonzero_tol)?;
        out.push(Some(a.matmul(&b)?.into_sparse()));
    }
    Ok(out)
}

/// Everything needed to take one split step for a given `(trotter, order)`.
#[derive(Debug, Clone)]
pub struct SplitSet {
    /// Trotter number this set was built for.
    pub trotter: usize,
    /// Splitting order this set was built for.
    pub order: usize,
    /// Splitting coefficients and index sequences.
    pub coeffs: SplitCoefficients,
    /// Interaction propagators; `None` means identity.
    pub inter_prop: Vec<Option<CMat>>,
}

impl SplitSet {
    /// Build a complete set, computing the interaction propagators.
    pub fn build(
        order: usize,
        trotter: usize,
        dt: f64,
        interaction: &CMat,
        method: PropMethod,
        nonzero_tol: f64,
    ) -> Result<Self> {
        let coeffs = split_coefficients(order, trotter)?;
        let inter_prop = interaction_propagators(
            &coeffs,
            order,
            trotter,
            dt,
            interaction,
            method,
            nonzero_tol,
        )?;
        Ok(SplitSet {
            trotter,
            order,
            coeffs,
            inter_prop,
        })
    }

    /// Build a set, reusing a cached copy of the interaction propagators when
    /// one exists.  This is the `prop_cache = 'store'` path.
    ///
    /// Native only: `wasm32` has no filesystem, and
    /// [`crate::types::PropCache::Store`] is rejected there.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn build_cached(
        order: usize,
        trotter: usize,
        dt: f64,
        interaction: &CMat,
        method: PropMethod,
        nonzero_tol: f64,
        cache_dir: &Path,
    ) -> Result<Self> {
        let coeffs = split_coefficients(order, trotter)?;
        let file = cache_dir.join(format!("prop_t{trotter}o{order}.bin"));
        if file.exists() {
            if let Ok(inter_prop) = read_propagator_cache(&file) {
                return Ok(SplitSet {
                    trotter,
                    order,
                    coeffs,
                    inter_prop,
                });
            }
        }
        let inter_prop = interaction_propagators(
            &coeffs,
            order,
            trotter,
            dt,
            interaction,
            method,
            nonzero_tol,
        )?;
        fs::create_dir_all(cache_dir)?;
        write_propagator_cache(&file, &inter_prop)?;
        Ok(SplitSet {
            trotter,
            order,
            coeffs,
            inter_prop,
        })
    }

    /// Convenience: the interaction propagator for stage index `i`.
    pub fn int_prop(&self, i: usize) -> Result<Option<&CMat>> {
        self.inter_prop
            .get(i)
            .map(|o| o.as_ref())
            .ok_or_else(|| QoalaError::Dimension(format!("interaction propagator {i} missing")))
    }
}

#[cfg(not(target_arch = "wasm32"))]
const CACHE_MAGIC: &[u8; 8] = b"QOALAP01";

/// Write interaction propagators to the scratch cache.
#[cfg(not(target_arch = "wasm32"))]
pub fn write_propagator_cache(path: &Path, props: &[Option<CMat>]) -> Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(CACHE_MAGIC)?;
    f.write_all(&(props.len() as u64).to_le_bytes())?;
    for p in props {
        match p {
            None => f.write_all(&[0u8])?,
            Some(m) => {
                f.write_all(&[1u8])?;
                let sp = m.to_sparse();
                let entries: Vec<(usize, usize, Complex64)> =
                    sp.triplet_iter().map(|(r, c, v)| (r, c, *v)).collect();
                f.write_all(&(sp.nrows() as u64).to_le_bytes())?;
                f.write_all(&(sp.ncols() as u64).to_le_bytes())?;
                f.write_all(&(entries.len() as u64).to_le_bytes())?;
                for (r, c, v) in entries {
                    f.write_all(&(r as u64).to_le_bytes())?;
                    f.write_all(&(c as u64).to_le_bytes())?;
                    f.write_all(&v.re.to_le_bytes())?;
                    f.write_all(&v.im.to_le_bytes())?;
                }
            }
        }
    }
    Ok(())
}

/// Read interaction propagators back from the scratch cache.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_propagator_cache(path: &Path) -> Result<Vec<Option<CMat>>> {
    let mut f = fs::File::open(path)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != CACHE_MAGIC {
        return Err(QoalaError::Io(format!(
            "{} is not a QOALA propagator cache",
            path.display()
        )));
    }
    let mut u64buf = [0u8; 8];
    let mut f64buf = [0u8; 8];
    let mut read_u64 = |f: &mut fs::File| -> Result<u64> {
        f.read_exact(&mut u64buf)?;
        Ok(u64::from_le_bytes(u64buf))
    };
    let count = read_u64(&mut f)? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut tag = [0u8; 1];
        f.read_exact(&mut tag)?;
        if tag[0] == 0 {
            out.push(None);
            continue;
        }
        let rows = read_u64(&mut f)? as usize;
        let cols = read_u64(&mut f)? as usize;
        let nnz = read_u64(&mut f)? as usize;
        let mut triplets = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            let r = read_u64(&mut f)? as usize;
            let c = read_u64(&mut f)? as usize;
            f.read_exact(&mut f64buf)?;
            let re = f64::from_le_bytes(f64buf);
            f.read_exact(&mut f64buf)?;
            let im = f64::from_le_bytes(f64buf);
            triplets.push((r, c, Complex64::new(re, im)));
        }
        out.push(Some(CMat::from_triplets(rows, cols, triplets)?));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{CDense, CMat};
    use crate::propagate::expm_pade;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    fn c(re: f64) -> Complex64 {
        Complex64::new(re, 0.0)
    }

    /// Two deliberately non-commuting Hermitian matrices.
    #[rustfmt::skip]
    fn test_pair() -> (CDense, CDense) {
        let a = CDense::from_row_slice(
            3,
            3,
            &[
                c(0.0), c(1.0), c(0.0),
                c(1.0), c(0.0), c(0.7),
                c(0.0), c(0.7), c(0.0),
            ],
        );
        let b = CDense::from_row_slice(
            3,
            3,
            &[
                c(1.3), c(0.0), c(0.2),
                c(0.0), c(-0.4), c(0.0),
                c(0.2), c(0.0), c(0.9),
            ],
        );
        (a, b)
    }

    /// Assemble one split step exactly as the objective functions do.
    fn split_step(order: usize, trotter: usize, dt: f64, a: &CDense, b: &CDense) -> CDense {
        let coeffs = split_coefficients(order, trotter).unwrap();
        let interaction = CMat::Dense(a.clone());
        let props = interaction_propagators(
            &coeffs,
            order,
            trotter,
            dt,
            &interaction,
            PropMethod::Pade,
            1e-16,
        )
        .unwrap();

        let dt_trt = dt / trotter as f64;
        let spn: Vec<CDense> = coeffs
            .c_spn
            .iter()
            .map(|cs| expm_pade(&b.map(|z| z * Complex64::new(0.0, -cs * dt_trt))).unwrap())
            .collect();
        let apply = |p: &Option<CMat>, x: CDense| -> CDense {
            match p {
                None => x,
                Some(m) => m.apply(&x).unwrap(),
            }
        };

        let n = a.nrows();
        let mut x = CDense::identity(n, n);
        x = apply(&props[coeffs.ind_int[0]], x);
        for i in 1..coeffs.num_a() {
            x = &spn[coeffs.ind_spn[i - 1]] * x;
            x = apply(&props[coeffs.ind_int[i]], x);
        }
        x = &spn[coeffs.ind_spn[coeffs.num_a() - 1]] * x;
        apply(&props[coeffs.ind_int[coeffs.num_b() - 1]], x)
    }

    fn exact_step(dt: f64, a: &CDense, b: &CDense) -> CDense {
        expm_pade(&(a + b).map(|z| z * Complex64::new(0.0, -dt))).unwrap()
    }

    #[test]
    fn coefficients_sum_to_one() {
        // Whatever the order or Trotter number, the applied time steps must
        // add up to the full step: sum(c_spn) = trotter and sum(c_cpl) =
        // trotter, where the trailing index may address the merged
        // first-plus-last propagator.
        for order in [0usize, 1, 2, 3, 4, 6] {
            for trotter in [1usize, 2, 3] {
                let k = split_coefficients(order, trotter).unwrap();
                let spn: f64 = k.ind_spn.iter().map(|&i| k.c_spn[i]).sum();
                assert_relative_eq!(spn, trotter as f64, epsilon = 1e-12, max_relative = 1e-12);

                // Order zero drops the interaction, so its coupling
                // constants are all zero by construction.
                if order > 0 {
                    let merged = k.c_cpl[k.ind_int[0]] + k.c_cpl[*k.ind_int.last().unwrap()];
                    let cpl: Complex64 = k
                        .ind_int
                        .iter()
                        .map(|&i| {
                            if i < k.c_cpl.len() {
                                k.c_cpl[i]
                            } else {
                                merged
                            }
                        })
                        .sum();
                    assert_relative_eq!(cpl.re, trotter as f64, epsilon = 1e-12);
                    assert_relative_eq!(cpl.im, 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn splittings_reach_their_advertised_order() {
        let (a, b) = test_pair();
        for order in [1usize, 2, 3, 4, 6] {
            let err = |dt: f64| (split_step(order, 1, dt, &a, &b) - exact_step(dt, &a, &b)).norm();
            let dt1 = 0.05;
            let dt2 = 0.025;
            let e1 = err(dt1);
            let e2 = err(dt2);
            // Halving the step should shrink the error by 2^(order+1).
            let observed = (e1 / e2).log2();
            assert!(
                (observed - (order as f64 + 1.0)).abs() < 0.4,
                "order {order}: observed convergence {observed:.2}, expected {:.2} (errors {e1:.3e}, {e2:.3e})",
                order as f64 + 1.0
            );
        }
    }

    #[test]
    fn order_zero_ignores_the_interaction() {
        let (a, b) = test_pair();
        let dt = 0.1;
        let got = split_step(0, 1, dt, &a, &b);
        let want = expm_pade(&b.map(|z| z * Complex64::new(0.0, -dt))).unwrap();
        assert_relative_eq!((got - want).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn trotterisation_improves_accuracy() {
        let (a, b) = test_pair();
        let dt = 0.4;
        let e1 = (split_step(2, 1, dt, &a, &b) - exact_step(dt, &a, &b)).norm();
        let e2 = (split_step(2, 2, dt, &a, &b) - exact_step(dt, &a, &b)).norm();
        let e4 = (split_step(2, 4, dt, &a, &b) - exact_step(dt, &a, &b)).norm();
        assert!(e2 < e1, "{e2} !< {e1}");
        assert!(e4 < e2, "{e4} !< {e2}");
        // Second-order Trotter: doubling the repeats should quarter the error.
        assert!((((e1 / e2).log2()) - 2.0).abs() < 0.4);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn propagator_cache_round_trips() {
        let dir = std::env::temp_dir().join("qoala_split_cache_test");
        let _ = fs::create_dir_all(&dir);
        let (a, _) = test_pair();
        let coeffs = split_coefficients(4, 2).unwrap();
        let props =
            interaction_propagators(&coeffs, 4, 2, 0.1, &CMat::Dense(a), PropMethod::Pade, 1e-14)
                .unwrap();
        let path = dir.join("prop_t2o4.bin");
        write_propagator_cache(&path, &props).unwrap();
        let back = read_propagator_cache(&path).unwrap();
        assert_eq!(back.len(), props.len());
        for (x, y) in props.iter().zip(back.iter()) {
            match (x, y) {
                (None, None) => {}
                (Some(p), Some(q)) => {
                    assert_relative_eq!((p.to_dense() - q.to_dense()).norm(), 0.0, epsilon = 1e-15)
                }
                _ => panic!("cache round trip changed an entry"),
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
