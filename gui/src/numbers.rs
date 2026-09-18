//! How the interface writes numbers.
//!
//! Every quantity is entered and shown in SI units, which spans many orders
//! of magnitude: a pulse can last 3e-9 s or 3e-3 s, a field be 50 Hz or
//! 2e9 Hz.  Numbers of ordinary size are written out in full; anything below
//! 0.001, or from 100000 up, goes to scientific notation.  Typing either form
//! works - `2e6`, `2E6` and `2000000` are the same number - because egui reads
//! what is typed with Rust's own number parser.

/// Smallest magnitude written out in full.
const PLAIN_FROM: f64 = 1e-3;
/// Smallest magnitude written in scientific notation from above.
const PLAIN_BELOW: f64 = 1e5;

/// `x` in full between 0.001 and 99999, in scientific notation outside that,
/// always in the shortest form that reads back as exactly `x`: `3e-6`, `2e6`,
/// `1.5e-5`, `17000`, `0.25`.
pub fn format(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if !x.is_finite() || (PLAIN_FROM..PLAIN_BELOW).contains(&x.abs()) {
        format!("{x}")
    } else {
        format!("{x:e}")
    }
}

/// As [`format()`], rounded to four significant figures first: for read-outs
/// of computed values, where every digit would be noise.
pub fn rounded(x: f64) -> String {
    format(format!("{x:.3e}").parse().unwrap_or(x))
}

/// A plot tick at `value` on a grid of spacing `step`.
///
/// Grid arithmetic leaves the tick at zero a rounding error away from it, so
/// anything within a millionth of a step of zero is zero.
pub fn tick(value: f64, step: f64) -> String {
    if value.abs() < step.abs() * 1e-6 {
        return "0".into();
    }
    rounded(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_numbers_are_written_out_and_others_are_scientific() {
        for (x, want) in [
            (0.0, "0"),
            (-0.0, "0"),
            (0.001, "0.001"),
            (0.25, "0.25"),
            (-3000.0, "-3000"),
            (17000.0, "17000"),
            (99999.0, "99999"),
            (1e5, "1e5"),
            (2e6, "2e6"),
            (2.5e9, "2.5e9"),
            (0.000999, "9.99e-4"),
            (3e-6, "3e-6"),
            (-1.5e-5, "-1.5e-5"),
            (1e-9, "1e-9"),
        ] {
            assert_eq!(format(x), want, "{x}");
        }
    }

    /// What is shown is what is stored: reading it back gives the same bits.
    #[test]
    fn the_written_form_reads_back_exactly() {
        for x in [
            3e-6,
            1.0 / 3.0,
            17000.0,
            123456.789,
            6.02214076e23,
            -2.5e-12,
        ] {
            assert_eq!(format(x).parse::<f64>().unwrap().to_bits(), x.to_bits());
        }
        // And scientific entry is ordinary entry.
        assert_eq!("2e6".parse::<f64>().unwrap(), 2_000_000.0);
        assert_eq!("2E6".parse::<f64>().unwrap(), 2_000_000.0);
    }

    #[test]
    fn read_outs_and_ticks_carry_four_significant_figures() {
        assert_eq!(rounded(1.0 / 3.0), "0.3333");
        // Grid arithmetic's kind of noise: shown in full it is a long number.
        let noisy = (0.1 + 0.2) * 1e-4;
        assert_ne!(format(noisy), "3e-5");
        assert_eq!(rounded(noisy), "3e-5");
        assert_eq!(rounded(123456.789), "1.235e5");
        assert_eq!(tick(noisy, 1e-5), "3e-5");
        assert_eq!(tick(1.3e-21, 1e-5), "0");
        assert_eq!(tick(-20000.000000000004, 5000.0), "-20000");
    }
}
