//! Console reporting.
//!
//! The MATLAB package prints a parser banner, a long list of resolved settings
//! and a live iteration table, all with `fprintf(output, ...)` where `output`
//! is either `1` (stdout) or a file id.  This module provides the same sink
//! plus the number-formatting helpers needed to reproduce the layout:
//! MATLAB's `pad`, `num2str` with `%g`/`%e`/`%f` formats, and the SI-prefix
//! helper `findprefix`.

use std::cell::RefCell;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::rc::Rc;

/// Where report lines go.
#[derive(Clone)]
pub enum ReportSink {
    /// Standard output, MATLAB's file id 1.
    Stdout,
    /// An open file, appended to.
    File(Rc<RefCell<File>>),
    /// Discard everything; useful in tests and library use.
    Silent,
}

impl std::fmt::Debug for ReportSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportSink::Stdout => f.write_str("Stdout"),
            ReportSink::File(_) => f.write_str("File"),
            ReportSink::Silent => f.write_str("Silent"),
        }
    }
}

/// A report destination.
#[derive(Clone, Debug)]
pub struct Reporter {
    sink: ReportSink,
}

impl Default for Reporter {
    fn default() -> Self {
        Reporter {
            sink: ReportSink::Stdout,
        }
    }
}

impl Reporter {
    /// Report to standard output.
    pub fn stdout() -> Self {
        Reporter {
            sink: ReportSink::Stdout,
        }
    }

    /// Report nowhere.
    pub fn silent() -> Self {
        Reporter {
            sink: ReportSink::Silent,
        }
    }

    /// Append the report to a file.
    pub fn file(path: &Path) -> std::io::Result<Self> {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Reporter {
            sink: ReportSink::File(Rc::new(RefCell::new(f))),
        })
    }

    /// Write one line, ignoring write failures exactly as the MATLAB
    /// `try ... end` around `fprintf` does.
    pub fn line(&self, s: &str) {
        match &self.sink {
            ReportSink::Stdout => println!("{s}"),
            ReportSink::File(f) => {
                let _ = writeln!(f.borrow_mut(), "{s}");
            }
            ReportSink::Silent => {}
        }
    }

    /// A blank line.
    pub fn blank(&self) {
        self.line("");
    }

    /// A `label ... value` settings line: the label padded to 60 columns.
    pub fn field(&self, label: &str, value: &str) {
        self.line(&format!("{}{}", pad(label, 60), value));
    }

    /// A settings line whose value is padded to 20 columns, as several of the
    /// MATLAB lines are.
    pub fn field_padded(&self, label: &str, value: &str) {
        self.line(&format!("{}{}", pad(label, 60), pad(value, 20)));
    }

    /// A `WARNING` line followed by its `RESOLVE` line.
    pub fn warn_resolve(&self, warning: &str, resolution: &str) {
        self.line(&format!("{}{}", pad(warning, 60), "WARNING"));
        self.line(&format!(
            "{}{}",
            pad(&format!("... {resolution}"), 60),
            "RESOLVE"
        ));
    }

    /// A standalone `WARNING` line.
    pub fn warn(&self, warning: &str) {
        self.line(&format!("{}{}", pad(warning, 60), "WARNING"));
    }

    /// A value with an SI prefix and a unit, e.g. `140.000 Hz`.
    pub fn field_si(&self, label: &str, value: f64, unit: &str) {
        let (v, p) = find_prefix(value, 1000.0);
        self.field(label, &format!("{} {}{}", pad(&fmt_f(v, 3), 7), p, unit));
    }

    /// The QOALA banner.
    pub fn banner(&self) {
        self.line("===========================================");
        self.line("=                                         =");
        self.line("=               QOALA v1.0                =");
        self.line("=                                         =");
        self.line("=   M.Foroozandeh, D.L.Goodwin, P.Singh   =");
        self.line("=                                         =");
        self.line("===========================================");
        self.blank();
    }
}

/// MATLAB's `pad`: right-pad with spaces to `n` columns, never truncating.
pub fn pad(s: &str, n: usize) -> String {
    if s.chars().count() >= n {
        s.to_string()
    } else {
        let mut out = String::from(s);
        out.extend(std::iter::repeat_n(' ', n - s.chars().count()));
        out
    }
}

/// Left-pad with spaces to `n` columns.
pub fn pad_left(s: &str, n: usize) -> String {
    if s.chars().count() >= n {
        s.to_string()
    } else {
        let mut out = String::new();
        out.extend(std::iter::repeat_n(' ', n - s.chars().count()));
        out.push_str(s);
        out
    }
}

/// Fixed-point formatting, MATLAB `%.<prec>f`.
pub fn fmt_f(v: f64, prec: usize) -> String {
    format!("{v:.prec$}")
}

/// Fixed-point formatting with a forced sign, MATLAB `%+.<prec>f`.
pub fn fmt_f_signed(v: f64, prec: usize) -> String {
    format!("{v:+.prec$}")
}

/// C-style `%e`, which Rust's `{:e}` does not produce: two-digit exponent with
/// an explicit sign.
pub fn fmt_e(v: f64, prec: usize) -> String {
    if !v.is_finite() {
        return fmt_nonfinite(v);
    }
    if v == 0.0 {
        return format!("{:.*}e+00", prec, 0.0);
    }
    let exp = v.abs().log10().floor() as i32;
    let mantissa = v / 10f64.powi(exp);
    // Rounding the mantissa can push it to 10.0; renormalise if so.
    let (mantissa, exp) = {
        let rounded = format!("{mantissa:.prec$}")
            .parse::<f64>()
            .unwrap_or(mantissa);
        if rounded.abs() >= 10.0 {
            (rounded / 10.0, exp + 1)
        } else {
            (rounded, exp)
        }
    };
    format!(
        "{:.*}e{}{:02}",
        prec,
        mantissa,
        if exp < 0 { '-' } else { '+' },
        exp.abs()
    )
}

/// C-style `%g` with `sig` significant digits, which is what MATLAB's
/// `num2str(x,'%.9g')` produces.
pub fn fmt_g(v: f64, sig: usize) -> String {
    if !v.is_finite() {
        return fmt_nonfinite(v);
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let sig = sig.max(1);
    let exp = v.abs().log10().floor() as i32;
    if exp < -4 || exp >= sig as i32 {
        let s = fmt_e(v, sig - 1);
        // Strip trailing zeros from the mantissa, as %g does.
        if let Some(epos) = s.find('e') {
            let (mant, tail) = s.split_at(epos);
            let mant = if mant.contains('.') {
                mant.trim_end_matches('0').trim_end_matches('.')
            } else {
                mant
            };
            format!("{mant}{tail}")
        } else {
            s
        }
    } else {
        let decimals = (sig as i32 - 1 - exp).max(0) as usize;
        let s = format!("{v:.decimals$}");
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}

/// `%g` with a forced sign.
pub fn fmt_g_signed(v: f64, sig: usize) -> String {
    let s = fmt_g(v, sig);
    if s.starts_with('-') || s.starts_with('+') {
        s
    } else {
        format!("+{s}")
    }
}

fn fmt_nonfinite(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v > 0.0 {
        "Inf".to_string()
    } else {
        "-Inf".to_string()
    }
}

/// MATLAB's `int2str`.
pub fn int2str(v: impl Into<i64>) -> String {
    format!("{}", v.into())
}

/// Split a value into a mantissa and an SI prefix.
///
/// Port of the `findprefix` subfunction; `base` is 1000 for physical units and
/// 1024 for byte counts.
pub fn find_prefix(value: f64, base: f64) -> (f64, &'static str) {
    if !value.is_finite() {
        return (value, "");
    }
    const PREFIXES: [(&str, i32); 17] = [
        ("Y", 8),
        ("Z", 7),
        ("E", 6),
        ("P", 5),
        ("T", 4),
        ("G", 3),
        ("M", 2),
        ("k", 1),
        ("", 0),
        ("m", -1),
        ("u", -2),
        ("n", -3),
        ("p", -4),
        ("f", -5),
        ("a", -6),
        ("z", -7),
        ("y", -8),
    ];
    let v = value.abs();
    if v == 0.0 {
        return (value, "");
    }
    // Largest prefix whose scale the value still exceeds, exactly as the
    // MATLAB cascade of comparisons does.
    for (name, power) in PREFIXES {
        let scale = base.powi(power);
        if v >= scale {
            return (value / scale, name);
        }
    }
    (value / base.powi(-8), "y")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_matches_matlab() {
        assert_eq!(pad("abc", 6), "abc   ");
        assert_eq!(pad("abcdefg", 3), "abcdefg");
        assert_eq!(pad_left("7", 4), "   7");
    }

    #[test]
    fn scientific_format_uses_two_digit_exponents() {
        assert_eq!(fmt_e(1.0, 2), "1.00e+00");
        assert_eq!(fmt_e(0.001234, 2), "1.23e-03");
        assert_eq!(fmt_e(-98765.0, 4), "-9.8765e+04");
        assert_eq!(fmt_e(9.999, 2), "1.00e+01");
    }

    #[test]
    fn general_format_matches_c_percent_g() {
        assert_eq!(fmt_g(0.0, 9), "0");
        assert_eq!(fmt_g(1.0, 9), "1");
        assert_eq!(fmt_g(0.15, 8), "0.15");
        assert_eq!(fmt_g(1e-12, 8), "1e-12");
        assert_eq!(fmt_g(1234567890.0, 9), "1.23456789e+09");
        assert_eq!(fmt_g(1e4, 9), "10000");
        assert_eq!(fmt_g_signed(0.5, 9), "+0.5");
        assert_eq!(fmt_g(f64::INFINITY, 9), "Inf");
    }

    #[test]
    fn si_prefixes_pick_sensible_scales() {
        assert_eq!(find_prefix(140.0, 1000.0).1, "");
        assert_eq!(find_prefix(1400.0, 1000.0).1, "k");
        assert_eq!(find_prefix(0.012, 1000.0).1, "m");
        assert_eq!(find_prefix(2.0e-4, 1000.0).1, "u");
        let (v, p) = find_prefix(1_500_000.0, 1000.0);
        assert_eq!(p, "M");
        assert!((v - 1.5).abs() < 1e-12);
    }
}
