//! Small helpers the examples use to get results out of the process.
//!
//! The MATLAB scripts finish with `figure`, `stairs` and `exportgraphics`.
//! There is no equivalent here, so the examples write plain CSV instead: one
//! row per time slice, amplitudes in Hz, ready for whatever plotting tool the
//! reader prefers.

use crate::error::Result;
use nalgebra::{DMatrix, DVector};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// Write a waveform as CSV: a time column followed by one column per control
/// channel, scaled from the optimiser's dimensionless amplitudes back to Hz.
pub fn write_waveform_csv(
    path: impl AsRef<Path>,
    waveform: &DMatrix<f64>,
    dt: &DVector<f64>,
    amps_rad_per_s: &[f64],
    channel_names: &[&str],
) -> Result<()> {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut out = String::from("time_s");
    for (k, name) in channel_names.iter().enumerate() {
        let label = if name.is_empty() {
            format!("channel_{k}")
        } else {
            (*name).to_string()
        };
        let _ = write!(out, ",{label}");
    }
    out.push('\n');

    let mut t = 0.0;
    for n in 0..waveform.nrows() {
        let _ = write!(out, "{t:.9}");
        for k in 0..waveform.ncols() {
            let hz = waveform[(n, k)] * amps_rad_per_s.get(k).copied().unwrap_or(two_pi) / two_pi;
            let _ = write!(out, ",{hz:.9}");
        }
        out.push('\n');
        t += dt[n];
    }
    fs::write(path, out)?;
    Ok(())
}

/// Write a table of numbers as CSV with a header row.
pub fn write_table_csv(path: impl AsRef<Path>, header: &[&str], rows: &[Vec<f64>]) -> Result<()> {
    let mut out = header.join(",");
    out.push('\n');
    for row in rows {
        let cells: Vec<String> = row
            .iter()
            .map(|v| {
                if v.is_nan() {
                    String::from("NaN")
                } else {
                    crate::report::fmt_g(*v, 12)
                }
            })
            .collect();
        out.push_str(&cells.join(","));
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}
