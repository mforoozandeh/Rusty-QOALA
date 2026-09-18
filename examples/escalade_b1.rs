//! A broadband pulse made robust to radio-frequency field error.
//!
//! Rust port of the optimisations in `test_runs/test_escalade_visual.m`: z to
//! -y across 51 spins over 20 kHz in 100 microseconds, optimised for a 17 kHz
//! field alone, then across 31 fields from 0.8 to 1.2 times it.
//!
//! The script's figures become CSV files for each pulse: the waveform, the
//! excitation profile, the map of Iy over offset and field, and the on-
//! resonance dphi/dB1 curve.  Its Bruker shape export and the rf-coefficient
//! optimisation of `ESCALADE_dphidB1_opt.m` are not ported.

use nalgebra::{DMatrix, DVector};
use qoala::error::Result;
use qoala::escalade::profile::Magnetisation;
use qoala::escalade::{escalade, profile, Escalade, Optimised};
use qoala::examples_io::{write_table_csv, write_waveform_csv};

/// The scripts start from +z and plot Iy.
const PLUS_Z: Magnetisation = Magnetisation::new(0.0, 0.0, 1.0);
const PLUS_Y: Magnetisation = Magnetisation::new(0.0, 1.0, 0.0);

fn main() -> Result<()> {
    let rf = 17000.0;
    let np_pulse = 50;
    let nspins = 51;

    // `f0 = eps*rand(np,1) - 0.5`: a flat start at -0.5 on both quadratures.
    let sensitive = Escalade {
        nspins,
        np_pulse,
        tau_p: 100e-6,
        rf: vec![rf],
        sw: 20000.0,
        use_hessian: true,
        max_iter: 1000,
        start: Some(DMatrix::from_element(np_pulse, 2, -0.5)),
        ..Default::default()
    };
    let compensated = Escalade {
        rf: (0..31)
            .map(|i| rf * (0.8 + 0.4 * i as f64 / 30.0))
            .collect(),
        ..sensitive.clone()
    };

    for (name, spec) in [
        ("b1_sensitive", &sensitive),
        ("b1_compensated", &compensated),
    ] {
        let out = escalade(spec)?;
        println!("\n --- {name} ------");
        println!(
            " fidelity {:.6} after {} iterations in {:.2} s; largest amplitude {:.4}; stopped because {}",
            out.fidelity, out.counters.iter, out.elapsed_s, out.max_amplitude, out.exitflag
        );
        let worst = (0..=8)
            .map(|i| 0.8 + 0.05 * i as f64)
            .map(|scale| (scale, band_fidelity(spec, &out, rf * scale)))
            .fold(
                (1.0, f64::INFINITY),
                |acc, x| if x.1 < acc.1 { x } else { acc },
            );
        println!(
            " worst fidelity from 0.8 to 1.2 times the field: {:.6} at {:.2}",
            worst.1, worst.0
        );
        write_outputs(name, spec, &out, rf)?;
    }
    Ok(())
}

/// Mean of -Iy across the band at `rf_hz`.
fn band_fidelity(spec: &Escalade, out: &Optimised, rf_hz: f64) -> f64 {
    let n = spec.nspins;
    (0..n)
        .map(|i| {
            let offset = -spec.sw / 2.0 + spec.sw * i as f64 / (n - 1) as f64;
            -profile::final_magnetisation(&out.pulse, spec.tau_p, rf_hz, offset, PLUS_Z).y
        })
        .sum::<f64>()
        / n as f64
}

fn write_outputs(name: &str, spec: &Escalade, out: &Optimised, rf: f64) -> Result<()> {
    let np = out.pulse.nrows();
    let dt = DVector::from_element(np, spec.tau_p / np as f64);
    let amps = [2.0 * std::f64::consts::PI * rf; 2];
    write_waveform_csv(
        format!("escalade_{name}_pulse.csv"),
        &out.pulse,
        &dt,
        &amps,
        &["x", "y"],
    )?;

    let rows: Vec<Vec<f64>> =
        profile::offset_profile(&out.pulse, spec.tau_p, rf, spec.sw, spec.nspins, PLUS_Z)
            .iter()
            .map(|p| {
                vec![
                    p.offset_hz,
                    p.m.x,
                    p.m.y,
                    p.m.z,
                    p.m.transverse(),
                    p.m.phase(),
                ]
            })
            .collect();
    write_table_csv(
        format!("escalade_{name}_profile.csv"),
        &["offset_hz", "ix", "iy", "iz", "ixy", "phase"],
        &rows,
    )?;

    let map = profile::b1_map(
        &out.pulse,
        spec.tau_p,
        rf,
        spec.sw,
        spec.nspins,
        PLUS_Z,
        PLUS_Y,
    );
    let mut rows = Vec::new();
    for (r, scale) in map.scales.iter().enumerate() {
        for (c, offset) in map.offsets_hz.iter().enumerate() {
            rows.push(vec![
                *scale,
                *offset,
                offset / map.rf_max_hz,
                map.values[(r, c)],
            ]);
        }
    }
    write_table_csv(
        format!("escalade_{name}_b1_map.csv"),
        &["b1_scale", "offset_hz", "offset_over_rf_max", "iy"],
        &rows,
    )?;

    let rows: Vec<Vec<f64>> =
        profile::phase_sensitivity(&out.pulse, spec.tau_p, rf, spec.nspins, PLUS_Z)
            .into_iter()
            .map(|(scale, dphi)| vec![scale, dphi])
            .collect();
    write_table_csv(
        format!("escalade_{name}_dphi_db1.csv"),
        &["b1_scale", "dphi_deg"],
        &rows,
    )?;

    println!(" written escalade_{name}_{{pulse,profile,b1_map,dphi_db1}}.csv");
    Ok(())
}
