//! ESCALADE with and without the analytic Hessian.
//!
//! Rust port of `test_runs/test_escalade_grad_vs_grad_hess.m`: z to -y across
//! 51 spins spread over 20 kHz, with a 17 kHz field and 100 microseconds of
//! pulse in 50 points, optimised twice from the same random start - once by a
//! Newton trust region on the Hessian, once by L-BFGS on the gradient alone.

use nalgebra::DVector;
use qoala::error::Result;
use qoala::escalade::{escalade, profile, Escalade, Optimised};
use qoala::examples_io::write_waveform_csv;

fn main() -> Result<()> {
    let np_pulse = 50;
    let base = Escalade {
        nspins: 51,
        np_pulse,
        tau_p: 100e-6,
        rf: vec![17000.0],
        sw: 20000.0,
        max_iter: 1000,
        target_fidelity: 0.99,
        seed: Some(1),
        ..Default::default()
    };
    // The script draws f0 and g0 once and starts both runs from them.
    let start = base.resolve()?.start;
    let with = |use_hessian| Escalade {
        use_hessian,
        start: Some(start.clone()),
        ..base.clone()
    };

    println!("\n --- HESSIANS ------");
    let hess = escalade(&with(true))?;
    report(&base, &hess);

    println!("\n --- GRADIENTS ONLY ------");
    let grad = escalade(&with(false))?;
    report(&base, &grad);

    let dt = DVector::from_element(np_pulse, base.tau_p / np_pulse as f64);
    let amps = [2.0 * std::f64::consts::PI * base.rf[0]; 2];
    for (name, out) in [("hessian", &hess), ("gradient", &grad)] {
        let path = format!("escalade_{name}_pulse.csv");
        write_waveform_csv(&path, &out.pulse, &dt, &amps, &["x", "y"])?;
        println!("waveform written to {path}");
    }
    Ok(())
}

fn report(spec: &Escalade, out: &Optimised) {
    // Independent check: the mean of -Iy across the band is the fidelity.
    let n = spec.nspins;
    let measured = (0..n)
        .map(|i| {
            let offset = -spec.sw / 2.0 + spec.sw * i as f64 / (n - 1) as f64;
            -profile::final_magnetisation(&out.pulse, spec.tau_p, spec.rf[0], offset).y
        })
        .sum::<f64>()
        / n as f64;
    println!(
        " finished in {:.2} seconds: {} iterations, {} function, {} gradient and {} Hessian evaluations",
        out.elapsed_s, out.counters.iter, out.counters.fx, out.counters.gfx, out.counters.hfx
    );
    println!(
        " fidelity {:.9}, checked from the magnetisation {:.9}",
        out.fidelity, measured
    );
    println!(
        " largest amplitude {:.4} of the nominal field; stopped because {}",
        out.max_amplitude, out.exitflag
    );
}
