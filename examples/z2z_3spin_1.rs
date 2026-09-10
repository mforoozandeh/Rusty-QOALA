//! Z-to-Z magnetisation transfer along a three-spin chain.
//!
//! Rust port of `examples/z2z_3spin_1.m`.  Three two-level systems in a linear
//! chain, weakly coupled at 140 Hz and -160 Hz, each with its own x and y
//! control; 22 ms in 110 slices, transferring z-magnetisation from the first
//! spin to the third.

use nalgebra::DVector;
use num_complex::Complex64;
use qoala::drivers::{state2state_xy, StateTransfer, Tuning};
use qoala::error::Result;
use qoala::linalg::{CDense, CMat};
use qoala::objfun::FidelityKind;
use qoala::spinops;
use qoala::types::{PropMethod, StateSpace};
use qoala::waveform::check_fidelity;

fn main() -> Result<()> {
    let nspins = 3;
    let duration = 0.022;
    let nsteps = 110;
    let amplitudes = vec![1000.0, 1000.0, 1000.0];
    let two_pi = 2.0 * std::f64::consts::PI;

    // Linear chain: 140 Hz between spins 1-2, -160 Hz between spins 2-3.
    let interaction = spinops::zz_coupling(nspins, 0, 1)?
        .scale(Complex64::new(two_pi * 140.0, 0.0))
        .add(&spinops::zz_coupling(nspins, 1, 2)?.scale(Complex64::new(two_pi * -160.0, 0.0)))?;

    let cartops = spinops::one_pair_per_spin(nspins);
    let initial = spinops::z_state(nspins, 0);
    let target = spinops::z_state(nspins, 2);

    let optimised = state2state_xy(StateTransfer {
        omega: vec![0.0; nspins],
        initial: initial.clone(),
        target: target.clone(),
        amplitudes: amplitudes.clone(),
        duration,
        increments: nsteps,
        interaction: interaction.clone(),
        cartops,
        init_pulse: None,
        seed: Some(3),
        tuning: Tuning::default(),
    })?;

    let ops = spinops::cartesian_operators(nspins);
    let controls: Vec<CMat> = ops
        .iter()
        .flat_map(|(x, y, _)| [x.clone(), y.clone()])
        .collect();
    let amps: Vec<f64> = amplitudes
        .iter()
        .flat_map(|a| [two_pi * a, two_pi * a])
        .collect();
    let dt = DVector::from_element(nsteps, duration / nsteps as f64);

    let fidelity = check_fidelity(
        StateSpace::Liouville,
        FidelityKind::State,
        64,
        &interaction,
        &controls,
        &amps,
        &optimised.waveform,
        &dt,
        PropMethod::Krylov,
        &normalise(&initial),
        &normalise(&target),
    )?;

    println!("... calculating the fidelity %");
    println!("fidelity calculated to be {:.8}%", 100.0 * fidelity);

    qoala::examples_io::write_waveform_csv(
        "z2z_3spin_1_pulse.csv",
        &optimised.waveform,
        &dt,
        &amps,
        &[
            "x-control 1",
            "y-control 1",
            "x-control 2",
            "y-control 2",
            "x-control 3",
            "y-control 3",
        ],
    )?;
    println!("waveform written to z2z_3spin_1_pulse.csv");
    Ok(())
}

fn normalise(x: &CDense) -> CDense {
    let n = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    x.map(|z| z / Complex64::new(n, 0.0))
}
