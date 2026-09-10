//! SWAP gate between the two ends of a three-spin chain.
//!
//! Rust port of `examples/swap_3spin_1.m`.  Spins 1 and 3 are not directly
//! coupled - the gate has to go through spin 2 - so this needs a long pulse:
//! 26.4 ms in 264 slices.

use nalgebra::DVector;
use num_complex::Complex64;
use qoala::drivers::{universal_gate_xy, GateSynthesis, Tuning};
use qoala::error::Result;
use qoala::linalg::{CDense, CMat};
use qoala::objfun::FidelityKind;
use qoala::spinops;
use qoala::types::{PropMethod, StateSpace};
use qoala::waveform::check_fidelity;

fn main() -> Result<()> {
    let nspins = 3;
    let duration = 0.0264;
    let nsteps = 264;
    let amplitudes = vec![1000.0, 1000.0, 1000.0];
    let two_pi = 2.0 * std::f64::consts::PI;

    let interaction = spinops::zz_coupling(nspins, 0, 1)?
        .scale(Complex64::new(two_pi * 140.0, 0.0))
        .add(&spinops::zz_coupling(nspins, 1, 2)?.scale(Complex64::new(two_pi * -160.0, 0.0)))?;

    // SWAP between the two ends of the chain: spins 1 and 3 are not directly
    // coupled, so the pulse has to route the gate through spin 2.
    let target = spinops::swap_gate(nspins, 0, 2)?;

    let cartops = spinops::one_pair_per_spin(nspins);

    let optimised = universal_gate_xy(GateSynthesis {
        omega: vec![0.0; nspins],
        target: target.clone(),
        amplitudes: amplitudes.clone(),
        duration,
        increments: nsteps,
        interaction: interaction.clone(),
        cartops,
        init_pulse: None,
        seed: Some(5),
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
        FidelityKind::Gate,
        64,
        &interaction,
        &controls,
        &amps,
        &optimised.waveform,
        &dt,
        PropMethod::Krylov,
        &CDense::identity(64, 64),
        &target,
    )?;

    println!("... calculating the fidelity %");
    println!("fidelity calculated to be {:.8}%", 100.0 * fidelity);

    qoala::examples_io::write_waveform_csv(
        "swap_3spin_1_pulse.csv",
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
    println!("waveform written to swap_3spin_1_pulse.csv");
    Ok(())
}
