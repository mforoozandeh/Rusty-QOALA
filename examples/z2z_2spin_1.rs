//! Z-to-Z magnetisation transfer across a weak (J-coupled) pair of spins.
//!
//! Rust port of `examples/z2z_2spin_1.m`.  Two two-level systems coupled only
//! through `zz` terms at 140 Hz, each with its own x and y control, 10 ms of
//! pulse in 50 slices.

use nalgebra::DVector;
use num_complex::Complex64;
use qoala::drivers::{state2state_xy, StateTransfer, Tuning};
use qoala::error::Result;
use qoala::linalg::CMat;
use qoala::objfun::FidelityKind;
use qoala::spinops;
use qoala::types::{PropMethod, StateSpace};
use qoala::waveform::check_fidelity;

fn main() -> Result<()> {
    let nspins = 2;
    let duration = 0.01;
    let nsteps = 50;
    let amplitudes = vec![1000.0, 1000.0];

    // Weak coupling: only zz terms, 140 Hz between the two spins.
    let interaction = spinops::zz_coupling(nspins, 0, 1)?
        .scale(Complex64::new(2.0 * std::f64::consts::PI * 140.0, 0.0));

    // One x/y control pair per spin.
    let cartops = spinops::one_pair_per_spin(nspins);

    // Transfer z-magnetisation from spin 1 to spin 2.
    let initial = spinops::z_state(nspins, 0);
    let target = spinops::z_state(nspins, 1);

    let optimised = state2state_xy(StateTransfer {
        omega: vec![0.0, 0.0],
        initial: initial.clone(),
        target: target.clone(),
        amplitudes: amplitudes.clone(),
        duration,
        increments: nsteps,
        interaction: interaction.clone(),
        cartops,
        init_pulse: None,
        seed: Some(1),
        tuning: Tuning::default(),
    })?;

    // Independent check by exact propagation, as the script does.
    let ops = spinops::cartesian_operators(nspins);
    let controls: Vec<CMat> = vec![
        ops[0].0.clone(),
        ops[0].1.clone(),
        ops[1].0.clone(),
        ops[1].1.clone(),
    ];
    let two_pi = 2.0 * std::f64::consts::PI;
    let amps: Vec<f64> = amplitudes
        .iter()
        .flat_map(|a| [two_pi * a, two_pi * a])
        .collect();
    let dt = DVector::from_element(nsteps, duration / nsteps as f64);

    let fidelity = check_fidelity(
        StateSpace::Liouville,
        FidelityKind::State,
        16,
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
        "z2z_2spin_1_pulse.csv",
        &optimised.waveform,
        &dt,
        &amps,
        &["x-control 1", "y-control 1", "x-control 2", "y-control 2"],
    )?;
    println!("waveform written to z2z_2spin_1_pulse.csv");
    Ok(())
}

fn normalise(x: &qoala::linalg::CDense) -> qoala::linalg::CDense {
    let n = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    x.map(|z| z / Complex64::new(n, 0.0))
}
