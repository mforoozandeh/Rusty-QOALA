//! Magnetisation transfer across a strongly coupled pair of spins, driven by a
//! single control pair.
//!
//! Rust port of `examples/z2z_2spin_2.m`.  Two two-level systems coupled
//! through xx, yy and zz terms at 20 Hz, sitting at +2000 Hz and -1200 Hz, and
//! one x/y control pair that drives both of them at once - which exercises the
//! multi-spin branch of the Rodrigues derivative.

use nalgebra::DVector;
use num_complex::Complex64;
use qoala::drivers::{state2state_xy, StateTransfer, Tuning};
use qoala::error::Result;
use qoala::linalg::CDense;
use qoala::objfun::FidelityKind;
use qoala::spinops;
use qoala::types::{PropMethod, StateSpace};
use qoala::waveform::check_fidelity;

fn main() -> Result<()> {
    let nspins = 2;
    let duration = 0.01;
    let nsteps = 50;
    let omega = vec![2000.0, -1200.0];
    let amplitudes = vec![1000.0];

    // Strong (isotropic) coupling at 20 Hz: xx, yy and zz terms.
    let interaction = spinops::isotropic_coupling(nspins, 0, 1)?
        .scale(Complex64::new(2.0 * std::f64::consts::PI * 20.0, 0.0));

    // A single control pair driving both spins.
    let cartops = spinops::one_pair_all_spins(nspins);

    // Both states live on the same basis element, with opposite sign.
    let mut initial = CDense::zeros(16, 1);
    initial[(9, 0)] = Complex64::new(-0.5, 0.0);
    let mut target = CDense::zeros(16, 1);
    target[(9, 0)] = Complex64::new(0.5, 0.0);

    let optimised = state2state_xy(StateTransfer {
        omega: omega.clone(),
        initial: initial.clone(),
        target: target.clone(),
        amplitudes: amplitudes.clone(),
        duration,
        increments: nsteps,
        interaction: interaction.clone(),
        cartops,
        init_pulse: None,
        seed: Some(2),
        tuning: Tuning::default(),
    })?;

    // Exact check: the drift now carries the two resonance offsets as well.
    let two_pi = 2.0 * std::f64::consts::PI;
    let ops = spinops::cartesian_operators(nspins);
    let drift = interaction
        .add(&ops[0].2.scale(Complex64::new(two_pi * omega[0], 0.0)))?
        .add(&ops[1].2.scale(Complex64::new(two_pi * omega[1], 0.0)))?;
    let controls = vec![ops[0].0.add(&ops[1].0)?, ops[0].1.add(&ops[1].1)?];
    let amps = vec![two_pi * amplitudes[0], two_pi * amplitudes[0]];
    let dt = DVector::from_element(nsteps, duration / nsteps as f64);

    let fidelity = check_fidelity(
        StateSpace::Liouville,
        FidelityKind::State,
        16,
        &drift,
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
        "z2z_2spin_2_pulse.csv",
        &optimised.waveform,
        &dt,
        &amps,
        &["x-control", "y-control"],
    )?;
    println!("waveform written to z2z_2spin_2_pulse.csv");
    Ok(())
}

fn normalise(x: &CDense) -> CDense {
    let n = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    x.map(|z| z / Complex64::new(n, 0.0))
}
