//! End-to-end optimisation and configuration behaviour.

mod common;

use common::Fixture;
use nalgebra::DVector;
use num_complex::Complex64;
use qoala::config::{optimconset, ControlOptions, DriftOptions, PowerLevels, TimeDependent};
use qoala::drivers::{random_pulse, state2state_xy, GrapeXyCost, StateTransfer, Tuning};
use qoala::objfun::FidelityKind;
use qoala::optim::newton::fmaxnewton;
use qoala::report::Reporter;
use qoala::spinops;
use qoala::types::*;
use qoala::waveform::check_fidelity;

fn silent_tuning(max_iter: usize) -> Tuning {
    Tuning {
        max_iter,
        output: Reporter::silent(),
        ..Default::default()
    }
}

#[test]
fn two_spin_transfer_converges_and_survives_an_exact_check() {
    let nspins = 2;
    let duration = 0.01;
    let nsteps = 50;
    let two_pi = 2.0 * std::f64::consts::PI;

    let interaction = spinops::zz_coupling(nspins, 0, 1)
        .unwrap()
        .scale(Complex64::new(two_pi * 140.0, 0.0));
    let initial = spinops::z_state(nspins, 0);
    let target = spinops::z_state(nspins, 1);

    let optimised = state2state_xy(StateTransfer {
        omega: vec![0.0, 0.0],
        initial: initial.clone(),
        target: target.clone(),
        amplitudes: vec![1000.0, 1000.0],
        duration,
        increments: nsteps,
        interaction: interaction.clone(),
        cartops: spinops::one_pair_per_spin(nspins),
        init_pulse: None,
        seed: Some(1),
        tuning: silent_tuning(60),
    })
    .expect("optimisation");

    // The optimiser's own view.
    let reported = optimised
        .data
        .fidelity_history()
        .into_iter()
        .filter(|v| !v.is_nan())
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        reported > 0.999,
        "the optimiser only reached a fidelity of {reported:.6}"
    );

    // An independent exact propagation of the pulse it produced.
    let ops = spinops::cartesian_operators(nspins);
    let controls = vec![
        ops[0].0.clone(),
        ops[0].1.clone(),
        ops[1].0.clone(),
        ops[1].1.clone(),
    ];
    let amps = vec![two_pi * 1000.0; 4];
    let dt = DVector::from_element(nsteps, duration / nsteps as f64);
    let normalise = |x: &qoala::linalg::CDense| {
        let n = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        x.map(|z| z / Complex64::new(n, 0.0))
    };
    let checked = check_fidelity(
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
    )
    .unwrap();

    assert!(
        checked > 0.999,
        "exact propagation of the optimised pulse gives only {checked:.6}"
    );
    assert!(
        (checked - reported).abs() < 5e-4,
        "the optimiser reported {reported:.8} but exact propagation gives {checked:.8}"
    );
}

#[test]
fn adaptivity_climbs_the_ladder_as_the_fidelity_improves() {
    // The ladder is only climbed when the tolerance tightens, and the
    // tolerance is scaled by the infidelity - so the optimisation has to get
    // somewhere before a finer splitting is called for.
    let fixture = Fixture {
        nsteps: 50,
        duration: 0.01,
        splitset: Some(vec![2, 3, 4]),
        split_order: 2,
        ..Default::default()
    };
    let mut sys = fixture.build().unwrap();
    sys.max_iter = 60;
    assert!(sys.drift_sys[0].adapt, "the ladder should be active");
    assert_eq!(
        sys.drift_sys[0].split_order, 2,
        "it should start at the bottom"
    );
    assert_eq!(sys.drift_sys[0].adaptset.len(), 3);

    let guess = random_pulse(sys.pulse_nsteps, sys.operators.len(), Some(11));
    let run = fmaxnewton(&mut sys, &GrapeXyCost, &guess).unwrap();

    assert!(
        sys.drift_sys[0].split_order > 2,
        "the splitting order never rose above its starting value"
    );
    let best = run
        .data
        .fidelity_history()
        .into_iter()
        .filter(|v| !v.is_nan())
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(best > 0.99, "adaptive run only reached {best:.6}");
}

#[test]
fn adaptive_and_fixed_high_order_reach_a_similar_fidelity() {
    let run_with = |splitset: Option<Vec<usize>>, order: usize| {
        let fixture = Fixture {
            nsteps: 50,
            duration: 0.01,
            splitset,
            split_order: order,
            ..Default::default()
        };
        let mut sys = fixture.build().unwrap();
        sys.max_iter = 80;
        let guess = random_pulse(sys.pulse_nsteps, sys.operators.len(), Some(21));
        let run = fmaxnewton(&mut sys, &GrapeXyCost, &guess).unwrap();
        run.data
            .fidelity_history()
            .into_iter()
            .filter(|v| !v.is_nan())
            .fold(f64::NEG_INFINITY, f64::max)
    };

    let adaptive = run_with(Some(vec![2, 3, 4]), 2);
    let fixed = run_with(None, 4);
    assert!(
        adaptive > 0.99 && fixed > 0.99,
        "adaptive reached {adaptive:.6} and a fixed fourth-order splitting reached {fixed:.6}"
    );
    assert!(
        (adaptive - fixed).abs() < 0.01,
        "adaptive reached {adaptive:.6} but fixed fourth order reached {fixed:.6}"
    );
}

#[test]
fn counters_and_histories_are_recorded() {
    let fixture = Fixture {
        nsteps: 10,
        ..Default::default()
    };
    let mut sys = fixture.build().unwrap();
    sys.max_iter = 5;
    let guess = random_pulse(sys.pulse_nsteps, sys.operators.len(), Some(31));
    let run = fmaxnewton(&mut sys, &GrapeXyCost, &guess).unwrap();

    assert!(run.data.count.iter >= 1);
    assert!(run.data.count.fx >= run.data.count.gfx);
    assert_eq!(run.data.count.hfx, 0, "LBFGS should not ask for Hessians");
    assert_eq!(run.data.x_shape, (10, 4));
    assert_eq!(run.waveform.shape(), (10, 4));

    // The wall-clock column must be non-decreasing where it is populated.
    let wall: Vec<f64> = run
        .data
        .wall_clock_history()
        .into_iter()
        .filter(|v| !v.is_nan())
        .collect();
    for pair in wall.windows(2) {
        assert!(pair[1] >= pair[0], "wall clock went backwards: {wall:?}");
    }
}

#[test]
fn a_zero_iteration_run_just_evaluates_once() {
    let fixture = Fixture {
        nsteps: 8,
        ..Default::default()
    };
    let mut sys = fixture.build().unwrap();
    sys.max_iter = 0;
    let guess = random_pulse(sys.pulse_nsteps, sys.operators.len(), Some(41));
    let run = fmaxnewton(&mut sys, &GrapeXyCost, &guess).unwrap();

    assert_eq!(run.data.count.iter, 0);
    assert_eq!(run.data.count.fx, 1);
    assert_eq!(run.data.count.gfx, 0);
    assert_eq!(run.waveform, guess, "the waveform should be untouched");
    assert!(!run.data.fidelity_history()[0].is_nan());
}

#[test]
fn missing_configuration_is_reported_rather_than_guessed() {
    // No basis.
    let opts = ControlOptions {
        output: Some(Reporter::silent()),
        ..Default::default()
    };
    assert!(optimconset(opts).is_err());

    // Basis but no targets.
    let opts = ControlOptions {
        basis: Some(Basis::Sphten),
        nspins: Some(1),
        optimcon_fun: Some(ObjectiveFn::StateAuxmat),
        output: Some(Reporter::silent()),
        ..Default::default()
    };
    assert!(optimconset(opts).is_err());
}

#[test]
fn spherical_tensors_force_liouville_space() {
    let nspins = 2;
    let two_pi = 2.0 * std::f64::consts::PI;
    let ops = spinops::cartesian_operators(nspins);
    let drift = DriftOptions {
        drift: Some(TimeDependent::Constant(
            spinops::zz_coupling(nspins, 0, 1)
                .unwrap()
                .scale(Complex64::new(two_pi * 140.0, 0.0)),
        )),
        ..Default::default()
    };
    let sys = optimconset(ControlOptions {
        basis: Some(Basis::Sphten),
        // Deliberately wrong: the parser must correct it.
        space: Some(StateSpace::Hilbert),
        nspins: Some(nspins),
        operators: Some(vec![ops[0].0.clone(), ops[0].1.clone()]),
        pwr_levels: Some(PowerLevels::Uniform(two_pi * 1000.0)),
        pulse_dur: Some(1e-3),
        pulse_nsteps: Some(4),
        optimcon_fun: Some(ObjectiveFn::StateAuxmat),
        rho_init: Some(vec![spinops::z_state(nspins, 0)]),
        rho_targ: Some(vec![spinops::z_state(nspins, 1)]),
        drift_sys: vec![drift],
        output: Some(Reporter::silent()),
        ..Default::default()
    })
    .expect("parse");
    assert_eq!(sys.space, StateSpace::Liouville);
}

#[test]
fn the_split_operator_objective_demands_a_spin_control_map() {
    let nspins = 2;
    let two_pi = 2.0 * std::f64::consts::PI;
    let ops = spinops::cartesian_operators(nspins);
    let drift = DriftOptions {
        interaction: Some(TimeDependent::Constant(
            spinops::zz_coupling(nspins, 0, 1)
                .unwrap()
                .scale(Complex64::new(two_pi * 140.0, 0.0)),
        )),
        offset: Some(DVector::zeros(nspins)),
        ..Default::default()
    };
    let err = optimconset(ControlOptions {
        basis: Some(Basis::Sphten),
        nspins: Some(nspins),
        operators: Some(vec![ops[0].0.clone(), ops[0].1.clone()]),
        pwr_levels: Some(PowerLevels::Uniform(two_pi * 1000.0)),
        pulse_dur: Some(1e-3),
        pulse_nsteps: Some(4),
        optimcon_fun: Some(ObjectiveFn::StateQoala),
        rho_init: Some(vec![spinops::z_state(nspins, 0)]),
        rho_targ: Some(vec![spinops::z_state(nspins, 1)]),
        drift_sys: vec![drift],
        output: Some(Reporter::silent()),
        ..Default::default()
    });
    assert!(err.is_err(), "spin_control should be required");
}

#[test]
fn penalty_weights_default_to_the_slice_count() {
    let nspins = 2;
    let two_pi = 2.0 * std::f64::consts::PI;
    let ops = spinops::cartesian_operators(nspins);
    let drift = DriftOptions {
        drift: Some(TimeDependent::Constant(
            spinops::zz_coupling(nspins, 0, 1)
                .unwrap()
                .scale(Complex64::new(two_pi * 140.0, 0.0)),
        )),
        ..Default::default()
    };
    let sys = optimconset(ControlOptions {
        basis: Some(Basis::Sphten),
        nspins: Some(nspins),
        operators: Some(vec![ops[0].0.clone(), ops[0].1.clone()]),
        pwr_levels: Some(PowerLevels::Uniform(two_pi * 1000.0)),
        pulse_dur: Some(1e-3),
        pulse_nsteps: Some(37),
        penalties: Some(vec![Penalty::Sns]),
        optimcon_fun: Some(ObjectiveFn::StateAuxmat),
        rho_init: Some(vec![spinops::z_state(nspins, 0)]),
        rho_targ: Some(vec![spinops::z_state(nspins, 1)]),
        drift_sys: vec![drift],
        output: Some(Reporter::silent()),
        ..Default::default()
    })
    .expect("parse");
    assert_eq!(sys.p_weights, vec![37.0]);
    // And the defaults the MATLAB documents.
    assert_eq!(sys.tol_x, 1e-3);
    assert_eq!(sys.tol_g, 1e-6);
    assert_eq!(sys.max_iter, 100);
    assert_eq!(sys.n_grads, 20);
    assert_eq!(sys.step_method, PropMethod::Krylov);
    assert_eq!(sys.auxmat_method, PropMethod::Taylor);
    assert_eq!(sys.sparsity, 0.15);
    assert_eq!(sys.prop_zeroed, 1e-12);
}
