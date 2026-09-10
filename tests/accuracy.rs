//! Accuracy of the split-operator objective against exact propagation.
//!
//! The point of QOALA is that a cheap, low-order splitting is good enough
//! early in an optimisation and can be refined on demand.  These tests check
//! that the refinement actually converges: raising the splitting order, the
//! Trotter number or the time resolution must drive both the fidelity and the
//! gradient towards the exact auxiliary-matrix answer.

mod common;

use common::{test_waveform, Fixture};
use qoala::drivers::GrapeXyCost;
use qoala::objfun::FidelityKind;
use qoala::optim::CostFunction;
use qoala::types::ObjectiveFn;
use qoala::waveform::check_fidelity;

/// Exact fidelity for a fixture's waveform, from the auxiliary-matrix method.
fn exact_fidelity(nsteps: usize, gate: bool) -> (nalgebra::DMatrix<f64>, f64) {
    let objective = if gate {
        ObjectiveFn::UgateAuxmat
    } else {
        ObjectiveFn::StateAuxmat
    };
    let sys = Fixture {
        nsteps,
        gate,
        objective,
        ..Default::default()
    }
    .build()
    .unwrap();
    let wf = test_waveform(nsteps, sys.operators.len());
    let f = GrapeXyCost.value(&wf, &sys, objective).unwrap().1[0];
    (wf, f)
}

#[test]
fn raising_the_splitting_order_improves_the_fidelity() {
    let nsteps = 6;
    let (wf, exact) = exact_fidelity(nsteps, false);
    let mut previous = f64::INFINITY;

    for order in [1usize, 2, 3, 4, 6] {
        let sys = Fixture {
            nsteps,
            split_order: order,
            ..Default::default()
        }
        .build()
        .unwrap();
        let approx = GrapeXyCost
            .value(&wf, &sys, ObjectiveFn::StateQoala)
            .unwrap()
            .1[0];
        let err = (approx - exact).abs();
        assert!(
            err < previous,
            "split order {order} was not more accurate than the previous order \
             ({err:.3e} vs {previous:.3e})"
        );
        previous = err;
    }
    // Six orders of magnitude gained over the first-order splitting, on a
    // deliberately coarse six-slice grid.
    assert!(
        previous < 1e-5,
        "order-6 error {previous:.3e} is larger than expected"
    );
}

#[test]
fn raising_the_trotter_number_improves_the_fidelity() {
    let nsteps = 6;
    let (wf, exact) = exact_fidelity(nsteps, false);
    let mut previous = f64::INFINITY;

    for trotter in [1usize, 2, 4, 8] {
        let sys = Fixture {
            nsteps,
            split_order: 2,
            trotter,
            ..Default::default()
        }
        .build()
        .unwrap();
        let approx = GrapeXyCost
            .value(&wf, &sys, ObjectiveFn::StateQoala)
            .unwrap()
            .1[0];
        let err = (approx - exact).abs();
        assert!(
            err < previous,
            "Trotter number {trotter} was not more accurate ({err:.3e} vs {previous:.3e})"
        );
        previous = err;
    }
}

#[test]
fn refining_the_time_grid_converges_to_the_exact_fidelity() {
    let mut previous = f64::INFINITY;
    for nsteps in [6usize, 12, 24, 48] {
        let (wf, exact) = exact_fidelity(nsteps, false);
        let sys = Fixture {
            nsteps,
            split_order: 6,
            trotter: 2,
            ..Default::default()
        }
        .build()
        .unwrap();
        let approx = GrapeXyCost
            .value(&wf, &sys, ObjectiveFn::StateQoala)
            .unwrap()
            .1[0];
        let err = (approx - exact).abs();
        assert!(
            err < previous,
            "{nsteps} slices was not more accurate ({err:.3e} vs {previous:.3e})"
        );
        previous = err;
    }
    assert!(
        previous < 1e-10,
        "a well-resolved sixth-order splitting is still {previous:.3e} from exact"
    );
}

#[test]
fn split_and_exact_gradients_agree_when_well_resolved() {
    let nsteps = 48;
    let exact_sys = Fixture {
        nsteps,
        objective: ObjectiveFn::StateAuxmat,
        ..Default::default()
    }
    .build()
    .unwrap();
    let wf = test_waveform(nsteps, exact_sys.operators.len());
    let exact = GrapeXyCost
        .value_grad(&wf, &exact_sys, ObjectiveFn::StateAuxmat)
        .unwrap()
        .2[0]
        .clone();

    let sys = Fixture {
        nsteps,
        split_order: 6,
        trotter: 2,
        ..Default::default()
    }
    .build()
    .unwrap();
    let split = GrapeXyCost
        .value_grad(&wf, &sys, ObjectiveFn::StateQoala)
        .unwrap()
        .2[0]
        .clone();

    let relative = (&split - &exact).amax() / exact.amax();
    assert!(
        relative < 1e-7,
        "sixth-order split gradient differs from the exact one by {relative:.3e} relative"
    );
}

#[test]
fn gate_splitting_converges_to_the_exact_fidelity() {
    let nsteps = 48;
    let (wf, exact) = exact_fidelity(nsteps, true);
    let sys = Fixture {
        nsteps,
        gate: true,
        objective: ObjectiveFn::UgateQoala,
        split_order: 6,
        trotter: 2,
        ..Default::default()
    }
    .build()
    .unwrap();
    let approx = GrapeXyCost
        .value(&wf, &sys, ObjectiveFn::UgateQoala)
        .unwrap()
        .1[0];
    assert!(
        (approx - exact).abs() < 1e-9,
        "gate fidelities differ: split {approx:.12} vs exact {exact:.12}"
    );
}

#[test]
fn the_independent_check_agrees_with_the_objective() {
    // waveform_fidelity is the reference the example scripts quote, so it must
    // agree with the objective function it is checking.
    let fixture = Fixture {
        objective: ObjectiveFn::StateAuxmat,
        ..Default::default()
    };
    let sys = fixture.build().unwrap();
    let wf = test_waveform(sys.pulse_nsteps, sys.operators.len());
    let from_objective = GrapeXyCost
        .value(&wf, &sys, ObjectiveFn::StateAuxmat)
        .unwrap()
        .1[0];

    let drift = fixture.drift().unwrap();
    let from_check = check_fidelity(
        sys.space,
        FidelityKind::State,
        sys.dim,
        &drift,
        &fixture.controls(),
        &sys.power_row(0),
        &wf,
        &sys.pulse_dt,
        sys.step_method,
        &sys.initials[0],
        &sys.targets[0],
    )
    .unwrap();

    assert!(
        (from_objective - from_check).abs() < 1e-9,
        "objective {from_objective:.12} vs independent check {from_check:.12}"
    );
}

#[test]
fn propagation_methods_agree_with_each_other() {
    // Krylov, Taylor and Pade must give the same fidelity for the same pulse.
    let fixture = Fixture {
        objective: ObjectiveFn::StateAuxmat,
        ..Default::default()
    };
    let sys = fixture.build().unwrap();
    let wf = test_waveform(sys.pulse_nsteps, sys.operators.len());
    let drift = fixture.drift().unwrap();

    let mut values = Vec::new();
    for method in [
        qoala::types::PropMethod::Krylov,
        qoala::types::PropMethod::Taylor,
        qoala::types::PropMethod::Pade,
    ] {
        values.push(
            check_fidelity(
                sys.space,
                FidelityKind::State,
                sys.dim,
                &drift,
                &fixture.controls(),
                &sys.power_row(0),
                &wf,
                &sys.pulse_dt,
                method,
                &sys.initials[0],
                &sys.targets[0],
            )
            .unwrap(),
        );
    }
    for v in &values[1..] {
        assert!(
            (v - values[0]).abs() < 1e-10,
            "propagation methods disagree: {values:?}"
        );
    }
}
