//! Gradient verification.
//!
//! Every objective function's analytic gradient is checked against a central
//! finite difference of its own fidelity.  This is the strongest statement
//! available without a second implementation: the gradient must be the exact
//! derivative of the quantity the optimiser is actually climbing.

mod common;

use common::{test_waveform, Fixture};
use qoala::drivers::GrapeXyCost;
use qoala::optim::CostFunction;
use qoala::types::{ObjectiveFn, Penalty};

/// Central-difference gradient of the objective fidelity.
fn numeric_gradient(
    cost: &GrapeXyCost,
    sys: &qoala::config::ControlSystem,
    wf: &nalgebra::DMatrix<f64>,
    objfun: ObjectiveFn,
    h: f64,
) -> nalgebra::DMatrix<f64> {
    let mut g = nalgebra::DMatrix::zeros(wf.nrows(), wf.ncols());
    for c in 0..wf.ncols() {
        for r in 0..wf.nrows() {
            let mut wp = wf.clone();
            let mut wm = wf.clone();
            wp[(r, c)] += h;
            wm[(r, c)] -= h;
            let fp = cost.value(&wp, sys, objfun).unwrap().1[0];
            let fm = cost.value(&wm, sys, objfun).unwrap().1[0];
            g[(r, c)] = (fp - fm) / (2.0 * h);
        }
    }
    g
}

#[test]
fn qoala_state_gradient_matches_finite_differences() {
    for order in [0usize, 1, 2, 3, 4, 6] {
        for trotter in [1usize, 2] {
            let fixture = Fixture {
                split_order: order,
                trotter,
                ..Default::default()
            };
            let sys = fixture.build().expect("control system");
            let wf = test_waveform(sys.pulse_nsteps, sys.operators.len());
            let cost = GrapeXyCost;

            let analytic = cost
                .value_grad(&wf, &sys, ObjectiveFn::StateQoala)
                .expect("gradient")
                .2[0]
                .clone();
            let numeric = numeric_gradient(&cost, &sys, &wf, ObjectiveFn::StateQoala, 1e-6);

            let err = (&analytic - &numeric).amax();
            let scale = numeric.amax().max(1e-8);
            assert!(
                err / scale < 1e-5,
                "split order {order}, trotter {trotter}: relative gradient error {:.3e} \
                 (max analytic {:.3e}, max numeric {:.3e})",
                err / scale,
                analytic.amax(),
                numeric.amax()
            );
        }
    }
}

#[test]
fn qoala_gate_gradient_matches_finite_differences() {
    for order in [2usize, 3, 4] {
        let fixture = Fixture {
            split_order: order,
            gate: true,
            objective: ObjectiveFn::UgateQoala,
            ..Default::default()
        };
        let sys = fixture.build().expect("control system");
        let wf = test_waveform(sys.pulse_nsteps, sys.operators.len());
        let cost = GrapeXyCost;

        let analytic = cost
            .value_grad(&wf, &sys, ObjectiveFn::UgateQoala)
            .expect("gradient")
            .2[0]
            .clone();
        let numeric = numeric_gradient(&cost, &sys, &wf, ObjectiveFn::UgateQoala, 1e-6);

        let err = (&analytic - &numeric).amax();
        let scale = numeric.amax().max(1e-8);
        assert!(
            err / scale < 1e-5,
            "gate, split order {order}: relative gradient error {:.3e}",
            err / scale
        );
    }
}

#[test]
fn auxmat_state_gradient_matches_finite_differences() {
    let fixture = Fixture {
        objective: ObjectiveFn::StateAuxmat,
        ..Default::default()
    };
    let sys = fixture.build().expect("control system");
    let wf = test_waveform(sys.pulse_nsteps, sys.operators.len());
    let cost = GrapeXyCost;

    let analytic = cost
        .value_grad(&wf, &sys, ObjectiveFn::StateAuxmat)
        .expect("gradient")
        .2[0]
        .clone();
    let numeric = numeric_gradient(&cost, &sys, &wf, ObjectiveFn::StateAuxmat, 1e-6);

    let err = (&analytic - &numeric).amax();
    let scale = numeric.amax().max(1e-8);
    assert!(
        err / scale < 1e-5,
        "auxiliary matrix state gradient: relative error {:.3e}",
        err / scale
    );
}

#[test]
fn auxmat_gate_gradient_matches_finite_differences() {
    let fixture = Fixture {
        gate: true,
        objective: ObjectiveFn::UgateAuxmat,
        ..Default::default()
    };
    let sys = fixture.build().expect("control system");
    let wf = test_waveform(sys.pulse_nsteps, sys.operators.len());
    let cost = GrapeXyCost;

    let analytic = cost
        .value_grad(&wf, &sys, ObjectiveFn::UgateAuxmat)
        .expect("gradient")
        .2[0]
        .clone();
    let numeric = numeric_gradient(&cost, &sys, &wf, ObjectiveFn::UgateAuxmat, 1e-6);

    let err = (&analytic - &numeric).amax();
    let scale = numeric.amax().max(1e-8);
    assert!(
        err / scale < 1e-5,
        "auxiliary matrix gate gradient: relative error {:.3e}",
        err / scale
    );
}

#[test]
fn penalty_gradients_reach_the_optimiser_intact() {
    // With a penalty switched on, the combined gradient the optimiser sees
    // must be the derivative of the combined objective.
    let fixture = Fixture {
        penalties: vec![Penalty::Sns],
        ..Default::default()
    };
    let sys = fixture.build().expect("control system");
    // A waveform that actually breaches the bounds.
    let wf = test_waveform(sys.pulse_nsteps, sys.operators.len()).map(|v| v * 2.5);
    let cost = GrapeXyCost;

    let (_, values, grads) = cost
        .value_grad(&wf, &sys, ObjectiveFn::StateQoala)
        .expect("gradient");
    assert_eq!(values.len(), 2, "fidelity plus one penalty");
    let combined: nalgebra::DMatrix<f64> = &grads[0] - &grads[1];

    let h = 1e-6;
    let mut numeric = nalgebra::DMatrix::zeros(wf.nrows(), wf.ncols());
    for c in 0..wf.ncols() {
        for r in 0..wf.nrows() {
            let mut wp = wf.clone();
            let mut wm = wf.clone();
            wp[(r, c)] += h;
            wm[(r, c)] -= h;
            let vp = cost.value(&wp, &sys, ObjectiveFn::StateQoala).unwrap().1;
            let vm = cost.value(&wm, &sys, ObjectiveFn::StateQoala).unwrap().1;
            let fp = vp[0] - vp[1];
            let fm = vm[0] - vm[1];
            numeric[(r, c)] = (fp - fm) / (2.0 * h);
        }
    }
    let err = (&combined - &numeric).amax();
    assert!(
        err / numeric.amax().max(1e-8) < 1e-5,
        "combined objective gradient error {err:.3e}"
    );
}
