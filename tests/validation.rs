//! Inputs the library must refuse with an error, rather than panic on or
//! quietly compute something else from.

mod common;

use common::{test_waveform, Fixture};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use qoala::config::{optimconset, ControlOptions, ControlSystem, PowerLevels, TimeDependent};
use qoala::drivers::{state2state_xy, GrapeXyCost, StateTransfer, Tuning};
use qoala::error::QoalaError;
use qoala::linalg::{CDense, CMat};
use qoala::objfun::TrajData;
use qoala::optim::newton::fmaxnewton;
use qoala::optim::{CostFunction, ValueGrad, ValueGradHess};
use qoala::report::Reporter;
use qoala::spinops;
use qoala::types::*;
use std::cell::Cell;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

#[test]
fn invalid_numbers_and_shapes_are_errors_not_panics() {
    type Spoil = fn(&mut ControlOptions);
    let cases: [(&str, Spoil); 16] = [
        ("a NaN reg_alpha", |o| o.reg_alpha = Some(f64::NAN)),
        ("a reg_phi of zero", |o| o.reg_phi = Some(0.0)),
        ("an infinite reg_max_cond", |o| {
            o.reg_max_cond = Some(f64::INFINITY)
        }),
        ("an empty time-dependent interaction", |o| {
            o.drift_sys[0].interaction = Some(TimeDependent::PerStep(Vec::new()))
        }),
        ("an interaction that is not finite", |o| {
            o.drift_sys[0].interaction = Some(TimeDependent::Constant(
                CMat::identity(16).scale(Complex64::new(f64::NAN, 0.0)),
            ))
        }),
        ("an initial state that is not finite", |o| {
            o.rho_init = Some(vec![CDense::from_element(
                16,
                1,
                Complex64::new(f64::NAN, 0.0),
            )])
        }),
        ("an empty pulse_dt", |o| {
            o.pulse_dt = Some(DVector::zeros(0))
        }),
        ("a NaN in pulse_dt", |o| {
            o.pulse_dt = Some(DVector::from_element(6, f64::NAN))
        }),
        ("a NaN pulse_dur", |o| o.pulse_dur = Some(f64::NAN)),
        ("a NaN power level", |o| {
            o.pwr_levels = Some(PowerLevels::Uniform(f64::NAN))
        }),
        ("an empty power level list", |o| {
            o.pwr_levels = Some(PowerLevels::Ensemble(vec![vec![]; 4]))
        }),
        ("a NaN in a power level list", |o| {
            o.pwr_levels = Some(PowerLevels::Ensemble(vec![vec![f64::NAN]; 4]))
        }),
        ("a target state of the wrong dimension", |o| {
            o.rho_targ = Some(vec![CDense::zeros(4, 1)])
        }),
        ("a spin_control map with too few rows", |o| {
            o.spin_control = Some(DMatrix::from_element(1, 4, true))
        }),
        ("an interaction of the wrong dimension", |o| {
            o.drift_sys[0].interaction = Some(TimeDependent::Constant(CMat::identity(4)))
        }),
        ("a control operator that is not finite", |o| {
            if let Some(ops) = &mut o.operators {
                ops[0] = ops[0].scale(Complex64::new(f64::INFINITY, 0.0));
            }
        }),
    ];
    for (what, spoil) in cases {
        let mut opts = Fixture::default().options().unwrap();
        spoil(&mut opts);
        assert!(optimconset(opts).is_err(), "{what} was accepted");
    }
}

/// The parser decides on the entries themselves, so a NaN cannot hide in a
/// matrix whatever its norms happen to report.
#[test]
fn matrices_holding_a_nan_are_refused_by_the_parser_itself() {
    fn nan_matrix() -> CMat {
        CMat::identity(16).scale(Complex64::new(f64::NAN, 0.0))
    }
    assert!(!nan_matrix().is_finite(), "the entry check misses the NaN");
    type Spoil = fn(&mut ControlOptions);
    let cases: [(&str, Spoil); 4] = [
        ("a control operator", |o| {
            let ops = o.operators.as_mut().expect("operators");
            ops[0] = ops[0].scale(Complex64::new(f64::NAN, 0.0));
        }),
        ("a drift", |o| {
            o.drift_sys[0].drift = Some(TimeDependent::Constant(nan_matrix()))
        }),
        ("an interaction", |o| {
            o.drift_sys[0].interaction = Some(TimeDependent::Constant(nan_matrix()))
        }),
        ("a single-spin term", |o| {
            o.drift_sys[0].singlespin = Some(nan_matrix())
        }),
    ];
    for (what, spoil) in cases {
        let mut opts = Fixture {
            objective: ObjectiveFn::StateAuxmat,
            ..Default::default()
        }
        .options()
        .unwrap();
        spoil(&mut opts);
        // BadValue is the parser's own verdict; anything else would mean the
        // NaN got past it and broke something further down.
        assert!(
            matches!(optimconset(opts), Err(QoalaError::BadValue(_))),
            "{what} holding a NaN was not caught as non-finite"
        );
    }
}

#[test]
fn a_time_dependent_drift_needs_exactly_one_matrix_per_slice() {
    // The auxiliary-matrix objective supports a time-dependent drift, so only
    // the slice count decides whether it parses.
    let fixture = Fixture {
        objective: ObjectiveFn::StateAuxmat,
        ..Default::default()
    };
    let with_matrices = |n: usize| {
        let mut opts = fixture.options().unwrap();
        let drift = fixture.drift().unwrap();
        opts.drift_sys[0].drift = Some(TimeDependent::PerStep(vec![drift; n]));
        optimconset(opts)
    };
    assert!(with_matrices(fixture.nsteps).is_ok());
    for n in [0, fixture.nsteps - 1, fixture.nsteps + 1] {
        assert!(
            with_matrices(n).is_err(),
            "{n} drift matrices accepted for {} slices",
            fixture.nsteps
        );
    }
}

#[test]
fn a_non_uniform_time_grid_is_refused_only_by_operator_splitting() {
    let grid = DVector::from_vec(vec![4e-4, 6e-4, 8e-4, 8e-4, 6e-4, 4e-4]);

    let mut opts = Fixture::default().options().unwrap();
    opts.pulse_dt = Some(grid.clone());
    assert!(matches!(
        optimconset(opts),
        Err(QoalaError::NotImplemented(_))
    ));

    // The auxiliary-matrix objective takes every slice's own duration.
    let mut opts = Fixture {
        objective: ObjectiveFn::StateAuxmat,
        ..Default::default()
    }
    .options()
    .unwrap();
    opts.pulse_dt = Some(grid);
    assert!(optimconset(opts).is_ok());
}

#[test]
fn power_level_ensembles_are_refused_rather_than_cut_to_one_member() {
    let mut opts = Fixture::default().options().unwrap();
    opts.pwr_levels = Some(PowerLevels::Ensemble(vec![
        vec![
            TWO_PI * 900.0,
            TWO_PI * 1100.0
        ];
        4
    ]));
    let sys = optimconset(opts).unwrap();
    assert_eq!(sys.pwr_levels.nrows(), 16);

    let wf = test_waveform(sys.pulse_nsteps, sys.operators.len());
    assert!(matches!(
        GrapeXyCost.value(&wf, &sys, sys.optimcon_fun),
        Err(QoalaError::NotImplemented(_))
    ));
}

#[test]
fn a_waveform_of_the_wrong_shape_is_an_error() {
    let sys = Fixture::default().build().unwrap();
    let wf = test_waveform(sys.pulse_nsteps + 1, sys.operators.len());
    assert!(matches!(
        GrapeXyCost.value(&wf, &sys, sys.optimcon_fun),
        Err(QoalaError::Dimension(_))
    ));

    let nspins = 2;
    let run = state2state_xy(StateTransfer {
        omega: vec![0.0; nspins],
        initial: spinops::z_state(nspins, 0),
        target: spinops::z_state(nspins, 1),
        amplitudes: vec![1000.0; nspins],
        duration: 0.01,
        increments: 10,
        interaction: spinops::zz_coupling(nspins, 0, 1)
            .unwrap()
            .scale(Complex64::new(TWO_PI * 140.0, 0.0)),
        cartops: spinops::one_pair_per_spin(nspins),
        init_pulse: Some(DMatrix::zeros(11, 2 * nspins)),
        seed: None,
        tuning: Tuning {
            max_iter: 1,
            output: Reporter::silent(),
            ..Default::default()
        },
    });
    assert!(matches!(run, Err(QoalaError::Dimension(_))));
}

/// Promises the objective rises along its gradient, then reports a lower
/// value at every point after the first, so no step can be accepted.
struct NothingBetter {
    calls: Cell<usize>,
}

impl CostFunction for NothingBetter {
    fn value(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        objfun: ObjectiveFn,
    ) -> qoala::error::Result<(TrajData, Vec<f64>)> {
        let (data, fidelities, _) = self.value_grad(wf, sys, objfun)?;
        Ok((data, fidelities))
    }

    fn value_grad(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        _objfun: ObjectiveFn,
    ) -> qoala::error::Result<ValueGrad> {
        let first = self.calls.replace(self.calls.get() + 1) == 0;
        let fidelity = if first { 1.0 } else { 0.0 };
        let data = TrajData {
            final_state: Some(CDense::from_element(1, 1, Complex64::new(fidelity, 0.0))),
            ..Default::default()
        };
        let npen = sys.penalties.len();
        let shape = (wf.nrows(), wf.ncols());
        let mut fidelities = vec![fidelity];
        fidelities.extend(std::iter::repeat_n(0.0, npen));
        let mut grads = vec![DMatrix::from_element(shape.0, shape.1, 1.0)];
        grads.extend(std::iter::repeat_n(DMatrix::zeros(shape.0, shape.1), npen));
        Ok((data, fidelities, grads))
    }
}

#[test]
fn a_failed_line_search_reports_the_point_it_keeps() {
    let mut sys = Fixture::default().build().unwrap();
    let guess = test_waveform(sys.pulse_nsteps, sys.operators.len());
    let cost = NothingBetter {
        calls: Cell::new(0),
    };
    let run = fmaxnewton(&mut sys, &cost, &guess).unwrap();

    assert!(matches!(run.exitflag, ExitFlag::LineSearchFailed));
    assert!(
        cost.calls.get() > 2,
        "the line search should try several steps"
    );
    assert_eq!(run.waveform, guess);

    let row = run.data.count.iter;
    assert_eq!(
        run.data.fx_store[(row, 1)],
        1.0,
        "the recorded fidelity belongs to a rejected trial"
    );
    assert_eq!(run.data.fx_sep_pen[0], 1.0);
    let state = run
        .data
        .traj_data
        .final_state
        .expect("diagnostics at the kept point");
    assert_eq!(state[(0, 0)].re, 1.0);
}

/// Reports a fidelity that is not a number, with a perfectly ordinary
/// gradient.
/// A Hessian that is finite on its own, but large enough that the RFO's
/// `alpha^2 H` overflows while the augmented matrix is assembled.
struct HugeHessian;

impl CostFunction for HugeHessian {
    fn value(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        objfun: ObjectiveFn,
    ) -> qoala::error::Result<(TrajData, Vec<f64>)> {
        let (data, fidelities, _) = self.value_grad(wf, sys, objfun)?;
        Ok((data, fidelities))
    }

    fn value_grad(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        _objfun: ObjectiveFn,
    ) -> qoala::error::Result<ValueGrad> {
        let npen = sys.penalties.len();
        let shape = (wf.nrows(), wf.ncols());
        let mut fidelities = vec![0.5];
        fidelities.extend(std::iter::repeat_n(0.0, npen));
        let mut grads = vec![DMatrix::from_element(shape.0, shape.1, 1.0)];
        grads.extend(std::iter::repeat_n(DMatrix::zeros(shape.0, shape.1), npen));
        Ok((TrajData::default(), fidelities, grads))
    }

    fn value_grad_hess(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        objfun: ObjectiveFn,
    ) -> qoala::error::Result<ValueGradHess> {
        let (data, fidelities, grads) = self.value_grad(wf, sys, objfun)?;
        let n = wf.nrows() * wf.ncols();
        let mut hesses = vec![DMatrix::from_diagonal_element(n, n, 1e200)];
        hesses.extend(std::iter::repeat_n(
            DMatrix::zeros(n, n),
            sys.penalties.len(),
        ));
        Ok((data, fidelities, grads, hesses))
    }
}

/// Every number reaching the regularisation is finite; the overflow happens
/// inside it, before the eigensolver is asked for anything.
#[test]
fn an_overflow_while_regularising_the_hessian_stops_the_run() {
    let mut sys = Fixture::default().build().unwrap();
    sys.method = OptMethod::NewtonRaphson;
    sys.reg_alpha = 1e200;
    let guess = test_waveform(sys.pulse_nsteps, sys.operators.len());
    assert!(matches!(
        fmaxnewton(&mut sys, &HugeHessian, &guess),
        Err(QoalaError::Numerical(_))
    ));
}

struct NotFinite;

impl CostFunction for NotFinite {
    fn value(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        objfun: ObjectiveFn,
    ) -> qoala::error::Result<(TrajData, Vec<f64>)> {
        let (data, fidelities, _) = self.value_grad(wf, sys, objfun)?;
        Ok((data, fidelities))
    }

    fn value_grad(
        &self,
        wf: &DMatrix<f64>,
        sys: &ControlSystem,
        _objfun: ObjectiveFn,
    ) -> qoala::error::Result<ValueGrad> {
        let npen = sys.penalties.len();
        let shape = (wf.nrows(), wf.ncols());
        let mut fidelities = vec![f64::NAN];
        fidelities.extend(std::iter::repeat_n(0.0, npen));
        let mut grads = vec![DMatrix::from_element(shape.0, shape.1, 1.0)];
        grads.extend(std::iter::repeat_n(DMatrix::zeros(shape.0, shape.1), npen));
        Ok((TrajData::default(), fidelities, grads))
    }
}

/// A NaN has to stop the run where it appeared: further down it becomes a
/// comparison that quietly takes the wrong branch.
#[test]
fn an_objective_that_is_not_finite_stops_the_run() {
    let mut sys = Fixture::default().build().unwrap();
    let guess = test_waveform(sys.pulse_nsteps, sys.operators.len());
    assert!(matches!(
        fmaxnewton(&mut sys, &NotFinite, &guess),
        Err(QoalaError::Numerical(_))
    ));
}
