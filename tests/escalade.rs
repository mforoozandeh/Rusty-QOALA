//! ESCALADE end to end: convergence, universal rotations, the amplitude
//! limit, B1 compensation, progress reporting and cancellation.
//!
//! The fidelity each run reports is checked against an independent
//! measurement of what its pulse does to magnetisation, so a run cannot pass
//! by agreeing with itself.

use qoala::escalade::profile::Magnetisation;
use qoala::escalade::{
    escalade, escalade_with_progress, profile, rotation, Escalade, Goal, Optimised,
};
use qoala::optim::{IterationReport, ProgressSink};
use qoala::types::ExitFlag;

/// The amplitude limit is a penalty, so a run that stops at its target
/// fidelity can sit a little outside it.
const AMPLITUDE_LIMIT: f64 = 1.02;

/// The problem `test_runs/test_escalade_grad_vs_grad_hess.m` runs: z to -y
/// across 20 kHz with a 17 kHz field in 100 microseconds.
fn broadband(use_hessian: bool) -> Escalade {
    Escalade {
        nspins: 51,
        np_pulse: 50,
        tau_p: 100e-6,
        rf: vec![17000.0],
        sw: 20000.0,
        use_hessian,
        max_iter: 1000,
        seed: Some(7),
        ..Default::default()
    }
}

fn offsets(spec: &Escalade) -> Vec<f64> {
    let n = spec.nspins;
    (0..n)
        .map(|i| -spec.sw / 2.0 + spec.sw * i as f64 / (n - 1) as f64)
        .collect()
}

const PLUS_X: Magnetisation = Magnetisation::new(1.0, 0.0, 0.0);
const PLUS_Y: Magnetisation = Magnetisation::new(0.0, 1.0, 0.0);
const PLUS_Z: Magnetisation = Magnetisation::new(0.0, 0.0, 1.0);
const MINUS_Y: Magnetisation = Magnetisation::new(0.0, -1.0, 0.0);

/// The default transfer, z to -y.
const EXCITATION: [(Magnetisation, Magnetisation); 1] = [(PLUS_Z, MINUS_Y)];

/// A universal 90-degree rotation about x, axis by axis.
const ROTATION_ABOUT_X: [(Magnetisation, Magnetisation); 3] =
    [(PLUS_X, PLUS_X), (PLUS_Y, PLUS_Z), (PLUS_Z, MINUS_Y)];

fn about_x() -> Goal {
    Goal::Rotation(rotation([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2))
}

/// How far the magnetisation lands along its target at each offset,
/// averaged over the transfers, at a field of `scale` times nominal.
fn per_offset(
    spec: &Escalade,
    out: &Optimised,
    scale: f64,
    pairs: &[(Magnetisation, Magnetisation)],
) -> Vec<f64> {
    let rf = spec.rf[0] * scale;
    offsets(spec)
        .into_iter()
        .map(|o| {
            pairs
                .iter()
                .map(|&(from, to)| {
                    profile::final_magnetisation(&out.pulse, spec.tau_p, rf, o, from).dot(&to)
                })
                .sum::<f64>()
                / pairs.len() as f64
        })
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// How far the magnetisation lands along its target, averaged over the band
/// and the transfers: the fidelity, measured from the magnetisation rather
/// than the objective.
fn measured(
    spec: &Escalade,
    out: &Optimised,
    scale: f64,
    pairs: &[(Magnetisation, Magnetisation)],
) -> f64 {
    mean(&per_offset(spec, out, scale, pairs))
}

/// The excitation fidelity, measured.
fn measured_fidelity(spec: &Escalade, out: &Optimised, scale: f64) -> f64 {
    measured(spec, out, scale, &EXCITATION)
}

/// A rotation's fidelity, measured from what it does to x, y and z.
///
/// Where the three transfers average `F`, the rotation is `acos((3F - 1) /
/// 2)` from its target, so the propagator overlap `Re tr(W^dagger U) / 2`
/// is `sqrt((3F + 1) / 4)` - provided the propagator has the target's sign,
/// which magnetisation cannot show.  So this agrees with the objective only
/// if no part of the band settled on the other sign.
fn measured_rotation(spec: &Escalade, out: &Optimised) -> f64 {
    let overlaps: Vec<f64> = per_offset(spec, out, 1.0, &ROTATION_ABOUT_X)
        .into_iter()
        .map(|f| ((3.0 * f + 1.0) / 4.0).sqrt())
        .collect();
    mean(&overlaps)
}

/// A rotation reached its target, by its own account and by what it does to
/// every axis.
fn rotated(spec: &Escalade, out: &Optimised) {
    assert_eq!(out.exitflag, ExitFlag::FidelityTolerance, "{out:?}");
    assert!(
        (measured_rotation(spec, out) - out.fidelity).abs() < 1e-9,
        "the objective and the magnetisation disagree"
    );
    // Each axis gets there, not just the average: a rotation `theta` from
    // its target moves no axis further than `cos theta = 2 a^2 - 1` from
    // where it should be, and averaged over the band that is at least
    // `2 F^2 - 1` for a fidelity `F`.
    let worst = 2.0 * out.fidelity * out.fidelity - 1.0;
    for pair in ROTATION_ABOUT_X {
        let each = measured(spec, out, 1.0, &[pair]);
        assert!(each >= worst - 1e-9, "{pair:?}: {each} below {worst}");
    }
    assert!(
        out.max_amplitude <= AMPLITUDE_LIMIT,
        "{}",
        out.max_amplitude
    );
}

fn converges(use_hessian: bool) {
    let spec = broadband(use_hessian);
    let out = escalade(&spec).unwrap();
    assert_eq!(out.exitflag, ExitFlag::FidelityTolerance, "{out:?}");
    assert!(out.fidelity >= 0.99, "fidelity {}", out.fidelity);
    assert!(
        (measured_fidelity(&spec, &out, 1.0) - out.fidelity).abs() < 1e-9,
        "the objective and the magnetisation disagree"
    );
    assert!(
        out.max_amplitude <= AMPLITUDE_LIMIT,
        "amplitude {} overshoots the limit",
        out.max_amplitude
    );
}

#[test]
fn broadband_excitation_converges_with_the_hessian() {
    converges(true);
}

#[test]
fn broadband_excitation_converges_on_the_gradient_alone() {
    converges(false);
}

/// A broadband universal rotation: every axis turned 90 degrees about x by
/// the one pulse, not just z brought down onto -y.
#[test]
fn a_universal_rotation_converges() {
    let spec = Escalade {
        nspins: 21,
        sw: 10000.0,
        tau_p: 200e-6,
        np_pulse: 60,
        goal: about_x(),
        ..broadband(true)
    };
    rotated(&spec, &escalade(&spec).unwrap());
}

/// A rotation across a band nearly twice the field.  Scored on the three
/// transfers x, y and z, every start tried settled at a fidelity of 0.638,
/// with the propagator's sign flipped across the outer part of the band and
/// the spins at the boundary turned 180 degrees from the target.
#[test]
fn a_wide_band_rotation_does_not_split_into_opposite_signs() {
    let spec = Escalade {
        nspins: 51,
        sw: 30000.0,
        tau_p: 200e-6,
        np_pulse: 100,
        goal: about_x(),
        max_iter: 2000,
        seed: Some(3),
        ..broadband(false)
    };
    rotated(&spec, &escalade(&spec).unwrap());
}

/// A starting pulse reaching twice the amplitude limit is pulled back inside
/// it.
#[test]
fn the_amplitude_limit_is_enforced() {
    for use_hessian in [true, false] {
        let mut spec = broadband(use_hessian);
        spec.nspins = 21;
        spec.np_pulse = 30;
        spec.start = Some(spec.resolve().unwrap().start * 1.4);
        let out = escalade(&spec).unwrap();
        assert!(out.fidelity >= 0.99, "fidelity {}", out.fidelity);
        assert!(
            out.max_amplitude <= AMPLITUDE_LIMIT,
            "amplitude {}",
            out.max_amplitude
        );
    }
}

/// Optimising across a spread of fields buys robustness to field error, which
/// is what `test_escalade_visual.m` demonstrates.
#[test]
fn optimising_over_a_field_spread_makes_the_pulse_robust() {
    let mut sensitive = broadband(true);
    sensitive.nspins = 21;
    sensitive.np_pulse = 40;
    sensitive.target_fidelity = 0.995;
    let mut compensated = sensitive.clone();
    compensated.rf = (0..7)
        .map(|i| 17000.0 * (0.8 + 0.4 * i as f64 / 6.0))
        .collect();

    let a = escalade(&sensitive).unwrap();
    let b = escalade(&compensated).unwrap();

    let worst = |spec: &Escalade, out: &Optimised| {
        (0..9)
            .map(|i| measured_fidelity(spec, out, 0.8 + 0.4 * i as f64 / 8.0))
            .fold(f64::INFINITY, f64::min)
    };
    // Measure both at the nominal field of the sensitive run.
    let wa = worst(&sensitive, &a);
    let wb = worst(&sensitive, &b);
    assert!(
        wb > wa + 0.01,
        "compensated worst case {wb} should beat sensitive {wa}"
    );
}

struct CancelAfter {
    reports: Vec<IterationReport>,
    rows: usize,
}

impl ProgressSink for CancelAfter {
    fn on_iteration(&mut self, report: &IterationReport) {
        self.reports.push(report.clone());
    }
    fn should_cancel(&self) -> bool {
        self.reports.len() >= self.rows
    }
}

#[test]
fn every_iteration_is_reported_and_the_budget_is_kept() {
    for use_hessian in [true, false] {
        let mut spec = broadband(use_hessian);
        spec.nspins = 11;
        spec.max_iter = 3;
        spec.target_fidelity = 2.0;
        let mut sink = CancelAfter {
            reports: Vec::new(),
            rows: usize::MAX,
        };
        let out = escalade_with_progress(&spec, &mut sink).unwrap();
        assert_eq!(out.exitflag, ExitFlag::MaxIterations);
        assert_eq!(out.counters.iter, 3);
        let iterations: Vec<usize> = sink.reports.iter().map(|r| r.iteration).collect();
        assert_eq!(iterations, vec![0, 1, 2, 3]);
        assert!(sink.reports.iter().all(|r| r.fidelity.is_finite()));
        // The objective only ever goes down.
        for pair in sink.reports.windows(2) {
            assert!(pair[1].total >= pair[0].total - 1e-12);
        }
    }
}

#[test]
fn a_sink_can_cancel_a_run() {
    let mut spec = broadband(false);
    spec.target_fidelity = 2.0;
    let mut sink = CancelAfter {
        reports: Vec::new(),
        rows: 4,
    };
    let out = escalade_with_progress(&spec, &mut sink).unwrap();
    assert_eq!(out.exitflag, ExitFlag::Cancelled);
    assert_eq!(out.counters.iter, 3);
    assert_eq!(out.pulse.shape(), (50, 2));
}
