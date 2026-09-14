//! ESCALADE end to end: convergence, the amplitude limit, B1 compensation,
//! progress reporting and cancellation.
//!
//! The fidelity each run reports is checked against an independent
//! measurement of what its pulse does to magnetisation, so a run cannot pass
//! by agreeing with itself.

use qoala::escalade::{escalade, escalade_with_progress, profile, Escalade, Optimised};
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

/// Mean of -Iy over the band at a field of `scale` times nominal: the
/// fidelity, measured from the magnetisation rather than the objective.
fn measured_fidelity(spec: &Escalade, out: &Optimised, scale: f64) -> f64 {
    let offs = offsets(spec);
    offs.iter()
        .map(|&o| -profile::final_magnetisation(&out.pulse, spec.tau_p, spec.rf[0] * scale, o).y)
        .sum::<f64>()
        / offs.len() as f64
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
