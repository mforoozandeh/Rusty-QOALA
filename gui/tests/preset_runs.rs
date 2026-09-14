//! End-to-end: every preset has to build and converge.
//!
//! These are the presets the front page offers, so a broken one is a broken
//! landing experience.

use qoala_gui::presets;
use qoala_gui::problem::Problem;
use qoala_gui::run::{run_to_sink, MessageSink, RunMessage};

#[derive(Default)]
struct Collect {
    messages: Vec<RunMessage>,
}

impl MessageSink for Collect {
    fn send(&mut self, message: RunMessage) {
        self.messages.push(message);
    }
}

/// Run a problem and return what came back.
fn run(problem: &Problem) -> Collect {
    let mut sink = Collect::default();
    run_to_sink(problem, &mut sink);
    sink
}

/// Run a QOALA preset with a small iteration budget.
fn run_qoala(mut setup: qoala_gui::setup::Setup, max_iter: usize) -> Collect {
    setup.max_iter = max_iter;
    run(&Problem::Qoala(setup))
}

fn finished(out: &Collect, name: &str) -> qoala_gui::run::Finished {
    out.messages
        .iter()
        .find_map(|m| match m {
            RunMessage::Finished(f) => Some((**f).clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{name} never finished: {:?}", out.messages.last()))
}

fn progress(out: &Collect) -> Vec<&qoala_gui::run::Progress> {
    out.messages
        .iter()
        .filter_map(|m| match m {
            RunMessage::Progress(p) => Some(p),
            _ => None,
        })
        .collect()
}

#[test]
fn every_preset_is_a_valid_setup() {
    for problem in presets::every() {
        assert!(
            problem.problems().is_empty(),
            "{}: {:?}",
            problem.name(),
            problem.problems()
        );
    }
}

#[test]
fn every_preset_runs_and_reports_progress() {
    for problem in presets::every() {
        let name = problem.name().to_string();
        let budget = 4;
        let out = run(&match problem {
            Problem::Qoala(mut s) => {
                s.max_iter = budget;
                Problem::Qoala(s)
            }
            Problem::Escalade(mut s) => {
                s.max_iter = budget;
                // Out of reach, so the budget is what stops it.
                s.target_fidelity = 1.0;
                Problem::Escalade(s)
            }
        });

        let failures: Vec<&RunMessage> = out
            .messages
            .iter()
            .filter(|m| matches!(m, RunMessage::Failed(_)))
            .collect();
        assert!(failures.is_empty(), "{name}: {failures:?}");

        let progress = progress(&out);
        // One row per iteration plus the initial evaluation.
        assert_eq!(progress.len(), budget + 1, "{name}");
        assert!(progress.iter().all(|p| p.fidelity.is_finite()), "{name}");

        let done = finished(&out, &name);
        assert_eq!(done.iterations, budget, "{name}");
        assert!(done.fidelity.is_finite(), "{name}");
    }
}

#[test]
fn qoala_progress_carries_the_splitting() {
    let out = run_qoala(presets::default_setup(), 2);
    assert!(progress(&out).iter().all(|p| p.split_order >= 1));
}

/// The two-spin default is the one somebody sees first, so it has to reach a
/// high fidelity in the budget it ships with.
#[test]
fn the_default_preset_converges() {
    let setup = presets::default_setup();
    let out = run_qoala(setup, 100);
    let done = finished(&out, "default");
    assert!(
        done.fidelity > 0.999,
        "fidelity only reached {}",
        done.fidelity
    );
    assert_eq!(done.waveform.len(), 50);
    assert_eq!(done.waveform[0].len(), 4);
}

/// The ESCALADE default reaches its target, and its pulse is x and y.
#[test]
fn the_escalade_default_converges() {
    let setup = presets::escalade_b1_sensitive();
    let out = run(&Problem::Escalade(setup.clone()));
    let done = finished(&out, "escalade");
    assert!(
        done.fidelity >= setup.target_fidelity,
        "fidelity only reached {}",
        done.fidelity
    );
    assert_eq!(done.waveform.len(), setup.nslices);
    assert_eq!(done.waveform[0].len(), 2);
}

/// A setup the interface would refuse must be refused here too, with a
/// message rather than a panic.
#[test]
fn an_impossible_setup_fails_with_a_message() {
    let mut setup = presets::default_setup();
    setup.spin_control[1] = vec![false, false];
    let out = run_qoala(setup, 1);
    assert!(matches!(out.messages.first(), Some(RunMessage::Failed(_))));

    let mut escalade = presets::escalade_b1_sensitive();
    escalade.to = escalade.from;
    let out = run(&Problem::Escalade(escalade));
    assert!(matches!(out.messages.first(), Some(RunMessage::Failed(_))));
}

/// A gate built through the gate menu has to be a workable target, not just
/// a matrix that exists.
#[test]
fn the_gate_menu_produces_runnable_targets() {
    use qoala_gui::setup::{GateChoice, Target};
    let mut setup = presets::swap_2spin_1();
    for gate in [
        GateChoice::Swap,
        GateChoice::ISwap,
        GateChoice::SqrtSwap,
        GateChoice::Cnot,
        GateChoice::Cz,
    ] {
        setup.target = Target::Gate {
            gate,
            qubits: vec![0, 1],
        };
        let out = run_qoala(setup.clone(), 2);
        let done = finished(&out, gate.name());
        assert!(done.fidelity.is_finite(), "{}", gate.name());
    }
}
