//! End-to-end: every preset has to build a control system and converge.
//!
//! These are the presets the front page offers, so a broken one is a broken
//! landing experience.

use qoala_gui::presets;
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

/// Run a preset with a small iteration budget and return what came back.
fn run(mut setup: qoala_gui::setup::Setup, max_iter: usize) -> Collect {
    setup.max_iter = max_iter;
    let mut sink = Collect::default();
    run_to_sink(&setup, &mut sink);
    sink
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

#[test]
fn every_preset_is_a_valid_setup() {
    for setup in presets::all() {
        assert!(
            setup.problems().is_empty(),
            "{}: {:?}",
            setup.name,
            setup.problems()
        );
    }
}

#[test]
fn every_preset_runs_and_reports_progress() {
    for setup in presets::all() {
        let name = setup.name.clone();
        let budget = 4;
        let out = run(setup, budget);

        let failures: Vec<&RunMessage> = out
            .messages
            .iter()
            .filter(|m| matches!(m, RunMessage::Failed(_)))
            .collect();
        assert!(failures.is_empty(), "{name}: {failures:?}");

        let progress: Vec<_> = out
            .messages
            .iter()
            .filter_map(|m| match m {
                RunMessage::Progress(p) => Some(p),
                _ => None,
            })
            .collect();
        // One row per iteration plus the initial evaluation.
        assert_eq!(progress.len(), budget + 1, "{name}");
        assert!(progress.iter().all(|p| p.fidelity.is_finite()), "{name}");
        assert!(progress.iter().all(|p| p.split_order >= 1), "{name}");

        let done = finished(&out, &name);
        assert_eq!(done.iterations, budget, "{name}");
        assert!(done.fidelity.is_finite(), "{name}");
    }
}

/// The two-spin default is the one somebody sees first, so it has to reach a
/// high fidelity in the budget it ships with.
#[test]
fn the_default_preset_converges() {
    let setup = presets::default_setup();
    let out = run(setup, 100);
    let done = finished(&out, "default");
    assert!(
        done.fidelity > 0.999,
        "fidelity only reached {}",
        done.fidelity
    );
    assert_eq!(done.waveform.len(), 50);
    assert_eq!(done.waveform[0].len(), 4);
}

/// A setup the interface would refuse must be refused here too, with a
/// message rather than a panic.
#[test]
fn an_impossible_setup_fails_with_a_message() {
    let mut setup = presets::default_setup();
    setup.spin_control[1] = vec![false, false];
    let out = run(setup, 1);
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
        let out = run(setup.clone(), 2);
        let done = finished(&out, gate.name());
        assert!(done.fidelity.is_finite(), "{}", gate.name());
    }
}
