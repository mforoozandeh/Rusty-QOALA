//! The native runner: a background thread, a channel, and cooperative
//! cancellation.
//!
//! The browser runner cannot be exercised without a browser; this covers the
//! half of the plumbing that can be.

#![cfg(not(target_arch = "wasm32"))]

use std::time::{Duration, Instant};

use qoala_gui::presets;
use qoala_gui::run::RunMessage;
use qoala_gui::runner::Runner;

/// Collect messages until the run ends or `limit` passes.
fn drain_until_done(runner: &mut Runner, limit: Duration) -> Vec<RunMessage> {
    let started = Instant::now();
    let mut all = Vec::new();
    loop {
        all.extend(runner.drain());
        if all
            .iter()
            .any(|m| matches!(m, RunMessage::Finished(_) | RunMessage::Failed(_)))
        {
            return all;
        }
        if started.elapsed() > limit {
            panic!("the run did not finish within {limit:?}");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_run_happens_off_the_calling_thread_and_reports_back() {
    let mut setup = presets::default_setup();
    setup.max_iter = 10;
    let mut runner = Runner::start(setup);

    let messages = drain_until_done(&mut runner, Duration::from_secs(60));
    let progress = messages
        .iter()
        .filter(|m| matches!(m, RunMessage::Progress(_)))
        .count();
    assert_eq!(progress, 11, "one row per iteration plus the initial one");

    let finished = messages
        .iter()
        .find_map(|m| match m {
            RunMessage::Finished(f) => Some(f),
            _ => None,
        })
        .expect("finished");
    assert_eq!(finished.iterations, 10);
    assert_eq!(finished.waveform.len(), 50);
}

/// Cancelling natively is cooperative, so the waveform reached so far comes
/// back with the `Finished` message rather than being thrown away.  That is
/// what `Runner::CANCEL_KEEPS_WAVEFORM` promises the interface, and what the
/// waveform assertions at the end of this test check.
#[test]
fn cancelling_stops_the_run_and_keeps_the_waveform() {
    let mut setup = presets::swap_3spin_1();
    setup.max_iter = 5000;
    let mut runner = Runner::start(setup);

    // Let it get going, then ask it to stop.
    let started = Instant::now();
    let mut all = Vec::new();
    while all
        .iter()
        .filter(|m| matches!(m, RunMessage::Progress(_)))
        .count()
        < 2
    {
        all.extend(runner.drain());
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "never started"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    runner.cancel();

    all.extend(drain_until_done(&mut runner, Duration::from_secs(120)));
    let finished = all
        .iter()
        .find_map(|m| match m {
            RunMessage::Finished(f) => Some(f),
            _ => None,
        })
        .expect("finished");

    assert_eq!(finished.exit_message, "cancelled by the caller");
    assert!(
        finished.iterations < 5000,
        "cancellation did not shorten the run: {}",
        finished.iterations
    );
    assert_eq!(finished.waveform.len(), 264);
    assert!(finished.waveform.iter().all(|row| row.len() == 6));
}
