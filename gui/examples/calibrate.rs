//! Measure how long the presets actually take, to calibrate the Run-button
//! estimate in `estimate.rs`, and print the share link for each one.  A
//! development tool, not part of the application.

use qoala_gui::estimate::estimate_seconds;
use qoala_gui::presets;
use qoala_gui::run::{run_to_sink, MessageSink, RunMessage};

struct Quiet;
impl MessageSink for Quiet {
    fn send(&mut self, _m: RunMessage) {}
}

fn main() {
    println!(
        "{:<40} {:>8} {:>10} {:>10} {:>8}",
        "preset", "iters", "measured", "estimate", "ratio"
    );
    for mut setup in presets::all() {
        setup.max_iter = 20;
        let started = std::time::Instant::now();
        run_to_sink(&setup, &mut Quiet);
        let measured = started.elapsed().as_secs_f64();
        let estimate = estimate_seconds(&setup);
        println!(
            "{:<40} {:>8} {:>9.3}s {:>9.3}s {:>8.2}",
            setup.name,
            setup.max_iter,
            measured,
            estimate,
            estimate / measured
        );
    }

    println!("\nshare links");
    for setup in presets::all() {
        println!(
            "{:<40} #{}",
            setup.name,
            qoala_gui::platform::fragment_for_setup(&setup).expect("fragment")
        );
    }
}
