//! Measure how long the presets actually take, to calibrate the Run-button
//! estimate in `estimate.rs`, and print the share link for each one.  A
//! development tool, not part of the application.

use qoala_gui::estimate::estimate_seconds;
use qoala_gui::presets;
use qoala_gui::problem::Problem;
use qoala_gui::run::{run_to_sink, MessageSink, RunMessage};

struct Quiet;
impl MessageSink for Quiet {
    fn send(&mut self, _m: RunMessage) {}
}

const ITERATIONS: usize = 20;

fn main() {
    println!(
        "{:<40} {:>8} {:>10} {:>10} {:>8}",
        "preset", "iters", "measured", "estimate", "ratio"
    );
    for problem in presets::every() {
        // Every run gets the same budget, and ESCALADE is kept from stopping
        // at its target so that the budget is what is measured.
        let problem = match problem {
            Problem::Qoala(mut s) => {
                s.max_iter = ITERATIONS;
                Problem::Qoala(s)
            }
            Problem::Escalade(mut s) => {
                s.max_iter = ITERATIONS;
                s.target_fidelity = 1.0;
                Problem::Escalade(s)
            }
        };
        let started = std::time::Instant::now();
        run_to_sink(&problem, &mut Quiet);
        let measured = started.elapsed().as_secs_f64();
        let estimate = estimate_seconds(&problem);
        println!(
            "{:<40} {:>8} {:>9.3}s {:>9.3}s {:>8.2}",
            problem.name(),
            ITERATIONS,
            measured,
            estimate,
            estimate / measured
        );
    }

    println!("\nshare links");
    for problem in presets::every() {
        println!(
            "{:<40} #{}",
            problem.name(),
            qoala_gui::platform::fragment_for_problem(&problem).expect("fragment")
        );
    }
}
