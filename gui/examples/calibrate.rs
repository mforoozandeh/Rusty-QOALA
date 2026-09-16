//! Measure how long the presets actually take, to calibrate the Run-button
//! estimate in `estimate.rs`, and print the share link for each one.  A
//! development tool, not part of the application.
//!
//! The estimate's constants are for one core, and ESCALADE otherwise runs on
//! every core, so calibrate with
//!
//! ```text
//! RAYON_NUM_THREADS=1 cargo run -p qoala-gui --release --example calibrate
//! ```

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
    // One untimed iteration of everything first.  The first run in a process
    // also starts rayon's thread pool and faults in fresh memory, and would
    // otherwise bill that to whichever preset happens to come first.
    for problem in presets::every() {
        run_to_sink(&with_budget(problem, 1), &mut Quiet);
    }

    println!(
        "{:<40} {:>8} {:>10} {:>10} {:>8}",
        "preset", "iters", "measured", "estimate", "ratio"
    );
    for problem in presets::every() {
        let problem = with_budget(problem, ITERATIONS);
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

/// `problem` with an iteration budget of `iterations`, and ESCALADE kept from
/// stopping at its target so that the budget is what is measured.
fn with_budget(problem: Problem, iterations: usize) -> Problem {
    match problem {
        Problem::Qoala(mut s) => {
            s.max_iter = iterations;
            Problem::Qoala(s)
        }
        Problem::Escalade(mut s) => {
            s.max_iter = iterations;
            s.target_fidelity = 1.0;
            Problem::Escalade(s)
        }
    }
}
