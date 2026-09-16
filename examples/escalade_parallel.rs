//! ESCALADE on one thread and on many.
//!
//! The B1-compensated problem of `escalade_b1` - z to -y across 51 spins over
//! 20 kHz, at 31 fields from 0.8 to 1.2 times 17 kHz, with a 50-point pulse
//! and the Newton trust region on the Hessian - is what the thread pool is
//! for: every evaluation propagates each spin at each field, 1581 work items
//! that do not depend on each other.  Each work item is a time-ordered
//! product over the pulse, and stays on one thread.
//!
//! First the objective alone, at each order, on one thread and on more, up to
//! one per core.  Then a whole optimisation on one thread and on all of them,
//! which has to come back with the same pulse to the last bit: the thread
//! count decides where the work runs, never how the sums are grouped.
//!
//! ```text
//! cargo run --release --example escalade_parallel
//! cargo run --release --example escalade_parallel -- --iters 1000 --fields 1
//! ```
//!
//! `--fields` sets the number of field strengths (1 is the B1-sensitive
//! problem), `--spins` the number of offsets, and `--iters` the iteration
//! budget of the optimisations.

use std::time::{Duration, Instant};

use nalgebra::DMatrix;
use qoala::error::{QoalaError, Result};
use qoala::escalade::objective::{gradhess, Evaluation};
use qoala::escalade::{escalade, Escalade, Optimised};
use qoala::optim::ObjectiveRequest;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Each timing repeats an evaluation for at least this long.
const MEASURE_FOR: Duration = Duration::from_millis(500);

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |key: &str| -> Option<usize> {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
    };
    let nfields = flag("--fields").unwrap_or(31).max(1);
    let nspins = flag("--spins").unwrap_or(51);
    let max_iter = flag("--iters").unwrap_or(200);

    let rf = 17000.0;
    let np_pulse = 50;
    let spec = Escalade {
        nspins,
        np_pulse,
        tau_p: 100e-6,
        rf: if nfields == 1 {
            vec![rf]
        } else {
            (0..nfields)
                .map(|i| rf * (0.8 + 0.4 * i as f64 / (nfields - 1) as f64))
                .collect()
        },
        sw: 20000.0,
        use_hessian: true,
        max_iter,
        start: Some(DMatrix::from_element(np_pulse, 2, -0.5)),
        ..Default::default()
    };
    let settings = spec.resolve()?;

    let cores = rayon::current_num_threads();
    let mut counts: Vec<usize> = std::iter::successors(Some(1), |n| Some(n * 2))
        .take_while(|&n| n < cores)
        .collect();
    counts.push(cores);
    let pools = counts
        .iter()
        .map(|&n| pool(n))
        .collect::<Result<Vec<_>>>()?;

    println!(
        "{} spins x {} fields, {} points; rayon's pool has {cores} threads",
        settings.nspins,
        settings.rf.len(),
        settings.np_pulse
    );

    println!("\n --- objective, milliseconds per evaluation (speed-up) ------");
    println!(
        " {:>7}  {:>18}  {:>18}  {:>18}",
        "threads", "value", "gradient", "hessian"
    );
    let orders = [
        ObjectiveRequest::Value,
        ObjectiveRequest::Gradient,
        ObjectiveRequest::Hessian,
    ];
    let mut serial = [0.0; 3];
    let mut reference: Vec<Evaluation> = Vec::new();
    for (&threads, pool) in counts.iter().zip(&pools) {
        let mut row = format!(" {threads:>7}");
        for (i, &order) in orders.iter().enumerate() {
            let (ms, eval) = pool.install(|| time_evaluation(&settings, order))?;
            if threads == 1 {
                serial[i] = ms;
                reference.push(eval);
            } else if !same(&eval, &reference[i]) {
                return Err(QoalaError::Numerical(format!(
                    "{threads} threads changed the {order:?} evaluation"
                )));
            }
            row += &format!("  {ms:>9.3} ({:>5.2}x)", serial[i] / ms);
        }
        println!("{row}");
    }
    println!(" every evaluation identical to the one-thread result");

    println!("\n --- optimisation, at most {max_iter} iterations ------");
    let one = pools[0].install(|| escalade(&spec))?;
    report(1, &one, one.elapsed_s);
    let all = pools[pools.len() - 1].install(|| escalade(&spec))?;
    report(cores, &all, one.elapsed_s);
    if one.pulse != all.pulse || one.counters.iter != all.counters.iter {
        return Err(QoalaError::Numerical(
            "the optimisations on one thread and on all of them diverged".into(),
        ));
    }
    println!(" identical pulses after identical iterations");
    Ok(())
}

fn pool(threads: usize) -> Result<ThreadPool> {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| QoalaError::BadValue(e.to_string()))
}

/// Mean milliseconds per evaluation of the start pulse, after a warm-up, and
/// the evaluation itself.
fn time_evaluation(
    settings: &qoala::escalade::Settings,
    order: ObjectiveRequest,
) -> Result<(f64, Evaluation)> {
    let eval = gradhess(settings, &settings.start, order)?;
    let started = Instant::now();
    let mut reps = 0;
    while reps < 3 || started.elapsed() < MEASURE_FOR {
        std::hint::black_box(gradhess(settings, &settings.start, order)?);
        reps += 1;
    }
    Ok((started.elapsed().as_secs_f64() * 1e3 / reps as f64, eval))
}

fn same(a: &Evaluation, b: &Evaluation) -> bool {
    a.value.to_bits() == b.value.to_bits() && a.grad == b.grad && a.hess == b.hess
}

fn report(threads: usize, out: &Optimised, serial_s: f64) {
    println!(
        " {threads:>2} threads: {:>7.2} s ({:.2}x), {} iterations ({} value, {} gradient, {} Hessian evaluations), fidelity {:.6}, stopped because {}",
        out.elapsed_s,
        serial_s / out.elapsed_s,
        out.counters.iter,
        out.counters.fx,
        out.counters.gfx,
        out.counters.hfx,
        out.fidelity,
        out.exitflag
    );
}
