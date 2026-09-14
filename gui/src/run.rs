//! Running an optimisation and getting the numbers back out.
//!
//! The optimiser is a single blocking call, so the work happens off the UI
//! thread and reaches the interface as a stream of [`RunMessage`]s.  How that
//! stream is produced differs by platform - a `std::thread` natively, a Web
//! Worker in the browser - but [`run_to_sink`] below is the same code on both
//! sides of that line.

use qoala::drivers::{
    random_pulse, state2state_xy_with_progress, universal_gate_xy_with_progress, GateSynthesis,
    StateTransfer, Tuning,
};
use qoala::optim::{IterationReport, ProgressSink};
use qoala::report::Reporter;
use qoala::types::Penalty;
use serde::{Deserialize, Serialize};

use crate::setup::{Setup, Target};

/// One row of progress, in a form that survives `postMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    /// Optimiser iteration; 0 is the initial evaluation.
    pub iteration: usize,
    /// Objective fidelity.
    pub fidelity: f64,
    /// Summed penalty terms, negated as the console table prints them.
    pub penalty: f64,
    /// Fidelity minus penalties.
    pub total: f64,
    /// Two-norm of the gradient.
    pub gradient_norm: f64,
    /// Line-search step length.
    pub alpha: Option<f64>,
    /// Splitting order the adaptive step has reached.
    pub split_order: usize,
    /// Trotter number the adaptive step has reached.
    pub trotter_number: usize,
    /// Wall-clock seconds since the run started.
    pub elapsed_s: f64,
    /// Objective-function evaluations so far.
    pub count_fx: usize,
    /// Gradient evaluations so far.
    pub count_gfx: usize,
}

impl From<&IterationReport> for Progress {
    fn from(r: &IterationReport) -> Self {
        Progress {
            iteration: r.iteration,
            fidelity: r.fidelity,
            penalty: r.penalty,
            total: r.total,
            gradient_norm: r.gradient_norm,
            alpha: r.alpha,
            split_order: r.split_order,
            trotter_number: r.trotter_number,
            elapsed_s: r.elapsed_s,
            count_fx: r.counters.fx,
            count_gfx: r.counters.gfx,
        }
    }
}

/// What a finished run produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finished {
    /// The waveform, row-major: `waveform[slice][channel]`, dimensionless in
    /// `[-1, 1]` as the optimiser works in.
    pub waveform: Vec<Vec<f64>>,
    /// Why the optimiser stopped.
    pub exit_message: String,
    /// Final objective fidelity.
    pub fidelity: f64,
    /// Total wall-clock seconds.
    pub elapsed_s: f64,
    /// Iterations completed.
    pub iterations: usize,
}

/// Everything the interface can be told by a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RunMessage {
    /// Transport detail: the Web Worker has installed its message handler and
    /// is safe to send a setup to.  A worker that is still loading its own
    /// WebAssembly drops anything posted to it, so the page waits for this
    /// before handing over the work.  Nothing else produces it and the
    /// interface ignores it.
    Ready,
    /// One iteration's numbers.
    Progress(Progress),
    /// The run finished on its own terms.
    Finished(Box<Finished>),
    /// The run could not start, or blew up.
    Failed(String),
}

/// Somewhere for [`run_to_sink`] to put messages.
///
/// Native sends them down a channel; the worker posts them to the page.
pub trait MessageSink {
    /// Hand over one message.
    fn send(&mut self, message: RunMessage);
    /// Has the caller asked for the run to stop?
    fn cancelled(&self) -> bool {
        false
    }
}

/// Bridge from the optimiser's [`ProgressSink`] to a [`MessageSink`].
struct Bridge<'a, S: MessageSink>(&'a mut S);

impl<S: MessageSink> ProgressSink for Bridge<'_, S> {
    fn on_iteration(&mut self, report: &IterationReport) {
        self.0.send(RunMessage::Progress(Progress::from(report)));
    }
    fn should_cancel(&self) -> bool {
        self.0.cancelled()
    }
}

/// Run the optimisation described by `setup`, reporting into `sink`.
///
/// Always finishes by sending exactly one [`RunMessage::Finished`] or
/// [`RunMessage::Failed`].
pub fn run_to_sink<S: MessageSink>(setup: &Setup, sink: &mut S) {
    let problems = setup.problems();
    if !problems.is_empty() {
        sink.send(RunMessage::Failed(problems.join("; ")));
        return;
    }
    match run_inner(setup, sink) {
        Ok(finished) => sink.send(RunMessage::Finished(Box::new(finished))),
        Err(e) => sink.send(RunMessage::Failed(e)),
    }
}

fn run_inner<S: MessageSink>(setup: &Setup, sink: &mut S) -> Result<Finished, String> {
    let interaction = setup.interaction().map_err(|e| e.to_string())?;
    let cartops = setup.cartops();
    let guess = random_pulse(setup.nslices, setup.nchannels(), setup.seed);

    let penalties = match setup.penalty {
        crate::setup::PenaltyChoice::None => vec![Penalty::None],
        other => vec![other.penalty()],
    };
    let tuning = Tuning {
        penalties,
        max_iter: setup.max_iter,
        splitset: setup.splitset.clone(),
        // Nothing here writes to a filesystem, and `store` is not available
        // on wasm at all.
        prop_cache: qoala::types::PropCache::Carry,
        output: Reporter::silent(),
        ..Default::default()
    };

    let mut bridge = Bridge(sink);
    let optimised = match &setup.target {
        Target::Transfer { from, to } => {
            let spec = StateTransfer {
                omega: setup.offsets_hz.clone(),
                initial: from.state(setup.nspins),
                target: to.state(setup.nspins),
                amplitudes: setup.amplitudes_hz.clone(),
                duration: setup.duration_s,
                increments: setup.nslices,
                interaction,
                cartops,
                init_pulse: Some(guess),
                seed: setup.seed,
                tuning,
            };
            state2state_xy_with_progress(spec, &mut bridge)
        }
        Target::Gate { .. } => {
            let target = setup
                .gate_target()
                .map_err(|e| e.to_string())?
                .ok_or_else(|| "gate target missing".to_string())?;
            let spec = GateSynthesis {
                omega: setup.offsets_hz.clone(),
                target,
                amplitudes: setup.amplitudes_hz.clone(),
                duration: setup.duration_s,
                increments: setup.nslices,
                interaction,
                cartops,
                init_pulse: Some(guess),
                seed: setup.seed,
                tuning,
            };
            universal_gate_xy_with_progress(spec, &mut bridge)
        }
    }
    .map_err(|e| e.to_string())?;

    let (rows, cols) = optimised.waveform.shape();
    let waveform = (0..rows)
        .map(|r| (0..cols).map(|c| optimised.waveform[(r, c)]).collect())
        .collect();

    let iterations = optimised.data.count.iter;
    let fidelity = optimised.data.fx_store[(iterations, 1)];
    let elapsed_s = optimised.data.timer[(iterations, 1)];

    Ok(Finished {
        waveform,
        exit_message: optimised.exitflag.message().to_string(),
        fidelity,
        elapsed_s: if elapsed_s.is_nan() { 0.0 } else { elapsed_s },
        iterations,
    })
}
