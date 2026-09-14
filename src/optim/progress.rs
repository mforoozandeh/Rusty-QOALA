//! Structured progress reporting and cancellation.
//!
//! The MATLAB original's only channel out of a running optimisation is the
//! `fprintf` iteration table, which is fine for a terminal and useless to
//! anything that wants to plot the numbers or stop the run.  This module adds
//! a data channel alongside the text one: [`fmaxnewton_with_progress`] hands
//! every iteration to a [`ProgressSink`], and asks it before each iteration
//! whether the run should stop.
//!
//! [`fmaxnewton_with_progress`]: crate::optim::newton::fmaxnewton_with_progress

use super::Counters;

/// One row of the optimiser's progress, as numbers rather than a formatted
/// line.
///
/// The fields are exactly what the console table prints, so a sink can
/// reproduce the table or plot any of it.
#[derive(Debug, Clone, PartialEq)]
pub struct IterationReport {
    /// Optimiser iteration index; `0` is the initial evaluation, before the
    /// first line search.
    pub iteration: usize,
    /// The objective fidelity on its own, `data.fx_sep_pen[0]`.
    pub fidelity: f64,
    /// Summed penalty terms, negated as the table prints them.
    pub penalty: f64,
    /// The combined objective, fidelity minus penalties.
    pub total: f64,
    /// The independent fidelity check, or the fidelity again when no check is
    /// configured.
    pub fidelity_check: f64,
    /// Two-norm of the gradient, or `0.0` when this row has no gradient.
    pub gradient_norm: f64,
    /// Line-search step length; `None` on the initial evaluation.
    pub alpha: Option<f64>,
    /// Splitting order currently chosen by the adaptive step.
    pub split_order: usize,
    /// Trotter number currently chosen by the adaptive step.
    pub trotter_number: usize,
    /// Wall-clock seconds since the optimisation started.
    pub elapsed_s: f64,
    /// Evaluation counters as of this row.
    pub counters: Counters,
}

/// Something that watches an optimisation and can stop it.
pub trait ProgressSink {
    /// Called once per row of the iteration table.
    fn on_iteration(&mut self, report: &IterationReport);

    /// Asked at the top of every iteration.  Returning `true` stops the run
    /// with [`crate::types::ExitFlag::Cancelled`], keeping the waveform
    /// reached so far.
    fn should_cancel(&self) -> bool {
        false
    }
}

/// A sink that does nothing, so [`crate::optim::newton::fmaxnewton`] can
/// delegate to the progress-aware entry point.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProgress;

impl ProgressSink for NoProgress {
    fn on_iteration(&mut self, _report: &IterationReport) {}
}

/// A sink that keeps every report, for tests and for callers that only want
/// the trajectory at the end.
#[derive(Debug, Clone, Default)]
pub struct RecordingSink {
    /// Every report received, in order.
    pub reports: Vec<IterationReport>,
}

impl ProgressSink for RecordingSink {
    fn on_iteration(&mut self, report: &IterationReport) {
        self.reports.push(report.clone());
    }
}
