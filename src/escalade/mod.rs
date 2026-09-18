//! # ESCALADE
//!
//! Broadband single-qubit pulse design: one x/y pulse that takes every spin
//! across a band of resonance offsets - and, optionally, across a spread of
//! radio-frequency field strengths - from an initial state to a target state,
//! or to a target propagator: a universal rotation, which turns every state
//! alike.
//!
//! A Rust port of the ESCALADE MATLAB code by M. Foroozandeh and P. Singh.
//! QOALA extends the same closed-form single-spin propagators to coupled
//! spins; ESCALADE is the uncoupled case, where they are all there is, and
//! so the analytic Hessian is cheap enough to use.
//!
//! The method: M. Foroozandeh and P. Singh, "Optimal control of spins by
//! analytical Lie algebraic derivatives", *Automatica* 129, 109611 (2021),
//! <https://doi.org/10.1016/j.automatica.2021.109611>.  Its extension to
//! coupled spins, QOALA: D. L. Goodwin, P. Singh and M. Foroozandeh,
//! "Adaptive optimal control of entangled qubits", *Science Advances* 8(49),
//! eabq4244 (2022), <https://doi.org/10.1126/sciadv.abq4244>.
//!
//! * [`escalade`] and [`escalade_with_progress`] - describe the problem, get
//!   an optimised pulse back.
//! * [`settings`] - the problem description and its defaults.
//! * [`solve`] - the optimiser, and how the amplitude limit is kept.
//! * [`objective`] - fidelity, gradient and Hessian.
//! * [`propagators`] - the single-spin propagators and their derivatives.
//! * [`profile`] - what a pulse does across offsets and field strengths.

pub mod objective;
pub mod profile;
pub mod propagators;
pub mod settings;
pub mod solve;

pub use settings::{magnetisation, rotation, Escalade, Goal, Settings, States};
pub use solve::{escalade, escalade_with_progress, optimise, Optimised};
