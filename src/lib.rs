//! # QOALA - Quantum Optimal control by Adaptive Low-cost Algorithm
//!
//! A Rust port of the QOALA MATLAB package (M. Foroozandeh, D. L. Goodwin,
//! P. Singh).  QOALA is a GRAPE optimal-control package for coupled two-level
//! systems whose central idea is that a pulse optimisation does not need an
//! accurate propagator until it is close to the optimum: it splits the
//! Hamiltonian into a single-spin part with a closed form and an interaction
//! part handled by a few precomputed propagators, then picks the splitting
//! order and Trotter number on the fly.
//!
//! ## Where to start
//!
//! * [`drivers`] - `state2state_xy` and `universal_gate_xy` wrap the whole
//!   pipeline for a chain of coupled two-level systems with an x and a y
//!   control on each.
//! * [`config`] - `ControlOptions` in, `ControlSystem` out, for anything the
//!   drivers do not cover.
//! * [`optim::newton::fmaxnewton`] - the optimiser itself.
//! * [`spinops`] - the spin operators, states, couplings and gates the
//!   examples are built from.
//! * [`gates`] - turn any Hilbert-space unitary into the Liouville-space
//!   superoperator a gate optimisation targets, plus a library of the usual
//!   one- and two-qubit gates.
//!
//! ## The numerical layers
//!
//! * [`objfun`] - the objective functions: [`objfun::qoala`] for the adaptive
//!   split-operator method, [`objfun::auxmat`] for the exact auxiliary-matrix
//!   reference.
//! * [`rodrigues`] - closed-form single-spin propagators and their control
//!   derivatives.
//! * [`splittings`] - operator splittings of orders 0, 1, 2, 3, 4 and 6, with
//!   Trotterisation.
//! * [`propagate`] - Pade, Taylor and Krylov matrix exponentials.
//! * [`penalty`] - waveform penalty terms with gradients and Hessians.
//! * [`linalg`] - the dense/sparse complex matrix type everything is built on.
//!
//! Differences from the MATLAB original are listed in `DEVIATIONS.md`.

pub mod config;
pub mod drivers;
pub mod error;
/// CSV writers the examples use; not available on `wasm32`, which has no
/// filesystem.
#[cfg(not(target_arch = "wasm32"))]
pub mod examples_io;
pub mod gates;
pub mod linalg;
pub mod objfun;
pub mod optim;
pub mod penalty;
pub mod prop_index;
pub mod propagate;
pub mod report;
pub mod rodrigues;
pub mod spinops;
pub mod splittings;
pub mod time;
pub mod types;
pub mod waveform;
