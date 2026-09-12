# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0]

### Added

- **Structured progress and cancellation.** `optim::ProgressSink` receives an
  `optim::IterationReport` - fidelity, penalty, total, check, gradient norm,
  step length, splitting order, Trotter number, elapsed seconds and the
  evaluation counters - for every row the console table prints.
  `optim::newton::fmaxnewton_with_progress` takes such a sink and stops the run
  when it asks to, returning `ExitFlag::Cancelled` and the waveform reached so
  far. `fmaxnewton` is unchanged and delegates to it.
- **Gate library.** `gates::superoperator_from_unitary` turns any
  Hilbert-space unitary into the Liouville-space superoperator for
  `rho -> U rho U^dagger`, in the same normalised product spherical-tensor
  basis the rest of the crate uses; it reproduces `spinops::swap_gate`, which
  is what pins the basis ordering and phases. On top of it: `SWAP`, `iSWAP`,
  `sqrt(SWAP)`, `CNOT`, `CZ`, `X`, `Y`, `Z`, `H` and an arbitrary-axis
  rotation, helpers to embed one- and two-qubit gates on chosen qubits of an
  n-qubit register, and a `Gate` menu type for a user interface to offer.
- **WebAssembly support.** The crate builds for `wasm32-unknown-unknown`. A
  clock shim (`time`) routes every timing call through `web-time` on that
  target, `getrandom`'s JS backend supplies entropy for the random starting
  waveform, and the filesystem paths - `examples_io`, `Reporter::file`,
  `ReportSink::File`, the propagator cache and `PropCache::Store` - are gated
  off it. `PropCache::Store` returns `QoalaError::NotImplemented` on wasm
  rather than failing to compile; `Carry` and `Calc` work unchanged.

- **A graphical front end**, in `gui/`: a workspace member, not part of the
  published crate. The same source builds as a desktop application
  (`cargo run -p qoala-gui`) and as a static web page (`trunk build --release
  --config gui/Trunk.toml`) that runs entirely in the visitor's browser with
  no server. It ships the five examples as presets, plots infidelity and the
  waveform, shows the splitting order and Trotter number live, estimates the
  runtime before you press Run, exports CSV and JSON, and
  encodes the whole setup in the URL fragment. The browser build runs the
  optimisation in an ordinary Web Worker, so the tab stays responsive without
  needing cross-origin isolation.
- **`drivers::state2state_xy_with_progress`** and
  **`drivers::universal_gate_xy_with_progress`**, plus
  `drivers::state_transfer_system` and `drivers::gate_synthesis_system`, which
  return the configured `ControlSystem` and starting waveform without running
  anything.

### Changed

- The console iteration table is now `optim::newton::TableSink`, one
  `ProgressSink` among others. The output is byte-for-byte what it was.
- `ExitFlag` is `#[non_exhaustive]` and gained a `Cancelled` variant. Matches
  on it outside the crate now need a wildcard arm.
- `optim::Counters` derives `PartialEq` and `Eq`.

## [1.0.0]

First release: a complete Rust port of the QOALA MATLAB package (v1.0) by
M. Foroozandeh, D. L. Goodwin and P. Singh.

### Added

- **Objective functions.** `objfun::qoala`, the adaptive split-operator GRAPE
  method for state transfer and gate synthesis, and `objfun::auxmat`, the exact
  auxiliary-matrix reference it is measured against.
- **Optimiser.** LBFGS and Newton-Raphson maximisation with rational-function
  Hessian regularisation, and a Wolfe-condition line search with cubic
  interpolation.
- **Numerics.** Euler-Rodrigues single-spin propagators and their control
  derivatives; operator splittings of orders 0, 1, 2, 3, 4 and 6 with
  Trotterisation; Pade, Taylor and Krylov matrix exponentials.
- **Configuration.** A typed replacement for `optimconset`, reproducing the
  MATLAB's defaults, resolutions and printed report.
- **Penalties.** `NS`, `DNS`, `SNS`, `SNSA`, `SNSM` and `ADIAB`, with gradients
  and, where they exist, Hessians.
- **Drivers and operators.** `state2state_xy` and `universal_gate_xy`, plus
  `spinops` for the spin operators, states, couplings and gates the examples
  are built from.
- **Examples.** All five simple MATLAB example scripts, and a single
  `paper_benchmarks` binary covering the six benchmark scripts.
- **Tests.** 81 tests: analytic gradients against finite differences, splittings
  against their order conditions, propagators against independently computed
  matrix exponentials, coupling superoperators against the literal matrices in
  the MATLAB scripts, and end-to-end convergence.

### Changed from the MATLAB original

Five corrections and a number of design changes, all listed with reasons in
[`DEVIATIONS.md`](DEVIATIONS.md). In summary: the backward pass now walks the
split sequence in reverse (exact at every splitting order, not just the
symmetric ones); the Lie-Trotter repeat sequence no longer indexes past the end
of its coefficient arrays; the line search tests the step it is actually
trialling; the `NS`, `DNS` and `ADIAB` penalties work in the waveform layout
QOALA actually uses; and the SWAP construction applies its rotations in the
right order.
