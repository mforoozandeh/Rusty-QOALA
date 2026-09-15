# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0]

### Added

- **ESCALADE**, in `escalade`: a port of the ESCALADE MATLAB code (M.
  Foroozandeh, P. Singh) for broadband single-spin pulses. `escalade::escalade`
  and `escalade_with_progress` optimise one x/y pulse across a band of offsets
  and, optionally, a spread of field strengths, with the analytic gradient and
  Hessian of `gradhess_vectorized_B1_parallel.m`. Progress goes to the same
  `optim::ProgressSink` as QOALA's. `escalade::profile` gives a pulse's
  excitation profile, its map of Iy over offset and field, and its phase
  sensitivity to the field.
- **Examples** `escalade_grad_vs_hess` and `escalade_b1`, porting the MATLAB's
  two `test_runs/` scripts.
- **The application runs ESCALADE.** An algorithm switch, ESCALADE presets
  and editor, and a choice of waveform, offset profile or B1-robustness view
  for the result. Links, stored sessions and exported setups from earlier
  versions still load.

### Changed

- New dependencies `argmin` and `argmin-math`, which supply ESCALADE's
  optimiser in place of MATLAB's `fmincon`.
- **Minimum supported Rust version is now 1.87** (was 1.82), because argmin
  0.11 uses `is_multiple_of`, stabilised in 1.87. The application's minimum
  rises from 1.85 to match.
- ESCALADE limits the field amplitude, `sqrt(x^2 + y^2)`, with a penalty,
  where the MATLAB bounds each quadrature. See `DEVIATIONS.md`.
- **`GrapeXyCost` refuses what it would have ignored**: more than one row of
  power levels, or more than one initial and target, now return
  `QoalaError::NotImplemented`. Both used to optimise the first member alone.
- **`optimconset` refuses a non-uniform `pulse_dt` for the split-operator
  objectives** (`QoalaError::NotImplemented`); it used to warn and build the
  interaction propagators for the average slice. Slices equal to 1e-12 count
  as uniform. The auxiliary-matrix objectives are unaffected.
- **`optimconset` checks numbers and shapes before parsing**: empty or
  non-finite time grids, non-positive or non-finite power levels, empty
  power-level lists, non-finite control operators, and drift, state,
  propagator and `spin_control` shapes that disagree with the control
  operators are errors rather than panics or later failures.
- The propagator cache format is now version 2 and records the inputs each
  file was built from. Version 1 files are rebuilt on first use.
  `splittings::write_propagator_cache` and `read_propagator_cache` are no
  longer public.

### Fixed

- Krylov propagation in Hilbert space applied `U rho` instead of
  `U rho U^dagger`. It is the default `step_method`, so value-only
  auxiliary-matrix evaluations and fidelity checks in Hilbert space disagreed
  with the gradient path.
- The propagator cache (`prop_cache = 'store'`) reused a file for the same
  Trotter number and order even when the time step or interaction had
  changed. Files are also now written atomically.
- After a failed line search the recorded fidelity, check and diagnostics
  came from the last rejected trial rather than the waveform that was kept.
- The norms propagate a NaN instead of losing it in an `f64::max` reduction,
  which returns the other operand when one side is NaN and so let a matrix
  holding a NaN report a finite norm. `CMat::norm_1` and `norm_2_dense` now
  return NaN for such a matrix, and infinity for one that is merely infinite.
  `expm_pade` checks its entries rather than its norm, and `expm_taylor` and
  `krylov_step` refuse a non-finite generator: Krylov used to reduce to zero
  sub-steps and return the state untouched, as though it had propagated.
- Non-finite numbers no longer travel on through the optimiser. An objective
  value, gradient or Hessian that is not finite is refused at the point it was
  evaluated; the Hessian regularisation errors rather than shifting the
  spectrum by a non-finite eigenvalue; and `cond_2_symmetric` reports an
  infinite condition number for a non-finite matrix, where a NaN used to lose
  every comparison and report the best conditioning there is.
- ESCALADE's `max_amplitude` propagates a NaN instead of reducing it away,
  which made a NaN pulse look like a zero-amplitude one, comfortably inside
  the limit, and a starting pulse holding a non-finite point is now refused.
- The objective functions return `QoalaError::Dimension` for a waveform whose
  slice count differs from the time grid, and the drivers check `init_pulse`
  against the pulse shape, instead of panicking.
- The application no longer panics on a stored session, link or worker
  message whose arrays do not match its spin and pair counts; a link like
  that is reported on screen and not loaded.

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

- `ExitFlag` is `#[non_exhaustive]` and gained a `Cancelled` variant. Matches
  on it outside the crate now need a wildcard arm.
- The console iteration table is now `optim::newton::TableSink`, one
  `ProgressSink` among others. The output is byte-for-byte what it was.
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
