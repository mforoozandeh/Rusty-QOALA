# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0]

### Added

- **ESCALADE optimises universal rotations.** `Goal::Rotation(W)` asks for
  every spin to end with the propagator `W`, whatever state it starts in:
  x to x, y to z and z to -y for `rotation([1.0, 0.0, 0.0], FRAC_PI_2)`, the
  new helper building `W` from an axis and an angle. The fidelity is the
  propagator overlap `Re tr(W^dagger U) / 2`, averaged over spins and
  weighted over fields as before, with its gradient and exact Hessian. It is
  what a rotation run reports and stops on, and it is stricter than the
  average over the three axes: 0.999 is 0.9973 there.

  It tells `U` from `-U`, which turn every axis alike. Scoring the three
  transfers x, y and z instead cannot, and lets parts of the band settle on
  opposite signs: the spins between them are then held 180 degrees from the
  target, where that score's gradient vanishes. A 90-degree rotation across
  30 kHz with a 17 kHz field in 200 microseconds stalled at 0.638 that way
  from every start tried, and reaches 0.9993 in 2000 L-BFGS iterations
  scored on the propagator.
- **The application designs universal rotations.** Single-qubit problems
  have a goal: State transfer, as before, or Universal rotation, set by where
  +x, +y and +z go. Anything that is not a proper rotation is refused with a
  message. The rotation is optimised on its propagator, taking the sign
  that turns by at most 180 degrees. For a rotation, the offset profile and
  B1 views have a menu choosing which axis they follow. A new preset rotates
  by 90 degrees about x across 10 kHz.

### Changed

- **Breaking: `Escalade` takes `goal: Goal`** in place of `initial` and
  `target`. Replace `initial: a, target: b` with
  `goal: Goal::Transfer { initial: a, target: b }`. `Settings::initial` and
  `Settings::target` are likewise `Settings::goal`, a `Goal<Vec<Op2>>` holding
  each spin's states. State transfers give bit-for-bit the same results as
  1.3.0.
- **Breaking: `escalade::profile` takes a starting magnetisation.**
  `final_magnetisation`, `offset_profile` and `phase_sensitivity` take a
  `start`, and `b1_map` a `start` and an `along`; pass
  `Magnetisation::new(0.0, 0.0, 1.0)` and, for the map,
  `Magnetisation::new(0.0, 1.0, 0.0)` for what 1.3.0 computed. `B1Map::iy` is
  now `B1Map::values`. `Magnetisation` gains `new` and `dot`.
- **The application's B1 map shows the final magnetisation along the target**
  rather than Iy, so that red means the transfer succeeded whatever the
  target. For the z to -y presets this is the old map with its sign flipped.
  Links, stored sessions and exported setups from earlier versions still
  load, as state transfers.

## [1.3.0]

### Added

- **ESCALADE runs on every core.** With the new `parallel` feature, on by
  default, each evaluation of the objective, its gradient and its Hessian
  cuts its (field, spin) pairs into up to 64 pieces and runs the pieces on
  rayon's thread pool. Long pulses get fewer pieces, so that the dense
  Hessian accumulators stay within 256 MB together. Sums are grouped by the
  size of the problem rather than the thread count, so a run gives
  bit-for-bit the same pulse on one thread, on many, or with the feature off.
  The B1-compensated `escalade_b1` run goes from 38 s to 4.8 s on a 12-core
  machine. Where it applies:
  - library and command line, native: every core by default;
    `default-features = false` (`--no-default-features` in this repository)
    runs on one core;
  - desktop application: every core;
  - browser application (`wasm32`): one core, even when the page is
    cross-origin isolated.

  QOALA's coupled-spin optimisation is unchanged, on one core everywhere.
- **Example** `escalade_parallel`: times the objective at every order on 1,
  2, 4, ... threads, then a whole optimisation on one thread and on all of
  them, and fails if any result differs.

### Changed

- New optional dependency `rayon`, behind the default `parallel` feature and
  never built for `wasm32`. `default-features = false` drops it.
- The ESCALADE objective sums its work items in a fixed tree rather than left
  to right, so its last digits, and after many iterations the path of an
  optimisation, differ slightly from 1.2.0: the 1000-iteration B1-compensated
  run of `escalade_b1` now ends at fidelity 0.98875 rather than 0.98924.
- **Small ESCALADE problems are slightly slower on one core** - in the
  browser, with `--no-default-features`, or on a single rayon thread. The
  pieces are fixed so that every build groups the sums the same way, so a
  single field of 51 spins becomes 51 one-spin pieces, each allocating,
  zeroing and merging its own Hessian accumulator where 1.2.0 used one. That
  costs about 5%: 25.8 ms against 24.5 ms for 20 iterations of the broadband
  preset. With more fields each piece holds more spins and the cost spreads
  out; on several cores the speed-up outweighs it.
- **The application says where single-qubit optimisation runs.** With
  Single-qubit selected, the browser build notes that it runs on one core and
  the desktop build spreads it over every core; the desktop build gives the
  size of rayon's thread pool, which follows `RAYON_NUM_THREADS`. This
  replaces the warning that a page without cross-origin isolation runs on one
  core, which wrongly implied that isolation would give more.
- **The Run-button estimate for single-qubit runs accounts for cores** on the
  desktop.
  Its constants are for one core, as before; the desktop estimate is divided
  by rayon's thread count times an efficiency of 0.6. The browser estimate is
  unchanged. Calibrate with `RAYON_NUM_THREADS=1`.
- **`deploy/README.md` and the header files say that cross-origin isolation
  does not speed up the current web build.**
- **The application is called Rusty-QOALA** - in the window title, the
  browser tab and the loading page - **and names problems, not methods.**
  The switch reads Single-qubit and Multi-qubit instead of ESCALADE and
  QOALA, and the interface says qubits where it said spins, preset names
  included. Links, stored sessions and exported setups from earlier versions
  still load.
- **The application works in SI units throughout**: durations in seconds
  (they were ms for multi-qubit pulses and µs for single-qubit ones),
  bandwidth and field amplitude in Hz (they were kHz), and the plot axes to
  match. Numbers can be typed in scientific notation, `2e6` or `3e-6`, and
  are shown that way below 0.001 and from 100000 up. Durations accept
  1e-12 to 1e3 s and field amplitudes up to 1e12 Hz, so nanosecond pulses
  can be set up. Dragging a value moves it by half a percent of itself.
  Stored setups were already in SI, so nothing needs converting.

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
