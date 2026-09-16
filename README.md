# qoala

[![CI](https://github.com/mforoozandeh/Rusty-QOALA/actions/workflows/ci.yml/badge.svg)](https://github.com/mforoozandeh/Rusty-QOALA/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/qoala.svg)](https://crates.io/crates/qoala)
[![Documentation](https://docs.rs/qoala/badge.svg)](https://docs.rs/qoala)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Q**uantum **O**ptimal control by **A**daptive **L**ow-cost **A**lgorithm — a
Rust port of the QOALA MATLAB package by M. Foroozandeh, D. L. Goodwin and
P. Singh.

QOALA is a GRAPE optimal-control package for coupled two-level systems. Its
idea is that most of the work in a pulse optimisation is wasted: early
iterations are nowhere near the optimum, so propagating the system to machine
precision to evaluate them is pointless. QOALA splits the Hamiltonian into a
single-spin part, which has a closed form and needs no matrix exponential at
all, and an interaction part handled by a small set of precomputed propagators —
then chooses the splitting order and Trotter number on the fly, so the
propagator is only ever as accurate as the optimiser currently needs.

On the paper's own two-spin benchmark this port reaches an infidelity of
3e-8 in 0.11 s of objective time, against 2.3 s for the exact
auxiliary-matrix method: a **21x speed-up** at the same accuracy.

The crate also ports **ESCALADE**, by M. Foroozandeh and P. Singh: the
uncoupled, single-spin method QOALA extends to coupled spins. With nothing
coupling the spins, the closed-form propagators are all there is, so ESCALADE
can afford the exact Hessian. It designs one pulse for a whole band of
resonance offsets, and optionally for a spread of field strengths as well,
which is how a pulse is made robust to B1 inhomogeneity. See
[ESCALADE](#escalade).

## Installation

```toml
[dependencies]
qoala = "1.1"
```

There is also a graphical front end - the same code as a desktop application
or as a web page that runs entirely in the browser. See
[The application](#the-application).

Or from git, until it is published:

```toml
[dependencies]
qoala = { git = "https://github.com/mforoozandeh/Rusty-QOALA" }
```

## Quick start

```rust
use qoala::drivers::{state2state_xy, StateTransfer, Tuning};
use qoala::spinops;
use num_complex::Complex64;

// Two spins, weakly coupled at 140 Hz, with an x/y control pair on each.
let interaction = spinops::zz_coupling(2, 0, 1)?
    .scale(Complex64::new(2.0 * std::f64::consts::PI * 140.0, 0.0));

let optimised = state2state_xy(StateTransfer {
    omega: vec![0.0, 0.0],                       // offsets, Hz
    initial: spinops::z_state(2, 0),             // z-magnetisation on spin 1
    target:  spinops::z_state(2, 1),             // ... moved onto spin 2
    amplitudes: vec![1000.0, 1000.0],            // maximum control power, Hz
    duration: 0.01,                              // seconds
    increments: 50,                              // time slices
    interaction,
    cartops: spinops::one_pair_per_spin(2),
    init_pulse: None,                            // random start
    seed: Some(1),
    tuning: Tuning::default(),
})?;

println!("{:?}", optimised.waveform.shape());    // (50, 4)
# Ok::<(), qoala::error::QoalaError>(())
```

For anything the drivers do not cover, build a `ControlOptions`, run it through
`optimconset`, and hand the result to `fmaxnewton` — the same two steps as the
MATLAB.

## Examples

```text
cargo run --release --example z2z_2spin_1        # z-to-z transfer, weak coupling
cargo run --release --example z2z_2spin_2        # strong coupling, one shared control pair
cargo run --release --example z2z_3spin_1        # three-spin chain
cargo run --release --example swap_2spin_1       # SWAP gate
cargo run --release --example swap_3spin_1       # SWAP across a three-spin chain
cargo run --release --example paper_benchmarks -- state-2spin-hetero
cargo run --release --example escalade_grad_vs_hess   # ESCALADE, see below
```

Each writes its optimised pulse to CSV and prints a fidelity computed by an
*independent* exact propagation of the pulse it produced — not by the objective
function that produced it. Current results, on a random starting pulse:

| example | fidelity (independently checked) |
|---|---|
| `z2z_2spin_1` | 99.999999 % |
| `z2z_2spin_2` | 99.999981 % |
| `z2z_3spin_1` | 99.999989 % |
| `swap_2spin_1` | 99.617 % (100-iteration budget, as in the MATLAB) |
| `swap_3spin_1` | 99.986 % |

`paper_benchmarks` covers all six scripts in the MATLAB's
`examples/qoala_paper_results/`, which differ only in the spin system and
target. Pass `state-2spin-hetero`, `state-2spin-homo`, `state-3spin-hetero`,
`state-4spin-mixed`, `gate-2spin-hetero` or `gate-3spin-hetero`, plus optional
`--runs`, `--steps` and `--iters`.

## ESCALADE

```rust
use qoala::escalade::{escalade, magnetisation, Escalade, States};

// z to -y across 20 kHz, with a 17 kHz field, in 100 microseconds.
let optimised = escalade(&Escalade {
    nspins: 51,                                  // spins spread across the band
    sw: 20000.0,                                 // bandwidth, Hz
    rf: vec![17000.0],                           // field, Hz; several values for B1 robustness
    tau_p: 100e-6,                               // seconds
    np_pulse: 50,                                // pulse points
    initial: States::Single(magnetisation(0.0, 0.0, 1.0)),
    target: States::Single(magnetisation(0.0, -1.0, 0.0)),
    use_hessian: true,                           // Newton trust region on the exact Hessian
    seed: Some(1),
    ..Default::default()
})?;

println!("{:.4} after {} iterations", optimised.fidelity, optimised.counters.iter);
# Ok::<(), qoala::error::QoalaError>(())
```

```text
cargo run --release --example escalade_grad_vs_hess   # Hessian against gradient only
cargo run --release --example escalade_b1             # B1-sensitive against B1-compensated
cargo run --release --example escalade_parallel       # one thread against every core
```

These port the two scripts in the MATLAB's `test_runs/`. On the first, both
optimisers reach 99 % in under a tenth of a second. The second writes the
pulse, the excitation profile, the map of Iy over offset and field, and the
dphi/dB1 curve as CSV; optimising over 0.8 to 1.2 times the field lifts the
worst-case fidelity across that range from 0.84 to 0.95.

`escalade::profile` computes the offset profile, the B1 map and the phase
sensitivity of any pulse, from the same propagators the optimiser uses.

**Threads.** One spin at one field - a work item - is a time-ordered product
over the pulse, but the work items are independent of each other. The `parallel` feature, on by
default, spreads them over [rayon](https://docs.rs/rayon)'s thread pool - one
thread per core, or `RAYON_NUM_THREADS`. Where that applies:

- **Library and command line, native** - on every core by default.
  `default-features = false` in a dependency, or `--no-default-features` in
  this repository, drops rayon and runs on one core.
- **Desktop application** (`cargo run -p qoala-gui --release`) - on every
  core.
- **Browser application** - on one core, even when the page is cross-origin
  isolated. The WebAssembly build has no threads.

Only ESCALADE is spread over cores; QOALA's coupled-spin optimisation runs on
one core everywhere. The sums are grouped by the size of the problem and
never by the thread count, so a run returns the same pulse to the last bit on
one core or many. That fixed grouping has a small cost on one core: a single
field of 51 spins runs about 5% slower there than a plain loop, since each
spin becomes a piece with its own Hessian accumulator.

`escalade_parallel` times the B1-compensated problem both ways. On an idle
12-core Apple M2 Max (8 performance, 4 efficiency), a Hessian evaluation of
its 1581 work items (51 spins × 31 fields) drops from 41 ms to 4.8 ms, and 200 Newton iterations from
7.1 s to 0.9 s, with identical pulses. `escalade_b1` falls from 38 s with the
original sequential code to 4.8 s.

| MATLAB | Rust |
|---|---|
| `main/ESCALADE.m` | `escalade/solve.rs` |
| `main/readDefaults.m`, `main/readSettings.m` | `escalade/settings.rs` |
| `main/gradhess_vectorized_B1_parallel.m`, `main/gradhess_vectorized.m` | `escalade/objective.rs` |
| `main/derivatives_vectorized.m`, `main/LL_vectorized.m` | `escalade/propagators.rs` |
| `visualisation/ESCALADE_pulse_sim.m`, `visualisation/ESCALADE_Bloch_B1.m` | `escalade/profile.rs` |

Two things work differently, both described in
[`DEVIATIONS.md`](DEVIATIONS.md). `fmincon` is replaced by
[argmin](https://argmin-rs.org): a trust region on the Hessian, or L-BFGS on
the gradient. And the amplitude limit is on the field itself,
`sqrt(x^2 + y^2) <= rf`, kept by a penalty, where the MATLAB bounds each
quadrature separately.

## Repository layout

```text
src/
  config.rs       parser: ControlOptions -> ControlSystem
  drivers.rs      state2state_xy, universal_gate_xy, the GRAPE cost function
  objfun/         the objective functions
    qoala.rs        adaptive split-operator method
    auxmat.rs       exact auxiliary-matrix reference
  optim/          optimiser
    newton.rs       LBFGS and Newton-Raphson
    linesearch.rs   Wolfe-condition line search
  escalade/       ESCALADE: uncoupled spins across a band
    solve.rs        the driver, and argmin's trust region and L-BFGS
    objective.rs    fidelity, gradient and Hessian over offsets and fields
    propagators.rs  SU(2) propagators and their first and second derivatives
    profile.rs      offset profiles, B1 maps, phase sensitivity
  rodrigues.rs    closed-form single-spin propagators and derivatives
  splittings.rs   operator splittings, orders 0-6, with Trotterisation
  propagate.rs    Pade, Taylor and Krylov matrix exponentials
  penalty.rs      waveform penalties
  prop_index.rs   sparse propagator index tables
  spinops.rs      spin operators, states, couplings, gates
  gates.rs        Hilbert unitaries -> Liouville superoperators, gate library
  linalg.rs       dense/sparse complex matrix type
  time.rs         clock shim, so the crate builds for wasm32
  types.rs        typed options; report.rs, error.rs, waveform.rs
examples/         the MATLAB example scripts, plus the benchmark harness
tests/            gradients, accuracy, end-to-end optimisation
gui/              the application: desktop and browser, same source
  src/setup.rs      the QOALA problem description, serde-serialisable
  src/escalade.rs   the ESCALADE problem description, and its analysis
  src/problem.rs    either of the two, as stored, shared and run
  src/presets.rs    the examples above, as one-click setups
  src/run.rs        the optimisation, and the message stream out of it
  src/app.rs        panels and plots
  src/export.rs     CSV and JSON export
  src/runner_*.rs   background thread natively, Web Worker in the browser
deploy/           cross-origin isolation headers, one file per host
```

## The application

`gui/` is a front end over the same library: describe a spin system, pick a
target, watch it converge, take the pulse away as a file. It runs two ways
from one source.

**Desktop.**

```bash
cargo run -p qoala-gui --release
```

**Browser.** Everything runs in the visitor's browser. There is no server, no
account and no upload; hosting is static files.

```bash
cargo install --locked trunk
trunk serve --config gui/Trunk.toml
```

Then open <http://localhost:8080>.

To build for deployment, `trunk build --release --config gui/Trunk.toml` and
serve `gui/dist` from anywhere - including `python3 -m http.server`. See
[`deploy/README.md`](deploy/README.md).

What it gives you:

- Both algorithms behind one switch. ESCALADE edits a band, a field (with an
  optional B1 spread) and a pulse; QOALA edits a coupled spin system.
- A visual editor for QOALA's control map - which channel drives which spin -
  which is the least obvious part of the library's API.
- Presets for the five QOALA examples and the two ESCALADE runs of
  `test_escalade_visual.m`; the two-spin transfer is loaded on startup, so
  the first click gives a converged pulse in about a second.
- For an ESCALADE pulse, a choice of three views under the convergence plot:
  the waveform, the excitation profile across offsets, and B1 robustness - a
  map of Iy over offset and field, beside the dphi/dB1 curve.
- Infidelity against iteration on a log axis, the waveform as stairs in Hz,
  and the splitting order and Trotter number live, climbing as the fidelity
  improves.
- A runtime estimate on the Run button, and a Cancel that works.
- Exports: the waveform as CSV in Hz against seconds, and the whole setup as
  JSON.
- The whole setup encoded in the URL fragment, so a link reproduces it
  exactly and can be sent to somebody else. Nothing is stored anywhere but
  the link and the visitor's own browser.

Three limits worth knowing. The browser build runs on one core, cross-origin
isolated or not, while the desktop build spreads ESCALADE over every core -
several times faster for a B1-compensated pulse. QOALA runs on one core in
both. Four-spin systems are dimension 256 with dense 256x256 propagators per
slice; the browser build refuses them and points at the desktop one. And
cancelling in the browser terminates the Web Worker, so the convergence curve
is kept but the partial waveform is not - natively, cancellation is
cooperative and the waveform comes back.

## How the code maps to the MATLAB

| MATLAB | Rust |
|---|---|
| `kernel/optimconset.m` | `config.rs` — `ControlOptions` in, `ControlSystem` out |
| `kernel/fmaxnewton.m` | `optim/newton.rs` |
| `kernel/fmaxlinesearch.m` | `optim/linesearch.rs` |
| `kernel/objective.m` | `optim/mod.rs` — `objective`, `OptData` |
| `kernel/penalty_fun.m` | `penalty.rs` |
| `kernel/propagate.m` | `propagate.rs` |
| `kernel/obj_fun/grape_{state,ugate}_qoala.m` | `objfun/qoala.rs` |
| `kernel/obj_fun/grape_{state,ugate}_auxmat.m` | `objfun/auxmat.rs` |
| `utilities/rodrigues.m` | `rodrigues.rs` |
| `utilities/operator_splittings.m` | `splittings.rs` |
| `utilities/propagator_ind.m` | `prop_index.rs` |
| `utilities/waveform_fidelity.m` | `waveform.rs` |
| `utilities/{state2state,universal_gate}_xy.m` | `drivers.rs` |
| operator construction in the example scripts | `spinops.rs` |
| — | `types.rs`, `error.rs`, `linalg.rs`, `report.rs` |

The two `grape_state_*` / `grape_ugate_*` pairs are the same algorithm with
different fidelity functionals, so each is one module parameterised by
`FidelityKind` rather than two near-identical files.

## Design notes

**Options are typed.** MATLAB's magic strings become enums, so `'lbgfs'` is a
compile error rather than a run-time surprise, and `ctrl_param` becomes a
struct. The parser still prints the same banner, the same resolved settings and
the same `WARNING`/`RESOLVE` commentary.

**Sparsity is explicit.** `CMat` is dense or CSR, and `densify_if` reproduces
the "more than `sparsity * dim^2` non-zeros means go dense" rule the objective
functions apply to every propagator.

**Propagator derivatives stay factored.** The MATLAB builds `dP{k}` as an
`nsteps x dim^2` array — 65536 columns wide for four spins.
`ControlDerivative` keeps the `D` block and the operators and forms the
`dim x dim` matrix on demand. Same arithmetic, much less memory.

**`nargout` becomes an argument.** `EvalOrder::{Value, Gradient}` and
`PenaltyOrder::{Value, Gradient, Hessian}` say what the caller wants.

## Verification

`cargo test` runs 121 tests. There is no MATLAB in the loop; every check is
self-contained and tests the mathematics rather than a stored output.

- **Gradients against finite differences.** Every objective function's analytic
  gradient is checked against a central difference of its own fidelity —
  split-operator and auxiliary-matrix, state transfer and gate synthesis, at
  splitting orders 0, 1, 2, 3, 4 and 6 with Trotter numbers 1 and 2, with and
  without penalties. Every penalty's gradient *and Hessian* likewise.
- **Propagators against `expm`.** The Euler-Rodrigues elements are checked
  against an independently computed matrix exponential of the same generator,
  and its derivative against a finite difference of that exponential. Padé,
  Taylor and Krylov are checked against each other and against closed-form
  Pauli rotations.
- **Splittings against their order conditions.** Each splitting is assembled
  and compared with the exact exponential at two step sizes; the observed
  convergence rate must match the advertised order for orders 1, 2, 3, 4 and 6.
  Trotterisation must improve second-order accuracy quadratically.
- **Operators against the MATLAB literals.** The `zz` and isotropic coupling
  superoperators built by `spinops` are compared entry by entry with the 16x16
  matrices written out longhand in `z2z_2spin_1.m` and `z2z_2spin_2.m`. The
  SWAP gate is checked by confirming it actually swaps z-magnetisation — which
  caught a real ordering bug that a fidelity number alone would have hidden.
- **Convergence to the exact answer.** Raising the splitting order, the Trotter
  number or the time resolution must monotonically close the gap to the
  auxiliary-matrix result; a well-resolved sixth-order splitting must agree to
  1e-10 in fidelity and 1e-7 in gradient.
- **End to end.** Optimisations must converge above 99.9 %, the adaptive ladder
  must climb as the infidelity falls, and adaptive and fixed fourth-order runs
  must land within 1 % of each other.
- **ESCALADE.** The SU(2) propagators are checked against `expm`, their first
  and second control derivatives against finite differences on both sides of
  the zero-field series switch, and the full Hessian of the objective against
  a finite difference of its gradient over several spins and two weighted
  fields. Runs with and without the Hessian must reach 99 %, with a fidelity
  that matches an independent measurement of the magnetisation to 1e-9; a
  pulse started at twice the amplitude limit must be brought back to it; and
  optimising over a field spread must improve the worst case across it.

## Differences from the MATLAB

Five corrections (an inexact backward pass for asymmetric splittings, an
unreachable Lie-Trotter branch, a stale variable in the line search, a
transposed penalty layout, and a gate-construction ordering), plus the features
the MATLAB parses but never implements. All of them are listed, with reasons,
in [`DEVIATIONS.md`](DEVIATIONS.md).

## References

- QOALA: D. L. Goodwin, P. Singh and M. Foroozandeh, "Adaptive optimal
  control of entangled qubits", *Science Advances* **8**(49), eabq4244 (2022).
  <https://doi.org/10.1126/sciadv.abq4244>
- ESCALADE: M. Foroozandeh and P. Singh, "Optimal control of spins by
  analytical Lie algebraic derivatives", *Automatica* **129**, 109611 (2021).
  <https://doi.org/10.1016/j.automatica.2021.109611>
- Auxiliary matrix method: <http://dx.doi.org/10.1063/1.4928978>
- Auxiliary matrix Krylov method: <http://dx.doi.org/10.5258/soton/t0003>
- Spillout-norm square penalty: <http://dx.doi.org/10.1063/1.4949534>
- Splittings with complex coefficients: Blanes (2021); Blanes (2019);
  Castella (2009)
- The optimiser derives from `fminlbfgs` (D. Kroon, University of Twente, 2010)
  and its Spinach implementation (Goodwin, Kuprov); several routines are noted
  in the MATLAB as based on Spinach and should be cited accordingly.

## Relationship to the original package

This is an independent reimplementation, not a wrapper: there is no MATLAB in
the loop at build time or run time. It follows the original's structure closely
enough that a change made to one can be found in the other, and every place it
departs is recorded in [`DEVIATIONS.md`](DEVIATIONS.md).

If you use this for published work, please cite the QOALA and ESCALADE papers
and the underlying methods listed above, not just this port:

- D. L. Goodwin, P. Singh and M. Foroozandeh, "Adaptive optimal control of
  entangled qubits", *Science Advances* **8**(49), eabq4244 (2022).
  <https://doi.org/10.1126/sciadv.abq4244>
- M. Foroozandeh and P. Singh, "Optimal control of spins by analytical Lie
  algebraic derivatives", *Automatica* **129**, 109611 (2021).
  <https://doi.org/10.1016/j.automatica.2021.109611>

## Licence

MIT, following the original package. See [`LICENSE`](LICENSE) — the original
copyright is retained alongside the port's.
