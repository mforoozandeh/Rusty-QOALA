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
  src/setup.rs      the problem description, serde-serialisable
  src/presets.rs    the five examples above, as one-click setups
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

- A visual editor for the control map - which channel drives which spin -
  which is the least obvious part of the library's API.
- Presets for all five examples; the two-spin transfer is loaded on startup,
  so the first click gives a converged pulse in about a second.
- Infidelity against iteration on a log axis, the waveform as stairs in Hz,
  and the splitting order and Trotter number live, climbing as the fidelity
  improves.
- A runtime estimate on the Run button, and a Cancel that works.
- Exports: the waveform as CSV in Hz against seconds, and the whole setup as
  JSON.
- The whole setup encoded in the URL fragment, so a link reproduces it
  exactly and can be sent to somebody else. Nothing is stored anywhere but
  the link and the visitor's own browser.

Two limits worth knowing. Four-spin systems are dimension 256 with dense
256x256 propagators per slice; the browser build refuses them and points at
the desktop one. And cancelling in the browser terminates the Web Worker, so
the convergence curve is kept but the partial waveform is not - natively,
cancellation is cooperative and the waveform comes back.

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

`cargo test` runs 81 tests. There is no MATLAB in the loop; every check is
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

## Differences from the MATLAB

Five corrections (an inexact backward pass for asymmetric splittings, an
unreachable Lie-Trotter branch, a stale variable in the line search, a
transposed penalty layout, and a gate-construction ordering), plus the features
the MATLAB parses but never implements. All of them are listed, with reasons,
in [`DEVIATIONS.md`](DEVIATIONS.md).

## References

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

If you use this for published work, please cite the QOALA paper and the
underlying methods listed above, not just this port.

## Licence

MIT, following the original package. See [`LICENSE`](LICENSE) — the original
copyright is retained alongside the port's.
