# Contributing

## Before you open a pull request

```
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test --release
```

CI runs exactly these, plus a documentation build with warnings denied and
three of the examples end to end.

## Tests are the argument

There is no MATLAB in the loop, so every claim this crate makes has to be
checked against mathematics rather than against a stored output. A new
numerical routine needs a test that could actually fail: a gradient against a
finite difference, a splitting against its order condition, a propagator
against an independently computed matrix exponential. A test that only asserts
the code runs is not much of a test.

Hand-aligned matrix literals are wrapped in `#[rustfmt::skip]` so that they
stay readable as grids. If you add one, do the same.

## Departures from the MATLAB

This crate is a port, so a difference in behaviour is a bug unless it is a
deliberate, documented decision. If you find one that is not in
[`DEVIATIONS.md`](DEVIATIONS.md), please open an issue; if you introduce one on
purpose, add it there with the reasoning.

## Provenance

Several routines are derived from the Spinach package (Goodwin, Kuprov) and
from `fminlbfgs` (D. Kroon, University of Twente), as noted in the MATLAB
source. Keep those attributions in the module documentation.
