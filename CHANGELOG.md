# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
