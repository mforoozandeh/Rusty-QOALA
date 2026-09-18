# Where this port differs from the MATLAB

Everything here is a deliberate, documented departure from
`QOALA v1.0` (M. Foroozandeh, D. L. Goodwin, P. Singh). Nothing below changes
the method; the list exists so that a difference in output can be traced to a
decision rather than a mystery.

The port reproduces the MATLAB exactly for every configuration the shipped
example scripts use. The departures are in code paths the examples never
reach, plus one line-search fix.

## Corrections

### Backward pass for asymmetric splittings

`kernel/obj_fun/grape_*_qoala.m` walks the interaction propagators in
**forward** index order during the backward sweep, and takes a separate branch
for splitting orders below two.

For the symmetric splittings - orders 2, 4 and 6, which is every order the
package uses by default - the index sequences are palindromes, so walking them
forwards and backwards is the same thing and the gradient is exact. For the
asymmetric orders 0, 1 and 3 it is not the adjoint of the forward step, and the
gradient is only correct to the order of the splitting.

`src/objfun/qoala.rs` always walks the sequence in reverse. That agrees with
the MATLAB wherever the MATLAB is exact, and gives the exact gradient of the
split propagator everywhere else. `tests/gradients.rs` checks the gradient
against finite differences at orders 0, 1, 2, 3, 4 and 6 with Trotter numbers 1
and 2; all of them pass to 1e-5 relative, which the forward-order version would
not.

### Lie-Trotter with a Trotter number above one

`utilities/operator_splittings.m` builds the order-1 repeat sequence with
single-spin index 2 and interaction index 3. That order defines one single-spin
constant and two coupling constants, so those indices do not exist and any
Trotter number above one indexes past the end of the array. The repeats here
are the intended Lie-Trotter sequence. Order 1 with Trotter number 1, which is
what the MATLAB actually reaches, is unchanged.

### Line search Armijo test

`kernel/fmaxlinesearch.m` tests the sufficient-increase condition against
`alpha`, the *initial* step length, rather than `alpha_2`, the step being
trialled. `alpha` is never updated inside the bracketing loop, so from the
second iteration onwards the test uses a stale, smaller step, which makes it
easier to pass and lengthens the bracketing phase.

`src/optim/linesearch.rs` tests the step actually under trial. This is the only
correction that changes the iterate sequence for a stock configuration, so a
run will not follow the MATLAB step for step. Both converge; this one brackets
in fewer objective calls.

### `NS` and `DNS` penalty orientation

`kernel/penalty_fun.m` is inconsistent about the waveform layout. `SNS` and
`SNSM` are written for `nsteps x nchannels`, which is what QOALA always hands
them, but `NS`, `DNS` and `ADIAB` normalise and differentiate along the other
axis - they were written for Spinach's transposed convention. In the MATLAB,
`DNS` therefore smooths *across control channels* rather than across time,
which is not what a smoothing penalty is for, and `norm(dwf,2)^2` takes a
spectral norm while its own gradient is that of a Frobenius norm.

Every penalty in `src/penalty.rs` works in the `nsteps x nchannels` layout and
does what its documentation says. No shipped example is affected: they all use
`SNS`.

### Rodrigues derivative block indexing

`utilities/rodrigues.m` selects the `D`-matrix block for a control pair by the
*control pair* number while taking the operators for that pair from the *spin*
list. These coincide whenever control pair `k` drives spin `k`, which is how
`spin_control` is built in every driver and example. `src/rodrigues.rs` indexes
by the spin, which is correct for any ordering and identical for all supported
ones.

### SWAP gate construction

The example scripts build their gate target from a chain of nested `expm`
calls. `spinops::swap_gate` reproduces that chain and is checked by actually
swapping z-magnetisation between the two spins
(`spinops::tests::swap_gate_exchanges_the_two_spins`), which is a stronger
statement than matching the source line by line.

## Not implemented

These are unreachable or absent in the MATLAB too, and the port declines
explicitly rather than pretending.

- **Second propagator derivatives.** `rodrigues.m` has a Hessian branch, but it
  references `S1` and a `sigma` that has gone out of scope, and is flagged in
  the source as not general for multi-spin systems. It would error in MATLAB.
  Consequently the Newton-Raphson path is implemented and tested for its linear
  algebra, but no shipped objective function can feed it a Hessian.
- **`SNSA` recursion and `ADIAB`.** `penalty_fun.m` calls `penalty`,
  `penaltyox`, `cartesian2spherical` and `spherical2cartesian`, none of which
  are in the package. `SNSA` is implemented here directly rather than by
  recursion, and `ADIAB` is reconstructed from its documented intent -
  penalising the roughness of the effective-field polar angle across the pulse,
  with the gradient transformed back to Cartesian coordinates.
- **The `square` fidelity functional.** `optimconset.m` accepts and reports it,
  but no objective function implements anything except `real`. The parser here
  rejects it rather than silently computing the wrong thing.
- **Time-dependent interactions with operator splitting.** The propagators
  would differ at every slice, and the MATLAB's cache layout cannot express
  that. Rejected with a clear error; time-dependent drifts work fine with the
  auxiliary-matrix objective.
- **Non-uniform time grids with operator splitting.** `optimconset.m` warns
  that this is not fully coded and carries on: the interaction propagators are
  built for the average slice while the single-spin rotations use each
  slice's own, so the result describes neither grid. Rejected here. The
  auxiliary-matrix objectives use every slice's own duration and accept any
  grid.
- **The cavity response.** `optimconset.m` parses `cavity_decay_rate`,
  `cavity_function` and `cavity_n_interp`, but no objective function applies
  them. The fields are accepted and reported, with a warning, and
  `cavity_n_interp` still feeds the memory estimate.
- **Ensembles.** The parser handles an ensemble of drift systems and the
  reports are written for it, but `optimfun_grape_xy` errors out on more than
  one member. Same here, and the cost function likewise refuses more than one
  row of power levels, or more than one initial and target state, rather than
  optimising the first and ignoring the rest.
- **`adapt_method = 'exact'` roll-back.** After the ladder walk, the MATLAB
  leaves the splitting parameters where the walk ended but hands the gradient
  the trajectories stored from the *previous* rung. This port keeps the
  parameters where the MATLAB leaves them and computes the gradient with the
  splitting that matches the stored trajectories, which is the only
  self-consistent reading. The default `'gain'` method, used by every example,
  is unaffected.

## ESCALADE

`src/escalade/` ports the ESCALADE MATLAB code (M. Foroozandeh, P. Singh).
The objective, gradient and Hessian are the MATLAB's, term for term; the
Hessian is checked against finite differences of the gradient. What differs
is the optimiser around them.

### The optimiser

`ESCALADE.m` calls `fmincon`: `trust-region-reflective` with the analytic
Hessian, the default algorithm without. `fmincon` is not portable, so
`escalade/solve.rs` uses [argmin](https://argmin-rs.org) instead: a Newton
trust region with a Steihaug-CG subproblem on the Hessian, and L-BFGS with a
More-Thuente line search on the gradient alone. The iterate sequence therefore
differs from the MATLAB's; the problem being solved does not, apart from the
amplitude limit below.

### The amplitude limit

`fmincon` is given box bounds of `[-1, 1]` on each quadrature, which lets a
pulse point reach `sqrt(2)` times the nominal field in the corners of the box.
The physical limit is on the field itself, so the port limits
`sqrt(x^2 + y^2) <= 1` instead. argmin's gradient solvers take no bounds, so
the limit is the `SNSA` spillout penalty from `penalty.rs`: zero anywhere
inside the unit circle, quadratic outside it, weight 100 by default
(`Escalade::amplitude_weight`). A penalty is soft - a run that stops at its
target fidelity can sit a percent or two above the limit - so every result
reports its `max_amplitude`.

### Stopping

The MATLAB's `OutputFcn` stops once the fidelity passes `targetfidelity`. Here
the run stops once the fidelity less the penalty does, so that an
out-of-bounds pulse cannot stop early. `fmincon`'s `OptimalityTolerance` and
`FunctionTolerance` of `1e-10` become a gradient-norm test and L-BFGS's cost
test, and a run that has not lowered the objective in 30 iterations stops as
stalled (`ExitFlag::StepTolerance`).

### Smaller changes

- **No finite-difference mode.** `usegrad = false` asks `fmincon` to
  difference the objective itself. The analytic gradient is always used.
- **Threads over work items, not fields.** The MATLAB's `parfor` spreads the
  fields over workers and vectorises each field's spins. Here every
  (field, spin) pair is a work item, the items are cut into up to 64
  contiguous pieces, and with the `parallel` feature each piece is a rayon
  task, so a single field is parallel too. A Hessian piece holds a dense
  `2N x 2N` accumulator, so long pulses get fewer pieces, keeping them to
  256 MB together. The grouping follows the problem's size and the order
  asked for, never the thread count, so the result is the same to the last
  bit on any number of threads and without the feature. `wasm32` has no
  threads and runs sequentially.
- **One objective, not two.** `gradhess_vectorized.m` is
  `gradhess_vectorized_B1_parallel.m` with a single unit-weight field, so only
  the latter is ported.
- **Zero field.** The MATLAB adds `eps` to every offset to keep `|s|` off
  zero, but the derivative coefficients still lose every digit to
  cancellation as `|s|` shrinks. The port drops the shift and switches to
  their Taylor series below `|s| = 0.1`.
- **Units and signs.** Explicit offsets (`Om`) are given in Hz rather than
  rad/s, and the target fidelity and reported fidelity are positive:
  `targetfidelity = -0.99` is `target_fidelity: 0.99`.
- **Weights are checked.** `readSettings` silently replaces `rfweights` of the
  wrong length with equal weights; the port refuses them.
- **Universal rotations.** The MATLAB takes one initial and one target state
  per spin. The port also takes a target propagator, `Goal::Rotation`,
  scored on `Re tr(W^dagger U) / 2`, so that one pulse can be optimised to
  turn every axis alike. A state transfer computes exactly what the MATLAB
  does.
- **Visualisation.** `escalade::profile` computes what
  `ESCALADE_pulse_sim.m` and `ESCALADE_Bloch_B1.m` plot, over the same grids,
  but propagates with the optimiser's own SU(2) propagators rather than the
  scripts' separate rotation-matrix Bloch simulation and its phase
  convention. The Bruker shape export, the rf-coefficient optimisation of
  `ESCALADE_dphidB1_opt.m` and the Bloch-sphere trajectories are not ported.
- **Start.** `test_escalade_visual.m` starts from a flat pulse at -0.5; the
  `escalade_b1` example does too. The application's presets use a seeded
  random start, which reaches the target far sooner under the amplitude limit.

## Deliberate design changes

These do not change any number, only how the code is shaped.

- **Options are typed.** MATLAB strings (`'lbfgs'`, `'taylor'`, `'sphten'`)
  become enums, and `ctrl_param` becomes `ControlOptions`. "Unrecognised
  option" is a compile error rather than a run-time warning, so the parser's
  unparsed-field sweep has no counterpart.
- **`nargout` becomes an argument.** MATLAB branches on how many outputs the
  caller asked for; here `EvalOrder` and `PenaltyOrder` say so explicitly.
- **Propagator derivatives are kept factored.** The MATLAB materialises
  `dP{k}` as an `nsteps x dim^2` array - 65536 columns wide for four spins.
  `ControlDerivative` stores the `D` block and the operators instead and forms
  the `dim x dim` matrix on demand. Identical arithmetic, far less memory
  traffic.
- **Propagator caching is a binary file.** `prop_cache = 'store'` writes
  `scratch/<job_id>/prop_t<q>o<p>.bin` in a small format defined in
  `splittings.rs`, in place of MATLAB's `-v7.3` MAT files. The file also
  records the time step, interaction, propagation method and cutoff it was
  built from, and is rebuilt when any of them differ, so an inherited `job_id`
  cannot hand one problem another's propagators.
- **Plots become CSV.** The example scripts finish with `figure`, `stairs` and
  `exportgraphics`. There is no equivalent, so the examples write waveforms and
  convergence tables as CSV.
- **The report prefix is fixed.** MATLAB derives the `[ ... ]` prefix on each
  report line from `dbstack`. Here it is the function name plus the objective,
  which is the same information without the stack walk.
- **Job identifiers.** MATLAB hashes the clock and process id with MD5; this
  uses a 128-bit mix of the same inputs, rendered the same way.
- **Progress is a data channel, not only text.** MATLAB's only way out of a
  running optimisation is the `fprintf` iteration table. `optim::ProgressSink`
  receives the same rows as an `IterationReport` struct, and can stop the run
  (`ExitFlag::Cancelled`, which the MATLAB has no counterpart for). The text
  table is itself one such sink, so its output is unchanged.
- **There is a graphical front end.** The MATLAB package is a set of scripts.
  `gui/` adds an interface over the same library, as a desktop application and
  as a browser page. It changes no numerics: it builds the same
  `ControlOptions` the drivers do and calls the same optimiser.
- **WebAssembly is a supported target.** On `wasm32-unknown-unknown` there is
  no filesystem and no monotonic clock, so `examples_io`, `Reporter::file`,
  `ReportSink::File` and the propagator cache are compiled out, timings go
  through `web-time`, and `PropCache::Store` returns
  `QoalaError::NotImplemented`. The numerics are identical on both targets.
