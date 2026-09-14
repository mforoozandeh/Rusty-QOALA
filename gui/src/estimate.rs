//! A runtime estimate for the Run button.
//!
//! Surprise is what loses people, not slowness, so the button says roughly
//! how long a run will take before they press it.
//!
//! # QOALA
//!
//! Cost is `slices * iterations * dimension^exponent`, and the exponent
//! differs by what is being propagated.  A state transfer pushes a vector
//! through each slice; a gate pushes a whole propagator, so it costs far
//! more and grows faster.  Neither exponent is the naive 2 or 3, because
//! QOALA's single-spin propagators are block-diagonal and its interaction
//! propagators are sparse - which is the method's entire point.
//!
//! # ESCALADE
//!
//! Every spin at every field is propagated through every slice, so a
//! gradient costs `spins * fields * slices`; the Hessian pairs every slice
//! with every other, so it costs that times `slices` again.  ESCALADE stops at
//! its target fidelity, usually long before its iteration budget, so for it
//! the estimate is an upper bound: see [`is_upper_bound`].
//!
//! The constants were measured with `cargo run -p qoala-gui --release
//! --example calibrate` on an Apple-silicon laptop, against the shipped
//! presets.  Treat the answer as an order of magnitude, not a stopwatch.

use crate::escalade::EscaladeSetup;
use crate::problem::Problem;
use crate::setup::{Setup, Target};

/// Seconds per slice per iteration at `dimension^TRANSFER_EXPONENT`, for a
/// state transfer.
const TRANSFER_COEFFICIENT: f64 = 1.52e-6;
/// How a state transfer's cost grows with the Liouville dimension.
const TRANSFER_EXPONENT: f64 = 1.1;

/// Seconds per slice per iteration at `dimension^GATE_EXPONENT`, for gate
/// synthesis.
const GATE_COEFFICIENT: f64 = 1.0e-7;
/// How gate synthesis's cost grows with the Liouville dimension.
const GATE_EXPONENT: f64 = 2.25;

/// Splitting order the constants were measured at, so the shipped presets
/// come out with a factor of one.
const REFERENCE_ORDER: f64 = 4.0;

/// Seconds per iteration per spin, field and slice, on the gradient alone.
const ESCALADE_GRADIENT_COEFFICIENT: f64 = 1.5e-7;
/// Seconds per iteration per spin, field and slice squared, with the Hessian.
const ESCALADE_HESSIAN_COEFFICIENT: f64 = 1.1e-8;

/// Rough allowance for WebAssembly being slower than native code.
pub const WEB_SLOWDOWN: f64 = 2.5;

/// Roughly how long `problem` will take to run natively, in seconds.
pub fn estimate_seconds(problem: &Problem) -> f64 {
    match problem {
        Problem::Qoala(setup) => qoala_seconds(setup),
        Problem::Escalade(setup) => escalade_seconds(setup),
    }
}

/// As [`estimate_seconds`], with the WebAssembly allowance applied when
/// `is_web`.
pub fn estimate_seconds_on(problem: &Problem, is_web: bool) -> f64 {
    estimate_seconds(problem) * if is_web { WEB_SLOWDOWN } else { 1.0 }
}

/// Whether the estimate is the most a run can take rather than what it will
/// take: true for ESCALADE, which stops once it reaches its target.
pub fn is_upper_bound(problem: &Problem) -> bool {
    matches!(problem, Problem::Escalade(_))
}

fn qoala_seconds(setup: &Setup) -> f64 {
    let dim = setup.dimension() as f64;
    let (coefficient, exponent) = match setup.target {
        Target::Transfer { .. } => (TRANSFER_COEFFICIENT, TRANSFER_EXPONENT),
        Target::Gate { .. } => (GATE_COEFFICIENT, GATE_EXPONENT),
    };
    // More interaction stages per slice at higher splitting order.
    let order = setup.splitset.iter().max().copied().unwrap_or(2).max(1) as f64;
    coefficient
        * dim.powf(exponent)
        * setup.nslices as f64
        * setup.max_iter as f64
        * (order / REFERENCE_ORDER)
}

fn escalade_seconds(setup: &EscaladeSetup) -> f64 {
    let slices = setup.nslices as f64;
    let work = setup.nspins as f64 * setup.fields_hz().len() as f64 * slices;
    let per_iteration = if setup.use_hessian {
        ESCALADE_HESSIAN_COEFFICIENT * work * slices
    } else {
        ESCALADE_GRADIENT_COEFFICIENT * work
    };
    per_iteration * setup.max_iter as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;
    use crate::setup::GateChoice;

    fn q(setup: &Setup) -> f64 {
        estimate_seconds(&Problem::Qoala(setup.clone()))
    }

    fn e(setup: &EscaladeSetup) -> f64 {
        estimate_seconds(&Problem::Escalade(setup.clone()))
    }

    /// The estimate has to be monotonic in the things that actually make a
    /// run longer, or it is worse than no estimate at all.
    #[test]
    fn the_estimate_grows_with_the_work() {
        let base = presets::z2z_2spin_1();
        let baseline = q(&base);
        assert!(baseline > 0.0);

        let mut more_slices = base.clone();
        more_slices.nslices *= 4;
        assert!((q(&more_slices) / baseline - 4.0).abs() < 1e-9);

        let mut more_iterations = base.clone();
        more_iterations.max_iter *= 3;
        assert!((q(&more_iterations) / baseline - 3.0).abs() < 1e-9);

        let mut more_spins = base.clone();
        more_spins.nspins = 3;
        more_spins.resize();
        assert!(q(&more_spins) > 3.0 * baseline);

        let problem = Problem::Qoala(base);
        assert!(estimate_seconds_on(&problem, true) > estimate_seconds_on(&problem, false));
    }

    /// A gate costs more than a transfer of the same size, and the gap grows
    /// with the register.
    #[test]
    fn a_gate_costs_more_than_a_transfer() {
        let transfer = presets::z2z_2spin_1();
        let mut gate = transfer.clone();
        gate.target = Target::Gate {
            gate: GateChoice::Swap,
            qubits: vec![0, 1],
        };
        let small = q(&gate) / q(&transfer);
        assert!(small > 1.0, "a two-spin gate should cost more: {small}");

        let mut transfer3 = transfer.clone();
        transfer3.nspins = 3;
        transfer3.resize();
        let mut gate3 = gate.clone();
        gate3.nspins = 3;
        gate3.resize();
        let large = q(&gate3) / q(&transfer3);
        assert!(large > small, "the gap should widen: {small} -> {large}");
    }

    #[test]
    fn escalade_scales_with_spins_fields_and_slices() {
        let base = presets::escalade_b1_sensitive();
        let baseline = e(&base);
        assert!(baseline > 0.0);

        let mut fields = base.clone();
        fields.b1_spread = 0.2;
        fields.b1_fields = 10;
        assert!((e(&fields) / baseline - 10.0).abs() < 1e-9);

        // The Hessian grows with the square of the slice count.
        let mut slices = base.clone();
        slices.nslices *= 2;
        assert!((e(&slices) / baseline - 4.0).abs() < 1e-9);
        let mut gradient = base.clone();
        gradient.use_hessian = false;
        let mut gradient_slices = gradient.clone();
        gradient_slices.nslices *= 2;
        assert!((e(&gradient_slices) / e(&gradient) - 2.0).abs() < 1e-9);

        assert!(is_upper_bound(&Problem::Escalade(base)));
        assert!(!is_upper_bound(&Problem::Qoala(presets::default_setup())));
    }

    /// The constants are a fit to measurements, so the presets they were
    /// fitted against must come back near what was measured.  The numbers on
    /// the right are from `--example calibrate` at 20 iterations.
    #[test]
    fn the_presets_land_near_their_measured_times() {
        let measured = [
            ("z to z, 2 spins, weak coupling", 0.032),
            ("SWAP, 2 spins", 0.055),
            ("z to z, 3 spins", 0.334),
            ("SWAP, 3 spins, end to end", 5.472),
        ];
        for (name, seconds) in measured {
            let mut setup = presets::qoala()
                .into_iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("no preset called {name}"));
            setup.max_iter = 20;
            let ratio = q(&setup) / seconds;
            assert!(
                (0.5..=2.0).contains(&ratio),
                "{name}: estimate is {ratio:.2} times the measured {seconds} s"
            );
        }

        let measured = [
            ("Broadband excitation", 0.027),
            ("B1-compensated broadband excitation", 0.850),
        ];
        for (name, seconds) in measured {
            let mut setup = presets::escalade()
                .into_iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("no preset called {name}"));
            setup.max_iter = 20;
            let ratio = e(&setup) / seconds;
            assert!(
                (0.5..=2.0).contains(&ratio),
                "{name}: estimate is {ratio:.2} times the measured {seconds} s"
            );
        }
    }
}
