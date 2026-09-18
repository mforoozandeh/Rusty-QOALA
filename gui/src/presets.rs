//! One-click setups: the five QOALA repository examples, the two ESCALADE
//! runs of `test_runs/test_escalade_visual.m`, and an ESCALADE universal
//! rotation.
//!
//! Every QOALA number here is copied from the corresponding file in
//! `examples/`, so clicking Run on a preset reproduces that example.

use crate::escalade::{Direction, EscaladeSetup, Goal};
use crate::problem::Problem;
use crate::setup::{
    pair_count, pair_index, Component, Coupling, GateChoice, Operator, PenaltyChoice, Setup, Target,
};

/// Build an otherwise-default setup for `nspins` spins and `npairs` control
/// pairs, with no couplings and nothing driven.
fn blank(name: &str, nspins: usize, npairs: usize) -> Setup {
    Setup {
        name: name.to_string(),
        nspins,
        offsets_hz: vec![0.0; nspins],
        couplings: vec![Coupling::default(); pair_count(nspins)],
        npairs,
        spin_control: vec![vec![false; npairs]; nspins],
        amplitudes_hz: vec![1000.0; npairs],
        target: Target::Transfer {
            from: Operator::single(0, Component::Z),
            to: Operator::single(nspins - 1, Component::Z),
        },
        duration_s: 0.01,
        nslices: 50,
        max_iter: 100,
        penalty: PenaltyChoice::Sns,
        splitset: vec![2, 3, 4],
        seed: Some(1),
    }
}

/// Give every spin its own control pair, as `spinops::one_pair_per_spin`
/// does.
fn one_pair_per_spin(setup: &mut Setup) {
    for s in 0..setup.nspins {
        for k in 0..setup.npairs {
            setup.spin_control[s][k] = s == k;
        }
    }
}

fn couple(setup: &mut Setup, i: usize, j: usize, j_hz: f64, strong: bool) {
    let index = pair_index(setup.nspins, i, j);
    setup.couplings[index] = Coupling { j_hz, strong };
}

/// `examples/z2z_2spin_1.rs`: z-to-z transfer across a weakly coupled pair.
pub fn z2z_2spin_1() -> Setup {
    let mut s = blank("z to z, 2 qubits, weak coupling", 2, 2);
    one_pair_per_spin(&mut s);
    couple(&mut s, 0, 1, 140.0, false);
    s.duration_s = 0.01;
    s.nslices = 50;
    s
}

/// `examples/z2z_2spin_2.rs`: strong coupling, one control pair driving both
/// spins, and a target that is a sign flip on a single basis element.
pub fn z2z_2spin_2() -> Setup {
    let mut s = blank("z to z, 2 qubits, strong coupling", 2, 1);
    for spin in 0..2 {
        s.spin_control[spin][0] = true;
    }
    couple(&mut s, 0, 1, 20.0, true);
    s.offsets_hz = vec![2000.0, -1200.0];
    s.amplitudes_hz = vec![1000.0];
    s.duration_s = 0.01;
    s.nslices = 50;
    s.seed = Some(2);
    // Both states sit on basis element 9 with opposite signs, exactly as the
    // example script sets them.
    s.target = Target::Transfer {
        from: Operator::Basis {
            coefficients: vec![(9, -0.5)],
        },
        to: Operator::Basis {
            coefficients: vec![(9, 0.5)],
        },
    };
    s
}

/// `examples/z2z_3spin_1.rs`: z-to-z transfer along a three-spin chain.
pub fn z2z_3spin_1() -> Setup {
    let mut s = blank("z to z, 3 qubits", 3, 3);
    one_pair_per_spin(&mut s);
    couple(&mut s, 0, 1, 140.0, false);
    couple(&mut s, 1, 2, -160.0, false);
    s.duration_s = 0.022;
    s.nslices = 110;
    s.target = Target::Transfer {
        from: Operator::single(0, Component::Z),
        to: Operator::single(2, Component::Z),
    };
    s
}

/// `examples/swap_2spin_1.rs`: SWAP between a weakly coupled pair.
pub fn swap_2spin_1() -> Setup {
    let mut s = blank("SWAP, 2 qubits", 2, 2);
    one_pair_per_spin(&mut s);
    couple(&mut s, 0, 1, 140.0, false);
    s.duration_s = 0.012;
    s.nslices = 60;
    s.target = Target::Gate {
        gate: GateChoice::Swap,
        qubits: vec![0, 1],
    };
    s
}

/// `examples/swap_3spin_1.rs`: SWAP between the end spins of a chain.
pub fn swap_3spin_1() -> Setup {
    let mut s = blank("SWAP, 3 qubits, end to end", 3, 3);
    one_pair_per_spin(&mut s);
    couple(&mut s, 0, 1, 140.0, false);
    couple(&mut s, 1, 2, -160.0, false);
    s.duration_s = 0.0264;
    s.nslices = 264;
    s.target = Target::Gate {
        gate: GateChoice::Swap,
        qubits: vec![0, 2],
    };
    s
}

/// Every QOALA preset, in the order the menu offers them.
///
/// The first is the default on load: it converges in a second or two, which
/// is the point of having a default at all.
pub fn qoala() -> Vec<Setup> {
    vec![
        z2z_2spin_1(),
        swap_2spin_1(),
        z2z_2spin_2(),
        z2z_3spin_1(),
        swap_3spin_1(),
    ]
}

/// The QOALA setup the application starts with.
pub fn default_setup() -> Setup {
    z2z_2spin_1()
}

/// `test_escalade_visual.m`, first run: z to -y across 20 kHz in 100
/// microseconds, for a 17 kHz field alone.
///
/// The script starts from a flat pulse at -0.5.  Under the amplitude limit a
/// random start converges far faster, so the presets use a seeded one.
pub fn escalade_b1_sensitive() -> EscaladeSetup {
    EscaladeSetup {
        name: "Broadband excitation".into(),
        nspins: 51,
        sw_hz: 20000.0,
        rf_hz: 17000.0,
        b1_spread: 0.0,
        b1_fields: 1,
        duration_s: 100e-6,
        nslices: 50,
        goal: Goal::Transfer,
        from: Direction::PlusZ,
        to: Direction::MinusY,
        rotation: [Direction::PlusX, Direction::PlusZ, Direction::MinusY],
        use_hessian: true,
        max_iter: 1000,
        target_fidelity: 0.99,
        seed: Some(7),
    }
}

/// `test_escalade_visual.m`, second run: the same excitation, optimised
/// across 31 fields from 0.8 to 1.2 times nominal.
pub fn escalade_b1_compensated() -> EscaladeSetup {
    EscaladeSetup {
        name: "B1-compensated broadband excitation".into(),
        b1_spread: 0.2,
        b1_fields: 31,
        ..escalade_b1_sensitive()
    }
}

/// A universal 90-degree rotation about x across 10 kHz: x stays, y goes to
/// z and z to -y, whatever the magnetisation started as.  Not one of the
/// MATLAB's runs, which have one initial and one target state.
pub fn escalade_universal_rotation() -> EscaladeSetup {
    EscaladeSetup {
        name: "Broadband universal 90-degree rotation about x".into(),
        sw_hz: 10000.0,
        duration_s: 200e-6,
        nslices: 60,
        goal: Goal::Rotation,
        ..escalade_b1_sensitive()
    }
}

/// Every ESCALADE preset, in menu order.
pub fn escalade() -> Vec<EscaladeSetup> {
    vec![
        escalade_b1_sensitive(),
        escalade_b1_compensated(),
        escalade_universal_rotation(),
    ]
}

/// Every preset of both kinds.
pub fn every() -> Vec<Problem> {
    qoala()
        .into_iter()
        .map(Problem::Qoala)
        .chain(escalade().into_iter().map(Problem::Escalade))
        .collect()
}
