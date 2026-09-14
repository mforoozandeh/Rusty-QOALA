//! The five repository examples, as one-click setups.
//!
//! Every number here is copied from the corresponding file in `examples/`, so
//! clicking Run on a preset reproduces that example.

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
    let mut s = blank("z to z, 2 spins, weak coupling", 2, 2);
    one_pair_per_spin(&mut s);
    couple(&mut s, 0, 1, 140.0, false);
    s.duration_s = 0.01;
    s.nslices = 50;
    s
}

/// `examples/z2z_2spin_2.rs`: strong coupling, one control pair driving both
/// spins, and a target that is a sign flip on a single basis element.
pub fn z2z_2spin_2() -> Setup {
    let mut s = blank("z to z, 2 spins, strong coupling", 2, 1);
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
    let mut s = blank("z to z, 3 spins", 3, 3);
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
    let mut s = blank("SWAP, 2 spins", 2, 2);
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
    let mut s = blank("SWAP, 3 spins, end to end", 3, 3);
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

/// Every preset, in the order the menu offers them.
///
/// The first is the default on load: it converges in a second or two, which
/// is the point of having a default at all.
pub fn all() -> Vec<Setup> {
    vec![
        z2z_2spin_1(),
        swap_2spin_1(),
        z2z_2spin_2(),
        z2z_3spin_1(),
        swap_3spin_1(),
    ]
}

/// The setup the application starts with.
pub fn default_setup() -> Setup {
    z2z_2spin_1()
}
