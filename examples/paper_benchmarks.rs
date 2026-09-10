//! The QOALA paper benchmarks: adaptive splitting against the exact method.
//!
//! One executable covering all six scripts in
//! `examples/qoala_paper_results/`.  They differ only in the spin system and
//! the target, so they are described here as data rather than as six
//! near-identical files.
//!
//! Each system is optimised five ways - the exact auxiliary-matrix method,
//! then fixed second-, third- and fourth-order splittings, then the adaptive
//! ladder over all three - from several random starting pulses.  The median
//! infidelity and the median time spent inside the objective function are
//! written to CSV, which is what the MATLAB scripts plot.
//!
//! ```text
//! cargo run --release --example paper_benchmarks -- state-2spin-hetero
//! cargo run --release --example paper_benchmarks -- gate-3spin-hetero --runs 8 --steps 264
//! ```
//!
//! Systems: `state-2spin-hetero`, `state-2spin-homo`, `state-3spin-hetero`,
//! `state-4spin-mixed`, `gate-2spin-hetero`, `gate-3spin-hetero`.
//!
//! The paper averages 28 runs; the default here is 4 so the example finishes
//! in a reasonable time.  Pass `--runs 28` to reproduce the published setup.
//! The four-spin system in the paper also carries a relaxation superoperator
//! from Spinach; without it that term is zero, exactly as the MATLAB script
//! falls back to when Spinach is absent.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use qoala::config::{
    optimconset, ControlOptions, ControlSystem, DriftOptions, PowerLevels, TimeDependent,
};
use qoala::drivers::{random_pulse, GrapeXyCost};
use qoala::error::{QoalaError, Result};
use qoala::linalg::{CDense, CMat};
use qoala::optim::newton::fmaxnewton;
use qoala::report::Reporter;
use qoala::spinops;
use qoala::types::*;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// One benchmark system.
struct System {
    name: &'static str,
    nspins: usize,
    /// Resonance offsets in Hz.
    omega: Vec<f64>,
    /// Maximum control amplitude in Hz.
    power_hz: f64,
    /// Pulse duration in seconds.
    duration: f64,
    /// Iteration budget used by the paper.
    max_iter: usize,
    /// Default number of time slices.
    nsteps: usize,
    /// Dense-storage threshold.
    sparsity: f64,
    /// Interaction Hamiltonian.
    interaction: CMat,
    /// Composite control operators, one per channel.
    operators: Vec<CMat>,
    /// Single-spin Cartesian operators for the split-operator objective.
    pauli: Vec<Vec<Option<CMat>>>,
    /// Which channels drive which spins.
    spin_control: DMatrix<bool>,
    /// Gate target, or state pair.
    goal: Goal,
}

enum Goal {
    Transfer { init: CDense, target: CDense },
    Gate { target: CDense },
}

fn normalise(x: &CDense) -> CDense {
    let n = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    x.map(|z| z / Complex64::new(n, 0.0))
}

/// Composite x/y operators for a group of spins, summed.
fn channel(ops: &[(CMat, CMat, CMat)], spins: &[usize], y: bool) -> Result<CMat> {
    let mut acc: Option<CMat> = None;
    for &s in spins {
        let op = if y { &ops[s].1 } else { &ops[s].0 };
        acc = Some(match acc {
            None => op.clone(),
            Some(a) => a.add(op)?,
        });
    }
    acc.ok_or_else(|| QoalaError::BadValue("empty control channel".into()))
}

/// Build the `pauli` array and `spin_control` map from a channel-to-spin
/// assignment.
fn control_map(
    nspins: usize,
    ops: &[(CMat, CMat, CMat)],
    groups: &[Vec<usize>],
) -> (Vec<Vec<Option<CMat>>>, DMatrix<bool>) {
    let npairs = groups.len();
    let kctrls = 2 * npairs;
    let mut pauli: Vec<Vec<Option<CMat>>> = vec![vec![None; 3 * npairs]; nspins];
    let mut spin_control = DMatrix::from_element(nspins, kctrls, false);
    for (k, group) in groups.iter().enumerate() {
        for &s in group {
            pauli[s][3 * k] = Some(ops[s].0.clone());
            pauli[s][3 * k + 1] = Some(ops[s].1.clone());
            pauli[s][3 * k + 2] = Some(ops[s].2.clone());
            spin_control[(s, 2 * k)] = true;
            spin_control[(s, 2 * k + 1)] = true;
        }
    }
    (pauli, spin_control)
}

fn build_system(name: &str) -> Result<System> {
    match name {
        // Two heteronuclear spins, 140 Hz zz coupling, one control pair each.
        "state-2spin-hetero" | "gate-2spin-hetero" => {
            let nspins = 2;
            let ops = spinops::cartesian_operators(nspins);
            let interaction =
                spinops::zz_coupling(nspins, 0, 1)?.scale(Complex64::new(TWO_PI * 140.0, 0.0));
            let operators = vec![
                channel(&ops, &[0], false)?,
                channel(&ops, &[0], true)?,
                channel(&ops, &[1], false)?,
                channel(&ops, &[1], true)?,
            ];
            let (pauli, spin_control) = control_map(nspins, &ops, &[vec![0], vec![1]]);
            let gate = name.starts_with("gate");
            Ok(System {
                name: if gate {
                    "gate-2spin-hetero"
                } else {
                    "state-2spin-hetero"
                },
                nspins,
                omega: vec![0.0, 0.0],
                power_hz: 1e3,
                duration: if gate { 12e-3 } else { 10e-3 },
                max_iter: if gate { 100 } else { 50 },
                nsteps: 50,
                sparsity: 0.15,
                interaction,
                operators,
                pauli,
                spin_control,
                goal: if gate {
                    Goal::Gate {
                        target: spinops::swap_gate(nspins, 0, 1)?,
                    }
                } else {
                    Goal::Transfer {
                        init: normalise(&spinops::z_state(nspins, 0)),
                        target: normalise(&spinops::z_state(nspins, 1)),
                    }
                },
            })
        }

        // Two homonuclear spins sharing one control pair, isotropic coupling.
        "state-2spin-homo" => {
            let nspins = 2;
            let ops = spinops::cartesian_operators(nspins);
            let interaction = spinops::isotropic_coupling(nspins, 0, 1)?
                .scale(Complex64::new(TWO_PI * 20.0, 0.0));
            let operators = vec![
                channel(&ops, &[0, 1], false)?,
                channel(&ops, &[0, 1], true)?,
            ];
            let (pauli, spin_control) = control_map(nspins, &ops, &[vec![0, 1]]);
            // z-magnetisation on spin 1, raising operator on spin 2.
            let lz_lp = spinops::kron_states(&[spinops::z_state_single(), spinops::p_state()]);
            Ok(System {
                name: "state-2spin-homo",
                nspins,
                omega: vec![2000.0, -1200.0],
                power_hz: 1e3,
                duration: 10e-3,
                max_iter: 125,
                nsteps: 50,
                sparsity: 0.15,
                interaction,
                operators,
                pauli,
                spin_control,
                goal: Goal::Transfer {
                    init: normalise(&lz_lp),
                    target: normalise(&lz_lp).map(|z| -z),
                },
            })
        }

        // Three heteronuclear spins in a chain, one control pair each.
        "state-3spin-hetero" | "gate-3spin-hetero" => {
            let nspins = 3;
            let ops = spinops::cartesian_operators(nspins);
            let interaction = spinops::zz_coupling(nspins, 0, 1)?
                .scale(Complex64::new(TWO_PI * 140.0, 0.0))
                .add(
                    &spinops::zz_coupling(nspins, 1, 2)?
                        .scale(Complex64::new(TWO_PI * -160.0, 0.0)),
                )?;
            let operators = vec![
                channel(&ops, &[0], false)?,
                channel(&ops, &[0], true)?,
                channel(&ops, &[1], false)?,
                channel(&ops, &[1], true)?,
                channel(&ops, &[2], false)?,
                channel(&ops, &[2], true)?,
            ];
            let (pauli, spin_control) = control_map(nspins, &ops, &[vec![0], vec![1], vec![2]]);
            let gate = name.starts_with("gate");
            Ok(System {
                name: if gate {
                    "gate-3spin-hetero"
                } else {
                    "state-3spin-hetero"
                },
                nspins,
                omega: vec![0.0; 3],
                power_hz: 1e3,
                duration: if gate { 26.4e-3 } else { 22e-3 },
                max_iter: if gate { 250 } else { 125 },
                nsteps: if gate { 264 } else { 110 },
                sparsity: 0.15,
                interaction,
                operators,
                pauli,
                spin_control,
                goal: if gate {
                    Goal::Gate {
                        target: spinops::swap_gate(nspins, 0, 2)?,
                    }
                } else {
                    Goal::Transfer {
                        init: normalise(&spinops::z_state(nspins, 0)),
                        target: normalise(&spinops::z_state(nspins, 2)),
                    }
                },
            })
        }

        // Four spins: two homonuclear pairs, heteronuclear across them, two
        // control channels each driving a pair.
        "state-4spin-mixed" => {
            let nspins = 4;
            let ops = spinops::cartesian_operators(nspins);
            let interaction = spinops::zz_coupling(nspins, 0, 2)?
                .scale(Complex64::new(TWO_PI * 150.0, 0.0))
                .add(
                    &spinops::zz_coupling(nspins, 1, 3)?.scale(Complex64::new(TWO_PI * 150.0, 0.0)),
                )?
                .add(
                    &spinops::isotropic_coupling(nspins, 0, 1)?
                        .scale(Complex64::new(TWO_PI * 7.0, 0.0)),
                )?
                .add(
                    &spinops::isotropic_coupling(nspins, 2, 3)?
                        .scale(Complex64::new(TWO_PI * 50.0, 0.0)),
                )?;
            let operators = vec![
                channel(&ops, &[0, 1], false)?,
                channel(&ops, &[0, 1], true)?,
                channel(&ops, &[2, 3], false)?,
                channel(&ops, &[2, 3], true)?,
            ];
            let (pauli, spin_control) = control_map(nspins, &ops, &[vec![0, 1], vec![2, 3]]);

            let unit = spinops::unit_state();
            let rz = spinops::z_state_single();
            let rp = spinops::p_state();
            let rm = spinops::m_state();
            let k = |a: &CDense, b: &CDense, c: &CDense, d: &CDense| {
                spinops::kron_states(&[a.clone(), b.clone(), c.clone(), d.clone()])
            };
            // Singlet-like starting state on the first homonuclear pair.
            let rho_s12 = k(&unit, &unit, &unit, &unit).map(|z| z * 0.5)
                - (k(&rz, &rz, &unit, &unit).map(|z| z * 2.0)
                    + k(&rp, &rm, &unit, &unit)
                    + k(&rm, &rp, &unit, &unit));
            let target = normalise(&spinops::z_state(nspins, 3)).map(|z| z * 2.0);

            Ok(System {
                name: "state-4spin-mixed",
                nspins,
                omega: vec![-900.0, -1200.0, -4500.0, -6000.0],
                power_hz: 10e3,
                duration: 100e-3,
                max_iter: 200,
                nsteps: 50,
                sparsity: 0.2,
                interaction,
                operators,
                pauli,
                spin_control,
                goal: Goal::Transfer {
                    init: normalise(&rho_s12),
                    target,
                },
            })
        }

        other => Err(QoalaError::BadValue(format!("unknown system: {other}"))),
    }
}

/// The five ways each system is optimised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Exact,
    Split(usize),
    Adaptive,
}

impl Method {
    fn label(self) -> String {
        match self {
            Method::Exact => "auxmat".into(),
            Method::Split(n) => format!("order{n}"),
            Method::Adaptive => "qoala".into(),
        }
    }
}

fn build_options(sys: &System, method: Method, nsteps: usize, max_iter: usize) -> ControlOptions {
    let gate = matches!(sys.goal, Goal::Gate { .. });
    let objective = match (method, gate) {
        (Method::Exact, false) => ObjectiveFn::StateAuxmat,
        (Method::Exact, true) => ObjectiveFn::UgateAuxmat,
        (_, false) => ObjectiveFn::StateQoala,
        (_, true) => ObjectiveFn::UgateQoala,
    };

    let offsets = DVector::from_iterator(sys.nspins, sys.omega.iter().map(|w| TWO_PI * w));
    let ops = spinops::cartesian_operators(sys.nspins);
    let mut singlespin: Option<CMat> = None;
    for (s, w) in sys.omega.iter().enumerate() {
        if *w != 0.0 {
            let term = ops[s].2.scale(Complex64::new(TWO_PI * w, 0.0));
            singlespin = Some(match singlespin {
                None => term,
                Some(acc) => acc.add(&term).expect("offset sum"),
            });
        }
    }

    let mut drift = DriftOptions {
        interaction: Some(TimeDependent::Constant(sys.interaction.clone())),
        singlespin,
        offset: Some(offsets),
        ..Default::default()
    };
    match method {
        Method::Exact => {}
        Method::Split(n) => drift.split_order = Some(n),
        Method::Adaptive => {
            drift.split_order = Some(2);
            drift.splitset = Some(vec![2, 3, 4]);
        }
    }

    let mut opts = ControlOptions {
        basis: Some(Basis::Sphten),
        space: Some(StateSpace::Liouville),
        nspins: Some(sys.nspins),
        operators: Some(sys.operators.clone()),
        pwr_levels: Some(PowerLevels::Uniform(TWO_PI * sys.power_hz)),
        pulse_dur: Some(sys.duration),
        pulse_nsteps: Some(nsteps),
        penalties: Some(vec![Penalty::Sns]),
        prop_zeroed: Some(1e-12),
        sparsity: Some(sys.sparsity),
        auxmat_method: Some(PropMethod::Krylov),
        method: Some(OptMethod::Lbfgs),
        n_grads: Some(25),
        tol_g: Some(1e-12),
        tol_x: Some(1e-12),
        max_iter: Some(max_iter),
        optimcon_fun: Some(objective),
        drift_sys: vec![drift],
        output: Some(Reporter::silent()),
        ..Default::default()
    };

    if objective.is_qoala() {
        opts.prop_cache = Some(PropCache::Carry);
        opts.pauli_operators = Some(sys.pauli.clone());
        opts.spin_control = Some(sys.spin_control.clone());
        opts.ctrl_axes = Some(
            (0..sys.operators.len() / 2)
                .flat_map(|_| [Axis::X, Axis::Y])
                .collect(),
        );
        // Report the true fidelity alongside the approximate one, as the
        // MATLAB benchmarks do.
        opts.fidelity_chk = Some(ObjectiveFn::WaveformFidelity);
    }

    match &sys.goal {
        Goal::Transfer { init, target } => {
            opts.rho_init = Some(vec![init.clone()]);
            opts.rho_targ = Some(vec![target.clone()]);
        }
        Goal::Gate { target } => {
            opts.prop_targ = Some(vec![target.clone()]);
        }
    }
    opts
}

fn median(mut values: Vec<f64>) -> f64 {
    values.retain(|v| !v.is_nan());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args
        .first()
        .cloned()
        .unwrap_or_else(|| "state-2spin-hetero".into());
    let flag = |key: &str| -> Option<usize> {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
    };
    let runs = flag("--runs").unwrap_or(4);
    let sys = build_system(&name)?;
    let nsteps = flag("--steps").unwrap_or(sys.nsteps);
    let max_iter = flag("--iters").unwrap_or(sys.max_iter);

    println!(
        "{}: {} spins, {} slices, {} iterations, {runs} runs",
        sys.name, sys.nspins, nsteps, max_iter
    );

    let methods = [
        Method::Exact,
        Method::Split(2),
        Method::Split(3),
        Method::Split(4),
        Method::Adaptive,
    ];

    // infidelity[method][iteration][run] and the matching inner timings.
    let mut infidelity: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); max_iter + 1]; methods.len()];
    let mut inner_time: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); max_iter + 1]; methods.len()];

    for (m, method) in methods.iter().enumerate() {
        let started = std::time::Instant::now();
        for run in 0..runs {
            let mut control: ControlSystem =
                optimconset(build_options(&sys, *method, nsteps, max_iter))?;
            let guess = random_pulse(nsteps, sys.operators.len(), Some(run as u64 + 1));
            let out = fmaxnewton(&mut control, &GrapeXyCost, &guess)?;

            // Carry the last achieved value forward once the run terminates,
            // exactly as the MATLAB post-processing does.
            let mut last_f = f64::NAN;
            let mut last_t = f64::NAN;
            for it in 0..=max_iter {
                let f = out.data.fx_chk_store[it];
                if !f.is_nan() {
                    last_f = f;
                }
                let t = out.data.inner_time_history()[it];
                if !t.is_nan() {
                    last_t = t;
                }
                infidelity[m][it].push((1.0 - last_f).max(f64::EPSILON));
                inner_time[m][it].push(last_t);
            }
        }
        println!(
            "  {:<8} {:>8.2} s",
            method.label(),
            started.elapsed().as_secs_f64()
        );
    }

    // Median convergence curves.
    let mut header = vec!["iteration".to_string()];
    for method in &methods {
        header.push(format!("infidelity_{}", method.label()));
        header.push(format!("inner_time_{}", method.label()));
    }
    let header_refs: Vec<&str> = header.iter().map(|s| s.as_str()).collect();

    let mut rows = Vec::with_capacity(max_iter + 1);
    for it in 0..=max_iter {
        let mut row = vec![it as f64];
        for m in 0..methods.len() {
            row.push(median(infidelity[m][it].clone()));
            row.push(median(inner_time[m][it].clone()));
        }
        rows.push(row);
    }

    let path = format!("{}_convergence.csv", sys.name);
    qoala::examples_io::write_table_csv(&path, &header_refs, &rows)?;
    println!("median convergence written to {path}");

    // Speed-up of each split method over the exact one, at matched accuracy.
    let final_row = rows.last().expect("at least one iteration");
    println!("\nfinal median infidelity and time inside the objective:");
    for (m, method) in methods.iter().enumerate() {
        println!(
            "  {:<8} 1-F = {:.3e}   inner time = {:.3} s",
            method.label(),
            final_row[1 + 2 * m],
            final_row[2 + 2 * m]
        );
    }
    let exact_time = final_row[2];
    let adaptive_time = final_row[2 + 2 * (methods.len() - 1)];
    if adaptive_time > 0.0 {
        println!(
            "\nQOALA is {:.1}x faster inside the objective than the exact method",
            exact_time / adaptive_time
        );
    }
    Ok(())
}
