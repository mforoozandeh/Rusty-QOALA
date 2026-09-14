//! The problem description the user edits, and how it becomes a QOALA run.
//!
//! Everything here is `serde`-serialisable so that a whole setup can go into
//! `localStorage`, into a URL fragment, or into an exported JSON file, and
//! come back identical.

use nalgebra::DMatrix;
use num_complex::Complex64;
use qoala::error::Result;
use qoala::gates::Gate;
use qoala::linalg::{CDense, CMat};
use qoala::spinops;
use qoala::types::Penalty;
use serde::{Deserialize, Serialize};

/// Largest register the browser build will attempt.
///
/// Four spins means dimension 256 and dense 256x256 propagators per slice,
/// which is native-only territory.
pub const MAX_WEB_SPINS: usize = 3;

/// Largest register the native build offers.
pub const MAX_SPINS: usize = 4;

/// Which Cartesian component a product-operator term uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Component {
    /// x magnetisation.
    X,
    /// y magnetisation.
    Y,
    /// z magnetisation.
    Z,
}

impl Component {
    /// All three, for a menu.
    pub const ALL: [Component; 3] = [Component::X, Component::Y, Component::Z];

    /// Label for a menu.
    pub fn name(self) -> &'static str {
        match self {
            Component::X => "x",
            Component::Y => "y",
            Component::Z => "z",
        }
    }

    fn single_spin_state(self) -> CDense {
        match self {
            Component::X => spinops::x_state_single(),
            Component::Y => spinops::y_state_single(),
            Component::Z => spinops::z_state_single(),
        }
    }
}

/// A product operator: a Cartesian component on each of a set of spins, unit
/// on the rest.
///
/// One term is single-spin magnetisation (`Iz` on spin 1); two terms make a
/// product operator (`2 I1z I2z`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductOperator {
    /// `(spin, component)` for each spin the operator acts on.
    pub terms: Vec<(usize, Component)>,
}

impl ProductOperator {
    /// Magnetisation of one spin along one axis.
    pub fn single(spin: usize, component: Component) -> Self {
        ProductOperator {
            terms: vec![(spin, component)],
        }
    }

    /// The Liouville-space state vector, before normalisation.
    pub fn state(&self, nspins: usize) -> CDense {
        let states: Vec<CDense> = (0..nspins)
            .map(|s| match self.terms.iter().find(|(spin, _)| *spin == s) {
                Some((_, component)) => component.single_spin_state(),
                None => spinops::unit_state(),
            })
            .collect();
        spinops::kron_states(&states)
    }

    /// A short label, e.g. `I1z` or `2 I1z I2z`.
    pub fn label(&self) -> String {
        let mut sorted = self.terms.clone();
        sorted.sort_by_key(|(spin, _)| *spin);
        let body: Vec<String> = sorted
            .iter()
            .map(|(spin, c)| format!("I{}{}", spin + 1, c.name()))
            .collect();
        match body.len() {
            0 => "unit".to_string(),
            1 => body[0].clone(),
            n => format!("{} {}", 1 << (n - 1), body.join(" ")),
        }
    }
}

/// A gate from the library, in a form that survives a round trip through
/// JSON.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GateChoice {
    /// [`Gate::Swap`].
    Swap,
    /// [`Gate::ISwap`].
    ISwap,
    /// [`Gate::SqrtSwap`].
    SqrtSwap,
    /// [`Gate::Cnot`].
    Cnot,
    /// [`Gate::Cz`].
    Cz,
    /// [`Gate::X`].
    X,
    /// [`Gate::Y`].
    Y,
    /// [`Gate::Z`].
    Z,
    /// [`Gate::H`].
    H,
    /// A rotation about an arbitrary axis.
    Rotation {
        /// x component of the axis.
        nx: f64,
        /// y component of the axis.
        ny: f64,
        /// z component of the axis.
        nz: f64,
        /// Angle in radians.
        angle: f64,
    },
}

impl GateChoice {
    /// Every parameter-free gate, for a menu.
    pub const MENU: [GateChoice; 9] = [
        GateChoice::Swap,
        GateChoice::ISwap,
        GateChoice::SqrtSwap,
        GateChoice::Cnot,
        GateChoice::Cz,
        GateChoice::X,
        GateChoice::Y,
        GateChoice::Z,
        GateChoice::H,
    ];

    /// The library gate this stands for.
    pub fn gate(self) -> Gate {
        match self {
            GateChoice::Swap => Gate::Swap,
            GateChoice::ISwap => Gate::ISwap,
            GateChoice::SqrtSwap => Gate::SqrtSwap,
            GateChoice::Cnot => Gate::Cnot,
            GateChoice::Cz => Gate::Cz,
            GateChoice::X => Gate::X,
            GateChoice::Y => Gate::Y,
            GateChoice::Z => Gate::Z,
            GateChoice::H => Gate::H,
            GateChoice::Rotation { nx, ny, nz, angle } => Gate::Rotation { nx, ny, nz, angle },
        }
    }

    /// How many qubits it acts on.
    pub fn qubits(self) -> usize {
        self.gate().qubits()
    }

    /// Label for a menu.
    pub fn name(self) -> &'static str {
        self.gate().name()
    }
}

/// An operator a transfer starts from or aims at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operator {
    /// A product operator, which is what the editor offers.
    Product(ProductOperator),
    /// Raw coefficients on the Liouville basis, for setups the
    /// product-operator editor cannot express.  The strong-coupling example
    /// ships one: both its states sit on a single basis element with opposite
    /// signs.
    Basis {
        /// `(basis index, coefficient)` pairs; everything else is zero.
        coefficients: Vec<(usize, f64)>,
    },
}

impl Operator {
    /// Magnetisation of one spin along one axis.
    pub fn single(spin: usize, component: Component) -> Self {
        Operator::Product(ProductOperator::single(spin, component))
    }

    /// The Liouville-space state vector, before normalisation.
    pub fn state(&self, nspins: usize) -> CDense {
        match self {
            Operator::Product(p) => p.state(nspins),
            Operator::Basis { coefficients } => {
                let dim = 4usize.pow(nspins as u32);
                let mut v = CDense::zeros(dim, 1);
                for (index, value) in coefficients {
                    if *index < dim {
                        v[(*index, 0)] = Complex64::new(*value, 0.0);
                    }
                }
                v
            }
        }
    }

    /// A short label for the interface.
    pub fn label(&self) -> String {
        match self {
            Operator::Product(p) => p.label(),
            Operator::Basis { coefficients } => {
                let body: Vec<String> = coefficients
                    .iter()
                    .map(|(i, v)| format!("{v:+} B{i}"))
                    .collect();
                body.join(" ")
            }
        }
    }

    /// The product operator inside, if there is one.
    pub fn as_product_mut(&mut self) -> Option<&mut ProductOperator> {
        match self {
            Operator::Product(p) => Some(p),
            Operator::Basis { .. } => None,
        }
    }
}

/// What the optimiser is aiming at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Target {
    /// Move one operator onto another.
    Transfer {
        /// Starting operator.
        from: Operator,
        /// Operator to end up with.
        to: Operator,
    },
    /// Synthesise a gate.
    Gate {
        /// Which gate.
        gate: GateChoice,
        /// Which qubits it acts on, in the gate's own order.
        qubits: Vec<usize>,
    },
}

/// One scalar coupling between a pair of spins.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Coupling {
    /// Coupling constant in Hz.
    pub j_hz: f64,
    /// `true` for the isotropic (strong-coupling) form, `false` for the
    /// weak-coupling `zz` form.
    pub strong: bool,
}

impl Default for Coupling {
    fn default() -> Self {
        Coupling {
            j_hz: 0.0,
            strong: false,
        }
    }
}

/// A penalty term and the weight it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PenaltyChoice {
    /// No penalty.
    None,
    /// Norm square.
    Ns,
    /// Norm square of the second derivative.
    Dns,
    /// Spillout norm square.
    Sns,
    /// Spillout norm square on the polar amplitude of an (x, y) pair.
    Snsa,
    /// Spillout norm square on the summed channel magnitude.
    Snsm,
}

impl PenaltyChoice {
    /// Every choice, for a menu.
    pub const MENU: [PenaltyChoice; 6] = [
        PenaltyChoice::None,
        PenaltyChoice::Ns,
        PenaltyChoice::Dns,
        PenaltyChoice::Sns,
        PenaltyChoice::Snsa,
        PenaltyChoice::Snsm,
    ];

    /// The library penalty this stands for.
    pub fn penalty(self) -> Penalty {
        match self {
            PenaltyChoice::None => Penalty::None,
            PenaltyChoice::Ns => Penalty::Ns,
            PenaltyChoice::Dns => Penalty::Dns,
            PenaltyChoice::Sns => Penalty::Sns,
            PenaltyChoice::Snsa => Penalty::Snsa,
            PenaltyChoice::Snsm => Penalty::Snsm,
        }
    }

    /// Label for a menu.
    pub fn name(self) -> &'static str {
        match self {
            PenaltyChoice::None => "none",
            PenaltyChoice::Ns => "NS - norm square",
            PenaltyChoice::Dns => "DNS - smoothing",
            PenaltyChoice::Sns => "SNS - spillout",
            PenaltyChoice::Snsa => "SNSA - polar spillout",
            PenaltyChoice::Snsm => "SNSM - summed spillout",
        }
    }
}

/// The whole problem: spin system, controls, target and pulse parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Setup {
    /// Name shown in the preset menu and used for export filenames.
    pub name: String,
    /// Number of spins.
    pub nspins: usize,
    /// Resonance offset of each spin, in Hz.
    pub offsets_hz: Vec<f64>,
    /// Couplings, indexed by [`pair_index`].
    pub couplings: Vec<Coupling>,
    /// Number of `(x, y)` control pairs.
    pub npairs: usize,
    /// `spin_control[spin][pair]`: does control pair `pair` drive `spin`?
    pub spin_control: Vec<Vec<bool>>,
    /// Maximum control amplitude per pair, in Hz.
    pub amplitudes_hz: Vec<f64>,
    /// What to optimise towards.
    pub target: Target,
    /// Pulse duration in seconds.
    pub duration_s: f64,
    /// Number of time slices.
    pub nslices: usize,
    /// Iteration budget.
    pub max_iter: usize,
    /// Penalty term.
    pub penalty: PenaltyChoice,
    /// Splitting orders the adaptive step may pick from.
    pub splitset: Vec<usize>,
    /// Seed for the random starting waveform; `None` draws from entropy.
    pub seed: Option<u64>,
}

/// Index into [`Setup::couplings`] for the pair `(i, j)`, `i < j`.
///
/// Pairs are laid out row by row of the upper triangle: for four spins,
/// `(0,1) (0,2) (0,3) (1,2) (1,3) (2,3)`.
pub fn pair_index(nspins: usize, i: usize, j: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    let mut index = 0;
    for a in 0..i {
        index += nspins - 1 - a;
    }
    index + (j - i - 1)
}

/// Number of distinct spin pairs in an `nspins` system.
pub fn pair_count(nspins: usize) -> usize {
    nspins * nspins.saturating_sub(1) / 2
}

impl Setup {
    /// Liouville-space dimension, `4^nspins`.
    pub fn dimension(&self) -> usize {
        4usize.pow(self.nspins as u32)
    }

    /// Number of waveform columns: two per control pair.
    pub fn nchannels(&self) -> usize {
        2 * self.npairs
    }

    /// Names for the waveform channels, in column order.
    pub fn channel_names(&self) -> Vec<String> {
        (0..self.npairs)
            .flat_map(|k| {
                [
                    format!("x-control {}", k + 1),
                    format!("y-control {}", k + 1),
                ]
            })
            .collect()
    }

    /// Anything that would make the run fail or be meaningless, as a message
    /// fit to put on screen.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !(2..=MAX_SPINS).contains(&self.nspins) {
            out.push(format!("between 2 and {MAX_SPINS} spins, please"));
        }
        if self.npairs == 0 {
            out.push("at least one control pair is needed".into());
        }
        for k in 0..self.npairs {
            if !(0..self.nspins).any(|s| self.spin_control[s][k]) {
                out.push(format!("control pair {} drives no spin", k + 1));
            }
        }
        for s in 0..self.nspins {
            if !(0..self.npairs).any(|k| self.spin_control[s][k]) {
                out.push(format!("spin {} has no control", s + 1));
            }
        }
        if self.duration_s <= 0.0 {
            out.push("the pulse needs a positive duration".into());
        }
        if self.nslices == 0 {
            out.push("the pulse needs at least one time slice".into());
        }
        if self.amplitudes_hz.iter().any(|a| *a <= 0.0) {
            out.push("every control pair needs a positive maximum amplitude".into());
        }
        if self.couplings.iter().all(|c| c.j_hz == 0.0) {
            out.push("with no couplings the spins cannot talk to each other".into());
        }
        match &self.target {
            Target::Transfer { from, to } => {
                if from == to {
                    out.push("the initial and target operators are the same".into());
                }
                for op in [from, to] {
                    match op {
                        Operator::Product(p) => {
                            if p.terms.is_empty() {
                                out.push("a product operator needs at least one term".into());
                            }
                            if p.terms.iter().any(|(s, _)| *s >= self.nspins) {
                                out.push(
                                    "a product operator names a spin that does not exist".into(),
                                );
                            }
                        }
                        Operator::Basis { coefficients } => {
                            if coefficients.is_empty() {
                                out.push("a basis-coefficient operator is empty".into());
                            }
                            if coefficients.iter().any(|(i, _)| *i >= self.dimension()) {
                                out.push(
                                    "a basis-coefficient operator points outside the space".into(),
                                );
                            }
                        }
                    }
                }
            }
            Target::Gate { gate, qubits } => {
                if qubits.len() != gate.qubits() {
                    out.push(format!(
                        "{} acts on {} qubit(s)",
                        gate.name(),
                        gate.qubits()
                    ));
                } else if qubits.iter().any(|q| *q >= self.nspins) {
                    out.push("the gate names a qubit that does not exist".into());
                } else if gate.qubits() == 2 && qubits[0] == qubits[1] {
                    out.push("a two-qubit gate needs two different qubits".into());
                }
            }
        }
        out
    }

    /// The interaction Hamiltonian, summed over every non-zero coupling.
    pub fn interaction(&self) -> Result<CMat> {
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut acc: Option<CMat> = None;
        for i in 0..self.nspins {
            for j in (i + 1)..self.nspins {
                let coupling = self.couplings[pair_index(self.nspins, i, j)];
                if coupling.j_hz == 0.0 {
                    continue;
                }
                let term = if coupling.strong {
                    spinops::isotropic_coupling(self.nspins, i, j)?
                } else {
                    spinops::zz_coupling(self.nspins, i, j)?
                }
                .scale(Complex64::new(two_pi * coupling.j_hz, 0.0));
                acc = Some(match acc {
                    None => term,
                    Some(a) => a.add(&term)?,
                });
            }
        }
        Ok(match acc {
            Some(a) => a,
            None => CMat::zeros(self.dimension(), self.dimension()),
        })
    }

    /// The Cartesian operator array the drivers want: one row per spin, three
    /// columns per control pair, `Some` where that pair drives that spin.
    ///
    /// This is the checkbox grid, translated.
    pub fn cartops(&self) -> Vec<Vec<Option<CMat>>> {
        let ops = spinops::cartesian_operators(self.nspins);
        (0..self.nspins)
            .map(|s| {
                let mut row: Vec<Option<CMat>> = vec![None; 3 * self.npairs];
                for k in 0..self.npairs {
                    if self.spin_control[s][k] {
                        row[3 * k] = Some(ops[s].0.clone());
                        row[3 * k + 1] = Some(ops[s].1.clone());
                        row[3 * k + 2] = Some(ops[s].2.clone());
                    }
                }
                row
            })
            .collect()
    }

    /// The gate target as a Liouville superoperator, when the target is a
    /// gate.
    pub fn gate_target(&self) -> Result<Option<CDense>> {
        match &self.target {
            Target::Transfer { .. } => Ok(None),
            Target::Gate { gate, qubits } => Ok(Some(
                gate.gate().superoperator(self.nspins, qubits)?.to_dense(),
            )),
        }
    }

    /// Control amplitudes in rad/s, one per waveform column, which is what
    /// the waveform has to be multiplied by to get back to physical units.
    pub fn channel_amplitudes_rad_per_s(&self) -> Vec<f64> {
        let two_pi = 2.0 * std::f64::consts::PI;
        self.amplitudes_hz
            .iter()
            .flat_map(|a| [two_pi * a, two_pi * a])
            .collect()
    }

    /// Slice widths, in seconds.
    pub fn dt(&self) -> Vec<f64> {
        vec![self.duration_s / self.nslices as f64; self.nslices]
    }

    /// Resize every per-spin and per-pair field after the spin or pair count
    /// changes, keeping what was already there.
    pub fn resize(&mut self) {
        self.offsets_hz.resize(self.nspins, 0.0);
        self.couplings
            .resize(pair_count(self.nspins), Coupling::default());
        self.amplitudes_hz.resize(self.npairs, 1000.0);
        self.spin_control.resize(self.nspins, Vec::new());
        for row in &mut self.spin_control {
            row.resize(self.npairs, false);
        }
        // Keep the target inside the register.
        match &mut self.target {
            Target::Transfer { from, to } => {
                for op in [from, to] {
                    if let Some(p) = op.as_product_mut() {
                        p.terms.retain(|(s, _)| *s < self.nspins);
                        if p.terms.is_empty() {
                            p.terms.push((0, Component::Z));
                        }
                    }
                }
            }
            Target::Gate { gate, qubits } => {
                qubits.resize(gate.qubits(), 0);
                for (n, q) in qubits.iter_mut().enumerate() {
                    if *q >= self.nspins {
                        *q = n.min(self.nspins - 1);
                    }
                }
            }
        }
    }

    /// A starting waveform of the right shape, for callers that want to see
    /// one before running.
    pub fn random_pulse(&self) -> DMatrix<f64> {
        qoala::drivers::random_pulse(self.nslices, self.nchannels(), self.seed)
    }
}
