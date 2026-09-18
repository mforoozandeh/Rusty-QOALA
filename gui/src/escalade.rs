//! The ESCALADE problem the user edits, how it becomes a run, and what a
//! finished pulse does.
//!
//! Like [`crate::setup`], the setup is `serde`-serialisable so that it can
//! travel through storage, a URL fragment, an exported file and the Web
//! Worker unchanged.

use nalgebra::DMatrix;
use qoala::escalade::profile::{self, B1Map, Magnetisation, ProfilePoint};
use qoala::escalade::propagators::Op2;
use qoala::escalade::{magnetisation, rotation, Escalade, States};
use serde::{Deserialize, Serialize};

/// Largest number of spins the editor offers.
pub const MAX_SPINS: usize = 401;

/// Largest number of fields the editor offers.
pub const MAX_FIELDS: usize = 101;

/// Grid resolution for the offset profile and the B1 map.
const PROFILE_POINTS: usize = 301;
const MAP_POINTS: usize = 101;

/// A magnetisation direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// +x.
    PlusX,
    /// -x.
    MinusX,
    /// +y.
    PlusY,
    /// -y.
    MinusY,
    /// +z.
    PlusZ,
    /// -z.
    MinusZ,
}

impl Direction {
    /// All six, for a menu.
    pub const ALL: [Direction; 6] = [
        Direction::PlusX,
        Direction::MinusX,
        Direction::PlusY,
        Direction::MinusY,
        Direction::PlusZ,
        Direction::MinusZ,
    ];

    /// Label for a menu.
    pub fn name(self) -> &'static str {
        match self {
            Direction::PlusX => "+x",
            Direction::MinusX => "-x",
            Direction::PlusY => "+y",
            Direction::MinusY => "-y",
            Direction::PlusZ => "+z",
            Direction::MinusZ => "-z",
        }
    }

    fn vector(self) -> [i8; 3] {
        match self {
            Direction::PlusX => [1, 0, 0],
            Direction::MinusX => [-1, 0, 0],
            Direction::PlusY => [0, 1, 0],
            Direction::MinusY => [0, -1, 0],
            Direction::PlusZ => [0, 0, 1],
            Direction::MinusZ => [0, 0, -1],
        }
    }

    /// As a unit magnetisation vector.
    pub fn magnetisation(self) -> Magnetisation {
        let [x, y, z] = self.vector().map(f64::from);
        Magnetisation::new(x, y, z)
    }

    fn state(self) -> States {
        let m = self.magnetisation();
        States::Single(magnetisation(m.x, m.y, m.z))
    }
}

/// What the pulse has to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Goal {
    /// Take one magnetisation direction to another.
    #[default]
    Transfer,
    /// Turn every direction the same way: +x, +y and +z each to its image
    /// under one rotation.
    Rotation,
}

impl Goal {
    /// Both, for a switch.
    pub const ALL: [Goal; 2] = [Goal::Transfer, Goal::Rotation];

    /// Label for the switch.
    pub fn name(self) -> &'static str {
        match self {
            Goal::Transfer => "State transfer",
            Goal::Rotation => "Universal rotation",
        }
    }
}

/// The axes a rotation is given by the images of.
pub const AXES: [Direction; 3] = [Direction::PlusX, Direction::PlusY, Direction::PlusZ];

/// 90 degrees about +x: x stays, y goes to z and z to -y.
fn default_rotation() -> [Direction; 3] {
    [Direction::PlusX, Direction::PlusZ, Direction::MinusY]
}

/// A band of uncoupled spins, one pulse for all of them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EscaladeSetup {
    /// Name shown in the preset menu and used for export filenames.
    pub name: String,
    /// Spins spread evenly across the bandwidth.
    pub nspins: usize,
    /// Bandwidth in Hz.
    pub sw_hz: f64,
    /// Nominal field in Hz, the largest amplitude the pulse may use.
    pub rf_hz: f64,
    /// Relative spread of field strengths to optimise over: `0.2` means 0.8
    /// to 1.2 times nominal.  Zero optimises for the nominal field alone.
    pub b1_spread: f64,
    /// How many field strengths sample the spread.
    pub b1_fields: usize,
    /// Pulse duration in seconds.
    pub duration_s: f64,
    /// Number of pulse points.
    pub nslices: usize,
    /// State transfer or universal rotation.  Setups from before rotations
    /// have no such field and are transfers.
    #[serde(default)]
    pub goal: Goal,
    /// Starting magnetisation of a transfer.
    pub from: Direction,
    /// Target magnetisation of a transfer.
    pub to: Direction,
    /// Where a rotation takes +x, +y and +z.  Kept while a transfer is
    /// showing.
    #[serde(default = "default_rotation")]
    pub rotation: [Direction; 3],
    /// Newton trust region on the analytic Hessian, rather than L-BFGS.
    pub use_hessian: bool,
    /// Iteration budget.
    pub max_iter: usize,
    /// Stop once the fidelity reaches this.
    pub target_fidelity: f64,
    /// Seed for the random starting pulse; `None` draws from entropy.
    pub seed: Option<u64>,
}

impl EscaladeSetup {
    /// The field strengths the run optimises over, in Hz.
    pub fn fields_hz(&self) -> Vec<f64> {
        let n = self.b1_fields;
        if self.b1_spread == 0.0 || n <= 1 {
            return vec![self.rf_hz];
        }
        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                self.rf_hz * (1.0 - self.b1_spread + 2.0 * self.b1_spread * t)
            })
            .collect()
    }

    /// Whether the run optimises over more than one field strength.
    pub fn is_b1_compensated(&self) -> bool {
        self.fields_hz().len() > 1
    }

    /// The `(from, to)` directions the pulse has to take: one for a transfer,
    /// one per axis for a rotation.
    pub fn transfers(&self) -> Vec<(Direction, Direction)> {
        match self.goal {
            Goal::Transfer => vec![(self.from, self.to)],
            Goal::Rotation => AXES.into_iter().zip(self.rotation).collect(),
        }
    }

    /// The library description of this problem.
    pub fn spec(&self) -> Escalade {
        Escalade {
            nspins: self.nspins,
            np_pulse: self.nslices,
            tau_p: self.duration_s,
            rf: self.fields_hz(),
            sw: self.sw_hz,
            goal: match self.goal {
                Goal::Transfer => qoala::escalade::Goal::Transfer {
                    initial: self.from.state(),
                    target: self.to.state(),
                },
                Goal::Rotation => qoala::escalade::Goal::Rotation(operator(self.rotation)),
            },
            seed: self.seed,
            use_hessian: self.use_hessian,
            max_iter: self.max_iter,
            target_fidelity: self.target_fidelity,
            ..Default::default()
        }
    }

    /// Anything that would make the run fail or be meaningless, as a message
    /// fit to put on screen.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !(1..=MAX_SPINS).contains(&self.nspins) {
            out.push(format!("between 1 and {MAX_SPINS} qubits, please"));
        }
        if !(self.sw_hz >= 0.0 && self.sw_hz.is_finite()) {
            out.push("the bandwidth cannot be negative".into());
        }
        if !(self.rf_hz > 0.0 && self.rf_hz.is_finite()) {
            out.push("the field needs a positive amplitude".into());
        }
        if !(0.0..1.0).contains(&self.b1_spread) {
            out.push("the field spread has to be at least 0 and below 100%".into());
        }
        if self.b1_spread > 0.0 && !(2..=MAX_FIELDS).contains(&self.b1_fields) {
            out.push(format!(
                "a field spread needs between 2 and {MAX_FIELDS} fields"
            ));
        }
        if !(self.duration_s > 0.0 && self.duration_s.is_finite()) {
            out.push("the pulse needs a positive duration".into());
        }
        if self.nslices == 0 {
            out.push("the pulse needs at least one point".into());
        }
        match self.goal {
            Goal::Transfer if self.from == self.to => {
                out.push("the initial and target magnetisation are the same".into());
            }
            Goal::Rotation if !is_rotation(self.rotation) => out.push(
                "the images of +x, +y and +z have to be a rotation: \
                 each axis once, and +x to +y to +z right-handed"
                    .into(),
            ),
            Goal::Rotation if self.rotation == AXES => {
                out.push("the rotation leaves every axis where it is".into());
            }
            _ => {}
        }
        if !(self.target_fidelity > 0.0 && self.target_fidelity <= 1.0) {
            out.push("the target fidelity has to be above 0 and at most 1".into());
        }
        out
    }

    /// Waveform columns: x and y.
    pub fn channel_names(&self) -> Vec<String> {
        vec!["x".into(), "y".into()]
    }

    /// The nominal field in rad/s for each waveform column.
    pub fn channel_amplitudes_rad_per_s(&self) -> Vec<f64> {
        vec![2.0 * std::f64::consts::PI * self.rf_hz; 2]
    }

    /// Slice widths, in seconds.
    pub fn dt(&self) -> Vec<f64> {
        vec![self.duration_s / self.nslices.max(1) as f64; self.nslices]
    }
}

/// Whether `images` of +x, +y and +z are a proper rotation: the image of
/// +z is the cross product of the other two, which also rules out a repeated
/// axis.
fn is_rotation(images: [Direction; 3]) -> bool {
    let [a, b, c] = images.map(Direction::vector);
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    cross == c
}

/// The propagator of the rotation taking +x, +y and +z to `images`, a
/// proper rotation, turning by at most 180 degrees: of its two signs, the
/// one nearer a pulse that does nothing.
fn operator(images: [Direction; 3]) -> Op2 {
    // Column c is the image of axis c.
    let [a, b, c] = images.map(|d| d.vector().map(f64::from));
    let r = |row: usize, col: usize| [a, b, c][col][row];
    let cos = ((r(0, 0) + r(1, 1) + r(2, 2) - 1.0) / 2.0).clamp(-1.0, 1.0);
    let angle = cos.acos();
    // Below a half turn the axis is the rotation's antisymmetric part; at a
    // half turn, where that vanishes, the rotation is 2 n n^T - 1.
    let skew = [r(2, 1) - r(1, 2), r(0, 2) - r(2, 0), r(1, 0) - r(0, 1)];
    let axis = if skew.iter().any(|v| *v != 0.0) || angle == 0.0 {
        skew
    } else {
        let i = (0..3)
            .max_by(|&p, &q| r(p, p).total_cmp(&r(q, q)))
            .unwrap_or(0);
        let ni = ((r(i, i) + 1.0) / 2.0).sqrt();
        std::array::from_fn(|j| if j == i { ni } else { r(i, j) / (2.0 * ni) })
    };
    rotation(axis, angle)
}

/// What a finished pulse does, worked out once when the run ends.
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    /// One for each transfer the pulse was optimised for.
    pub transfers: Vec<TransferAnalysis>,
    /// The largest amplitude in the pulse, as a fraction of the nominal field.
    pub max_amplitude: f64,
}

/// What the pulse does to magnetisation starting along one direction.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferAnalysis {
    /// Starting magnetisation.
    pub from: Direction,
    /// Where it should end up.
    pub to: Direction,
    /// Final magnetisation over one and a half times the bandwidth, at the
    /// nominal field.
    pub profile: Vec<ProfilePoint>,
    /// Final magnetisation along the target over offset and field strength:
    /// one where the transfer succeeds.
    pub map: B1Map,
    /// On-resonance phase sensitivity to the field, `(scale, degrees)`.
    pub dphi: Vec<(f64, f64)>,
}

/// Work out what `waveform` (row-major, `[slice][x, y]`) does under `setup`.
pub fn analyse(setup: &EscaladeSetup, waveform: &[Vec<f64>]) -> Analysis {
    let pulse = DMatrix::from_fn(waveform.len(), 2, |r, c| {
        waveform[r].get(c).copied().unwrap_or(0.0)
    });
    let (tau, rf, sw) = (setup.duration_s, setup.rf_hz, setup.sw_hz);
    let transfers = setup
        .transfers()
        .into_iter()
        .map(|(from, to)| {
            let (start, target) = (from.magnetisation(), to.magnetisation());
            TransferAnalysis {
                from,
                to,
                profile: profile::offset_profile(&pulse, tau, rf, sw, PROFILE_POINTS, start),
                map: profile::b1_map(&pulse, tau, rf, sw, MAP_POINTS, start, target),
                dphi: profile::phase_sensitivity(&pulse, tau, rf, MAP_POINTS, start),
            }
        })
        .collect();
    Analysis {
        transfers,
        max_amplitude: profile::max_amplitude(&pulse),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;

    #[test]
    fn fields_span_the_spread() {
        let s = presets::escalade_b1_compensated();
        let fields = s.fields_hz();
        assert_eq!(fields.len(), 31);
        assert!((fields[0] - 0.8 * s.rf_hz).abs() < 1e-9);
        assert!((fields[30] - 1.2 * s.rf_hz).abs() < 1e-9);
        assert_eq!(presets::escalade_b1_sensitive().fields_hz(), vec![17000.0]);
    }

    #[test]
    fn the_spec_resolves() {
        for setup in presets::escalade() {
            let settings = setup.spec().resolve().expect("resolves");
            assert_eq!(settings.np_pulse, setup.nslices);
            assert_eq!(settings.rf.len(), setup.fields_hz().len());
        }
    }

    #[test]
    fn nonsense_is_reported() {
        let mut s = presets::escalade_b1_sensitive();
        s.to = s.from;
        s.b1_spread = 1.5;
        s.b1_fields = 31;
        s.nslices = 0;
        assert_eq!(s.problems().len(), 3, "{:?}", s.problems());
    }

    #[test]
    fn the_analysis_covers_the_grids() {
        let s = presets::escalade_b1_sensitive();
        let wf = vec![vec![0.5, -0.2]; s.nslices];
        let a = analyse(&s, &wf);
        assert_eq!(a.transfers.len(), 1);
        let t = &a.transfers[0];
        assert_eq!((t.from, t.to), (Direction::PlusZ, Direction::MinusY));
        assert_eq!(t.profile.len(), PROFILE_POINTS);
        assert_eq!(t.map.values.shape(), (MAP_POINTS, MAP_POINTS));
        assert_eq!(t.dphi.len(), MAP_POINTS);
        assert!((a.max_amplitude - 0.29f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn a_universal_rotation_is_shown_as_three_transfers() {
        let s = presets::escalade_universal_rotation();
        assert!(s.problems().is_empty(), "{:?}", s.problems());
        use Direction::*;
        assert_eq!(
            s.transfers(),
            vec![(PlusX, PlusX), (PlusY, PlusZ), (PlusZ, MinusY)]
        );
        let settings = s.spec().resolve().unwrap();
        assert!(matches!(settings.goal, qoala::escalade::Goal::Rotation(_)));

        // Each transfer is analysed, and the map measures along its target:
        // a pulse that does nothing leaves x exactly on x, and y and z
        // exactly off their targets.
        let a = analyse(&s, &vec![vec![0.0, 0.0]; s.nslices]);
        assert_eq!(a.transfers.len(), 3);
        let centre = |t: &TransferAnalysis| t.map.values[(MAP_POINTS / 2, MAP_POINTS / 2)];
        assert!((centre(&a.transfers[0]) - 1.0).abs() < 1e-12);
        assert!(centre(&a.transfers[1]).abs() < 1e-12);
        assert!(centre(&a.transfers[2]).abs() < 1e-12);
    }

    /// Every proper rotation the menus can express becomes the propagator
    /// that turns each axis onto its image, with the sign of the smaller
    /// turn: the one nearer a pulse that does nothing.
    #[test]
    fn every_rotation_becomes_its_propagator() {
        let mut proper = 0;
        for a in Direction::ALL {
            for b in Direction::ALL {
                for c in Direction::ALL {
                    let images = [a, b, c];
                    if !is_rotation(images) {
                        continue;
                    }
                    proper += 1;
                    let w = operator(images);
                    for (axis, image) in AXES.into_iter().zip(images) {
                        let turned = w * op(axis) * w.adjoint();
                        assert!((turned - op(image)).norm() < 1e-12, "{images:?}: {axis:?}");
                    }
                    assert!(w.trace().re >= -1e-12, "{images:?}: {}", w.trace());
                }
            }
        }
        assert_eq!(proper, 24);
    }

    fn op(d: Direction) -> qoala::escalade::propagators::Op2 {
        let m = d.magnetisation();
        magnetisation(m.x, m.y, m.z)
    }

    #[test]
    fn only_a_proper_rotation_is_accepted() {
        use Direction::*;
        let mut s = presets::escalade_universal_rotation();
        // 90 degrees about -x, and a half turn about z.
        for good in [[PlusX, MinusZ, PlusY], [MinusX, MinusY, PlusZ]] {
            s.rotation = good;
            assert!(s.problems().is_empty(), "{good:?}: {:?}", s.problems());
        }
        // A repeated axis, a reflection, and doing nothing.
        for bad in [
            [PlusX, PlusX, MinusY],
            [PlusX, PlusZ, PlusY],
            [PlusX, PlusY, PlusZ],
        ] {
            s.rotation = bad;
            assert_eq!(s.problems().len(), 1, "{bad:?}: {:?}", s.problems());
        }
        // The transfer's own pair is not looked at in a rotation.
        s.rotation = [PlusX, PlusZ, MinusY];
        s.to = s.from;
        assert!(s.problems().is_empty(), "{:?}", s.problems());
    }

    /// Links, stored sessions and exported setups from before rotations
    /// have neither field, and are state transfers.
    #[test]
    fn a_setup_from_before_rotations_loads_as_a_transfer() {
        let mut json = serde_json::to_value(presets::escalade_b1_sensitive()).unwrap();
        let fields = json.as_object_mut().unwrap();
        assert!(fields.remove("goal").is_some() && fields.remove("rotation").is_some());
        let s: EscaladeSetup = serde_json::from_value(json).unwrap();
        assert_eq!(s.goal, Goal::Transfer);
        assert_eq!(s, presets::escalade_b1_sensitive());
    }
}
