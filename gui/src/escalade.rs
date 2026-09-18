//! The ESCALADE problem the user edits, how it becomes a run, and what a
//! finished pulse does.
//!
//! Like [`crate::setup`], the setup is `serde`-serialisable so that it can
//! travel through storage, a URL fragment, an exported file and the Web
//! Worker unchanged.

use nalgebra::DMatrix;
use qoala::escalade::profile::{self, B1Map, ProfilePoint};
use qoala::escalade::{magnetisation, Escalade, States};
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

    fn vector(self) -> [f64; 3] {
        match self {
            Direction::PlusX => [1.0, 0.0, 0.0],
            Direction::MinusX => [-1.0, 0.0, 0.0],
            Direction::PlusY => [0.0, 1.0, 0.0],
            Direction::MinusY => [0.0, -1.0, 0.0],
            Direction::PlusZ => [0.0, 0.0, 1.0],
            Direction::MinusZ => [0.0, 0.0, -1.0],
        }
    }

    fn state(self) -> States {
        let [x, y, z] = self.vector();
        States::Single(magnetisation(x, y, z))
    }
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
    /// Starting magnetisation.
    pub from: Direction,
    /// Target magnetisation.
    pub to: Direction,
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

    /// The library description of this problem.
    pub fn spec(&self) -> Escalade {
        Escalade {
            nspins: self.nspins,
            np_pulse: self.nslices,
            tau_p: self.duration_s,
            rf: self.fields_hz(),
            sw: self.sw_hz,
            initial: self.from.state(),
            target: self.to.state(),
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
        if self.from == self.to {
            out.push("the initial and target magnetisation are the same".into());
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

/// What a finished pulse does, worked out once when the run ends.
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    /// Final magnetisation over one and a half times the bandwidth, at the
    /// nominal field.
    pub profile: Vec<ProfilePoint>,
    /// Final y magnetisation over offset and field strength.
    pub map: B1Map,
    /// On-resonance phase sensitivity to the field, `(scale, degrees)`.
    pub dphi: Vec<(f64, f64)>,
    /// The largest amplitude in the pulse, as a fraction of the nominal field.
    pub max_amplitude: f64,
}

/// Work out what `waveform` (row-major, `[slice][x, y]`) does under `setup`.
pub fn analyse(setup: &EscaladeSetup, waveform: &[Vec<f64>]) -> Analysis {
    let pulse = DMatrix::from_fn(waveform.len(), 2, |r, c| {
        waveform[r].get(c).copied().unwrap_or(0.0)
    });
    let (tau, rf, sw) = (setup.duration_s, setup.rf_hz, setup.sw_hz);
    Analysis {
        profile: profile::offset_profile(&pulse, tau, rf, sw, PROFILE_POINTS),
        map: profile::b1_map(&pulse, tau, rf, sw, MAP_POINTS),
        dphi: profile::phase_sensitivity(&pulse, tau, rf, MAP_POINTS),
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
        assert_eq!(a.profile.len(), PROFILE_POINTS);
        assert_eq!(a.map.iy.shape(), (MAP_POINTS, MAP_POINTS));
        assert_eq!(a.dphi.len(), MAP_POINTS);
        assert!((a.max_amplitude - 0.29f64.sqrt()).abs() < 1e-12);
    }
}
