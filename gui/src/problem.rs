//! Which algorithm, and the problem for it: the one value the interface
//! stores, shares, exports and hands to a run.

use serde::{Deserialize, Serialize};

use crate::escalade::EscaladeSetup;
use crate::setup::Setup;

/// The two optimisers the application offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algorithm {
    /// Coupled spins, adaptive operator splitting.
    Qoala,
    /// Uncoupled spins across a band, analytic Hessian.
    Escalade,
}

impl Algorithm {
    /// Both, for a switch.
    pub const ALL: [Algorithm; 2] = [Algorithm::Qoala, Algorithm::Escalade];

    /// Label for the switch.
    pub fn name(self) -> &'static str {
        match self {
            Algorithm::Qoala => "QOALA",
            Algorithm::Escalade => "ESCALADE",
        }
    }

    /// One line on what it is for.
    pub fn blurb(self) -> &'static str {
        match self {
            Algorithm::Qoala => "coupled spins: adaptive split-operator pulse optimisation",
            Algorithm::Escalade => "single spins across a band: broadband and B1-robust pulses",
        }
    }
}

/// A problem for one of the algorithms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Problem {
    /// A QOALA setup.
    Qoala(Setup),
    /// An ESCALADE setup.
    Escalade(EscaladeSetup),
}

impl Problem {
    /// Read a problem from JSON.
    ///
    /// Links, stored sessions and exported files from before ESCALADE hold a
    /// bare QOALA [`Setup`]; those still load.
    pub fn from_json(json: &str) -> Option<Problem> {
        serde_json::from_str::<Problem>(json)
            .ok()
            .or_else(|| serde_json::from_str::<Setup>(json).ok().map(Problem::Qoala))
    }

    /// Which algorithm this is for.
    pub fn algorithm(&self) -> Algorithm {
        match self {
            Problem::Qoala(_) => Algorithm::Qoala,
            Problem::Escalade(_) => Algorithm::Escalade,
        }
    }

    /// The name shown in the preset menu and used for filenames.
    pub fn name(&self) -> &str {
        match self {
            Problem::Qoala(s) => &s.name,
            Problem::Escalade(s) => &s.name,
        }
    }

    /// Anything that would stop the run, fit for the screen.
    pub fn problems(&self) -> Vec<String> {
        match self {
            Problem::Qoala(s) => s.problems(),
            Problem::Escalade(s) => s.problems(),
        }
    }

    /// Waveform column names.
    pub fn channel_names(&self) -> Vec<String> {
        match self {
            Problem::Qoala(s) => s.channel_names(),
            Problem::Escalade(s) => s.channel_names(),
        }
    }

    /// What each waveform column is multiplied by to reach rad/s.
    pub fn channel_amplitudes_rad_per_s(&self) -> Vec<f64> {
        match self {
            Problem::Qoala(s) => s.channel_amplitudes_rad_per_s(),
            Problem::Escalade(s) => s.channel_amplitudes_rad_per_s(),
        }
    }

    /// Slice widths in seconds.
    pub fn dt(&self) -> Vec<f64> {
        match self {
            Problem::Qoala(s) => s.dt(),
            Problem::Escalade(s) => s.dt(),
        }
    }

    /// Pulse duration in seconds.
    pub fn duration_s(&self) -> f64 {
        match self {
            Problem::Qoala(s) => s.duration_s,
            Problem::Escalade(s) => s.duration_s,
        }
    }

    /// Number of time slices.
    pub fn nslices(&self) -> usize {
        match self {
            Problem::Qoala(s) => s.nslices,
            Problem::Escalade(s) => s.nslices,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;

    #[test]
    fn both_kinds_survive_json() {
        for problem in presets::every() {
            let json = serde_json::to_string(&problem).unwrap();
            assert_eq!(Problem::from_json(&json), Some(problem));
        }
    }

    /// A bare QOALA setup is what links and storage held before ESCALADE.
    #[test]
    fn a_legacy_setup_still_loads() {
        let setup = presets::default_setup();
        let json = serde_json::to_string(&setup).unwrap();
        assert_eq!(Problem::from_json(&json), Some(Problem::Qoala(setup)));
        assert_eq!(Problem::from_json("not json"), None);
    }
}
