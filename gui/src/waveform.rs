//! Waveform amplitudes in physical units.
//!
//! The optimisers work in dimensionless amplitudes; everything a person
//! sees, the plot and the CSV, is in Hz against seconds.  The conversion
//! needs the problem that produced the waveform, which is the whole reason
//! [`crate::app`] keeps the two together.

use crate::problem::Problem;

/// Stair points for one channel: `[time in ms, amplitude in Hz]`, two per
/// slice so the trace is flat across each one.
pub fn stairs(problem: &Problem, waveform: &[Vec<f64>], channel: usize) -> Vec<[f64; 2]> {
    let two_pi = 2.0 * std::f64::consts::PI;
    let amps = problem.channel_amplitudes_rad_per_s();
    let dt = problem.dt();
    let scale = amps.get(channel).copied().unwrap_or(two_pi) / two_pi;

    // Slices are uniform, so a missing one can fall back to the nominal
    // width.  This should never fire - the interface keeps a result with the
    // problem that produced it - but a plot is not worth a panic.
    let nominal = problem.duration_s() / problem.nslices().max(1) as f64;

    let mut points = Vec::with_capacity(2 * waveform.len());
    let mut t = 0.0;
    for (n, row) in waveform.iter().enumerate() {
        let hz = row.get(channel).copied().unwrap_or(0.0) * scale;
        points.push([t * 1e3, hz]);
        t += dt.get(n).copied().unwrap_or(nominal);
        points.push([t * 1e3, hz]);
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;

    fn flat(nslices: usize, nchannels: usize) -> Vec<Vec<f64>> {
        vec![vec![0.5; nchannels]; nslices]
    }

    #[test]
    fn stairs_are_flat_across_each_slice_and_scaled_to_hz() {
        let setup = presets::z2z_2spin_1();
        let waveform = flat(setup.nslices, setup.nchannels());
        let problem = Problem::Qoala(setup.clone());
        let points = stairs(&problem, &waveform, 0);

        assert_eq!(points.len(), 2 * setup.nslices);
        // Half amplitude of a 1000 Hz channel.
        assert!(points.iter().all(|p| (p[1] - 500.0).abs() < 1e-9));
        // Time starts at zero and ends at the pulse duration, in ms.
        assert!((points[0][0] - 0.0).abs() < 1e-12);
        let last = points.last().unwrap()[0];
        assert!((last - setup.duration_s * 1e3).abs() < 1e-9, "{last}");
    }

    #[test]
    fn escalade_stairs_are_scaled_by_the_nominal_field() {
        let setup = presets::escalade_b1_compensated();
        let waveform = flat(setup.nslices, 2);
        let points = stairs(&Problem::Escalade(setup.clone()), &waveform, 1);
        assert!(points
            .iter()
            .all(|p| (p[1] - 0.5 * setup.rf_hz).abs() < 1e-9));
        let last = points.last().unwrap()[0];
        assert!((last - setup.duration_s * 1e3).abs() < 1e-9);
    }

    /// Editing the setup after a run must not be able to take the plot down
    /// with it.  This is what panicked: `time slices` dragged to its minimum
    /// of 2 while a fifty-slice result was on screen.
    #[test]
    fn a_waveform_longer_than_the_setup_does_not_panic() {
        let mut setup = presets::z2z_2spin_1();
        let waveform = flat(setup.nslices, setup.nchannels());
        setup.nslices = 2;
        let points = stairs(&Problem::Qoala(setup), &waveform, 0);
        assert_eq!(points.len(), 2 * waveform.len());
    }

    /// And the same for a channel the setup no longer has.
    #[test]
    fn a_waveform_wider_than_the_setup_does_not_panic() {
        let mut setup = presets::z2z_2spin_1();
        let waveform = flat(setup.nslices, setup.nchannels());
        setup.npairs = 1;
        setup.resize();
        let points = stairs(&Problem::Qoala(setup), &waveform, 3);
        assert_eq!(points.len(), 2 * waveform.len());
    }
}
