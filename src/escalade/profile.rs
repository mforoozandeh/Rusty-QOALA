//! What a pulse does: magnetisation across offsets and field strengths.
//!
//! The numbers behind the MATLAB `visualisation/` scripts - the excitation
//! profile of `ESCALADE_pulse_sim.m`, and the `dphi/dB1` curve and the
//! offset-by-field map of `ESCALADE_Bloch_B1.m` - over the same grids.
//!
//! The scripts run a separate rotation-matrix Bloch simulation with its own
//! phase convention.  Everything here propagates with the SU(2) propagators
//! the objective itself uses, so what is plotted is, by construction, what
//! was optimised.  As in the scripts, the magnetisation starts along +z.

use super::propagators::{slice, spin_operator, Op2};
use crate::optim::ObjectiveRequest;
use nalgebra::DMatrix;

/// A magnetisation vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Magnetisation {
    /// x component.
    pub x: f64,
    /// y component.
    pub y: f64,
    /// z component.
    pub z: f64,
}

impl Magnetisation {
    /// Length of the transverse part, the scripts' `Ixy`.
    pub fn transverse(&self) -> f64 {
        self.x.hypot(self.y)
    }

    /// The scripts' `Phase`: `angle(complex(Iy, Ix)) / pi`, in `[-1, 1]`.
    pub fn phase(&self) -> f64 {
        self.x.atan2(self.y) / std::f64::consts::PI
    }
}

/// Where +z magnetisation ends up after `pulse` (`np x 2`, dimensionless),
/// applied for `tau_p` seconds at a field of `rf_hz` to a spin `offset_hz`
/// off resonance.
pub fn final_magnetisation(
    pulse: &DMatrix<f64>,
    tau_p: f64,
    rf_hz: f64,
    offset_hz: f64,
) -> Magnetisation {
    let two_pi = 2.0 * std::f64::consts::PI;
    let n = pulse.nrows();
    let dt = tau_p / n.max(1) as f64;
    let mut u = Op2::identity();
    for k in 0..n {
        let step = slice(
            dt,
            two_pi * rf_hz,
            pulse[(k, 0)],
            pulse[(k, 1)],
            two_pi * offset_hz,
            ObjectiveRequest::Value,
        );
        u = step.u * u;
    }
    let rho = u * spin_operator([0.0, 0.0, 2.0]) * u.adjoint();
    let component = |axis: [f64; 3]| (spin_operator(axis) * rho).trace().re;
    Magnetisation {
        x: component([1.0, 0.0, 0.0]),
        y: component([0.0, 1.0, 0.0]),
        z: component([0.0, 0.0, 1.0]),
    }
}

/// One point of an excitation profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfilePoint {
    /// Resonance offset in Hz.
    pub offset_hz: f64,
    /// Final magnetisation.
    pub m: Magnetisation,
}

/// The excitation profile over one and a half times the bandwidth, as
/// `ESCALADE_pulse_sim` plots it.
pub fn offset_profile(
    pulse: &DMatrix<f64>,
    tau_p: f64,
    rf_hz: f64,
    sw_hz: f64,
    npoints: usize,
) -> Vec<ProfilePoint> {
    plot_offsets(sw_hz, npoints)
        .into_iter()
        .map(|offset_hz| ProfilePoint {
            offset_hz,
            m: final_magnetisation(pulse, tau_p, rf_hz, offset_hz),
        })
        .collect()
}

/// Final y magnetisation over resonance offset and field strength: the
/// contour and mesh panels of `ESCALADE_Bloch_B1`.
#[derive(Debug, Clone, PartialEq)]
pub struct B1Map {
    /// Offsets in Hz, one per column.
    pub offsets_hz: Vec<f64>,
    /// Field as a fraction of the nominal one, one per row.
    pub scales: Vec<f64>,
    /// `iy[(row, column)]` for `scales[row]` and `offsets_hz[column]`.
    pub iy: DMatrix<f64>,
    /// Peak field the pulse reaches at the nominal amplitude, in Hz: the
    /// scripts' `RF_max`, which scales their offset axis.
    pub rf_max_hz: f64,
}

/// The offset-by-field map, over one and a half times the bandwidth and
/// fields from half to one and a half times nominal.
pub fn b1_map(pulse: &DMatrix<f64>, tau_p: f64, rf_hz: f64, sw_hz: f64, npoints: usize) -> B1Map {
    let offsets_hz = plot_offsets(sw_hz, npoints);
    let scales = b1_scales(npoints);
    let iy = DMatrix::from_fn(scales.len(), offsets_hz.len(), |r, c| {
        final_magnetisation(pulse, tau_p, rf_hz * scales[r], offsets_hz[c]).y
    });
    B1Map {
        offsets_hz,
        scales,
        iy,
        rf_max_hz: rf_hz * max_amplitude(pulse),
    }
}

/// How fast the on-resonance phase moves with the field, in degrees per 1%
/// of field, at fields from half to one and a half times nominal.
///
/// `(scale, dphi)` pairs: the phase at `1.005 scale` minus the phase at
/// `0.995 scale`, wrapped into `[-180, 180]`, as `ESCALADE_Bloch_B1` draws it.
pub fn phase_sensitivity(
    pulse: &DMatrix<f64>,
    tau_p: f64,
    rf_hz: f64,
    npoints: usize,
) -> Vec<(f64, f64)> {
    let phase = |scale: f64| {
        let m = final_magnetisation(pulse, tau_p, rf_hz * scale, 0.0);
        m.x.atan2(m.y).to_degrees()
    };
    b1_scales(npoints)
        .into_iter()
        .map(|z| {
            let mut dphi = phase(1.005 * z) - phase(0.995 * z);
            if dphi < -180.0 {
                dphi += 360.0;
            } else if dphi > 180.0 {
                dphi -= 360.0;
            }
            (z, dphi)
        })
        .collect()
}

/// The largest `sqrt(f^2 + g^2)` in the pulse.
pub fn max_amplitude(pulse: &DMatrix<f64>) -> f64 {
    (0..pulse.nrows())
        .map(|r| pulse[(r, 0)].hypot(pulse[(r, 1)]))
        .fold(0.0, f64::max)
}

fn plot_offsets(sw_hz: f64, npoints: usize) -> Vec<f64> {
    linspace(-0.75 * sw_hz, 0.75 * sw_hz, npoints)
}

fn b1_scales(npoints: usize) -> Vec<f64> {
    linspace(0.5, 1.5, npoints)
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![b],
        _ => (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hard pulse along +x lasting a quarter period of its field.
    fn hard_90(n: usize) -> (DMatrix<f64>, f64, f64) {
        let rf = 10000.0;
        (
            DMatrix::from_fn(n, 2, |_, c| if c == 0 { 1.0 } else { 0.0 }),
            0.25 / rf,
            rf,
        )
    }

    #[test]
    fn a_hard_90_on_resonance_puts_z_onto_minus_y() {
        let (pulse, tau, rf) = hard_90(8);
        let m = final_magnetisation(&pulse, tau, rf, 0.0);
        assert!(m.x.abs() < 1e-12 && (m.y + 1.0).abs() < 1e-12 && m.z.abs() < 1e-12);
        assert!((m.transverse() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nothing_happens_without_a_pulse() {
        let pulse = DMatrix::zeros(4, 2);
        let m = final_magnetisation(&pulse, 1e-3, 10000.0, 1234.0);
        assert!((m.z - 1.0).abs() < 1e-12 && m.transverse() < 1e-12);
    }

    #[test]
    fn the_grids_span_what_the_scripts_plot() {
        let (pulse, tau, rf) = hard_90(4);
        let profile = offset_profile(&pulse, tau, rf, 20000.0, 11);
        assert_eq!(profile.len(), 11);
        assert!((profile[0].offset_hz + 15000.0).abs() < 1e-9);
        assert!((profile[10].offset_hz - 15000.0).abs() < 1e-9);
        // Magnetisation is conserved.
        for p in &profile {
            let len = (p.m.x.powi(2) + p.m.y.powi(2) + p.m.z.powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-12);
        }

        let map = b1_map(&pulse, tau, rf, 20000.0, 9);
        assert_eq!(map.iy.shape(), (9, 9));
        assert!((map.scales[0] - 0.5).abs() < 1e-15 && (map.scales[8] - 1.5).abs() < 1e-15);
        assert!((map.rf_max_hz - rf).abs() < 1e-9);
        // Centre of the map is the nominal field on resonance: -y.
        assert!((map.iy[(4, 4)] + 1.0).abs() < 1e-12);
    }

    /// A hard pulse's on-resonance phase does not move with the field until
    /// the flip angle passes 180 degrees, where it jumps.
    #[test]
    fn phase_sensitivity_is_flat_for_a_hard_pulse_below_180() {
        let (pulse, tau, rf) = hard_90(4);
        for (z, dphi) in phase_sensitivity(&pulse, tau, rf, 11) {
            assert!(dphi.abs() < 1e-9, "scale {z}: {dphi}");
        }
    }
}
