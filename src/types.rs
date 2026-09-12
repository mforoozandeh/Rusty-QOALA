//! Typed replacements for QOALA's MATLAB "magic string" options.
//!
//! In the MATLAB package these are all character strings compared with
//! `strcmp`/`ismember`, which means a typo is only discovered at run time and
//! usually deep inside a propagation loop.  Here each option set is an enum,
//! so the compiler rejects impossible configurations up front.  Every enum
//! keeps a `parse`/`as_str` pair so that the original spelling still works
//! when configuration is read from a file.

use crate::error::{QoalaError, Result};
use std::fmt;

macro_rules! str_enum {
    (
        $(#[$meta:meta])*
        $name:ident { $( $(#[$vmeta:meta])* $variant:ident => [$($alias:literal),+], canonical $canon:literal ),+ $(,)? }
        , error $errmsg:literal
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name { $( $(#[$vmeta])* $variant ),+ }

        impl $name {
            /// The canonical MATLAB spelling of this option.
            pub fn as_str(self) -> &'static str {
                match self { $( $name::$variant => $canon ),+ }
            }
            /// Parse one of the MATLAB spellings (case-insensitive).
            pub fn parse(s: &str) -> Result<Self> {
                let l = s.trim().to_ascii_lowercase();
                $( if [$($alias),+].iter().any(|a| *a == l) { return Ok($name::$variant); } )+
                Err(QoalaError::BadValue(format!(concat!($errmsg, ": {}"), s)))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) }
        }
    };
}

str_enum! {
    /// State-space formalism: Hilbert space (density matrices / propagators)
    /// or Liouville space (state vectors / superoperators).
    StateSpace {
        Hilbert   => ["hilbert"],   canonical "hilbert",
        Liouville => ["liouville"], canonical "liouville",
    }, error "only coded for Hilbert or Liouville spaces"
}

str_enum! {
    /// Basis set the operators are expressed in.
    Basis {
        /// Irreducible spherical tensors; the natural basis for QOALA because
        /// single-spin propagators are Wigner matrices.
        Sphten => ["sphten"], canonical "sphten",
        /// Product Zeeman basis.
        Zeeman => ["zeeman"], canonical "zeeman",
    }, error "only coded for Zeeman or spherical tensor basis sets"
}

str_enum! {
    /// Matrix-exponential algorithm used by [`crate::propagate`].
    PropMethod {
        /// Scaling-and-squaring Pade approximant (MATLAB `expm`).
        Pade   => ["pade"],   canonical "pade",
        /// Truncated Taylor series with scaling and squaring and element chopping.
        Taylor => ["taylor"], canonical "taylor",
        /// Krylov-style series applied directly to a state.
        Krylov => ["krylov"], canonical "krylov",
    }, error "expm method not coded"
}

str_enum! {
    /// Whether the interaction propagators are recomputed, carried in memory,
    /// or cached to the scratch directory.
    PropCache {
        /// Precompute every splitting variant and keep them in memory.
        Carry => ["carry"], canonical "carry",
        /// Precompute and save to `scratch/<job_id>/prop_t<q>o<p>.bin`.
        Store => ["store"], canonical "store",
        /// Recompute whenever the splitting parameters change.
        Calc  => ["calc"],  canonical "calc",
    }, error "unknown propagator precalculation method"
}

str_enum! {
    /// Which GRAPE flavour is being run.
    CtrlType {
        /// Point-to-point state transfer.
        PointToPoint      => ["pp", "point-to-point"], canonical "PP",
        /// Universal rotation (gate) synthesis.
        UniversalRotation => ["ur", "universal-rotation"], canonical "UR",
    }, error "unknown control algorithm type"
}

str_enum! {
    /// Fidelity functional.
    Fidelity {
        /// `Re<target|rho(T)>`, range [-1,+1].
        Real   => ["real"],   canonical "real",
        /// `|<target|rho(T)>|^2`, range [0,+1].
        Square => ["square"], canonical "square",
    }, error "control.fidelity can be 'real' or 'square'"
}

str_enum! {
    /// Optimisation algorithm.
    OptMethod {
        /// Limited-memory BFGS quasi-Newton.
        Lbfgs         => ["lbfgs"],          canonical "lbfgs",
        /// Newton-Raphson with rational-function (RFO) Hessian regularisation.
        NewtonRaphson => ["newton-raphson"], canonical "newton-raphson",
        /// Gauss-Newton (reported by the footer; shares the Newton code path).
        GaussNewton   => ["gauss-newton"],   canonical "gauss-newton",
    }, error "unknown optimisation method"
}

str_enum! {
    /// How the adaptive step decides that the current splitting is good enough.
    AdaptMethod {
        /// Compare against the previous, cheaper, estimate.
        Gain  => ["gain"],  canonical "gain",
        /// Compare against an exact propagation of the whole waveform.
        Exact => ["exact"], canonical "exact",
    }, error "allowable adapt_method values are 'gain' and 'exact'"
}

str_enum! {
    /// What the adaptivity tolerance is scaled by.
    AdaptScale {
        /// No scaling.
        None       => ["none"],       canonical "none",
        /// Scale by the current infidelity.
        Infidelity => ["infidelity"], canonical "infidelity",
        /// Scale by the norm of the last gradient.
        Gradient   => ["gradient"],   canonical "gradient",
    }, error "allowable adapt_scale values are 'none', 'infidelity' or 'gradient'"
}

str_enum! {
    /// Waveform penalty terms.
    Penalty {
        /// No penalty.
        None => ["none"], canonical "none",
        /// Norm square of the waveform.
        Ns   => ["ns"],   canonical "NS",
        /// Norm square of the waveform's second derivative (smoothing).
        Dns  => ["dns"],  canonical "DNS",
        /// Spillout norm square: penalise only excursions past the bounds.
        Sns  => ["sns"],  canonical "SNS",
        /// Spillout norm square on the polar amplitude of an (x,y) pair.
        Snsa => ["snsa"], canonical "SNSA",
        /// Spillout norm square on the summed channel magnitude.
        Snsm => ["snsm"], canonical "SNSM",
        /// Roughness of the effective-field polar angle (adiabaticity).
        Adiab => ["adiab"], canonical "ADIAB",
    }, error "unknown penalty function type"
}

str_enum! {
    /// Whether the gradient operators are stored as full composite-space
    /// matrices or as single-spin blocks.
    GradOps {
        Full   => ["full"],   canonical "full",
        Sparse => ["sparse"], canonical "sparse",
    }, error "gradops must be 'full' or 'sparse'"
}

str_enum! {
    /// Rotation axis a control channel acts on.
    Axis {
        X => ["x"], canonical "x",
        Y => ["y"], canonical "y",
        Z => ["z"], canonical "z",
    }, error "control axis must be 'x', 'y' or 'z'"
}

/// Which objective function to evaluate.
///
/// Replaces the MATLAB function handles that were interrogated with
/// `contains(func2str(f),{'escalade','qoala'})`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectiveFn {
    /// Adaptive split-operator GRAPE, point-to-point (`grape_state_qoala`).
    StateQoala,
    /// Adaptive split-operator GRAPE, universal rotation (`grape_ugate_qoala`).
    UgateQoala,
    /// Exact auxiliary-matrix GRAPE, point-to-point (`grape_state_auxmat`).
    StateAuxmat,
    /// Exact auxiliary-matrix GRAPE, universal rotation (`grape_ugate_auxmat`).
    UgateAuxmat,
    /// Plain forward propagation used as an independent fidelity check.
    WaveformFidelity,
}

impl ObjectiveFn {
    /// Name as it appears in the MATLAB reports.
    pub fn as_str(self) -> &'static str {
        match self {
            ObjectiveFn::StateQoala => "grape_state_qoala",
            ObjectiveFn::UgateQoala => "grape_ugate_qoala",
            ObjectiveFn::StateAuxmat => "grape_state_auxmat",
            ObjectiveFn::UgateAuxmat => "grape_ugate_auxmat",
            ObjectiveFn::WaveformFidelity => "waveform_fidelity",
        }
    }
    /// True for the adaptive split-operator (QOALA/ESCALADE) family, which is
    /// what `contains(func2str(f),{'escalade','qoala'})` tested for.
    pub fn is_qoala(self) -> bool {
        matches!(self, ObjectiveFn::StateQoala | ObjectiveFn::UgateQoala)
    }
    /// True when the objective works with propagators rather than states.
    pub fn is_ugate(self) -> bool {
        matches!(self, ObjectiveFn::UgateQoala | ObjectiveFn::UgateAuxmat)
    }
    /// Parse a MATLAB function-handle name.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().trim_start_matches('@') {
            "grape_state_qoala" | "grape_state_escalade" => Ok(ObjectiveFn::StateQoala),
            "grape_ugate_qoala" | "grape_ugate_escalade" => Ok(ObjectiveFn::UgateQoala),
            "grape_state_auxmat" => Ok(ObjectiveFn::StateAuxmat),
            "grape_ugate_auxmat" => Ok(ObjectiveFn::UgateAuxmat),
            "waveform_fidelity" => Ok(ObjectiveFn::WaveformFidelity),
            other => Err(QoalaError::BadValue(format!(
                "unknown objective function: {other}"
            ))),
        }
    }
}

impl fmt::Display for ObjectiveFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Operator-splitting method, tied one-to-one to its integer order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SplitMethod {
    /// Order 0: ignore the interaction entirely.
    Uncoupled,
    /// Order 1: Lie-Trotter.
    Lie,
    /// Order 2: Strang.
    Strang,
    /// Order 3: Blanes, with complex coefficients.
    Blanes3,
    /// Order 4: Blanes.
    Blanes4,
    /// Order 6: Omelyan.
    Omelyan6,
}

impl SplitMethod {
    /// Integer splitting order used everywhere in the numerics.
    pub fn order(self) -> usize {
        match self {
            SplitMethod::Uncoupled => 0,
            SplitMethod::Lie => 1,
            SplitMethod::Strang => 2,
            SplitMethod::Blanes3 => 3,
            SplitMethod::Blanes4 => 4,
            SplitMethod::Omelyan6 => 6,
        }
    }
    /// Recover the method from its order.
    pub fn from_order(order: usize) -> Result<Self> {
        Ok(match order {
            0 => SplitMethod::Uncoupled,
            1 => SplitMethod::Lie,
            2 => SplitMethod::Strang,
            3 => SplitMethod::Blanes3,
            4 => SplitMethod::Blanes4,
            6 => SplitMethod::Omelyan6,
            n => return Err(QoalaError::BadValue(format!("unknown split order: {n}"))),
        })
    }
    /// Name as printed by the MATLAB parser.
    pub fn as_str(self) -> &'static str {
        match self {
            SplitMethod::Uncoupled => "uncoupled",
            SplitMethod::Lie => "Lie-1st",
            SplitMethod::Strang => "Strang-2nd",
            SplitMethod::Blanes3 => "Blanes-3rd",
            SplitMethod::Blanes4 => "Blanes-4th",
            SplitMethod::Omelyan6 => "Omelyan-6th",
        }
    }
    /// Parse the MATLAB `split_method` strings.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "uncoupled" | "none" => SplitMethod::Uncoupled,
            "Lie" | "Lie-1st" => SplitMethod::Lie,
            "Strang" | "Strang-2nd" => SplitMethod::Strang,
            "Blanes-3rd" => SplitMethod::Blanes3,
            "Blanes" | "Blanes-4th" => SplitMethod::Blanes4,
            "Omelyan" | "Omelyan-6th" => SplitMethod::Omelyan6,
            other => {
                return Err(QoalaError::BadValue(format!(
                    "Unknown splitting method: {other}"
                )))
            }
        })
    }
}

impl fmt::Display for SplitMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why the optimiser stopped.
///
/// Marked `#[non_exhaustive]`: the GUI work added [`ExitFlag::Cancelled`] and
/// more reasons are plausible, so downstream matches need a wildcard arm and
/// a new variant is no longer a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExitFlag {
    /// `norm(gradient,2) < tol_g`.
    GradientTolerance,
    /// `norm(step,1) < tol_x`.
    StepTolerance,
    /// `fx > tol_f`.
    FidelityTolerance,
    /// Iteration budget exhausted.
    MaxIterations,
    /// The line search could not find an acceptable point.
    LineSearchFailed,
    /// A [`crate::optim::ProgressSink`] asked for the run to stop.
    Cancelled,
}

impl ExitFlag {
    /// The message printed in the MATLAB footer.
    pub fn message(self) -> &'static str {
        match self {
            ExitFlag::GradientTolerance => "norm(gradient,2) < tol_gfx",
            ExitFlag::StepTolerance => "norm(step,1) < tol_x",
            ExitFlag::FidelityTolerance => "fx > tol_f",
            ExitFlag::MaxIterations => "number of iterations exceeded",
            ExitFlag::LineSearchFailed => "line search found no minimum",
            ExitFlag::Cancelled => "cancelled by the caller",
        }
    }
}

impl fmt::Display for ExitFlag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

/// A scalar or per-element waveform bound (MATLAB allowed either).
#[derive(Debug, Clone, PartialEq)]
pub enum Bound {
    /// One value for every element of the waveform.
    Scalar(f64),
    /// One value per waveform element, shaped like the waveform.
    Array(nalgebra::DMatrix<f64>),
}

impl Bound {
    /// Value at waveform element `(row, col)`.
    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f64 {
        match self {
            Bound::Scalar(v) => *v,
            Bound::Array(a) => a[(row, col)],
        }
    }
    /// Largest bound value, for reporting.
    pub fn max(&self) -> f64 {
        match self {
            Bound::Scalar(v) => *v,
            Bound::Array(a) => a.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    }
    /// Smallest bound value, for reporting.
    pub fn min(&self) -> f64 {
        match self {
            Bound::Scalar(v) => *v,
            Bound::Array(a) => a.iter().cloned().fold(f64::INFINITY, f64::min),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_matlab_spellings() {
        assert_eq!(
            StateSpace::parse("liouville").unwrap(),
            StateSpace::Liouville
        );
        assert_eq!(Basis::parse("SPHTEN").unwrap(), Basis::Sphten);
        assert_eq!(PropMethod::parse("krylov").unwrap(), PropMethod::Krylov);
        assert!(PropMethod::parse("expm").is_err());
        assert_eq!(Penalty::parse("SNS").unwrap(), Penalty::Sns);
        assert_eq!(Penalty::Sns.as_str(), "SNS");
    }

    #[test]
    fn split_orders_match_matlab() {
        for (m, o) in [
            (SplitMethod::Uncoupled, 0),
            (SplitMethod::Lie, 1),
            (SplitMethod::Strang, 2),
            (SplitMethod::Blanes3, 3),
            (SplitMethod::Blanes4, 4),
            (SplitMethod::Omelyan6, 6),
        ] {
            assert_eq!(m.order(), o);
            assert_eq!(SplitMethod::from_order(o).unwrap(), m);
        }
        assert!(SplitMethod::from_order(5).is_err());
    }
}
