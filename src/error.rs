//! Error type for the QOALA crate.
//!
//! The MATLAB original signals problems with `error(...)` inside `grumble()`
//! consistency-enforcement subfunctions.  Here those become typed errors that
//! propagate through `Result`.

use std::fmt;

/// Everything that can go wrong inside QOALA.
#[derive(Debug, Clone, PartialEq)]
pub enum QoalaError {
    /// A required configuration field was not supplied.
    MissingField(String),
    /// A configuration field was supplied but is inconsistent or out of range.
    BadValue(String),
    /// A combination of options that the package does not implement.
    NotImplemented(String),
    /// Shapes of two operands do not line up.
    Dimension(String),
    /// A numerical routine failed (singular matrix, non-convergence, ...).
    Numerical(String),
    /// Filesystem problem while reading or writing the propagator cache.
    Io(String),
}

impl fmt::Display for QoalaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QoalaError::MissingField(s) => write!(f, "missing configuration: {s}"),
            QoalaError::BadValue(s) => write!(f, "invalid configuration: {s}"),
            QoalaError::NotImplemented(s) => write!(f, "not implemented: {s}"),
            QoalaError::Dimension(s) => write!(f, "dimension mismatch: {s}"),
            QoalaError::Numerical(s) => write!(f, "numerical failure: {s}"),
            QoalaError::Io(s) => write!(f, "io error: {s}"),
        }
    }
}

impl std::error::Error for QoalaError {}

impl From<std::io::Error> for QoalaError {
    fn from(e: std::io::Error) -> Self {
        QoalaError::Io(e.to_string())
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, QoalaError>;

/// Shorthand for building a [`QoalaError::MissingField`].
#[macro_export]
macro_rules! missing {
    ($($arg:tt)*) => { $crate::error::QoalaError::MissingField(format!($($arg)*)) };
}

/// Shorthand for building a [`QoalaError::BadValue`].
#[macro_export]
macro_rules! bad {
    ($($arg:tt)*) => { $crate::error::QoalaError::BadValue(format!($($arg)*)) };
}

/// Shorthand for building a [`QoalaError::NotImplemented`].
#[macro_export]
macro_rules! nyi {
    ($($arg:tt)*) => { $crate::error::QoalaError::NotImplemented(format!($($arg)*)) };
}
