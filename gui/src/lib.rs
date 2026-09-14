//! QOALA in a window, or in a browser tab.
//!
//! `cargo run -p qoala-gui` gives the desktop application; `trunk serve` in
//! this directory gives the web one.  Same source both ways - the only
//! platform-specific code is in [`platform`], the two `runner` modules, and
//! the file-download helper in [`export`].
//!
//! The crate is a library with two binaries on top: `qoala-gui` is the
//! interface, and `qoala-worker` is the WebAssembly Web Worker the browser
//! build runs the optimisation inside.

pub mod app;
pub mod estimate;
pub mod export;
pub mod platform;
pub mod presets;
pub mod run;
pub mod setup;
pub mod waveform;

#[cfg_attr(target_arch = "wasm32", path = "runner_web.rs")]
#[cfg_attr(not(target_arch = "wasm32"), path = "runner_native.rs")]
pub mod runner;
