//! Getting results out: CSV, and the browser/desktop plumbing that hands the
//! user a file.

use std::fmt::Write as _;

use crate::setup::Setup;

/// Waveform as CSV: a time column then one column per control channel, in Hz.
///
/// The same layout `qoala::examples_io::write_waveform_csv` writes, so the
/// two are interchangeable.
pub fn waveform_csv(setup: &Setup, waveform: &[Vec<f64>]) -> String {
    let two_pi = 2.0 * std::f64::consts::PI;
    let amps = setup.channel_amplitudes_rad_per_s();
    let dt = setup.dt();

    let mut out = String::from("time_s");
    for name in setup.channel_names() {
        let _ = write!(out, ",{name}");
    }
    out.push('\n');

    let mut t = 0.0;
    for (n, row) in waveform.iter().enumerate() {
        let _ = write!(out, "{t:.9}");
        for (k, value) in row.iter().enumerate() {
            let hz = value * amps.get(k).copied().unwrap_or(two_pi) / two_pi;
            let _ = write!(out, ",{hz:.9}");
        }
        out.push('\n');
        t += dt.get(n).copied().unwrap_or(0.0);
    }
    out
}

/// Hand `contents` to the user as a file called `filename`.
#[cfg(not(target_arch = "wasm32"))]
pub fn offer_file(filename: &str, contents: &str) {
    if let Some(path) = rfd::FileDialog::new().set_file_name(filename).save_file() {
        if let Err(e) = std::fs::write(&path, contents) {
            log::error!("could not write {}: {e}", path.display());
        }
    }
}

/// Hand `contents` to the user as a download called `filename`.
#[cfg(target_arch = "wasm32")]
pub fn offer_file(filename: &str, contents: &str) {
    use wasm_bindgen::JsCast;

    let go = || -> Option<()> {
        let window = web_sys::window()?;
        let document = window.document()?;
        let parts = js_sys::Array::new();
        parts.push(&wasm_bindgen::JsValue::from_str(contents));
        let options = web_sys::BlobPropertyBag::new();
        options.set_type("text/plain;charset=utf-8");
        let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options).ok()?;
        let url = web_sys::Url::create_object_url_with_blob(&blob).ok()?;
        let anchor: web_sys::HtmlAnchorElement =
            document.create_element("a").ok()?.dyn_into().ok()?;
        anchor.set_href(&url);
        anchor.set_download(filename);
        anchor.click();
        let _ = web_sys::Url::revoke_object_url(&url);
        Some(())
    };
    if go().is_none() {
        log::error!("could not offer {filename} as a download");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;

    fn ramp(setup: &Setup) -> Vec<Vec<f64>> {
        (0..setup.nslices)
            .map(|n| {
                let t = n as f64 / setup.nslices as f64;
                (0..setup.nchannels())
                    .map(|k| 0.5 * ((3.0 + k as f64) * t).sin())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn the_csv_has_a_header_and_one_row_per_slice() {
        let setup = presets::z2z_2spin_1();
        let csv = waveform_csv(&setup, &ramp(&setup));
        let lines: Vec<&str> = csv.trim_end().lines().collect();
        assert_eq!(lines.len(), setup.nslices + 1);
        assert_eq!(
            lines[0],
            "time_s,x-control 1,y-control 1,x-control 2,y-control 2"
        );
        for line in &lines[1..] {
            assert_eq!(line.split(',').count(), setup.nchannels() + 1);
        }
        // The first row is at t = 0 and the last at (n-1) dt.
        assert!(lines[1].starts_with("0.000000000,"));
        let last_t: f64 = lines[setup.nslices]
            .split(',')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let dt = setup.duration_s / setup.nslices as f64;
        assert!((last_t - dt * (setup.nslices - 1) as f64).abs() < 1e-12);
    }
}
