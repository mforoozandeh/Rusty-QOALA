//! What this build is running on, and the two browser facilities the
//! interface uses: the URL fragment and the clipboard-friendly link it makes.
//!
//! Natively, the URL functions work on a `qoala://` style string so that the
//! same button does something sensible in the desktop app.

use crate::setup::Setup;

/// Facts about the host that change what the interface offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Platform {
    /// Running in a browser rather than as a desktop application.
    pub is_web: bool,
    /// `crossOriginIsolated`: whether `SharedArrayBuffer` and therefore
    /// multi-core WebAssembly are available.  Always false natively, where it
    /// means nothing.
    pub cross_origin_isolated: bool,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn detect() -> Platform {
    Platform {
        is_web: false,
        cross_origin_isolated: false,
    }
}

#[cfg(target_arch = "wasm32")]
pub fn detect() -> Platform {
    use wasm_bindgen::JsCast;
    let isolated = web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w, &"crossOriginIsolated".into()).ok())
        .and_then(|v| v.dyn_into::<js_sys::Boolean>().ok())
        .map(|b| b.value_of())
        .unwrap_or(false);
    Platform {
        is_web: true,
        cross_origin_isolated: isolated,
    }
}

/// Percent-encode the characters a URL fragment cannot carry.
fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

/// Undo [`encode`].
fn decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut n = 0;
    while n < bytes.len() {
        if bytes[n] == b'%' {
            if n + 2 >= bytes.len() {
                return None;
            }
            let hex = std::str::from_utf8(&bytes[n + 1..n + 3]).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            n += 3;
        } else {
            out.push(bytes[n]);
            n += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// The fragment a setup encodes to, without the leading `#`.
pub fn fragment_for_setup(setup: &Setup) -> Result<String, String> {
    let json = serde_json::to_string(setup).map_err(|e| e.to_string())?;
    Ok(format!("setup={}", encode(&json)))
}

/// Read a setup out of a fragment, ignoring anything that is not one.
pub fn setup_from_fragment(fragment: &str) -> Option<Setup> {
    let fragment = fragment.trim_start_matches('#');
    for part in fragment.split('&') {
        if let Some(value) = part.strip_prefix("setup=") {
            let json = decode(value)?;
            return serde_json::from_str(&json).ok();
        }
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
pub fn setup_from_url() -> Option<Setup> {
    // The desktop application accepts one as a command-line argument, which
    // makes a shared link work by paste.
    let arg = std::env::args().nth(1)?;
    let fragment = arg.split_once('#').map(|(_, f)| f).unwrap_or(&arg);
    setup_from_fragment(fragment)
}

#[cfg(target_arch = "wasm32")]
pub fn setup_from_url() -> Option<Setup> {
    let hash = web_sys::window()?.location().hash().ok()?;
    setup_from_fragment(&hash)
}

/// A link that reproduces this setup, and which also updates the address bar
/// so the browser's own share and bookmark actions pick it up.
#[cfg(target_arch = "wasm32")]
pub fn url_for_setup(setup: &Setup) -> Result<String, String> {
    let fragment = fragment_for_setup(setup)?;
    let window = web_sys::window().ok_or("no window")?;
    let location = window.location();
    let base = format!(
        "{}{}",
        location.origin().map_err(|_| "no origin")?,
        location.pathname().map_err(|_| "no path")?
    );
    let url = format!("{base}#{fragment}");
    let _ = location.set_hash(&fragment);
    Ok(url)
}

/// Natively there is no address bar, so the link is the fragment on its own -
/// paste it after the address of a hosted copy, or hand it to the desktop
/// binary as an argument.
#[cfg(not(target_arch = "wasm32"))]
pub fn url_for_setup(setup: &Setup) -> Result<String, String> {
    Ok(format!("#{}", fragment_for_setup(setup)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;

    #[test]
    fn a_setup_survives_the_url_fragment() {
        for setup in presets::all() {
            let fragment = fragment_for_setup(&setup).unwrap();
            assert!(!fragment.contains('#'));
            let back = setup_from_fragment(&fragment).expect("decode");
            assert_eq!(back, setup);
            // And with the leading marker a browser would hand back.
            let back = setup_from_fragment(&format!("#{fragment}")).expect("decode");
            assert_eq!(back, setup);
        }
    }

    #[test]
    fn junk_fragments_are_ignored_rather_than_panicking() {
        assert!(setup_from_fragment("").is_none());
        assert!(setup_from_fragment("#other=1").is_none());
        assert!(setup_from_fragment("setup=not%20json").is_none());
        assert!(setup_from_fragment("setup=%").is_none());
        assert!(setup_from_fragment("setup=%ZZ").is_none());
    }
}
