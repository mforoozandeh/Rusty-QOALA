//! The optimisation worker.
//!
//! Trunk builds this as a second WebAssembly binary and the page starts it as
//! a Web Worker.  It receives a JSON [`Setup`] by `postMessage`, runs the
//! optimisation, and posts a JSON [`RunMessage`] back for every iteration and
//! once more at the end.
//!
//! It never cancels itself: the page terminates the worker instead, because a
//! worker blocked inside the optimiser cannot read its own message queue.

#[cfg(target_arch = "wasm32")]
use qoala_gui::{run, setup};

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!(
        "qoala-worker is the WebAssembly worker for the browser build; it does nothing natively."
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    console_error_panic_hook::set_once();

    let scope: web_sys::DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let handler_scope = scope.clone();

    let on_message =
        Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |event: web_sys::MessageEvent| {
            let reply = |message: &run::RunMessage| {
                if let Ok(json) = serde_json::to_string(message) {
                    let _ = handler_scope.post_message(&JsValue::from_str(&json));
                }
            };

            let Some(text) = event.data().as_string() else {
                reply(&run::RunMessage::Failed(
                    "the worker was sent something that was not a setup".into(),
                ));
                return;
            };
            match serde_json::from_str::<setup::Setup>(&text) {
                Ok(problem) => {
                    let mut sink = PostSink {
                        scope: handler_scope.clone(),
                    };
                    run::run_to_sink(&problem, &mut sink);
                }
                Err(e) => reply(&run::RunMessage::Failed(format!("unreadable setup: {e}"))),
            }
        });

    scope.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    // The closure has to outlive `main`, which returns immediately.
    on_message.forget();

    // Only now is it safe to be sent a setup: anything posted before this
    // point would have been dispatched with no handler installed and lost.
    if let Ok(json) = serde_json::to_string(&run::RunMessage::Ready) {
        let _ = scope.post_message(&JsValue::from_str(&json));
    }
}

/// A [`run::MessageSink`] that posts each message to the page.
#[cfg(target_arch = "wasm32")]
struct PostSink {
    scope: web_sys::DedicatedWorkerGlobalScope,
}

#[cfg(target_arch = "wasm32")]
impl run::MessageSink for PostSink {
    fn send(&mut self, message: run::RunMessage) {
        if let Ok(json) = serde_json::to_string(&message) {
            let _ = self
                .scope
                .post_message(&wasm_bindgen::JsValue::from_str(&json));
        }
    }
}
