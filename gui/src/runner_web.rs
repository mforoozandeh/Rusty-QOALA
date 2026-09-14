//! Web runner: the optimisation in a dedicated Web Worker.
//!
//! A worker is used rather than `wasm-bindgen-rayon` or `wasm_thread` on
//! purpose.  Those need `SharedArrayBuffer`, which needs the COOP/COEP
//! response headers, which not every host can set - and the atomics build of
//! the standard library, which needs a nightly toolchain.  A plain worker
//! needs none of that: it gets its own WebAssembly instance and its own
//! memory, talks by `postMessage`, and works on any static host including
//! `python -m http.server`.
//!
//! The cost is cancellation.  A worker running a synchronous optimisation
//! cannot process an incoming message, so Cancel terminates it outright.  The
//! convergence recorded so far survives; the partial waveform does not.  The
//! interface says so.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::run::RunMessage;
use crate::setup::Setup;

/// A run in progress.
pub struct Runner {
    worker: Option<web_sys::Worker>,
    inbox: Rc<RefCell<Vec<RunMessage>>>,
    // Kept alive for as long as the worker is: dropping it would detach the
    // handler.
    _on_message: Closure<dyn FnMut(web_sys::MessageEvent)>,
    _on_error: Closure<dyn FnMut(web_sys::Event)>,
}

/// The worker's own WebAssembly has to load before it can listen, and a
/// message posted before then is dropped rather than queued.  So the setup
/// waits here until the worker says it is ready.
type Pending = Rc<RefCell<Option<String>>>;

impl Runner {
    /// Start `setup` in a fresh worker.
    pub fn start(setup: Setup) -> Self {
        let inbox: Rc<RefCell<Vec<RunMessage>>> = Rc::new(RefCell::new(Vec::new()));

        let pending: Pending = Rc::new(RefCell::new(serde_json::to_string(&setup).ok()));
        if pending.borrow().is_none() {
            inbox
                .borrow_mut()
                .push(RunMessage::Failed("the setup would not serialise".into()));
        }

        let worker = spawn_worker();

        let box_for_message = Rc::clone(&inbox);
        let pending_for_message = Rc::clone(&pending);
        let worker_for_message = worker.clone();
        let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
            move |event: web_sys::MessageEvent| {
                let Some(text) = event.data().as_string() else {
                    return;
                };
                match serde_json::from_str::<RunMessage>(&text) {
                    Ok(RunMessage::Ready) => {
                        // Now, and only now, hand over the work.
                        let Some(json) = pending_for_message.borrow_mut().take() else {
                            return;
                        };
                        let posted = worker_for_message
                            .as_ref()
                            .map(|w| w.post_message(&JsValue::from_str(&json)).is_ok())
                            .unwrap_or(false);
                        if !posted {
                            box_for_message.borrow_mut().push(RunMessage::Failed(
                                "could not hand the setup to the worker".into(),
                            ));
                        }
                    }
                    Ok(message) => box_for_message.borrow_mut().push(message),
                    Err(e) => box_for_message
                        .borrow_mut()
                        .push(RunMessage::Failed(format!(
                            "unreadable worker message: {e}"
                        ))),
                }
            },
        );

        let box_for_error = Rc::clone(&inbox);
        let on_error = Closure::<dyn FnMut(web_sys::Event)>::new(move |_: web_sys::Event| {
            box_for_error.borrow_mut().push(RunMessage::Failed(
                "the optimisation worker stopped unexpectedly".into(),
            ));
        });

        match &worker {
            Some(worker) => {
                worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
                worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
            }
            None => inbox.borrow_mut().push(RunMessage::Failed(
                "this browser would not start a Web Worker".into(),
            )),
        }

        Runner {
            worker,
            inbox,
            _on_message: on_message,
            _on_error: on_error,
        }
    }

    /// Every message that has arrived since the last call.
    pub fn drain(&mut self) -> Vec<RunMessage> {
        std::mem::take(&mut *self.inbox.borrow_mut())
    }

    /// Stop the run.  The worker is blocked inside the optimiser and cannot
    /// hear a polite request, so it is terminated.
    pub fn cancel(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.terminate();
        }
    }

    /// Cancellation here throws the partial waveform away; see the module
    /// documentation.
    pub const CANCEL_KEEPS_WAVEFORM: bool = false;
}

impl Drop for Runner {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.terminate();
        }
    }
}

/// Start the worker from the loader script Trunk emits next to the page.
///
/// It has to be a *classic* worker: Trunk's `data-loader-shim` emits a stub
/// that calls `importScripts`, which a module worker does not have.
/// `window.qoala_worker_url` lets a host that puts the assets somewhere
/// unusual point at them without rebuilding.
fn spawn_worker() -> Option<web_sys::Worker> {
    let window = web_sys::window()?;
    let configured = js_sys::Reflect::get(&window, &JsValue::from_str("qoala_worker_url"))
        .ok()
        .and_then(|v| v.as_string());
    let url = configured.unwrap_or_else(|| "./qoala-worker_loader.js".to_string());
    web_sys::Worker::new(&url).ok()
}
