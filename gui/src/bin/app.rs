//! The interface binary: a window natively, a canvas in the browser.

use qoala_gui::app;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 840.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("QOALA"),
        ..Default::default()
    };
    eframe::run_native(
        "QOALA",
        options,
        Box::new(|cc| Ok(Box::new(app::QoalaApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    console_error_panic_hook::set_once();
    eframe::WebLogger::init(log::LevelFilter::Info).ok();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document");
        let canvas = document
            .get_element_by_id("qoala_canvas")
            .expect("the page needs a canvas with id qoala_canvas")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("qoala_canvas is not a canvas");

        let result = eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| Ok(Box::new(app::QoalaApp::new(cc)))),
            )
            .await;

        // Take the loading text down whichever way this went.
        if let Some(loading) = document.get_element_by_id("loading") {
            match result {
                Ok(_) => loading.remove(),
                Err(e) => {
                    loading.set_inner_html(&format!("<p>QOALA failed to start.</p><p>{e:?}</p>"))
                }
            }
        }
    });
}
