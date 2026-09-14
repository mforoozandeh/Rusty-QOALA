//! The application: panels, plots and the run loop.

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};

use crate::estimate::estimate_seconds_on;
use crate::export;
use crate::platform::{self, Platform};
use crate::presets;
use crate::run::{Finished, Progress, RunMessage};
use crate::runner::Runner;
use crate::setup::{
    pair_index, Component, GateChoice, Operator, PenaltyChoice, ProductOperator, Setup, Target,
    MAX_SPINS, MAX_WEB_SPINS,
};
use crate::waveform;

/// Where a run has got to.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Status {
    /// Nothing has been run yet.
    Idle,
    /// A run is in progress.
    Running,
    /// A run finished and its results are on screen.
    Done,
    /// A run failed.
    Failed,
}

/// The whole application.
pub struct QoalaApp {
    setup: Setup,
    presets: Vec<Setup>,
    runner: Option<Runner>,
    history: Vec<Progress>,
    /// A finished run, together with the setup that produced it.  They are
    /// one field because they must never disagree: the setup is what turns
    /// the waveform back into Hz and seconds, and the one on screen can be
    /// edited the moment the run ends.
    result: Option<(Setup, Finished)>,
    /// The setup the run in flight was started from.
    running_setup: Option<Setup>,
    message: Option<String>,
    status: Status,
    platform: Platform,
    show_channels: Vec<bool>,
}

impl QoalaApp {
    /// A fresh application, showing the default preset.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let platform = platform::detect();
        // A setup carried in the URL fragment wins over one in storage, so a
        // shared link always shows what the sender meant.
        let setup = platform::setup_from_url()
            .or_else(|| {
                cc.storage
                    .and_then(|s| s.get_string(STORAGE_KEY))
                    .and_then(|s| serde_json::from_str::<Setup>(&s).ok())
            })
            .unwrap_or_else(presets::default_setup);
        let nchannels = setup.nchannels();
        QoalaApp {
            setup,
            presets: presets::all(),
            runner: None,
            history: Vec::new(),
            result: None,
            running_setup: None,
            message: None,
            status: Status::Idle,
            platform,
            show_channels: vec![true; nchannels],
        }
    }

    fn max_spins(&self) -> usize {
        if self.platform.is_web {
            MAX_WEB_SPINS
        } else {
            MAX_SPINS
        }
    }

    fn start(&mut self) {
        self.history.clear();
        self.result = None;
        self.message = None;
        self.status = Status::Running;
        self.show_channels = vec![true; self.setup.nchannels()];
        self.running_setup = Some(self.setup.clone());
        self.runner = Some(Runner::start(self.setup.clone()));
    }

    fn cancel(&mut self) {
        if let Some(runner) = &mut self.runner {
            runner.cancel();
        }
        if !Runner::CANCEL_KEEPS_WAVEFORM {
            self.runner = None;
            self.status = Status::Done;
            self.message = Some(
                "cancelled - the convergence so far is kept, the partial waveform is not".into(),
            );
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        let Some(runner) = &mut self.runner else {
            return;
        };
        let mut finished = false;
        for message in runner.drain() {
            match message {
                RunMessage::Ready => {}
                RunMessage::Progress(p) => self.history.push(p),
                RunMessage::Finished(f) => {
                    self.message = Some(format!("stopped: {}", f.exit_message));
                    let setup = self
                        .running_setup
                        .clone()
                        .unwrap_or_else(|| self.setup.clone());
                    self.result = Some((setup, *f));
                    self.status = Status::Done;
                    finished = true;
                }
                RunMessage::Failed(e) => {
                    self.message = Some(e);
                    self.status = Status::Failed;
                    finished = true;
                }
            }
        }
        if finished {
            self.runner = None;
            self.running_setup = None;
        } else {
            // Keep the frame loop going while work is in flight, so the plots
            // move without the user having to wiggle the mouse.
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }
}

const STORAGE_KEY: &str = "qoala-setup";

impl eframe::App for QoalaApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(json) = serde_json::to_string(&self.setup) {
            storage.set_string(STORAGE_KEY, json);
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll(ctx);

        egui::TopBottomPanel::top("top").show(ctx, |ui| self.top_bar(ui));
        egui::SidePanel::left("inputs")
            .resizable(true)
            .default_width(360.0)
            .width_range(300.0..=620.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.inputs(ui));
            });
        egui::CentralPanel::default().show(ctx, |ui| self.outputs(ui));
    }
}

// ----------------------------------------------------------------- top ----

impl QoalaApp {
    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("QOALA");
            ui.label("adaptive split-operator pulse optimisation");
            ui.separator();

            let mut chosen: Option<Setup> = None;
            egui::ComboBox::from_id_salt("preset")
                .selected_text(self.setup.name.clone())
                .width(260.0)
                .show_ui(ui, |ui| {
                    for preset in &self.presets {
                        if ui
                            .selectable_label(preset.name == self.setup.name, &preset.name)
                            .clicked()
                        {
                            chosen = Some(preset.clone());
                        }
                    }
                });
            if let Some(preset) = chosen {
                self.setup = preset;
                self.history.clear();
                self.result = None;
                self.message = None;
                self.status = Status::Idle;
            }

            ui.separator();

            let running = self.status == Status::Running;
            let problems = self.setup.problems();
            let too_big = self.setup.nspins > self.max_spins();
            let blocked = !problems.is_empty() || too_big;

            if running {
                if ui.button("Cancel").clicked() {
                    self.cancel();
                }
            } else {
                let seconds = estimate_seconds_on(&self.setup, self.platform.is_web);
                let label = format!("Run  -  about {}", human_time(seconds));
                let button = ui.add_enabled(!blocked, egui::Button::new(label));
                if button.clicked() {
                    self.start();
                }
                if blocked {
                    let why = if too_big {
                        format!("{} spins needs the desktop build", self.setup.nspins)
                    } else {
                        problems.join("; ")
                    };
                    button.on_disabled_hover_text(why);
                }
            }

            if running {
                ui.spinner();
                if let Some(last) = self.history.last() {
                    ui.label(format!("iteration {}", last.iteration));
                }
            }
        });

        if let Some(message) = &self.message {
            let colour = if self.status == Status::Failed {
                egui::Color32::from_rgb(200, 80, 80)
            } else {
                ui.visuals().weak_text_color()
            };
            ui.colored_label(colour, message);
        }

        for line in self.setup.problems() {
            ui.colored_label(egui::Color32::from_rgb(200, 140, 60), format!("! {line}"));
        }

        if self.platform.is_web && !self.platform.cross_origin_isolated {
            ui.colored_label(
                egui::Color32::from_rgb(200, 140, 60),
                "This page is not cross-origin isolated, so the optimisation runs on one core. \
                 It still works; larger systems are just slower.",
            );
        }
        ui.add_space(4.0);
    }
}

// -------------------------------------------------------------- inputs ----

impl QoalaApp {
    fn inputs(&mut self, ui: &mut egui::Ui) {
        let editable = self.status != Status::Running;
        ui.add_enabled_ui(editable, |ui| {
            self.spin_system(ui);
            ui.add_space(8.0);
            self.control_channels(ui);
            ui.add_space(8.0);
            self.target(ui);
            ui.add_space(8.0);
            self.pulse(ui);
            ui.add_space(8.0);
            self.sharing(ui);
        });
    }

    fn spin_system(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Spin system")
            .default_open(true)
            .show(ui, |ui| {
                let max = self.max_spins();
                ui.horizontal(|ui| {
                    ui.label("spins");
                    let mut nspins = self.setup.nspins;
                    if ui
                        .add(egui::DragValue::new(&mut nspins).range(2..=max))
                        .changed()
                    {
                        self.setup.nspins = nspins;
                        self.setup.resize();
                    }
                    if max < MAX_SPINS {
                        ui.label(format!("(up to {max} here)"));
                    }
                });

                ui.label("resonance offsets, Hz");
                egui::Grid::new("offsets").num_columns(2).show(ui, |ui| {
                    for s in 0..self.setup.nspins {
                        ui.label(format!("spin {}", s + 1));
                        ui.add(
                            egui::DragValue::new(&mut self.setup.offsets_hz[s])
                                .speed(10.0)
                                .suffix(" Hz"),
                        );
                        ui.end_row();
                    }
                });

                ui.add_space(4.0);
                ui.label("J couplings, Hz");
                egui::Grid::new("couplings").num_columns(3).show(ui, |ui| {
                    for i in 0..self.setup.nspins {
                        for j in (i + 1)..self.setup.nspins {
                            let index = pair_index(self.setup.nspins, i, j);
                            ui.label(format!("{}-{}", i + 1, j + 1));
                            ui.add(
                                egui::DragValue::new(&mut self.setup.couplings[index].j_hz)
                                    .speed(1.0)
                                    .suffix(" Hz"),
                            );
                            let strong = &mut self.setup.couplings[index].strong;
                            let text = if *strong { "strong" } else { "weak" };
                            if ui.selectable_label(*strong, text).clicked() {
                                *strong = !*strong;
                            }
                            ui.end_row();
                        }
                    }
                });
                ui.label(
                    egui::RichText::new("weak: zz only.  strong: isotropic xx + yy + zz.")
                        .small()
                        .weak(),
                );
            });
    }

    fn control_channels(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Control channels")
            .default_open(true)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Each pair is one x and one y control.  Tick the spins it drives.",
                    )
                    .small()
                    .weak(),
                );
                ui.horizontal(|ui| {
                    ui.label("pairs");
                    let mut npairs = self.setup.npairs;
                    if ui
                        .add(egui::DragValue::new(&mut npairs).range(1..=self.setup.nspins))
                        .changed()
                    {
                        self.setup.npairs = npairs;
                        self.setup.resize();
                    }
                });

                egui::Grid::new("spin_control")
                    .num_columns(self.setup.npairs + 2)
                    .show(ui, |ui| {
                        ui.label("");
                        for k in 0..self.setup.npairs {
                            ui.label(format!("pair {}", k + 1));
                        }
                        ui.end_row();
                        for s in 0..self.setup.nspins {
                            ui.label(format!("spin {}", s + 1));
                            for k in 0..self.setup.npairs {
                                ui.checkbox(&mut self.setup.spin_control[s][k], "");
                            }
                            ui.end_row();
                        }
                        ui.label("max amp");
                        for k in 0..self.setup.npairs {
                            ui.add(
                                egui::DragValue::new(&mut self.setup.amplitudes_hz[k])
                                    .speed(50.0)
                                    .range(1.0..=1e7)
                                    .suffix(" Hz"),
                            );
                        }
                        ui.end_row();
                    });
            });
    }

    fn target(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Target")
            .default_open(true)
            .show(ui, |ui| {
                let is_gate = matches!(self.setup.target, Target::Gate { .. });
                ui.horizontal(|ui| {
                    if ui.selectable_label(!is_gate, "state transfer").clicked() && is_gate {
                        self.setup.target = Target::Transfer {
                            from: Operator::single(0, Component::Z),
                            to: Operator::single(self.setup.nspins - 1, Component::Z),
                        };
                    }
                    if ui.selectable_label(is_gate, "gate").clicked() && !is_gate {
                        self.setup.target = Target::Gate {
                            gate: GateChoice::Swap,
                            qubits: vec![0, self.setup.nspins.min(2) - 1],
                        };
                        self.setup.resize();
                    }
                });

                let nspins = self.setup.nspins;
                match &mut self.setup.target {
                    Target::Transfer { from, to } => {
                        for (label, op) in [("from", from), ("to", to)] {
                            ui.horizontal(|ui| {
                                ui.label(label);
                                match op {
                                    Operator::Product(p) => product_editor(ui, label, p, nspins),
                                    Operator::Basis { .. } => {
                                        ui.label(egui::RichText::new(op.label()).small().weak());
                                        if ui.small_button("edit as product operator").clicked() {
                                            *op = Operator::single(0, Component::Z);
                                        }
                                    }
                                }
                            });
                        }
                    }
                    Target::Gate { gate, qubits } => {
                        let mut next = *gate;
                        egui::ComboBox::from_id_salt("gate")
                            .selected_text(gate.name())
                            .show_ui(ui, |ui| {
                                for choice in GateChoice::MENU {
                                    ui.selectable_value(&mut next, choice, choice.name());
                                }
                            });
                        if next != *gate {
                            *gate = next;
                            qubits.resize(next.qubits(), 0);
                        }
                        ui.horizontal(|ui| {
                            ui.label("on qubits");
                            for (n, q) in qubits.iter_mut().enumerate() {
                                let mut one_based = *q + 1;
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut one_based)
                                            .range(1..=nspins)
                                            .prefix(if n == 0 { "" } else { ", " }),
                                    )
                                    .changed()
                                {
                                    *q = one_based - 1;
                                }
                            }
                        });
                    }
                }
            });
    }

    fn pulse(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Pulse")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("pulse").num_columns(2).show(ui, |ui| {
                    ui.label("duration");
                    let mut ms = self.setup.duration_s * 1e3;
                    if ui
                        .add(
                            egui::DragValue::new(&mut ms)
                                .speed(0.1)
                                .range(0.001..=1e5)
                                .suffix(" ms"),
                        )
                        .changed()
                    {
                        self.setup.duration_s = ms * 1e-3;
                    }
                    ui.end_row();

                    ui.label("time slices");
                    ui.add(egui::DragValue::new(&mut self.setup.nslices).range(2..=20000));
                    ui.end_row();

                    ui.label("iteration budget");
                    ui.add(egui::DragValue::new(&mut self.setup.max_iter).range(0..=100000));
                    ui.end_row();

                    ui.label("penalty");
                    egui::ComboBox::from_id_salt("penalty")
                        .selected_text(self.setup.penalty.name())
                        .show_ui(ui, |ui| {
                            for choice in PenaltyChoice::MENU {
                                ui.selectable_value(&mut self.setup.penalty, choice, choice.name());
                            }
                        });
                    ui.end_row();

                    ui.label("splitting orders");
                    ui.horizontal(|ui| {
                        for order in [1usize, 2, 3, 4, 6] {
                            let mut on = self.setup.splitset.contains(&order);
                            if ui.checkbox(&mut on, order.to_string()).changed() {
                                if on {
                                    self.setup.splitset.push(order);
                                    self.setup.splitset.sort_unstable();
                                } else if self.setup.splitset.len() > 1 {
                                    self.setup.splitset.retain(|o| *o != order);
                                }
                            }
                        }
                    });
                    ui.end_row();

                    ui.label("seed");
                    ui.horizontal(|ui| {
                        let mut fixed = self.setup.seed.is_some();
                        if ui.checkbox(&mut fixed, "fixed").changed() {
                            self.setup.seed = fixed.then_some(1);
                        }
                        if let Some(seed) = &mut self.setup.seed {
                            ui.add(egui::DragValue::new(seed));
                        }
                    });
                    ui.end_row();
                });
                ui.label(
                    egui::RichText::new(format!(
                        "dimension {}  -  {} waveform points",
                        self.setup.dimension(),
                        self.setup.nslices * self.setup.nchannels()
                    ))
                    .small()
                    .weak(),
                );
            });
    }

    fn sharing(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Share and export", |ui| {
            if ui.button("Copy link to this setup").clicked() {
                match platform::url_for_setup(&self.setup) {
                    Ok(url) => {
                        ui.ctx().copy_text(url);
                        self.message = Some("link copied".into());
                    }
                    Err(e) => self.message = Some(e),
                }
            }
            if ui.button("Download setup as JSON").clicked() {
                match serde_json::to_string_pretty(&self.setup) {
                    Ok(json) => {
                        export::offer_file(&format!("{}.json", slug(&self.setup.name)), &json)
                    }
                    Err(e) => self.message = Some(e.to_string()),
                }
            }
            ui.separator();
            let have_result = self.result.is_some();
            ui.add_enabled_ui(have_result, |ui| {
                // Exports describe the run, so they use its setup rather than
                // whatever is in the editor now.
                if ui.button("Download waveform as CSV").clicked() {
                    if let Some((setup, result)) = &self.result {
                        let csv = export::waveform_csv(setup, &result.waveform);
                        export::offer_file(&format!("{}.csv", slug(&setup.name)), &csv);
                    }
                }
            });
            if !have_result {
                ui.label(egui::RichText::new("run something first").small().weak());
            }
        });
    }
}

fn product_editor(ui: &mut egui::Ui, id: &str, op: &mut ProductOperator, nspins: usize) {
    ui.label(op.label());
    let nterms = op.terms.len();
    let mut remove: Option<usize> = None;
    for (n, (spin, component)) in op.terms.iter_mut().enumerate() {
        let mut one_based = *spin + 1;
        if ui
            .add(
                egui::DragValue::new(&mut one_based)
                    .range(1..=nspins)
                    .prefix("spin "),
            )
            .changed()
        {
            *spin = one_based - 1;
        }
        egui::ComboBox::from_id_salt(format!("{id}-component-{n}"))
            .selected_text(component.name())
            .width(44.0)
            .show_ui(ui, |ui| {
                for choice in Component::ALL {
                    ui.selectable_value(component, choice, choice.name());
                }
            });
        if nterms > 1 && ui.small_button("x").clicked() {
            remove = Some(n);
        }
    }
    if let Some(n) = remove {
        op.terms.remove(n);
    }
    if op.terms.len() < nspins && ui.small_button("+").clicked() {
        let used: Vec<usize> = op.terms.iter().map(|(s, _)| *s).collect();
        if let Some(free) = (0..nspins).find(|s| !used.contains(s)) {
            op.terms.push((free, Component::Z));
        }
    }
}

// ------------------------------------------------------------- outputs ----

impl QoalaApp {
    fn outputs(&mut self, ui: &mut egui::Ui) {
        self.live_numbers(ui);
        ui.separator();

        let available = ui.available_height();
        ui.allocate_ui(egui::vec2(ui.available_width(), available * 0.48), |ui| {
            ui.label("infidelity against iteration");
            self.convergence_plot(ui);
        });
        ui.separator();
        ui.label("waveform, Hz");
        self.channel_toggles(ui);
        self.waveform_plot(ui);
    }

    fn live_numbers(&mut self, ui: &mut egui::Ui) {
        let last = self.history.last();
        ui.horizontal_wrapped(|ui| {
            let big = |ui: &mut egui::Ui, label: &str, value: String| {
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new(label).small().weak());
                    ui.label(egui::RichText::new(value).heading());
                });
                ui.add_space(18.0);
            };

            let fidelity = last.map(|p| p.fidelity).unwrap_or(f64::NAN);
            big(
                ui,
                "fidelity",
                if fidelity.is_nan() {
                    "-".into()
                } else {
                    format!("{:.6} %", 100.0 * fidelity)
                },
            );
            big(
                ui,
                "infidelity",
                if fidelity.is_nan() {
                    "-".into()
                } else {
                    format!("{:.2e}", (1.0 - fidelity).max(0.0))
                },
            );
            big(
                ui,
                "splitting order",
                last.map(|p| p.split_order.to_string())
                    .unwrap_or_else(|| "-".into()),
            );
            big(
                ui,
                "Trotter number",
                last.map(|p| p.trotter_number.to_string())
                    .unwrap_or_else(|| "-".into()),
            );
            big(
                ui,
                "iteration",
                last.map(|p| p.iteration.to_string())
                    .unwrap_or_else(|| "-".into()),
            );
            big(
                ui,
                "elapsed",
                last.map(|p| format!("{:.1} s", p.elapsed_s))
                    .unwrap_or_else(|| "-".into()),
            );
            big(
                ui,
                "|grad|",
                last.map(|p| format!("{:.3e}", p.gradient_norm))
                    .unwrap_or_else(|| "-".into()),
            );
            if let Some(p) = last {
                big(
                    ui,
                    "f / grad evals",
                    format!("{} / {}", p.count_fx, p.count_gfx),
                );
            }
        });
    }

    /// Infidelity on a log axis.  egui_plot has no log scale, so the values
    /// are plotted as their base-10 logarithm and the labels undo it.
    fn convergence_plot(&mut self, ui: &mut egui::Ui) {
        let points: PlotPoints = self
            .history
            .iter()
            .map(|p| {
                let infidelity = (1.0 - p.fidelity).max(1e-16);
                [p.iteration as f64, infidelity.log10()]
            })
            .collect();

        Plot::new("convergence")
            .legend(Legend::default())
            .y_axis_formatter(|mark, _| {
                // Marks land on fractions once the plot is zoomed; only the
                // decades are worth a label.
                if (mark.value - mark.value.round()).abs() < 1e-6 {
                    format!("1e{:.0}", mark.value.round())
                } else {
                    String::new()
                }
            })
            .label_formatter(|_, value| {
                format!(
                    "iteration {:.0}\ninfidelity {:.3e}",
                    value.x,
                    10f64.powf(value.y)
                )
            })
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new("1 - fidelity", points));
            });
    }

    fn channel_toggles(&mut self, ui: &mut egui::Ui) {
        // The toggles belong to whatever is plotted, which is the finished
        // run if there is one.
        let names = match &self.result {
            Some((setup, _)) => setup.channel_names(),
            None => self.setup.channel_names(),
        };
        if self.show_channels.len() != names.len() {
            self.show_channels = vec![true; names.len()];
        }
        ui.horizontal_wrapped(|ui| {
            for (k, name) in names.iter().enumerate() {
                ui.checkbox(&mut self.show_channels[k], name);
            }
        });
    }

    fn waveform_plot(&mut self, ui: &mut egui::Ui) {
        let Some((setup, result)) = &self.result else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("the waveform appears when the run finishes").weak());
            });
            return;
        };

        // Scaled by the setup the run used, not the one being edited: the two
        // part company the moment somebody changes a number after a run.
        let names = setup.channel_names();

        Plot::new("waveform")
            .legend(Legend::default())
            .allow_scroll(false)
            .x_axis_label("time, ms")
            .show(ui, |plot_ui| {
                for (k, name) in names.iter().enumerate() {
                    if !self.show_channels.get(k).copied().unwrap_or(true) {
                        continue;
                    }
                    let points = waveform::stairs(setup, &result.waveform, k);
                    plot_ui.line(Line::new(name.clone(), PlotPoints::from(points)));
                }
            });
    }
}

// --------------------------------------------------------------- bits -----

fn human_time(seconds: f64) -> String {
    if seconds < 1.0 {
        "a moment".to_string()
    } else if seconds < 90.0 {
        format!("{seconds:.0} s")
    } else if seconds < 5400.0 {
        format!("{:.0} min", seconds / 60.0)
    } else {
        format!("{:.1} hours", seconds / 3600.0)
    }
}

/// A filename-safe version of a preset name.
pub fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}
