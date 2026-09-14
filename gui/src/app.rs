//! The application: panels, plots and the run loop.

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotImage, PlotPoint, PlotPoints};

use crate::escalade::{self, Analysis, Direction, EscaladeSetup};
use crate::estimate::{estimate_seconds_on, is_upper_bound};
use crate::export;
use crate::platform::{self, Platform};
use crate::presets;
use crate::problem::{Algorithm, Problem};
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

/// What the lower plot shows for an ESCALADE result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EscaladeView {
    /// The optimised x and y amplitudes.
    Waveform,
    /// Final magnetisation across offsets.
    Profile,
    /// Final y magnetisation over offset and field, and the phase slope.
    B1,
}

/// The whole application.
pub struct QoalaApp {
    algorithm: Algorithm,
    /// The QOALA setup being edited, kept while ESCALADE is showing.
    qoala: Setup,
    /// The ESCALADE setup being edited, kept while QOALA is showing.
    escalade: EscaladeSetup,
    presets: Vec<Problem>,
    runner: Option<Runner>,
    history: Vec<Progress>,
    /// A finished run, together with the problem that produced it.  They are
    /// one field because they must never disagree: the problem is what turns
    /// the waveform back into Hz and seconds, and the one on screen can be
    /// edited the moment the run ends.
    result: Option<(Problem, Finished)>,
    /// What an ESCALADE result's pulse does, worked out when it arrived.
    analysis: Option<Analysis>,
    /// The B1 map as an image, made the first time it is drawn.
    map_texture: Option<egui::TextureHandle>,
    /// The problem the run in flight was started from.
    running: Option<Problem>,
    message: Option<String>,
    status: Status,
    platform: Platform,
    show_channels: Vec<bool>,
    view: EscaladeView,
}

const QOALA_KEY: &str = "qoala-setup";
const ESCALADE_KEY: &str = "escalade-setup";
const ALGORITHM_KEY: &str = "algorithm";

impl QoalaApp {
    /// A fresh application, showing whatever was last open.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let platform = platform::detect();
        let stored = |key: &str| cc.storage.and_then(|s| s.get_string(key));

        let mut qoala = stored(QOALA_KEY)
            .and_then(|s| serde_json::from_str::<Setup>(&s).ok())
            .unwrap_or_else(presets::default_setup);
        let mut escalade = stored(ESCALADE_KEY)
            .and_then(|s| serde_json::from_str::<EscaladeSetup>(&s).ok())
            .unwrap_or_else(presets::escalade_b1_sensitive);
        let mut algorithm = stored(ALGORITHM_KEY)
            .and_then(|s| serde_json::from_str::<Algorithm>(&s).ok())
            .unwrap_or(Algorithm::Qoala);

        // A problem carried in the URL fragment wins over storage, so a
        // shared link always shows what the sender meant.
        match platform::problem_from_url() {
            Some(Problem::Qoala(s)) => {
                qoala = s;
                algorithm = Algorithm::Qoala;
            }
            Some(Problem::Escalade(s)) => {
                escalade = s;
                algorithm = Algorithm::Escalade;
            }
            None => {}
        }

        let mut app = QoalaApp {
            algorithm,
            qoala,
            escalade,
            presets: presets::every(),
            runner: None,
            history: Vec::new(),
            result: None,
            analysis: None,
            map_texture: None,
            running: None,
            message: None,
            status: Status::Idle,
            platform,
            show_channels: Vec::new(),
            view: EscaladeView::Waveform,
        };
        app.show_channels = vec![true; app.problem().channel_names().len()];
        app
    }

    /// The problem on screen.
    fn problem(&self) -> Problem {
        match self.algorithm {
            Algorithm::Qoala => Problem::Qoala(self.qoala.clone()),
            Algorithm::Escalade => Problem::Escalade(self.escalade.clone()),
        }
    }

    fn max_spins(&self) -> usize {
        if self.platform.is_web {
            MAX_WEB_SPINS
        } else {
            MAX_SPINS
        }
    }

    fn clear_results(&mut self) {
        self.history.clear();
        self.result = None;
        self.analysis = None;
        self.map_texture = None;
        self.message = None;
    }

    fn start(&mut self) {
        let problem = self.problem();
        self.clear_results();
        self.status = Status::Running;
        self.show_channels = vec![true; problem.channel_names().len()];
        self.running = Some(problem.clone());
        self.runner = Some(Runner::start(problem));
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
                    let problem = self.running.clone().unwrap_or_else(|| self.problem());
                    self.analysis = match &problem {
                        Problem::Escalade(setup) => Some(escalade::analyse(setup, &f.waveform)),
                        Problem::Qoala(_) => None,
                    };
                    self.map_texture = None;
                    self.result = Some((problem, *f));
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
            self.running = None;
        } else {
            // Keep the frame loop going while work is in flight, so the plots
            // move without the user having to wiggle the mouse.
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }
}

impl eframe::App for QoalaApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(json) = serde_json::to_string(&self.qoala) {
            storage.set_string(QOALA_KEY, json);
        }
        if let Ok(json) = serde_json::to_string(&self.escalade) {
            storage.set_string(ESCALADE_KEY, json);
        }
        if let Ok(json) = serde_json::to_string(&self.algorithm) {
            storage.set_string(ALGORITHM_KEY, json);
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
        let running = self.status == Status::Running;
        let problem = self.problem();

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!running, |ui| {
                for algorithm in Algorithm::ALL {
                    let label = egui::RichText::new(algorithm.name()).heading();
                    if ui
                        .selectable_label(self.algorithm == algorithm, label)
                        .clicked()
                        && self.algorithm != algorithm
                    {
                        self.algorithm = algorithm;
                        self.clear_results();
                        self.status = Status::Idle;
                    }
                }
            });
            ui.label(self.algorithm.blurb());
            ui.separator();

            let mut chosen: Option<Problem> = None;
            egui::ComboBox::from_id_salt("preset")
                .selected_text(problem.name())
                .width(260.0)
                .show_ui(ui, |ui| {
                    for preset in self
                        .presets
                        .iter()
                        .filter(|p| p.algorithm() == self.algorithm)
                    {
                        if ui
                            .selectable_label(preset.name() == problem.name(), preset.name())
                            .clicked()
                        {
                            chosen = Some(preset.clone());
                        }
                    }
                });
            if let Some(preset) = chosen {
                match preset {
                    Problem::Qoala(s) => self.qoala = s,
                    Problem::Escalade(s) => self.escalade = s,
                }
                self.clear_results();
                self.status = Status::Idle;
            }

            ui.separator();

            let problems = problem.problems();
            let too_big =
                self.algorithm == Algorithm::Qoala && self.qoala.nspins > self.max_spins();
            let blocked = !problems.is_empty() || too_big;

            if running {
                if ui.button("Cancel").clicked() {
                    self.cancel();
                }
            } else {
                let seconds = estimate_seconds_on(&problem, self.platform.is_web);
                let bound = if is_upper_bound(&problem) {
                    "up to"
                } else {
                    "about"
                };
                let label = format!("Run  -  {bound} {}", human_time(seconds));
                let button = ui.add_enabled(!blocked, egui::Button::new(label));
                if button.clicked() {
                    self.start();
                }
                if blocked {
                    let why = if too_big {
                        format!("{} spins needs the desktop build", self.qoala.nspins)
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

        for line in problem.problems() {
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
            match self.algorithm {
                Algorithm::Qoala => {
                    self.spin_system(ui);
                    ui.add_space(8.0);
                    self.control_channels(ui);
                    ui.add_space(8.0);
                    self.target(ui);
                    ui.add_space(8.0);
                    self.pulse(ui);
                }
                Algorithm::Escalade => {
                    self.escalade_band(ui);
                    ui.add_space(8.0);
                    self.escalade_field(ui);
                    ui.add_space(8.0);
                    self.escalade_pulse(ui);
                }
            }
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
                    let mut nspins = self.qoala.nspins;
                    if ui
                        .add(egui::DragValue::new(&mut nspins).range(2..=max))
                        .changed()
                    {
                        self.qoala.nspins = nspins;
                        self.qoala.resize();
                    }
                    if max < MAX_SPINS {
                        ui.label(format!("(up to {max} here)"));
                    }
                });

                ui.label("resonance offsets, Hz");
                egui::Grid::new("offsets").num_columns(2).show(ui, |ui| {
                    for s in 0..self.qoala.nspins {
                        ui.label(format!("spin {}", s + 1));
                        ui.add(
                            egui::DragValue::new(&mut self.qoala.offsets_hz[s])
                                .speed(10.0)
                                .suffix(" Hz"),
                        );
                        ui.end_row();
                    }
                });

                ui.add_space(4.0);
                ui.label("J couplings, Hz");
                egui::Grid::new("couplings").num_columns(3).show(ui, |ui| {
                    for i in 0..self.qoala.nspins {
                        for j in (i + 1)..self.qoala.nspins {
                            let index = pair_index(self.qoala.nspins, i, j);
                            ui.label(format!("{}-{}", i + 1, j + 1));
                            ui.add(
                                egui::DragValue::new(&mut self.qoala.couplings[index].j_hz)
                                    .speed(1.0)
                                    .suffix(" Hz"),
                            );
                            let strong = &mut self.qoala.couplings[index].strong;
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
                    let mut npairs = self.qoala.npairs;
                    if ui
                        .add(egui::DragValue::new(&mut npairs).range(1..=self.qoala.nspins))
                        .changed()
                    {
                        self.qoala.npairs = npairs;
                        self.qoala.resize();
                    }
                });

                egui::Grid::new("spin_control")
                    .num_columns(self.qoala.npairs + 2)
                    .show(ui, |ui| {
                        ui.label("");
                        for k in 0..self.qoala.npairs {
                            ui.label(format!("pair {}", k + 1));
                        }
                        ui.end_row();
                        for s in 0..self.qoala.nspins {
                            ui.label(format!("spin {}", s + 1));
                            for k in 0..self.qoala.npairs {
                                ui.checkbox(&mut self.qoala.spin_control[s][k], "");
                            }
                            ui.end_row();
                        }
                        ui.label("max amp");
                        for k in 0..self.qoala.npairs {
                            ui.add(
                                egui::DragValue::new(&mut self.qoala.amplitudes_hz[k])
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
                let is_gate = matches!(self.qoala.target, Target::Gate { .. });
                ui.horizontal(|ui| {
                    if ui.selectable_label(!is_gate, "state transfer").clicked() && is_gate {
                        self.qoala.target = Target::Transfer {
                            from: Operator::single(0, Component::Z),
                            to: Operator::single(self.qoala.nspins - 1, Component::Z),
                        };
                    }
                    if ui.selectable_label(is_gate, "gate").clicked() && !is_gate {
                        self.qoala.target = Target::Gate {
                            gate: GateChoice::Swap,
                            qubits: vec![0, self.qoala.nspins.min(2) - 1],
                        };
                        self.qoala.resize();
                    }
                });

                let nspins = self.qoala.nspins;
                match &mut self.qoala.target {
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
                    let mut ms = self.qoala.duration_s * 1e3;
                    if ui
                        .add(
                            egui::DragValue::new(&mut ms)
                                .speed(0.1)
                                .range(0.001..=1e5)
                                .suffix(" ms"),
                        )
                        .changed()
                    {
                        self.qoala.duration_s = ms * 1e-3;
                    }
                    ui.end_row();

                    ui.label("time slices");
                    ui.add(egui::DragValue::new(&mut self.qoala.nslices).range(2..=20000));
                    ui.end_row();

                    ui.label("iteration budget");
                    ui.add(egui::DragValue::new(&mut self.qoala.max_iter).range(0..=100000));
                    ui.end_row();

                    ui.label("penalty");
                    egui::ComboBox::from_id_salt("penalty")
                        .selected_text(self.qoala.penalty.name())
                        .show_ui(ui, |ui| {
                            for choice in PenaltyChoice::MENU {
                                ui.selectable_value(&mut self.qoala.penalty, choice, choice.name());
                            }
                        });
                    ui.end_row();

                    ui.label("splitting orders");
                    ui.horizontal(|ui| {
                        for order in [1usize, 2, 3, 4, 6] {
                            let mut on = self.qoala.splitset.contains(&order);
                            if ui.checkbox(&mut on, order.to_string()).changed() {
                                if on {
                                    self.qoala.splitset.push(order);
                                    self.qoala.splitset.sort_unstable();
                                } else if self.qoala.splitset.len() > 1 {
                                    self.qoala.splitset.retain(|o| *o != order);
                                }
                            }
                        }
                    });
                    ui.end_row();

                    ui.label("seed");
                    seed_editor(ui, &mut self.qoala.seed);
                    ui.end_row();
                });
                ui.label(
                    egui::RichText::new(format!(
                        "dimension {}  -  {} waveform points",
                        self.qoala.dimension(),
                        self.qoala.nslices * self.qoala.nchannels()
                    ))
                    .small()
                    .weak(),
                );
            });
    }

    fn escalade_band(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Band")
            .default_open(true)
            .show(ui, |ui| {
                let s = &mut self.escalade;
                egui::Grid::new("band").num_columns(2).show(ui, |ui| {
                    ui.label("spins");
                    ui.add(egui::DragValue::new(&mut s.nspins).range(1..=escalade::MAX_SPINS));
                    ui.end_row();

                    ui.label("bandwidth");
                    let mut khz = s.sw_hz * 1e-3;
                    if ui
                        .add(
                            egui::DragValue::new(&mut khz)
                                .speed(0.1)
                                .range(0.0..=1e4)
                                .suffix(" kHz"),
                        )
                        .changed()
                    {
                        s.sw_hz = khz * 1e3;
                    }
                    ui.end_row();

                    ui.label("from");
                    direction_menu(ui, "from", &mut s.from);
                    ui.end_row();

                    ui.label("to");
                    direction_menu(ui, "to", &mut s.to);
                    ui.end_row();
                });
                ui.label(
                    egui::RichText::new(format!(
                        "{} spins evenly from -{:.1} to +{:.1} kHz",
                        s.nspins,
                        s.sw_hz * 5e-4,
                        s.sw_hz * 5e-4
                    ))
                    .small()
                    .weak(),
                );
            });
    }

    fn escalade_field(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Field")
            .default_open(true)
            .show(ui, |ui| {
                let s = &mut self.escalade;
                egui::Grid::new("field").num_columns(2).show(ui, |ui| {
                    ui.label("amplitude");
                    let mut khz = s.rf_hz * 1e-3;
                    if ui
                        .add(
                            egui::DragValue::new(&mut khz)
                                .speed(0.1)
                                .range(0.001..=1e4)
                                .suffix(" kHz"),
                        )
                        .changed()
                    {
                        s.rf_hz = khz * 1e3;
                    }
                    ui.end_row();

                    ui.label("B1 compensation");
                    let mut on = s.b1_spread > 0.0;
                    if ui.checkbox(&mut on, "").changed() {
                        if on {
                            s.b1_spread = 0.2;
                            s.b1_fields = s.b1_fields.max(11);
                        } else {
                            s.b1_spread = 0.0;
                        }
                    }
                    ui.end_row();

                    if s.b1_spread > 0.0 {
                        ui.label("spread");
                        let mut percent = s.b1_spread * 100.0;
                        if ui
                            .add(
                                egui::DragValue::new(&mut percent)
                                    .speed(0.5)
                                    .range(1.0..=90.0)
                                    .prefix("+/- ")
                                    .suffix(" %"),
                            )
                            .changed()
                        {
                            s.b1_spread = percent / 100.0;
                        }
                        ui.end_row();

                        ui.label("fields");
                        ui.add(
                            egui::DragValue::new(&mut s.b1_fields).range(2..=escalade::MAX_FIELDS),
                        );
                        ui.end_row();
                    }
                });
                let note = if s.b1_spread > 0.0 {
                    format!(
                        "optimised over {} fields from {:.2} to {:.2} kHz",
                        s.b1_fields,
                        s.rf_hz * (1.0 - s.b1_spread) * 1e-3,
                        s.rf_hz * (1.0 + s.b1_spread) * 1e-3
                    )
                } else {
                    "optimised for the nominal field alone".to_string()
                };
                ui.label(egui::RichText::new(note).small().weak());
                ui.label(
                    egui::RichText::new("the pulse amplitude is kept within the nominal field")
                        .small()
                        .weak(),
                );
            });
    }

    fn escalade_pulse(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Pulse")
            .default_open(true)
            .show(ui, |ui| {
                let s = &mut self.escalade;
                egui::Grid::new("escalade_pulse")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("duration");
                        let mut us = s.duration_s * 1e6;
                        if ui
                            .add(
                                egui::DragValue::new(&mut us)
                                    .speed(1.0)
                                    .range(0.001..=1e7)
                                    .suffix(" µs"),
                            )
                            .changed()
                        {
                            s.duration_s = us * 1e-6;
                        }
                        ui.end_row();

                        ui.label("points");
                        ui.add(egui::DragValue::new(&mut s.nslices).range(1..=5000));
                        ui.end_row();

                        ui.label("optimiser");
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut s.use_hessian, true, "Newton (Hessian)");
                            ui.selectable_value(&mut s.use_hessian, false, "L-BFGS");
                        });
                        ui.end_row();

                        ui.label("target fidelity");
                        ui.add(
                            egui::DragValue::new(&mut s.target_fidelity)
                                .speed(0.001)
                                .range(0.5..=1.0)
                                .fixed_decimals(4),
                        );
                        ui.end_row();

                        ui.label("iteration budget");
                        ui.add(egui::DragValue::new(&mut s.max_iter).range(0..=100000));
                        ui.end_row();

                        ui.label("seed");
                        seed_editor(ui, &mut s.seed);
                        ui.end_row();
                    });
                ui.label(
                    egui::RichText::new(format!(
                        "{} spins x {} fields x {} points",
                        s.nspins,
                        s.fields_hz().len(),
                        s.nslices
                    ))
                    .small()
                    .weak(),
                );
            });
    }

    fn sharing(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Share and export", |ui| {
            let problem = self.problem();
            if ui.button("Copy link to this setup").clicked() {
                match platform::url_for_problem(&problem) {
                    Ok(url) => {
                        ui.ctx().copy_text(url);
                        self.message = Some("link copied".into());
                    }
                    Err(e) => self.message = Some(e),
                }
            }
            if ui.button("Download setup as JSON").clicked() {
                match serde_json::to_string_pretty(&problem) {
                    Ok(json) => {
                        export::offer_file(&format!("{}.json", slug(problem.name())), &json)
                    }
                    Err(e) => self.message = Some(e.to_string()),
                }
            }
            ui.separator();
            let have_result = self.result.is_some();
            ui.add_enabled_ui(have_result, |ui| {
                // Exports describe the run, so they use its problem rather
                // than whatever is in the editor now.
                if ui.button("Download waveform as CSV").clicked() {
                    if let Some((problem, result)) = &self.result {
                        let csv = export::waveform_csv(problem, &result.waveform);
                        export::offer_file(&format!("{}.csv", slug(problem.name())), &csv);
                    }
                }
            });
            if !have_result {
                ui.label(egui::RichText::new("run something first").small().weak());
            }
        });
    }
}

fn seed_editor(ui: &mut egui::Ui, seed: &mut Option<u64>) {
    ui.horizontal(|ui| {
        let mut fixed = seed.is_some();
        if ui.checkbox(&mut fixed, "fixed").changed() {
            *seed = fixed.then_some(1);
        }
        if let Some(seed) = seed {
            ui.add(egui::DragValue::new(seed));
        }
    });
}

fn direction_menu(ui: &mut egui::Ui, id: &str, direction: &mut Direction) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(direction.name())
        .width(60.0)
        .show_ui(ui, |ui| {
            for choice in Direction::ALL {
                ui.selectable_value(direction, choice, choice.name());
            }
        });
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

        if self.algorithm == Algorithm::Escalade {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.view, EscaladeView::Waveform, "Waveform");
                ui.selectable_value(&mut self.view, EscaladeView::Profile, "Offset profile");
                ui.selectable_value(&mut self.view, EscaladeView::B1, "B1 robustness");
            });
        }
        let view = match self.algorithm {
            Algorithm::Qoala => EscaladeView::Waveform,
            Algorithm::Escalade => self.view,
        };
        match view {
            EscaladeView::Waveform => {
                ui.label("waveform, Hz");
                self.channel_toggles(ui);
                self.waveform_plot(ui);
            }
            EscaladeView::Profile => self.profile_plot(ui),
            EscaladeView::B1 => self.b1_plots(ui),
        }
    }

    fn live_numbers(&mut self, ui: &mut egui::Ui) {
        let last = self.history.last();
        let adaptive = self.algorithm == Algorithm::Qoala;
        let max_amplitude = self.analysis.as_ref().map(|a| a.max_amplitude);
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
            if adaptive {
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
            }
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
            if let Some(amp) = max_amplitude {
                big(ui, "peak amplitude", format!("{:.1} % of B1", 100.0 * amp));
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
            Some((problem, _)) => problem.channel_names(),
            None => self.problem().channel_names(),
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
        let Some((problem, result)) = &self.result else {
            waiting(ui, "the waveform appears when the run finishes");
            return;
        };

        // Scaled by the problem the run used, not the one being edited: the
        // two part company the moment somebody changes a number after a run.
        let names = problem.channel_names();

        Plot::new("waveform")
            .legend(Legend::default())
            .allow_scroll(false)
            .x_axis_label("time, ms")
            .show(ui, |plot_ui| {
                for (k, name) in names.iter().enumerate() {
                    if !self.show_channels.get(k).copied().unwrap_or(true) {
                        continue;
                    }
                    let points = waveform::stairs(problem, &result.waveform, k);
                    plot_ui.line(Line::new(name.clone(), PlotPoints::from(points)));
                }
            });
    }

    /// Final magnetisation across offsets, as `ESCALADE_pulse_sim` plots it.
    fn profile_plot(&mut self, ui: &mut egui::Ui) {
        let (Some(analysis), Some((Problem::Escalade(setup), _))) = (&self.analysis, &self.result)
        else {
            waiting(ui, "the offset profile appears when the run finishes");
            return;
        };
        ui.label(format!(
            "final magnetisation from {} at the nominal field; the optimised band is +/- {:.1} kHz",
            setup.from.name(),
            setup.sw_hz * 5e-4
        ));

        type Series = fn(&qoala::escalade::profile::Magnetisation) -> f64;
        let series: [(&str, Series); 5] = [
            ("Ix", |m| m.x),
            ("Iy", |m| m.y),
            ("Iz", |m| m.z),
            ("|Ixy|", |m| m.transverse()),
            ("phase / pi", |m| m.phase()),
        ];
        let half_band = setup.sw_hz * 5e-4;

        Plot::new("profile")
            .legend(Legend::default())
            .allow_scroll(false)
            .x_axis_label("offset, kHz")
            .include_y(-1.05)
            .include_y(1.05)
            .show(ui, |plot_ui| {
                for (name, component) in series {
                    let points: Vec<[f64; 2]> = analysis
                        .profile
                        .iter()
                        .map(|p| [p.offset_hz * 1e-3, component(&p.m)])
                        .collect();
                    plot_ui.line(Line::new(name, PlotPoints::from(points)));
                }
                for edge in [-half_band, half_band] {
                    plot_ui.line(
                        Line::new(
                            "band edges",
                            PlotPoints::from(vec![[edge, -1.05], [edge, 1.05]]),
                        )
                        .color(egui::Color32::GRAY),
                    );
                }
            });
    }

    /// The offset-by-field map of Iy and the phase slope, as
    /// `ESCALADE_Bloch_B1` plots them.
    fn b1_plots(&mut self, ui: &mut egui::Ui) {
        let (Some(analysis), Some((Problem::Escalade(setup), _))) = (&self.analysis, &self.result)
        else {
            waiting(ui, "the B1 maps appear when the run finishes");
            return;
        };
        let texture = self
            .map_texture
            .get_or_insert_with(|| {
                ui.ctx().load_texture(
                    "b1-map",
                    colour_map(&analysis.map),
                    egui::TextureOptions::LINEAR,
                )
            })
            .id();

        let map = &analysis.map;
        let (x0, x1) = (
            map.offsets_hz.first().copied().unwrap_or(0.0) * 1e-3,
            map.offsets_hz.last().copied().unwrap_or(0.0) * 1e-3,
        );
        let (y0, y1) = (
            map.scales.first().copied().unwrap_or(0.5),
            map.scales.last().copied().unwrap_or(1.5),
        );
        let half_band = setup.sw_hz * 5e-4;
        let fields = setup.fields_hz();
        let (lo, hi) = (
            fields.first().copied().unwrap_or(setup.rf_hz) / setup.rf_hz,
            fields.last().copied().unwrap_or(setup.rf_hz) / setup.rf_hz,
        );
        let dphi = &analysis.dphi;

        ui.columns(2, |columns| {
            columns[0].label("final Iy over offset and field  (blue -1, white 0, red +1)");
            Plot::new("b1-map")
                .allow_scroll(false)
                .x_axis_label("offset, kHz")
                .y_axis_label("B1 / nominal")
                .show(&mut columns[0], |plot_ui| {
                    plot_ui.image(PlotImage::new(
                        "Iy",
                        texture,
                        PlotPoint::new((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                        egui::vec2((x1 - x0) as f32, (y1 - y0) as f32),
                    ));
                    let guide = egui::Color32::from_gray(40);
                    for edge in [-half_band, half_band] {
                        plot_ui.line(
                            Line::new("band", PlotPoints::from(vec![[edge, y0], [edge, y1]]))
                                .color(guide),
                        );
                    }
                    for field in [lo, hi] {
                        plot_ui.line(
                            Line::new(
                                "optimised fields",
                                PlotPoints::from(vec![[-half_band, field], [half_band, field]]),
                            )
                            .color(guide),
                        );
                    }
                });

            columns[1].label("on-resonance phase change across +/- 0.5 % of B1");
            Plot::new("dphi")
                .allow_scroll(false)
                .x_axis_label("B1 / nominal")
                .y_axis_label("degrees")
                .show(&mut columns[1], |plot_ui| {
                    let points: Vec<[f64; 2]> = dphi.iter().map(|&(z, d)| [z, d]).collect();
                    plot_ui.line(Line::new("dphi", PlotPoints::from(points)));
                });
        });
    }
}

// --------------------------------------------------------------- bits -----

fn waiting(ui: &mut egui::Ui, text: &str) {
    ui.centered_and_justified(|ui| {
        ui.label(egui::RichText::new(text).weak());
    });
}

/// Iy in `[-1, 1]` on a blue-white-red scale, highest field in the top row.
fn colour_map(map: &qoala::escalade::profile::B1Map) -> egui::ColorImage {
    let (rows, cols) = map.iy.shape();
    let mut pixels = Vec::with_capacity(rows * cols);
    for r in (0..rows).rev() {
        for c in 0..cols {
            pixels.push(diverging(map.iy[(r, c)]));
        }
    }
    egui::ColorImage::new([cols, rows], pixels)
}

fn diverging(v: f64) -> egui::Color32 {
    const NEGATIVE: [f64; 3] = [59.0, 76.0, 192.0];
    const MIDDLE: [f64; 3] = [235.0, 235.0, 235.0];
    const POSITIVE: [f64; 3] = [180.0, 4.0, 38.0];
    let t = v.clamp(-1.0, 1.0);
    let (end, u) = if t < 0.0 {
        (NEGATIVE, -t)
    } else {
        (POSITIVE, t)
    };
    let channel = |k: usize| (MIDDLE[k] + (end[k] - MIDDLE[k]) * u).round() as u8;
    egui::Color32::from_rgb(channel(0), channel(1), channel(2))
}

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
