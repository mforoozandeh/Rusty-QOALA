//! Native runner: the optimisation on a background thread, messages down a
//! channel.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use crate::run::{run_to_sink, MessageSink, RunMessage};
use crate::setup::Setup;

struct ChannelSink {
    tx: Sender<RunMessage>,
    cancel: Arc<AtomicBool>,
}

impl MessageSink for ChannelSink {
    fn send(&mut self, message: RunMessage) {
        // A closed channel means the interface has gone away; the cancel flag
        // will stop the run on the next iteration.
        let _ = self.tx.send(message);
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// A run in progress.
pub struct Runner {
    rx: Receiver<RunMessage>,
    cancel: Arc<AtomicBool>,
}

impl Runner {
    /// Start `setup` on a background thread.
    pub fn start(setup: Setup) -> Self {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut sink = ChannelSink {
            tx,
            cancel: Arc::clone(&cancel),
        };
        std::thread::spawn(move || run_to_sink(&setup, &mut sink));
        Runner { rx, cancel }
    }

    /// Every message that has arrived since the last call.
    pub fn drain(&mut self) -> Vec<RunMessage> {
        self.rx.try_iter().collect()
    }

    /// Ask the optimiser to stop at the next iteration.  It finishes the one
    /// it is on and returns the waveform it reached.
    pub fn cancel(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation keeps the partial waveform on this platform.
    pub const CANCEL_KEEPS_WAVEFORM: bool = true;
}
