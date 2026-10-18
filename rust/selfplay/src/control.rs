//! Local pipe protocol between a supervising client and its self-play child.

use std::{
    io::{self, BufRead, Read, Write},
    thread,
};

use rgo_artifacts::model_version_from_filename;
use rgo_protocol::selfplay::Command;
pub(super) use rgo_protocol::selfplay::Event;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinSet;

use super::{
    config::SelfPlayConfig,
    orchestrator::{ModelFile, SelfPlayOrchestrator},
};

const MAX_COMMAND_BYTES: u64 = 4096;

struct PendingEvent {
    event: Event,
    written: oneshot::Sender<io::Result<()>>,
}

#[derive(Clone)]
pub(super) struct EventPublisher(mpsc::Sender<PendingEvent>);

impl EventPublisher {
    fn start() -> io::Result<Self> {
        let (tx, rx) = mpsc::channel(1);
        // Dedicated stdio threads do not block Tokio's signal worker or keep
        // runtime shutdown waiting on an uncancellable stdin/stdout operation.
        thread::Builder::new()
            .name("selfplay-events".into())
            .spawn(move || write_events(io::stdout().lock(), rx))?;
        Ok(Self(tx))
    }

    /// Return after the event has been written and flushed to the client's pipe.
    pub(super) async fn emit(&self, event: Event) -> io::Result<()> {
        let (written, ack) = oneshot::channel();
        self.0
            .send(PendingEvent { event, written })
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "event writer closed"))?;
        ack.await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "event writer stopped"))?
    }

    /// The same acknowledged publication for callers on a dedicated OS thread.
    pub(super) fn blocking_emit(&self, event: Event) -> io::Result<()> {
        let (written, ack) = oneshot::channel();
        self.0
            .blocking_send(PendingEvent { event, written })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "event writer closed"))?;
        ack.blocking_recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "event writer stopped"))?
    }
}

fn write_events(mut output: impl Write, mut events: mpsc::Receiver<PendingEvent>) {
    while let Some(PendingEvent { event, written }) = events.blocking_recv() {
        let result = (|| {
            serde_json::to_writer(&mut output, &event).map_err(io::Error::other)?;
            output.write_all(b"\n")?;
            output.flush()
        })();
        let failed = result.is_err();
        let _ = written.send(result);
        if failed {
            break;
        }
    }
}

fn read_commands(mut input: impl BufRead, commands: mpsc::Sender<Result<Command, String>>) {
    loop {
        let mut line = Vec::new();
        let command = match input
            .by_ref()
            .take(MAX_COMMAND_BYTES + 1)
            .read_until(b'\n', &mut line)
        {
            Ok(0) => Ok(Command::Finish {}), // A disconnected supervisor stops new games.
            Ok(_) if line.len() as u64 > MAX_COMMAND_BYTES => {
                Err("control command exceeds 4096 bytes".into())
            }
            Ok(_) => serde_json::from_slice(&line)
                .map_err(|error| format!("invalid control command: {error}")),
            Err(error) => Err(format!("could not read control command: {error}")),
        };
        let terminal = !matches!(&command, Ok(Command::ModelReady { .. }));
        if commands.blocking_send(command).is_err() || terminal {
            break;
        }
    }
}

async fn apply_commands(
    mut commands: mpsc::Receiver<Result<Command, String>>,
    latest_tx: watch::Sender<Option<ModelFile>>,
    finish_tx: watch::Sender<bool>,
) -> Result<(), String> {
    while let Some(command) = commands.recv().await {
        match command? {
            Command::Finish {} => {
                let _ = finish_tx.send(true);
                return Ok(());
            }
            Command::ModelReady { path } => {
                let version = path
                    .file_name()
                    .and_then(model_version_from_filename)
                    .ok_or_else(|| {
                        format!("invalid canonical model filename: {}", path.display())
                    })?;
                if latest_tx
                    .borrow()
                    .as_ref()
                    .is_some_and(|current| version <= current.version)
                {
                    continue;
                }
                let metadata = tokio::fs::metadata(&path).await.map_err(|error| {
                    format!("announced model {} is unavailable: {error}", path.display())
                })?;
                if !metadata.is_file() {
                    return Err(format!("announced model {} is not a file", path.display()));
                }
                latest_tx
                    .send(Some(ModelFile { version, path }))
                    .map_err(|_| "model control receiver closed".to_owned())?;
            }
        }
    }
    Err("control reader stopped unexpectedly".into())
}

/// Start dedicated pipe I/O threads and supervise the self-play pipeline.
pub(super) async fn run_client_session(
    config: SelfPlayConfig,
    finish_tx: watch::Sender<bool>,
    finish_rx: watch::Receiver<bool>,
) -> Result<(), String> {
    let events = EventPublisher::start()
        .map_err(|error| format!("could not start event writer: {error}"))?;
    let result = supervise_self_play(config, finish_tx, finish_rx, &events).await;
    if let Err(message) = &result {
        let _ = events
            .emit(Event::Error {
                message: message.clone(),
            })
            .await;
    }
    result
}

async fn supervise_self_play(
    config: SelfPlayConfig,
    finish_tx: watch::Sender<bool>,
    finish_rx: watch::Receiver<bool>,
    events: &EventPublisher,
) -> Result<(), String> {
    let orchestrator = SelfPlayOrchestrator::new(config, events.clone());
    let (commands_tx, commands_rx) = mpsc::channel(1);
    thread::Builder::new()
        .name("selfplay-commands".into())
        .spawn(move || read_commands(io::stdin().lock(), commands_tx))
        .map_err(|error| format!("could not start command reader: {error}"))?;
    let (latest_tx, latest_rx) = watch::channel(None);
    let mut controls = JoinSet::new();
    controls.spawn(apply_commands(commands_rx, latest_tx.clone(), finish_tx));
    events
        .emit(Event::Ready {
            protocol_version: rgo_protocol::PROTOCOL_VERSION,
        })
        .await
        .map_err(|error| format!("could not write ready event: {error}"))?;
    let pipeline = orchestrator.run(latest_rx, finish_rx);
    tokio::pin!(pipeline);
    loop {
        tokio::select! {
            biased;
            Some(command_result) = controls.join_next() => {
                // controls exits successfully after applying the Finish command
                command_result.map_err(|error| format!("control task failed: {error}"))??;
            }
            result = &mut pipeline => {
                result.map_err(|error| format!("self-play failed: {error:?}"))?;
                break;
            }
        }
    }
    // Keep the model sender alive through graceful draining, even when the
    // finish command ended the command task. A closed model source is an error.
    drop(latest_tx);
    events
        .emit(Event::Stopped)
        .await
        .map_err(|error| format!("could not write stopped event: {error}"))
}

#[cfg(test)]
impl EventPublisher {
    pub(super) fn discard() -> Self {
        Self::with_output(io::sink())
    }

    pub(super) fn with_output(output: impl Write + Send + 'static) -> Self {
        let (tx, rx) = mpsc::channel(1);
        thread::spawn(move || write_events(output, rx));
        Self(tx)
    }
}
