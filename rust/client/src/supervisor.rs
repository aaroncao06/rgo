//! Child ownership, durable chunk notifications, model updates, and shutdown.

use crate::{config::Config, connection::Connection};
use rgo_artifacts::chunk_id_from_filename;
use rgo_protocol::{
    PROTOCOL_VERSION, ShutdownSignals,
    client_server::{Request, Response},
    selfplay::{Command, Event},
};
use std::{
    fs::{self, File},
    io,
    process::Stdio,
};
use tokio::{
    io::{AsyncWrite, BufReader},
    process,
    sync::{mpsc, watch},
    task::JoinSet,
    time,
};

/// Submit one completed local chunk and wait for the server's acceptance receipt.
async fn submit_chunk(connection: &mut Connection<'_>, path: &std::path::Path) -> io::Result<()> {
    match connection
        .request(&Request::AcceptChunk { path: path.into() })
        .await?
    {
        Response::ChunkAccepted {
            records,
            already_accepted,
        } => {
            eprintln!(
                "rgo-client: accepted {}: {records} records{}",
                path.display(),
                if already_accepted { " (retry)" } else { "" }
            );
            Ok(())
        }
        Response::Error { message } => Err(io::Error::other(message)),
        response => Err(io::Error::other(format!(
            "invalid chunk receipt: {response:?}"
        ))),
    }
}

/// Handle worker events and model polls on one server connection.
/// A request finishes or exhausts its retries before this loop handles another.
async fn run_supervision_loop(
    mut connection: Connection<'_>,
    mut events: mpsc::Receiver<io::Result<Event>>,
    commands: mpsc::Sender<Command>,
    finishing: watch::Receiver<bool>,
    config: &Config,
) -> io::Result<()> {
    let mut ready = false;
    let mut poll = time::interval(config.poll_interval());
    poll.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            event = events.recv() => match event.ok_or_else(|| io::Error::other("worker stdout closed before stopped"))?? {
                Event::Ready { protocol_version } if !ready && protocol_version == PROTOCOL_VERSION => {
                    ready = true;
                }
                Event::ChunkReady { path, .. } if ready => submit_chunk(&mut connection, &path).await?,
                Event::Stopped if ready => return Ok(()),
                Event::Error { message } => return Err(io::Error::other(format!("self-play failed: {message}"))),
                event => return Err(io::Error::other(format!("unexpected worker event: {event:?}"))),
            },
            _ = poll.tick(), if ready && !*finishing.borrow() => {
                let path = match connection.request(&Request::LatestModel {}).await? {
                    Response::Model { path } => path,
                    Response::Error { message } => return Err(io::Error::other(message)),
                    _ => return Err(io::Error::other("invalid model response")),
                };
                // Shutdown may have begun while the model request was in flight.
                if let Some(path) = path && !*finishing.borrow()
                    && commands.send(Command::ModelReady { path }).await.is_err()
                    && !*finishing.borrow()
                {
                    return Err(io::Error::other("worker command writer closed"));
                }
            }
        }
    }
}

/// Write commands to worker stdin, giving shutdown priority over model updates.
async fn write_commands(
    mut output: impl AsyncWrite + Unpin,
    mut commands: mpsc::Receiver<Command>,
    mut finishing: watch::Receiver<bool>,
) -> io::Result<()> {
    loop {
        let command = tokio::select! {
            biased;
            changed = finishing.changed() => {
                changed.map_err(|_| io::Error::other("supervisor stopped"))?;
                Command::Finish {}
            }
            command = commands.recv() => command.ok_or_else(|| io::Error::other("command queue closed"))?,
        };
        rgo_protocol::write(&mut output, &command).await?;
        if matches!(command, Command::Finish {}) {
            return Ok(());
        }
    }
}

/// Recover pending chunks, launch one worker, and supervise it through shutdown.
pub async fn run(config: &Config) -> io::Result<()> {
    let mut signals = ShutdownSignals::new()?;
    let mut connection = tokio::select! {
        biased;
        signal = signals.recv() => { signal?; return Ok(()); }
        connection = Connection::open(config) => connection?,
    };
    // Own the output directory before recovery or worker launch, so another
    // client cannot submit its files at the same time.
    let output_dir = config.prepare_output_dir()?;
    let _client_lock = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(output_dir.join(".client.lock"))?;
    _client_lock.try_lock().map_err(|error| {
        io::Error::other(format!(
            "client output directory is already supervised: {error}"
        ))
    })?;
    // An orphaned child retains its own lock while draining after pipe EOF.
    let recovery_worker_lock = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(output_dir.join(".selfplay.lock"))?;
    recovery_worker_lock.try_lock().map_err(|error| {
        io::Error::other(format!(
            "previous worker is still using output directory: {error}"
        ))
    })?;
    // Finish accepting chunks left by an earlier run before starting a writer.
    let mut pending = Vec::new();
    for entry in fs::read_dir(&output_dir)? {
        let entry = entry?;
        if chunk_id_from_filename(&entry.file_name()).is_some() {
            pending.push(entry.path());
        }
    }
    pending.sort_unstable();
    // Cancelling an in-flight request discards this connection. The server may
    // still accept it; durable receipts make the remaining files safe to retry.
    let recovery = async {
        for path in pending {
            submit_chunk(&mut connection, &path).await?;
        }
        // Also observe a pending signal when there were no files to recover.
        tokio::task::yield_now().await;
        Ok::<_, io::Error>(())
    };
    tokio::select! {
        biased;
        signal = signals.recv() => { signal?; return Ok(()); }
        result = recovery => result?,
    }
    // Start the worker only after recovery has completed. It reads the same
    // output path from its TOML and canonicalizes it before publishing chunks.
    // Configured relative binary paths use the client's working directory.
    let binary = match &config.selfplay_binary {
        Some(path) if path.is_absolute() => path.clone(),
        Some(path) => std::env::current_dir()?.join(path),
        None => std::env::current_exe()?.with_file_name(if cfg!(windows) {
            "rgo-selfplay.exe"
        } else {
            "rgo-selfplay"
        }),
    };
    drop(recovery_worker_lock);
    let mut child = process::Command::new(binary)
        .arg(&config.selfplay_config)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()?;
    let stdout = child.stdout.take().expect("piped stdout");
    let stdin = child.stdin.take().expect("piped stdin");
    // Separate pipe I/O from coordination. Bounded channels carry worker
    // events and model commands; a watch flag requests finish.
    let (events_tx, events_rx) = mpsc::channel(1);
    let (commands_tx, commands_rx) = mpsc::channel(1);
    let (finish_tx, finish_rx) = watch::channel(false);
    let mut tasks = JoinSet::new();
    let errors = events_tx.clone();
    tasks.spawn(async move {
        let mut stdout = BufReader::new(stdout);
        loop {
            let event = match rgo_protocol::read::<Event>(&mut stdout).await {
                Ok(Some(event)) => Ok(event),
                Ok(None) => Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "worker stdout closed",
                )),
                Err(error) => Err(error),
            };
            let failed = event.is_err();
            if events_tx.send(event).await.is_err() || failed {
                break;
            }
        }
    });
    let writer_finish = finish_rx.clone();
    tasks.spawn(async move {
        if let Err(error) = write_commands(stdin, commands_rx, writer_finish).await {
            let _ = errors.send(Err(error)).await;
        }
    });
    // Success requires both the worker's Stopped event and a successful exit.
    // The first signal requests a drain; the next interrupts the session.
    let result = {
        let session = async {
            tokio::try_join!(
                run_supervision_loop(connection, events_rx, commands_tx, finish_rx, config),
                async {
                    let status = child.wait().await?;
                    if status.success() {
                        Ok(())
                    } else {
                        Err(io::Error::other(format!("self-play exited with {status}")))
                    }
                },
            )?;
            Ok(())
        };
        tokio::pin!(session);
        tokio::select! {
            biased;
            result = &mut session => result,
            signal = signals.recv() => {
                signal?;
                let _ = finish_tx.send(true);
                eprintln!("rgo-client: finishing active games and accepting final chunks");
                tokio::select! {
                    result = &mut session => result,
                    signal = signals.recv() => {
                        signal?;
                        Err(io::Error::other("second shutdown signal; forcing worker exit"))
                    }
                }
            }
        }
    };
    // Stop pipe tasks and reap any remaining child before releasing the lock.
    tasks.shutdown().await;
    if child.try_wait()?.is_none() {
        child.kill().await?;
    }
    result
}
