//! Child ownership, durable chunk notifications, model updates, and shutdown.

use crate::{
    config::Config,
    connection::{Connection, next_retry_delay},
    model_cache::{ModelAcquisition, ModelCache},
};
use rgo_artifacts::{ModelVersion, chunk_id_from_filename, model_version_from_filename};
use rgo_protocol::{
    PROTOCOL_VERSION, ShutdownSignals,
    client_server::{Request, Response},
    selfplay::{Command, Event},
};
use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncWrite, BufReader},
    process,
    sync::{mpsc, watch},
    task::JoinSet,
    time,
};

/// Submit one completed local chunk and wait for the server's acceptance receipt.
async fn submit_chunk(connection: &mut Connection<'_>, path: &Path) -> io::Result<()> {
    match connection
        .request(&Request::SubmitChunk { path: path.into() })
        .await?
    {
        Response::ChunkAccepted {} => {
            eprintln!("rgo-client: accepted {}", path.display());
            Ok(())
        }
        Response::Error { message } => Err(io::Error::other(message)),
        response => Err(io::Error::other(format!(
            "invalid chunk receipt: {response:?}"
        ))),
    }
}

/// Poll again if a newer publication removed the advertised server name.
async fn acquire_latest_model(
    connection: &mut Connection<'_>,
    loaded: Option<ModelVersion>,
    mut acquire: impl AsyncFnMut(PathBuf, ModelVersion) -> io::Result<ModelAcquisition>,
) -> io::Result<Option<(ModelVersion, PathBuf)>> {
    let mut delay = Duration::ZERO;
    let mut denied_link = None;
    loop {
        let source = match connection.request(&Request::GetLatestModel {}).await? {
            Response::Model { path } => path,
            Response::Error { message } => return Err(io::Error::other(message)),
            _ => return Err(io::Error::other("invalid model response")),
        };
        let Some(source) = source else {
            return match denied_link {
                Some((_, error)) => Err(error),
                None => Ok(None),
            };
        };
        let version = source
            .file_name()
            .filter(|_| source.is_absolute())
            .and_then(model_version_from_filename)
            .ok_or_else(|| io::Error::other(format!("invalid model path: {}", source.display())))?;
        if let Some((denied_version, error)) = denied_link.take()
            && version <= denied_version
        {
            return Err(error);
        }
        if loaded.is_some_and(|current| version <= current) {
            return Ok(None);
        }
        match acquire(source, version).await? {
            ModelAcquisition::Cached(path) => return Ok(Some((version, path))),
            ModelAcquisition::SourceMissing => {
                delay = next_retry_delay(delay);
                time::sleep(delay).await;
            }
            ModelAcquisition::LinkAccessDenied(error) => {
                // Check for a replacement before treating the denial as a pruning race.
                denied_link = Some((version, error));
            }
        }
    }
}

/// Handle worker events and model polls on one server connection.
/// A request finishes or exhausts its retries before this loop handles another.
async fn run_supervision_loop(
    mut connection: Connection<'_>,
    mut events: mpsc::Receiver<io::Result<Event>>,
    commands: mpsc::Sender<Command>,
    mut finishing: watch::Receiver<bool>,
    models: &Arc<ModelCache>,
    config: &Config,
) -> io::Result<()> {
    let mut ready = false;
    let mut pending = None;
    let mut loaded = None;
    let mut poll = time::interval(config.poll_interval());
    poll.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            event = events.recv() => match event.ok_or_else(|| io::Error::other("worker stdout closed before stopped"))?? {
                Event::Ready { protocol_version } if !ready && protocol_version == PROTOCOL_VERSION => {
                    ready = true;
                }
                Event::ModelLoaded { version } if pending == Some(version) => {
                    loaded = pending.take();
                }
                Event::ChunkReady { path, .. } if ready => submit_chunk(&mut connection, &path).await?,
                Event::Stopped if ready => return Ok(()),
                Event::Error { message } => return Err(io::Error::other(format!("self-play failed: {message}"))),
                event => return Err(io::Error::other(format!("unexpected worker event: {event:?}"))),
            },
            // Poll when the worker is ready, no load is pending, and shutdown hasn't begun.
            _ = poll.tick(), if ready && pending.is_none() && !*finishing.borrow() => {
                // Look up and cache the latest model, stopping the wait on shutdown or timeout.
                let deadline = time::Instant::now() + config.retry_timeout();
                let model = tokio::select! {
                    biased;
                    changed = finishing.changed() => {
                        changed.map_err(|_| io::Error::other("supervisor stopped"))?;
                        continue;
                    }
                    _ = time::sleep_until(deadline) => return Err(io::Error::new(
                        io::ErrorKind::TimedOut, "model acquisition retry budget expired"
                    )),
                    result = acquire_latest_model(&mut connection, loaded, async |source, version| {
                        models.acquire(source, version).await
                    }) => result?,
                };
                // Queue the cached path and mark its version pending until ModelLoaded.
                if let Some((version, path)) = model {
                    commands
                        .try_send(Command::ModelReady { path })
                        .map_err(io::Error::other)?;
                    pending = Some(version);
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
    // Own the self-play directory before recovery or worker launch, so another
    // client cannot submit its files at the same time.
    let selfplay_dir =
        rgo_fs::prepare_directory(&config.data_dir.join(connection.run_name()).join("selfplay"))?;
    let client_lock = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(selfplay_dir.join(".client.lock"))?;
    client_lock.try_lock().map_err(|error| {
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
        .open(selfplay_dir.join(".selfplay.lock"))?;
    recovery_worker_lock.try_lock().map_err(|error| {
        io::Error::other(format!(
            "previous worker is still using output directory: {error}"
        ))
    })?;
    // Recover pending chunks for the selected run before starting a writer.
    let mut pending = Vec::new();
    for entry in fs::read_dir(&selfplay_dir)? {
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
    let model_dir = rgo_fs::prepare_directory(&selfplay_dir.join("models"))?;
    let (models, cache_closed) = ModelCache::new(model_dir, client_lock);
    // Start the worker with the self-play directory used for recovery.
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
        .arg(&selfplay_dir)
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
                run_supervision_loop(
                    connection,
                    events_rx,
                    commands_tx,
                    finish_rx,
                    &models,
                    config
                ),
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
    result?;

    // A running cache operation can outlive the worker and still hold the lock.
    drop(models);
    tokio::select! {
        biased;
        _ = cache_closed => Ok(()),
        signal = signals.recv() => {
            signal?;
            Err(io::Error::other("second shutdown signal; forcing client exit"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rgo_artifacts::model_path;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn denied_model_link_is_retried_only_if_the_version_advanced() {
        for replacement in [Some(43), Some(42), Some(41), None] {
            let directory = tempfile::tempdir().unwrap();
            let source = model_path(directory.path(), 42);
            let newer_source = replacement.map(|version| model_path(directory.path(), version));
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let config = Config {
                server: listener.local_addr().unwrap(),
                client_id: "test".into(),
                data_dir: directory.path().into(),
                selfplay_config: "unused.toml".into(),
                selfplay_binary: None,
                model_poll_interval_ms: 10,
                request_timeout_secs: 2,
                retry_timeout_secs: 2,
            };
            let server = tokio::spawn(async move {
                let (stream, _) = listener.accept().await.unwrap();
                let mut stream = BufReader::new(stream);
                let hello = rgo_protocol::read::<Request>(&mut stream).await.unwrap();
                assert!(matches!(hello, Some(Request::Hello { .. })));
                rgo_protocol::write(
                    stream.get_mut(),
                    &Response::Welcome {
                        protocol_version: PROTOCOL_VERSION,
                        run_name: "test".into(),
                    },
                )
                .await
                .unwrap();
                for path in [Some(source), newer_source] {
                    let request = rgo_protocol::read::<Request>(&mut stream).await.unwrap();
                    assert!(matches!(request, Some(Request::GetLatestModel {})));
                    rgo_protocol::write(stream.get_mut(), &Response::Model { path })
                        .await
                        .unwrap();
                }
            });

            let mut connection = Connection::open(&config).await.unwrap();
            let mut attempts = Vec::new();
            let result = acquire_latest_model(&mut connection, None, async |path, version| {
                attempts.push(version);
                if version == 42 {
                    Ok(ModelAcquisition::LinkAccessDenied(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "original hard-link error",
                    )))
                } else {
                    Ok(ModelAcquisition::Cached(path))
                }
            })
            .await;
            if replacement == Some(43) {
                assert_eq!(attempts, [42, 43]);
                assert_eq!(
                    result.unwrap(),
                    Some((43, model_path(directory.path(), 43)))
                );
            } else {
                assert_eq!(attempts, [42]);
                let error = result.unwrap_err();
                assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
                assert_eq!(error.to_string(), "original hard-link error");
            }
            server.await.unwrap();
        }
    }
}
