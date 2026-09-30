//! Standalone self-play application built on the reusable engine.

use rgo_engine::{game, inference, search};
use std::{env, fs, future::Future, io, path::Path, process};
use tokio::{sync::watch, task::JoinSet};

use config::SelfPlayConfig;
use orchestrator::SelfPlayOrchestrator;

/// Change this one value to reproduce or vary all self-play randomness.
const RNG_SEED: u64 = 0;

mod chunk_assembler;
mod chunk_sink;
mod config;
mod orchestrator;
mod params;
mod training_data;
mod worker;

const USAGE: &str = "Usage: rgo-selfplay <config.toml>";

fn main() {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        eprintln!("{USAGE}");
        process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("{USAGE}");
        process::exit(2);
    }
    if path == "--help" || path == "-h" {
        println!("{USAGE}");
        return;
    }

    let config = SelfPlayConfig::load(Path::new(&path)).unwrap_or_else(|error| fatal(error));
    let runtime = build_runtime()
        .unwrap_or_else(|error| fatal(format!("could not start Tokio runtime: {error}")));
    if let Err(error) = runtime.block_on(run(config)) {
        // The orchestrator returns promptly on failure and may leave worker
        // threads active. Exit before dropping the runtime or waiting for them.
        fatal(error);
    }
}

fn build_runtime() -> io::Result<tokio::runtime::Runtime> {
    // block_on runs orchestration on the calling thread. A separate runtime
    // worker keeps signal handling and filesystem tasks alive during model load.
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
}

fn fatal(error: impl std::fmt::Display) -> ! {
    eprintln!("rgo-selfplay: {error}");
    process::exit(1);
}

async fn run(config: SelfPlayConfig) -> Result<(), String> {
    let metadata = fs::metadata(&config.inference.model_dir).map_err(|error| {
        format!(
            "could not open model directory {}: {error}",
            config.inference.model_dir.display()
        )
    })?;
    if !metadata.is_dir() {
        return Err(format!(
            "model path is not a directory: {}",
            config.inference.model_dir.display()
        ));
    }
    prepare_output_dir(&config.output_dir).map_err(|error| {
        format!(
            "could not prepare output directory {}: {error}",
            config.output_dir.display()
        )
    })?;
    with_shutdown(|finish_rx| async move {
        eprintln!(
            "Starting {} self-play workers; watching {}; writing chunks to {}. Ctrl-C requests a graceful shutdown.",
            config.worker_threads * config.workers_per_thread,
            config.inference.model_dir.display(),
            config.output_dir.display(),
        );
        SelfPlayOrchestrator::new(
            config.search,
            config.self_play,
            config.inference,
            config.worker_threads,
            config.workers_per_thread,
            config.chunk,
            config.output_dir,
        )
        .run_local(finish_rx)
        .await
        .map_err(|error| format!("self-play failed: {error:?}"))
    })
    .await?;
    eprintln!("Self-play stopped; training chunks drained.");
    Ok(())
}

async fn with_shutdown<F: Future<Output = Result<(), String>>>(
    run: impl FnOnce(watch::Receiver<bool>) -> F,
) -> Result<(), String> {
    // Register before starting orchestration. The signal task runs on the
    // runtime worker, independently of synchronous model lifecycle operations.
    let mut signals = ShutdownSignals::new()
        .map_err(|error| format!("could not register shutdown signals: {error}"))?;
    let (finish_tx, finish_rx) = watch::channel(false);
    let mut shutdown_tasks = JoinSet::<()>::new();
    shutdown_tasks.spawn(async move {
        signals.recv().await.unwrap_or_else(|error| {
            fatal(format!("could not receive shutdown signal: {error}"))
        });
        let _ = finish_tx.send(true);
        eprintln!("Finishing active games and draining training chunks; a second shutdown signal forces exit.");
        signals.recv().await.unwrap_or_else(|error| {
            fatal(format!("could not receive shutdown signal: {error}"))
        });
        fatal("second shutdown signal received; exiting immediately");
    });
    // JoinSet aborts the signal task on success or error, keeping the sender
    // alive throughout orchestration without leaving a detached task behind.
    run(finish_rx).await
}

/// Provision directory entries durably before the file sink publishes chunks.
fn prepare_output_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    let parents = {
        let absolute = env::current_dir()?.join(path);
        let mut missing = absolute.as_path();
        let mut parents = Vec::new();
        while !missing.try_exists()? {
            let parent = missing
                .parent()
                .ok_or_else(|| io::Error::other("output directory has no parent"))?;
            parents.push(parent.to_path_buf());
            missing = parent;
        }
        parents
    };
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    for parent in parents {
        fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

struct ShutdownSignals {
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
    #[cfg(unix)]
    terminate: tokio::signal::unix::Signal,
    #[cfg(windows)]
    interrupt: tokio::signal::windows::CtrlC,
}

impl ShutdownSignals {
    fn new() -> io::Result<Self> {
        Ok(Self {
            #[cfg(unix)]
            interrupt: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?,
            #[cfg(unix)]
            terminate: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
            #[cfg(windows)]
            interrupt: tokio::signal::windows::ctrl_c()?,
        })
    }

    async fn recv(&mut self) -> io::Result<()> {
        #[cfg(unix)]
        {
            let received = tokio::select! {
                received = self.interrupt.recv() => received,
                received = self.terminate.recv() => received,
            };
            received.ok_or_else(|| io::Error::other("shutdown signal stream closed"))
        }
        #[cfg(windows)]
        self.interrupt
            .recv()
            .await
            .ok_or_else(|| io::Error::other("shutdown signal stream closed"))
    }
}

#[cfg(all(test, unix))]
mod shutdown_tests;
