//! Self-play worker process supervised by the client over local pipes.

use rgo_engine::{game, inference, search};
use rgo_protocol::ShutdownSignals;
use std::{env, fs, future::Future, io, path::Path, process};
use tokio::{sync::watch, task::JoinSet};

use config::SelfPlayConfig;

/// Change this one value to reproduce or vary all self-play randomness.
const RNG_SEED: u64 = 0;

mod chunk_encoder;
mod chunk_writer;
mod config;
mod control;
mod game_records;
mod orchestrator;
mod params;
mod worker;

const USAGE: &str = "Usage: rgo-selfplay <config.toml> [--model-dir <path>] [--output-dir <path>] (client-supervised worker)";

fn main() {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        eprintln!("{USAGE}");
        process::exit(2);
    };
    if path == "--help" || path == "-h" {
        println!("{USAGE}");
        return;
    }

    let mut model_dir = None;
    let mut output_dir = None;
    while let Some(flag) = args.next() {
        let destination = match flag.to_str() {
            Some("--model-dir") if model_dir.is_none() => &mut model_dir,
            Some("--output-dir") if output_dir.is_none() => &mut output_dir,
            _ => {
                eprintln!("{USAGE}");
                process::exit(2);
            }
        };
        *destination = Some(args.next().unwrap_or_else(|| {
            eprintln!("{USAGE}");
            process::exit(2);
        }));
    }
    let mut config = SelfPlayConfig::load(Path::new(&path)).unwrap_or_else(|error| fatal(error));
    if let Some(directory) = model_dir {
        config.inference.model_dir = directory.into();
    }
    if let Some(directory) = output_dir {
        config.output_dir = directory.into();
    }
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
    rgo_fs::prepare_directory(&config.output_dir).map_err(|error| {
        format!(
            "could not prepare output directory {}: {error}",
            config.output_dir.display()
        )
    })?;
    // The child owns this lock through its entire drain, even if its parent dies.
    let _output_lock = fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(config.output_dir.join(".selfplay.lock"))
        .map_err(|error| error.to_string())?;
    _output_lock
        .try_lock()
        .map_err(|error| format!("self-play output directory is already in use: {error}"))?;
    with_shutdown(|finish_tx, finish_rx| async move {
        eprintln!(
            "Starting {} self-play workers; receiving model commands for {}; writing chunks to {}.",
            config.worker_threads * config.workers_per_thread,
            config.inference.model_dir.display(),
            config.output_dir.display(),
        );
        control::run_client_session(config, finish_tx, finish_rx).await
    })
    .await?;
    eprintln!("Self-play stopped; training chunks drained.");
    Ok(())
}

async fn with_shutdown<F: Future<Output = Result<(), String>>>(
    run: impl FnOnce(watch::Sender<bool>, watch::Receiver<bool>) -> F,
) -> Result<(), String> {
    // Register before starting orchestration. The signal task runs on the
    // runtime worker, independently of synchronous model lifecycle operations.
    let mut signals = ShutdownSignals::new()
        .map_err(|error| format!("could not register shutdown signals: {error}"))?;
    let (finish_tx, finish_rx) = watch::channel(false);
    let signal_finish_tx = finish_tx.clone();
    let mut shutdown_tasks = JoinSet::<()>::new();
    shutdown_tasks.spawn(async move {
        signals.recv().await.unwrap_or_else(|error| {
            fatal(format!("could not receive shutdown signal: {error}"))
        });
        let _ = signal_finish_tx.send(true);
        eprintln!("Finishing active games and draining training chunks; a second shutdown signal forces exit.");
        signals.recv().await.unwrap_or_else(|error| {
            fatal(format!("could not receive shutdown signal: {error}"))
        });
        fatal("second shutdown signal received; exiting immediately");
    });
    // JoinSet aborts the signal task on success or error, keeping the sender
    // alive throughout orchestration without leaving a detached task behind.
    run(finish_tx, finish_rx).await
}

#[cfg(all(test, unix))]
#[path = "../tests/unit/shutdown.rs"]
mod shutdown_tests;
