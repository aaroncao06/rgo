//! Run from the repository root:
//! cargo run --offline --release --manifest-path rust/Cargo.toml \
//!     -p rgo-selfplay --example chunk_io_bench -- [existing-output-parent]
//! Uses the real encoder and the former sink's Tokio write/sync/rename/directory-sync
//! sequence, retained as a historical benchmark. The current writer uses std::fs.
//! Encoding, validation, setup, and cleanup are outside timed regions.
//! Reads are immediate, warm-cache reads into a reusable 256 KiB upload buffer.
//! The no-sync control is NOT a durable publication alternative.

use std::{io, path::Path, time::Instant};

pub use rgo_engine::{game, inference, search};
#[path = "../src/training_data.rs"]
#[allow(dead_code)] // The standalone benchmark uses only part of the encoder API.
mod training_data;

use game::{game_state::GameState, rules::Rules};
use inference::{inputs::NNInput, policy::MAX_POLICY_SIZE};
use rgo_artifacts::chunk_path;
use serde_json::json;
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
};
use training_data::{ChunkEncoder, TrainingSample, ValueTarget};

struct Scratch(std::path::PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn mock_chunk(records: usize) -> Vec<u8> {
    let state = GameState::new(Rules::TROMP_TAYLORISH_9);
    let policy_len = state.board().dim().pow(2) + 1;
    let mut policy_target = [half::f16::ZERO; MAX_POLICY_SIZE];
    policy_target[..policy_len].fill(half::f16::from_f64(1.0 / policy_len as f64));
    let mut sample = TrainingSample {
        input: NNInput::encode(&state),
        policy_target,
        value_target: ValueTarget {
            win_probability: 0.5,
            score_mean: 0.0,
            score_stdev: 1.0,
            ownership: [1; game::board::MAX_BOARD_AREA],
        },
    };
    let mut encoder = ChunkEncoder::new(records);
    for i in 0..records {
        sample.value_target.score_mean = (i % 81) as f32 - 40.0;
        sample.value_target.win_probability = (i % 101) as f32 / 100.0;
        encoder.push(&sample);
    }
    assert_eq!(encoder.record_count(), records);
    let bytes = encoder.finish();
    assert!(training_data::verify_chunk_checksum(bytes));
    bytes.to_vec()
}

// Columns: write, file sync, rename, directory sync, save total, read, save+read.
async fn measure(
    dir: &Path,
    bytes: &[u8],
    durable: bool,
    id: u128,
    buffer: &mut [u8],
) -> io::Result<[f64; 7]> {
    let pending = dir.join(".pending-chunk.tmp");
    let published = chunk_path(dir, id);
    let start = Instant::now();
    let mut file = fs::File::create(&pending).await?;
    file.write_all(bytes).await?;
    // Tokio may still have a buffered write in flight. sync_all waits for it in
    // the sink path; the no-sync control must explicitly flush before rename.
    if !durable {
        file.flush().await?;
    }
    let written = Instant::now();
    if durable {
        file.sync_all().await?;
    }
    let synced = Instant::now();
    drop(file);
    fs::rename(&pending, &published).await?;
    let renamed = Instant::now();
    if durable {
        fs::File::open(dir).await?.sync_all().await?;
    }
    let saved = Instant::now();
    let mut input = fs::File::open(&published).await?;
    let mut count = 0;
    loop {
        let n = input.read(buffer).await?;
        if n == 0 {
            break;
        }
        std::hint::black_box(&buffer[..n]);
        count += n;
    }
    drop(input);
    let read = Instant::now();
    assert_eq!(count, bytes.len());
    // Validate all bytes outside the timing, and remove every temporary file.
    assert_eq!(fs::read(&published).await?, bytes);
    fs::remove_file(&published).await?;
    let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1000.0;
    Ok([
        ms(start, written),
        ms(written, synced),
        ms(synced, renamed),
        ms(renamed, saved),
        ms(start, saved),
        ms(saved, read),
        ms(start, read),
    ])
}

fn summary(samples: &[[f64; 7]]) -> serde_json::Value {
    let mut result = serde_json::Map::new();
    for (col, name) in [
        "write_ms",
        "file_sync_ms",
        "rename_ms",
        "directory_sync_ms",
        "save_ms",
        "read_ms",
        "save_read_ms",
    ]
    .iter()
    .enumerate()
    {
        let mut values: Vec<_> = samples.iter().map(|row| row[col]).collect();
        values.sort_by(f64::total_cmp);
        result.insert(
            (*name).into(),
            json!({
                "median": values[values.len() / 2],
                "p95": values[(values.len() * 95).div_ceil(100) - 1],
                "mean": values.iter().sum::<f64>() / values.len() as f64,
            }),
        );
    }
    result.into()
}

fn main() -> io::Result<()> {
    let parent = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or(std::env::current_dir()?);
    let scratch_path = parent.join(format!(".rgo-chunk-bench-{}", std::process::id()));
    std::fs::create_dir(&scratch_path)?;
    // Only clean up a directory that this invocation successfully created.
    let scratch = Scratch(scratch_path);
    std::fs::File::open(&parent)?.sync_all()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let mut reports = Vec::new();
        let mut buffer = vec![0; 256 * 1024];
        let mut id = 0u128;
        for records in [100, 200, 25_000] {
            let bytes = mock_chunk(records);
            let iterations = if records < 1000 { 100 } else { 30 };
            let mut durable = Vec::new();
            let mut buffered = Vec::new();
            for i in 0..iterations + 3 {
                // Alternate order to reduce drift between control and sink path.
                for sync in if i % 2 == 0 { [true, false] } else { [false, true] } {
                    id += 1;
                    let row = measure(&scratch.0, &bytes, sync, id, &mut buffer).await?;
                    if i >= 3 {
                        if sync {
                            durable.push(row);
                        } else {
                            buffered.push(row);
                        }
                    }
                }
            }
            let report = json!({"records":records, "bytes":bytes.len(), "iterations":iterations,
                "durable":summary(&durable), "no_sync_control":summary(&buffered)});
            eprintln!("Finished {records} records ({} bytes), {iterations} measured iterations per mode", bytes.len());
            reports.push(report);
        }
        println!("{}", serde_json::to_string_pretty(&json!({
            "directory": parent.canonicalize()?,
            "read_buffer_bytes":buffer.len(),
            "warmup_iterations_per_mode":3,
            "notes":"Real encoder; historical Tokio sink I/O sequence without event delivery; immediate warm-cache reads; no network, no cold-cache or F_FULLFSYNC test; setup/encoding/validation/deletion excluded",
            "results":reports
        }))?);
        Ok(())
    })
}
