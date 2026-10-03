use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use rand::{TryRng, rngs::SysError, rngs::SysRng};
use rgo_artifacts::{ChunkId, chunk_path};
use tokio::sync::{mpsc, oneshot};

use super::{
    control::{Event, EventPublisher},
    training_data::{ChunkEncoder, EncodedChunk, TrainingSample},
};

const PENDING_CHUNK_FILE: &str = ".pending-chunk.tmp";

pub(super) struct CompletedGame {
    pub(super) samples: Vec<TrainingSample>,
    pub(super) recycle_tx: oneshot::Sender<Vec<TrainingSample>>,
}

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(
    tag = "mode",
    content = "records",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(super) enum ChunkMode {
    /// Publish every completed game immediately as one chunk.
    PerGame,
    /// Combine or split games to publish chunks of exactly this many records.
    FixedRecords(usize),
}

#[derive(Debug)]
pub(super) enum ChunkWriterError {
    Io(io::Error),
    RandomnessUnavailable(SysError),
}

impl From<io::Error> for ChunkWriterError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Encodes and durably publishes chunks on one dedicated OS thread.
///
/// Workers submit completed games through a bounded queue and receive their
/// emptied sample vectors back for reuse. One encoded byte allocation moves
/// between the active encoder and publication, without overlapping writes.
/// Closing the queue drains games and flushes the final partial fixed chunk.
///
/// The caller must durably provision the output directory before starting.
/// Exactly one writer may use a directory at a time. Random 128-bit IDs avoid
/// filename reuse after restarts or deletion; one temporary path bounds
/// abandoned staging data to at most one chunk.
pub(super) struct ChunkWriter {
    mode: ChunkMode,
    output_dir: PathBuf,
    completed_games_rx: mpsc::Receiver<CompletedGame>,
    events: EventPublisher,
    active_chunk: Option<ChunkEncoder>,
    bytes: Vec<u8>,
}

impl ChunkWriter {
    pub(super) fn new(
        mode: ChunkMode,
        output_dir: PathBuf,
        completed_games_rx: mpsc::Receiver<CompletedGame>,
        events: EventPublisher,
    ) -> Self {
        let active_chunk = match mode {
            ChunkMode::PerGame => None,
            ChunkMode::FixedRecords(records) => Some(ChunkEncoder::new(records)),
        };
        Self {
            mode,
            output_dir,
            completed_games_rx,
            events,
            active_chunk,
            bytes: Vec::new(),
        }
    }

    /// Blocking entry point; do not call on a Tokio runtime thread.
    pub(super) fn run(mut self) -> Result<(), ChunkWriterError> {
        let metadata = fs::metadata(&self.output_dir)?;
        if !metadata.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                "chunk output path is not a directory",
            )
            .into());
        }

        while let Some(game) = self.completed_games_rx.blocking_recv() {
            match self.mode {
                ChunkMode::PerGame => self.append_game_per_game(game)?,
                ChunkMode::FixedRecords(records) => self.append_game_fixed(game, records)?,
            }
        }
        if let Some(chunk) = self.active_chunk.take()
            && chunk.record_count() > 0
        {
            self.publish(chunk.finish())?;
        }
        Ok(())
    }

    fn append_game_per_game(&mut self, mut game: CompletedGame) -> Result<(), ChunkWriterError> {
        if game.samples.is_empty() {
            let _ = game.recycle_tx.send(game.samples);
            return Ok(());
        }
        let mut encoder =
            ChunkEncoder::with_buffer(game.samples.len(), std::mem::take(&mut self.bytes));
        for sample in &game.samples {
            encoder.push(sample);
        }
        game.samples.clear();
        let chunk = encoder.finish();
        let _ = game.recycle_tx.send(game.samples);
        self.publish(chunk)
    }

    fn append_game_fixed(
        &mut self,
        mut game: CompletedGame,
        records: usize,
    ) -> Result<(), ChunkWriterError> {
        loop {
            let mut encoder = self.active_chunk.take().unwrap_or_else(|| {
                ChunkEncoder::with_buffer(records, std::mem::take(&mut self.bytes))
            });
            // Preserve tail-first consumption: independently sampled records
            // need no game order, and truncation does not shift remaining data.
            let count = encoder.remaining_capacity().min(game.samples.len());
            let remaining = game.samples.len() - count;
            for sample in &game.samples[remaining..] {
                encoder.push(sample);
            }
            game.samples.truncate(remaining);

            let chunk = if encoder.record_count() == records {
                Some(encoder.finish())
            } else {
                self.active_chunk = Some(encoder);
                None
            };
            if game.samples.is_empty() {
                // Let the worker start its next game before publishing this
                // game's final chunk or waiting for the client pipe to flush.
                let _ = game.recycle_tx.send(game.samples);
                if let Some(chunk) = chunk {
                    self.publish(chunk)?;
                }
                return Ok(());
            }
            self.publish(chunk.expect("unfinished game must have filled a chunk"))?;
        }
    }

    fn publish(&mut self, mut chunk: EncodedChunk) -> Result<(), ChunkWriterError> {
        let id = random_chunk_id().map_err(ChunkWriterError::RandomnessUnavailable)?;
        let final_path = chunk_path(&self.output_dir, id);
        let temporary_path = self.output_dir.join(PENDING_CHUNK_FILE);

        let mut file = File::create(&temporary_path)?;
        file.write_all(&chunk.bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary_path, &final_path)?;
        sync_directory(&self.output_dir)?;
        self.events.blocking_emit(Event::ChunkReady {
            path: final_path,
            bytes: chunk.bytes.len(),
            records: chunk.records,
        })?;
        chunk.bytes.clear();
        self.bytes = chunk.bytes;
        Ok(())
    }
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

fn random_chunk_id() -> Result<ChunkId, SysError> {
    let mut bytes = [0; size_of::<ChunkId>()];
    SysRng.try_fill_bytes(&mut bytes)?;
    Ok(ChunkId::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests;
