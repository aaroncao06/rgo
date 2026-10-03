use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use rand::{TryRng, rngs::SysError, rngs::SysRng};
use rgo_artifacts::{ChunkId, chunk_path};
use tokio::sync::mpsc;

use super::{
    control::{Event, EventPublisher},
    training_data::{ChunkEncoder, TrainingSample},
};

const PENDING_CHUNK_FILE: &str = ".pending-chunk.tmp";

pub(super) struct CompletedGame {
    pub(super) samples: Vec<TrainingSample>,
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
/// Workers transfer completed games through a bounded queue. Sample vectors
/// are freed after encoding. One reusable encoder owns the encoded
/// byte allocation through publication, without overlapping writes.
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
    active_chunk: ChunkEncoder,
}

impl ChunkWriter {
    pub(super) fn new(
        mode: ChunkMode,
        output_dir: PathBuf,
        completed_games_rx: mpsc::Receiver<CompletedGame>,
        events: EventPublisher,
    ) -> Self {
        let active_chunk = match mode {
            // Per-game mode sets the actual capacity when a game arrives.
            ChunkMode::PerGame => ChunkEncoder::new(1),
            ChunkMode::FixedRecords(records) => ChunkEncoder::new(records),
        };
        Self {
            mode,
            output_dir,
            completed_games_rx,
            events,
            active_chunk,
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
        if self.active_chunk.record_count() > 0 {
            debug_assert!(matches!(self.mode, ChunkMode::FixedRecords(_)));
            self.publish()?;
        }
        Ok(())
    }

    fn append_game_per_game(&mut self, game: CompletedGame) -> Result<(), ChunkWriterError> {
        debug_assert!(
            !game.samples.is_empty(),
            "completed games must have samples"
        );
        self.active_chunk.reset(game.samples.len());
        for sample in &game.samples {
            self.active_chunk.push(sample);
        }
        drop(game);
        self.publish()
    }

    fn append_game_fixed(
        &mut self,
        mut game: CompletedGame,
        records: usize,
    ) -> Result<(), ChunkWriterError> {
        loop {
            // Preserve tail-first consumption: independently sampled records
            // need no game order, and truncation does not shift remaining data.
            let count = self
                .active_chunk
                .remaining_capacity()
                .min(game.samples.len());
            let remaining = game.samples.len() - count;
            for sample in &game.samples[remaining..] {
                self.active_chunk.push(sample);
            }
            game.samples.truncate(remaining);

            if game.samples.is_empty() {
                // Release samples before publishing the game's final chunk.
                drop(game);
                if self.active_chunk.record_count() == records {
                    self.publish()?;
                }
                return Ok(());
            }
            debug_assert!(
                self.active_chunk.record_count() == records,
                "unfinished game must have filled a chunk"
            );
            self.publish()?;
        }
    }

    fn publish(&mut self) -> Result<(), ChunkWriterError> {
        let records = self.active_chunk.record_count();
        let bytes = self.active_chunk.finish();
        let id = random_chunk_id().map_err(ChunkWriterError::RandomnessUnavailable)?;
        let final_path = chunk_path(&self.output_dir, id);
        let temporary_path = self.output_dir.join(PENDING_CHUNK_FILE);

        let mut file = File::create(&temporary_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary_path, &final_path)?;
        sync_directory(&self.output_dir)?;
        self.events.blocking_emit(Event::ChunkReady {
            path: final_path,
            bytes: bytes.len(),
            records,
        })?;
        // Per-game mode resets when the next game supplies its capacity.
        if let ChunkMode::FixedRecords(records) = self.mode {
            self.active_chunk.reset(records);
        }
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
