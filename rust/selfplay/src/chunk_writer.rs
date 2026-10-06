use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use rand::{TryRng, rngs::SysError, rngs::SysRng};
use rgo_artifacts::{ChunkId, chunk_path};
use tokio::sync::mpsc;

use super::{
    chunk_encoder::ChunkEncoder,
    control::{Event, EventPublisher},
    game_records::CompletedGame,
};
use crate::{
    game::{
        board::{BOARD_STORAGE_LEN, Board, Color, MAX_BOARD_AREA, Player},
        game_state::GameState,
        rules::Rules,
    },
    inference::{inputs::NNInput, policy::loc_to_spatial},
};

const PENDING_CHUNK_FILE: &str = ".pending-chunk.tmp";
const MAX_CHUNK_RECORDS: usize = 1024; // only applies to per game

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(
    tag = "mode",
    content = "records",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(super) enum ChunkMode {
    /// Publish on each completed game, splitting at MAX_CHUNK_RECORDS and carrying tails.
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
/// Workers transfer completed move records through a bounded queue. Replay
/// generates inputs directly into reusable scratch storage; records are freed
/// after encoding. One reusable encoder owns the encoded byte allocation
/// through publication, without overlapping writes.
/// Closing the queue drains games and flushes the final partial chunk.
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
    replay_state: GameState,
    input: NNInput,
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
        let replay_state = GameState::new(Rules::default());
        let input = NNInput::encode(&replay_state);
        Self {
            mode,
            output_dir,
            completed_games_rx,
            events,
            active_chunk,
            replay_state,
            input,
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
            self.append_game(game)?;
        }
        if self.active_chunk.record_count() > 0 {
            self.publish()?;
        }
        Ok(())
    }

    fn append_game(&mut self, game: CompletedGame) -> Result<(), ChunkWriterError> {
        debug_assert!(
            !game.records.is_empty(),
            "completed games must have records"
        );
        if let ChunkMode::PerGame = self.mode {
            self.active_chunk.set_capacity(
                (self.active_chunk.record_count() + game.records.len()).min(MAX_CHUNK_RECORDS),
            );
        }
        self.replay_state.reset(game.rules);
        let black_ownership = ownership_for_player(
            self.replay_state.board(),
            &game.final_ownership,
            Player::Black,
        );
        let white_ownership = ownership_for_player(
            self.replay_state.board(),
            &game.final_ownership,
            Player::White,
        );
        let white_score = self
            .replay_state
            .board()
            .locs()
            .map(|loc| match game.final_ownership[loc.index()] {
                Color::White => 1_i16,
                Color::Black => -1,
                Color::Empty => 0,
                Color::Wall => unreachable!("board iterator yielded a wall"),
            })
            .sum::<i16>() as f32
            + game.rules.komi;
        let white_win_target = if white_score > 0.0 {
            1.0
        } else if white_score < 0.0 {
            0.0
        } else {
            0.5
        };
        for (index, record) in game.records.iter().enumerate() {
            debug_assert_eq!(record.player, self.replay_state.next_player());
            self.input.encode_in_place(&self.replay_state);
            let (win_target, final_score, ownership) = match record.player {
                Player::Black => (1.0 - white_win_target, -white_score, &black_ownership),
                Player::White => (white_win_target, white_score, &white_ownership),
            };
            self.active_chunk.push(
                &self.input,
                &record.policy_target,
                win_target,
                final_score,
                ownership,
            );
            let played = self.replay_state.play(record.selected_move);
            debug_assert!(played, "recorded self-play move failed during replay");

            // Publish intermediate full chunks while retaining the remaining
            // move records. A full final chunk is published after releasing them.
            if self.active_chunk.remaining_capacity() == 0 && index + 1 < game.records.len() {
                let num_records = self.active_chunk.record_count();
                self.publish()?;
                self.active_chunk.reset(num_records);
            }
        }
        debug_assert!(self.replay_state.is_finished());
        debug_assert_eq!(
            self.replay_state.turn_number(),
            game.records.len(),
            "replayed game has the wrong length"
        );
        drop(game);
        if self.active_chunk.remaining_capacity() == 0 {
            self.publish()?;
            if let ChunkMode::FixedRecords(records) = self.mode {
                self.active_chunk.reset(records);
            }
        }
        Ok(())
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
        Ok(())
    }
}

fn ownership_for_player(
    board: &Board,
    final_ownership: &[Color; BOARD_STORAGE_LEN],
    player: Player,
) -> [u8; MAX_BOARD_AREA] {
    let mut ownership = [1; MAX_BOARD_AREA];
    for loc in board.locs() {
        ownership[loc_to_spatial(board, loc)] = match (final_ownership[loc.index()], player) {
            (Color::Empty, _) => 1,
            (Color::Black, Player::Black) | (Color::White, Player::White) => 2,
            (Color::Black, Player::White) | (Color::White, Player::Black) => 0,
            (Color::Wall, _) => unreachable!("board iterator yielded a wall"),
        };
    }
    ownership
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
