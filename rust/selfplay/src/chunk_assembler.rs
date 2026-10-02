use tokio::sync::{mpsc, oneshot};

use super::training_data::{ChunkEncoder, EncodedChunk, TrainingSample};

pub(super) struct CompletedGame {
    pub(super) samples: Vec<TrainingSample>,
    pub(super) recycle_tx: oneshot::Sender<Vec<TrainingSample>>, // channel to send back the new training sample vector
}

pub(super) struct TrainingChunk {
    pub(super) bytes: Vec<u8>,
    pub(super) records: usize,
    pub(super) recycle_tx: oneshot::Sender<Vec<u8>>,
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

struct EncodedStep {
    active_chunk: Option<ChunkEncoder>,
    completed_chunk: Option<EncodedChunk>,
    game: CompletedGame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChunkAssemblerError {
    OutputClosed,
}

/// Collects completed games into training chunks.
///
/// Per-game chunks are published as soon as a game finishes. Fixed-record
/// chunks contain exactly the configured number of samples, except for a final
/// partial chunk at shutdown; a game may span multiple fixed-record chunks.
///
/// Workers transfer game buffers through `completed_games_rx`; emptied buffers
/// are returned to their originating workers for reuse. The channel lifecycle
/// remains asynchronous and cancellable, while encoding and hashing run on
/// Tokio's blocking pool. At most two byte buffers circulate between this
/// assembler and the sink. The assembler waits for the previous buffer before
/// sending another, so only one chunk is ever awaiting a return.
/// Fixed-record mode alone retains an active chunk.
pub(super) struct ChunkAssembler {
    mode: ChunkMode,
    completed_games_rx: mpsc::Receiver<CompletedGame>,
    chunks_tx: mpsc::Sender<TrainingChunk>,
    active_chunk: Option<ChunkEncoder>,
    spare_bytes: Option<Vec<u8>>,
    in_flight: Option<oneshot::Receiver<Vec<u8>>>,
}

impl ChunkAssembler {
    pub(super) fn new(
        mode: ChunkMode,
        completed_games_rx: mpsc::Receiver<CompletedGame>,
        chunks_tx: mpsc::Sender<TrainingChunk>,
    ) -> Self {
        let active_chunk = match mode {
            ChunkMode::PerGame => None,
            ChunkMode::FixedRecords(chunk_size) => Some(ChunkEncoder::new(chunk_size)),
        };
        Self {
            mode,
            completed_games_rx,
            chunks_tx,
            active_chunk,
            spare_bytes: Some(Vec::new()),
            in_flight: None,
        }
    }

    pub(super) async fn run(mut self) -> Result<(), ChunkAssemblerError> {
        while let Some(game) = self.completed_games_rx.recv().await {
            match self.mode {
                ChunkMode::PerGame => self.append_game_per_game(game).await?,
                ChunkMode::FixedRecords(chunk_size) => {
                    self.append_game_fixed(game, chunk_size).await?
                }
            }
        }
        match self.mode {
            ChunkMode::PerGame => Ok(()),
            ChunkMode::FixedRecords(_) => self.flush().await,
        }
    }

    async fn append_game_per_game(
        &mut self,
        mut game: CompletedGame,
    ) -> Result<(), ChunkAssemblerError> {
        if game.samples.is_empty() {
            let _ = game.recycle_tx.send(game.samples);
            return Ok(());
        }

        let bytes = self.acquire_buffer();
        let (game, chunk) = tokio::task::spawn_blocking(move || {
            let mut encoder = ChunkEncoder::with_buffer(game.samples.len(), bytes);
            for sample in &game.samples {
                encoder.push(sample);
            }
            game.samples.clear();
            (game, encoder.finish())
        })
        .await
        .expect("chunk encoding task panicked");

        let _ = game.recycle_tx.send(game.samples);
        self.send_chunk(chunk).await
    }

    async fn append_game_fixed(
        &mut self,
        mut game: CompletedGame,
        chunk_size: usize,
    ) -> Result<(), ChunkAssemblerError> {
        loop {
            let active_chunk = match self.active_chunk.take() {
                Some(chunk) => chunk,
                None => ChunkEncoder::with_buffer(chunk_size, self.acquire_buffer()),
            };
            let result = tokio::task::spawn_blocking(move || {
                Self::encode_until_chunk(active_chunk, chunk_size, game)
            })
            .await
            .expect("chunk encoding task panicked");

            self.active_chunk = result.active_chunk;
            game = result.game;

            if game.samples.is_empty() {
                // Return the allocation before a potentially blocking output
                // send so the worker can begin its next game immediately.
                let _ = game.recycle_tx.send(game.samples);
                if let Some(chunk) = result.completed_chunk {
                    self.send_chunk(chunk).await?;
                }
                return Ok(());
            }

            self.send_chunk(
                result
                    .completed_chunk
                    .expect("unfinished game must have filled a chunk"),
            )
            .await?;
        }
    }

    fn encode_until_chunk(
        mut active_chunk: ChunkEncoder,
        chunk_size: usize,
        mut game: CompletedGame,
    ) -> EncodedStep {
        // Training records are sampled independently, so their order within a
        // chunk is irrelevant. Encode from the tail so removing them does not
        // shift the remaining samples or copy them into an intermediate chunk.
        let count = active_chunk.remaining_capacity().min(game.samples.len());
        let remaining = game.samples.len() - count;
        for sample in &game.samples[remaining..] {
            active_chunk.push(sample);
        }
        game.samples.truncate(remaining);

        let (active_chunk, completed_chunk) = if active_chunk.record_count() == chunk_size {
            (None, Some(active_chunk.finish()))
        } else {
            (Some(active_chunk), None)
        };

        EncodedStep {
            active_chunk,
            completed_chunk,
            game,
        }
    }

    fn acquire_buffer(&mut self) -> Vec<u8> {
        self.spare_bytes
            .take()
            .expect("the previous send must leave a spare byte buffer")
    }

    async fn send_chunk(&mut self, chunk: EncodedChunk) -> Result<(), ChunkAssemblerError> {
        if let Some(previous) = self.in_flight.take() {
            debug_assert!(self.spare_bytes.is_none());
            self.spare_bytes = Some(
                previous
                    .await
                    .map_err(|_| ChunkAssemblerError::OutputClosed)?,
            );
        } else if self.spare_bytes.is_none() {
            // Per-game mode starts with one buffer; its first publication
            // creates the second buffer for overlapping the next encode.
            // After the first send, in_flight remains Some on the success path.
            self.spare_bytes = Some(Vec::new());
        }
        let (recycle_tx, recycle_rx) = oneshot::channel();
        self.chunks_tx
            .send(TrainingChunk {
                bytes: chunk.bytes,
                records: chunk.records,
                recycle_tx,
            })
            .await
            .map_err(|_| ChunkAssemblerError::OutputClosed)?;
        self.in_flight = Some(recycle_rx);
        Ok(())
    }

    async fn flush(&mut self) -> Result<(), ChunkAssemblerError> {
        let Some(active_chunk) = self.active_chunk.take() else {
            return Ok(());
        };
        if active_chunk.record_count() == 0 {
            return Ok(());
        }

        let chunk = tokio::task::spawn_blocking(move || active_chunk.finish())
            .await
            .expect("chunk encoding task panicked");
        self.send_chunk(chunk).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        game::{game_state::GameState, rules::Rules},
        inference::{inputs::NNInput, policy::MAX_POLICY_SIZE},
        training_data::{TrainingSample, ValueTarget, encode_chunk, verify_chunk_checksum},
    };

    fn sample() -> TrainingSample {
        TrainingSample {
            input: NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH_9)),
            policy_target: [half::f16::ZERO; MAX_POLICY_SIZE],
            value_target: ValueTarget {
                win_probability: 0.5,
                score_mean: 0.0,
                score_stdev: 1.0,
                ownership: [1; crate::game::board::MAX_BOARD_AREA],
            },
        }
    }

    fn record_count(chunk: &TrainingChunk) -> u32 {
        let count = u32::from_le_bytes(chunk.bytes[16..20].try_into().unwrap());
        assert_eq!(chunk.records, count as usize);
        count
    }

    #[test]
    fn one_encoding_step_leaves_the_rest_of_a_large_game_unencoded() {
        let (recycle_tx, _recycle_rx) = oneshot::channel();
        let game = CompletedGame {
            samples: (0..5).map(|_| sample()).collect(),
            recycle_tx,
        };

        let step = ChunkAssembler::encode_until_chunk(ChunkEncoder::new(2), 2, game);

        assert_eq!(step.game.samples.len(), 3);
        assert!(step.active_chunk.is_none());
        let chunk = step.completed_chunk.expect("one full chunk");
        assert_eq!(chunk.records, 2);
        assert_eq!(
            u32::from_le_bytes(chunk.bytes[16..20].try_into().unwrap()),
            2
        );
        assert!(verify_chunk_checksum(&chunk.bytes));
    }

    #[tokio::test]
    async fn per_game_mode_publishes_each_game_without_waiting_or_splitting() {
        let (completed_tx, completed_rx) = mpsc::channel(1);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(1);
        let assembler = ChunkAssembler::new(ChunkMode::PerGame, completed_rx, chunks_tx);
        let assembler_task = tokio::spawn(assembler.run());

        for game_length in [1, 3] {
            let samples: Vec<_> = (0..game_length).map(|_| sample()).collect();
            let expected_bytes = encode_chunk(&samples);
            let allocation = samples.as_ptr();
            let (recycle_tx, recycle_rx) = oneshot::channel();
            completed_tx
                .send(CompletedGame {
                    samples,
                    recycle_tx,
                })
                .await
                .unwrap();

            // The input channel remains open: publication cannot rely on shutdown.
            let chunk = chunks_rx.recv().await.unwrap();
            assert_eq!(record_count(&chunk) as usize, game_length);
            assert_eq!(chunk.bytes, expected_bytes);
            let mut bytes = chunk.bytes;
            bytes.clear();
            chunk.recycle_tx.send(bytes).unwrap();
            let recycled = recycle_rx.await.unwrap();
            assert!(recycled.is_empty());
            assert_eq!(recycled.as_ptr(), allocation);
        }

        drop(completed_tx);
        assert_eq!(assembler_task.await.unwrap(), Ok(()));
        assert!(chunks_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn per_game_mode_recycles_an_empty_game_without_publishing() {
        let (completed_tx, completed_rx) = mpsc::channel(1);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(1);
        let assembler = ChunkAssembler::new(ChunkMode::PerGame, completed_rx, chunks_tx);
        let assembler_task = tokio::spawn(assembler.run());
        let (recycle_tx, recycle_rx) = oneshot::channel();
        completed_tx
            .send(CompletedGame {
                samples: Vec::new(),
                recycle_tx,
            })
            .await
            .unwrap();

        assert!(recycle_rx.await.unwrap().is_empty());
        assert!(matches!(
            chunks_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        drop(completed_tx);
        assert_eq!(assembler_task.await.unwrap(), Ok(()));
        assert!(chunks_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn per_game_mode_reuses_only_two_byte_buffers() {
        let (completed_tx, completed_rx) = mpsc::channel(1);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(1);
        let assembler = ChunkAssembler::new(ChunkMode::PerGame, completed_rx, chunks_tx);
        let assembler_task = tokio::spawn(assembler.run());

        let (recycle_tx, recycle_rx) = oneshot::channel();
        completed_tx
            .send(CompletedGame {
                samples: vec![sample()],
                recycle_tx,
            })
            .await
            .unwrap();
        let first = chunks_rx.recv().await.unwrap();
        let first_allocation = first.bytes.as_ptr();
        assert!(recycle_rx.await.unwrap().is_empty());

        let (recycle_tx, recycle_rx) = oneshot::channel();
        completed_tx
            .send(CompletedGame {
                samples: vec![sample()],
                recycle_tx,
            })
            .await
            .unwrap();
        // Encoding may finish, but publication waits for the first buffer.
        assert!(recycle_rx.await.unwrap().is_empty());
        assert!(matches!(
            chunks_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        let mut bytes = first.bytes;
        bytes.clear();
        first.recycle_tx.send(bytes).unwrap();

        let second = chunks_rx.recv().await.unwrap();
        assert_ne!(second.bytes.as_ptr(), first_allocation);

        let (recycle_tx, recycle_rx) = oneshot::channel();
        completed_tx
            .send(CompletedGame {
                samples: vec![sample()],
                recycle_tx,
            })
            .await
            .unwrap();
        assert!(recycle_rx.await.unwrap().is_empty());
        let mut bytes = second.bytes;
        bytes.clear();
        second.recycle_tx.send(bytes).unwrap();
        let third = chunks_rx.recv().await.unwrap();
        assert_eq!(third.bytes.as_ptr(), first_allocation);
        drop(completed_tx);
        assert_eq!(assembler_task.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn dropped_sink_buffer_stops_assembly_instead_of_hanging() {
        let (completed_tx, completed_rx) = mpsc::channel(1);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(1);
        let assembler = ChunkAssembler::new(ChunkMode::PerGame, completed_rx, chunks_tx);
        let assembler_task = tokio::spawn(assembler.run());

        let (recycle_tx, recycle_rx) = oneshot::channel();
        completed_tx
            .send(CompletedGame {
                samples: vec![sample()],
                recycle_tx,
            })
            .await
            .unwrap();
        // Dropping the chunk simulates sink failure before buffer return.
        drop(chunks_rx.recv().await.unwrap());
        assert!(recycle_rx.await.unwrap().is_empty());

        let (recycle_tx, recycle_rx) = oneshot::channel();
        completed_tx
            .send(CompletedGame {
                samples: vec![sample()],
                recycle_tx,
            })
            .await
            .unwrap();
        assert_eq!(
            assembler_task.await.unwrap(),
            Err(ChunkAssemblerError::OutputClosed)
        );
        assert!(recycle_rx.await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn emits_chunks_and_returns_empty_game_buffers() {
        let (completed_tx, completed_rx) = mpsc::channel(1);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(1);
        let (recycle_tx, recycle_rx) = oneshot::channel();
        let assembler = ChunkAssembler::new(ChunkMode::FixedRecords(2), completed_rx, chunks_tx);
        let assembler_task = tokio::spawn(assembler.run());
        let samples = vec![sample(), sample()];
        let allocation = samples.as_ptr();

        completed_tx
            .send(CompletedGame {
                samples,
                recycle_tx,
            })
            .await
            .unwrap();

        let chunk = chunks_rx.recv().await.unwrap();
        assert_eq!(record_count(&chunk), 2);
        assert!(verify_chunk_checksum(&chunk.bytes));
        let recycled = recycle_rx.await.unwrap();
        assert!(recycled.is_empty());
        assert_eq!(recycled.as_ptr(), allocation);

        drop(completed_tx);
        assert_eq!(assembler_task.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn combines_complete_games_in_one_chunk() {
        let (completed_tx, completed_rx) = mpsc::channel(2);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(1);
        let assembler = ChunkAssembler::new(ChunkMode::FixedRecords(2), completed_rx, chunks_tx);
        let assembler_task = tokio::spawn(assembler.run());
        let mut recycle_receivers = Vec::new();

        for _ in 0..2 {
            let (recycle_tx, recycle_rx) = oneshot::channel();
            recycle_receivers.push(recycle_rx);
            completed_tx
                .send(CompletedGame {
                    samples: vec![sample()],
                    recycle_tx,
                })
                .await
                .unwrap();
        }
        drop(completed_tx);

        let chunk = chunks_rx.recv().await.unwrap();
        assert_eq!(record_count(&chunk), 2);
        assert!(verify_chunk_checksum(&chunk.bytes));
        for receiver in recycle_receivers {
            assert!(receiver.await.unwrap().is_empty());
        }
        assert_eq!(assembler_task.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn splits_large_games_into_fixed_chunks_and_a_final_partial_chunk() {
        let (completed_tx, completed_rx) = mpsc::channel(1);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(3);
        let (recycle_tx, recycle_rx) = oneshot::channel();
        let assembler = ChunkAssembler::new(ChunkMode::FixedRecords(2), completed_rx, chunks_tx);
        let assembler_task = tokio::spawn(assembler.run());

        completed_tx
            .send(CompletedGame {
                samples: (0..5).map(|_| sample()).collect(),
                recycle_tx,
            })
            .await
            .unwrap();
        drop(completed_tx);

        let mut chunk_sizes = Vec::new();
        let mut byte_allocations = Vec::new();
        while let Some(chunk) = chunks_rx.recv().await {
            assert!(verify_chunk_checksum(&chunk.bytes));
            chunk_sizes.push(record_count(&chunk));
            byte_allocations.push(chunk.bytes.as_ptr());
            let mut bytes = chunk.bytes;
            bytes.clear();
            let _ = chunk.recycle_tx.send(bytes);
        }
        assert_eq!(chunk_sizes, [2, 2, 1]);
        assert_ne!(byte_allocations[0], byte_allocations[1]);
        assert_eq!(byte_allocations[0], byte_allocations[2]);
        assert!(recycle_rx.await.unwrap().is_empty());
        assert_eq!(assembler_task.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn writer_failure_cancels_the_unprocessed_game_recycle() {
        let (completed_tx, completed_rx) = mpsc::channel(2);
        let (chunks_tx, chunks_rx) = mpsc::channel(1);
        drop(chunks_rx);
        let assembler = ChunkAssembler::new(ChunkMode::FixedRecords(1), completed_rx, chunks_tx);

        let (first_recycle_tx, first_recycle_rx) = oneshot::channel();
        completed_tx
            .send(CompletedGame {
                samples: vec![sample()],
                recycle_tx: first_recycle_tx,
            })
            .await
            .unwrap();
        let (second_recycle_tx, second_recycle_rx) = oneshot::channel();
        completed_tx
            .send(CompletedGame {
                samples: vec![sample()],
                recycle_tx: second_recycle_tx,
            })
            .await
            .unwrap();
        let assembler_task = tokio::spawn(assembler.run());

        assert_eq!(
            assembler_task.await.unwrap(),
            Err(ChunkAssemblerError::OutputClosed)
        );
        assert!(first_recycle_rx.await.unwrap().is_empty());
        assert!(second_recycle_rx.await.is_err());
    }

    #[tokio::test]
    async fn aborting_while_idle_closes_the_pipeline() {
        let (completed_tx, completed_rx) = mpsc::channel(1);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(1);
        let assembler = ChunkAssembler::new(ChunkMode::FixedRecords(2), completed_rx, chunks_tx);
        let assembler_task = tokio::spawn(assembler.run());

        tokio::task::yield_now().await;
        assembler_task.abort();
        assert!(assembler_task.await.unwrap_err().is_cancelled());
        assert!(completed_tx.is_closed());
        assert!(chunks_rx.recv().await.is_none());
    }
}
