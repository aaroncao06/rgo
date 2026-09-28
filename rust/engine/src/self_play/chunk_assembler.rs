use tokio::sync::{mpsc, oneshot};

use super::training_data::{ChunkEncoder, TrainingSample};

pub(super) struct CompletedGame {
    pub(super) samples: Vec<TrainingSample>,
    pub(super) recycle_tx: oneshot::Sender<Vec<TrainingSample>>, // channel to send back the new training sample vector
}

pub(super) struct TrainingChunk {
    pub(super) bytes: Vec<u8>,
}

struct EncodedStep {
    active_chunk: ChunkEncoder,
    completed_chunk: Option<TrainingChunk>,
    game: CompletedGame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChunkAssemblerError {
    OutputClosed,
}

/// Collects completed games into training chunks.
///
/// Full chunks contain exactly `chunk_size` samples; only the final chunk emitted
/// during shutdown may be smaller. A game may span multiple chunks.
///
/// This task is the sole owner of the active chunk. Workers transfer game
/// buffers through `completed_games_rx`; emptied buffers are returned to their
/// originating workers for reuse. The channel lifecycle remains asynchronous
/// and cancellable, while bounded encoding and hashing jobs run on Tokio's
/// blocking pool.
pub(super) struct ChunkAssembler {
    chunk_size: usize,
    completed_games_rx: mpsc::Receiver<CompletedGame>,
    chunks_tx: mpsc::Sender<TrainingChunk>,
    active_chunk: Option<ChunkEncoder>,
}

impl ChunkAssembler {
    pub(super) fn new(
        chunk_size: usize,
        completed_games_rx: mpsc::Receiver<CompletedGame>,
        chunks_tx: mpsc::Sender<TrainingChunk>,
    ) -> Self {
        assert!(chunk_size > 0, "training chunks must be nonempty");
        Self {
            chunk_size,
            completed_games_rx,
            chunks_tx,
            active_chunk: Some(ChunkEncoder::new(chunk_size)),
        }
    }

    pub(super) async fn run(mut self) -> Result<(), ChunkAssemblerError> {
        while let Some(game) = self.completed_games_rx.recv().await {
            self.append_game(game).await?;
        }
        self.flush().await
    }

    async fn append_game(&mut self, mut game: CompletedGame) -> Result<(), ChunkAssemblerError> {
        loop {
            let active_chunk = self
                .active_chunk
                .take()
                .expect("chunk assembler always owns an active encoder");
            let chunk_size = self.chunk_size;
            let result = tokio::task::spawn_blocking(move || {
                Self::encode_until_chunk(active_chunk, chunk_size, game)
            })
            .await
            .expect("chunk encoding task panicked");

            self.active_chunk = Some(result.active_chunk);
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

        let completed_chunk = (active_chunk.record_count() == chunk_size).then(|| {
            let encoder = std::mem::replace(&mut active_chunk, ChunkEncoder::new(chunk_size));
            TrainingChunk {
                bytes: encoder.finish(),
            }
        });

        EncodedStep {
            active_chunk,
            completed_chunk,
            game,
        }
    }

    async fn send_chunk(&self, chunk: TrainingChunk) -> Result<(), ChunkAssemblerError> {
        self.chunks_tx
            .send(chunk)
            .await
            .map_err(|_| ChunkAssemblerError::OutputClosed)
    }

    async fn flush(&mut self) -> Result<(), ChunkAssemblerError> {
        let active_chunk = self
            .active_chunk
            .take()
            .expect("chunk assembler always owns an active encoder");
        if active_chunk.record_count() == 0 {
            return Ok(());
        }

        let chunk = tokio::task::spawn_blocking(move || TrainingChunk {
            bytes: active_chunk.finish(),
        })
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
        inference::{inputs::NNInput, policy::POLICY_SIZE},
        self_play::training_data::{TrainingSample, ValueTarget, verify_chunk_checksum},
    };

    fn sample() -> TrainingSample {
        TrainingSample {
            input: NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH)),
            policy_target: [0.0; POLICY_SIZE],
            value_target: ValueTarget {
                win_probability: 0.5,
                score_mean: 0.0,
                score_stdev: 1.0,
                ownership: [1; crate::inference::policy::BOARD_POLICY_SIZE],
            },
        }
    }

    fn record_count(chunk: &TrainingChunk) -> u32 {
        u32::from_le_bytes(chunk.bytes[16..20].try_into().unwrap())
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
        assert_eq!(step.active_chunk.record_count(), 0);
        let chunk = step.completed_chunk.expect("one full chunk");
        assert_eq!(record_count(&chunk), 2);
        assert!(verify_chunk_checksum(&chunk.bytes));
    }

    #[tokio::test]
    async fn emits_chunks_and_returns_empty_game_buffers() {
        let (completed_tx, completed_rx) = mpsc::channel(1);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(1);
        let (recycle_tx, recycle_rx) = oneshot::channel();
        let assembler = ChunkAssembler::new(2, completed_rx, chunks_tx);
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
        let assembler = ChunkAssembler::new(2, completed_rx, chunks_tx);
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
        let assembler = ChunkAssembler::new(2, completed_rx, chunks_tx);
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
        while let Some(chunk) = chunks_rx.recv().await {
            assert!(verify_chunk_checksum(&chunk.bytes));
            chunk_sizes.push(record_count(&chunk));
        }
        assert_eq!(chunk_sizes, [2, 2, 1]);
        assert!(recycle_rx.await.unwrap().is_empty());
        assert_eq!(assembler_task.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn writer_failure_cancels_the_unprocessed_game_recycle() {
        let (completed_tx, completed_rx) = mpsc::channel(2);
        let (chunks_tx, chunks_rx) = mpsc::channel(1);
        drop(chunks_rx);
        let assembler = ChunkAssembler::new(1, completed_rx, chunks_tx);

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
        let assembler = ChunkAssembler::new(2, completed_rx, chunks_tx);
        let assembler_task = tokio::spawn(assembler.run());

        tokio::task::yield_now().await;
        assembler_task.abort();
        assert!(assembler_task.await.unwrap_err().is_cancelled());
        assert!(completed_tx.is_closed());
        assert!(chunks_rx.recv().await.is_none());
    }
}
