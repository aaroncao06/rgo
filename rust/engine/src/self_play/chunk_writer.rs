use tokio::sync::{mpsc, oneshot};

use super::training_data::TrainingSample;

pub(super) struct CompletedGame {
    pub(super) samples: Vec<TrainingSample>,
    pub(super) recycle_tx: oneshot::Sender<Vec<TrainingSample>>, // channel to send back the new training sample vector
}

pub(super) struct TrainingChunk {
    pub(super) samples: Vec<TrainingSample>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChunkWriterError {
    OutputClosed,
}

/// Collects completed games into training chunks.
///
/// Full chunks contain exactly `chunk_size` samples; only the final chunk emitted
/// during shutdown may be smaller. A game may span multiple chunks.
///
/// This task is the sole owner of the active chunk. Workers transfer game
/// buffers through `completed_games_rx`; emptied buffers are returned to their
/// originating workers for reuse.
pub(super) struct ChunkWriter {
    chunk_size: usize,
    completed_games_rx: mpsc::Receiver<CompletedGame>,
    chunks_tx: mpsc::Sender<TrainingChunk>,
    active_samples: Vec<TrainingSample>,
}

impl ChunkWriter {
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
            active_samples: Vec::with_capacity(chunk_size),
        }
    }

    pub(super) async fn run(mut self) -> Result<(), ChunkWriterError> {
        while let Some(game) = self.completed_games_rx.recv().await {
            self.append_game(game).await?;
        }
        self.flush().await
    }

    async fn append_game(&mut self, mut game: CompletedGame) -> Result<(), ChunkWriterError> {
        // Training records are sampled independently, so their order within a
        // chunk is irrelevant. `pop` avoids shifting the remaining samples.
        while !game.samples.is_empty() {
            let available = self.chunk_size - self.active_samples.len();
            let count = available.min(game.samples.len());
            for _ in 0..count {
                self.active_samples
                    .push(game.samples.pop().expect("game still has samples"));
            }

            if game.samples.is_empty() {
                // Return the allocation before a potentially blocking output
                // send so the worker can begin its next game immediately.
                let _ = game.recycle_tx.send(game.samples);
                if self.active_samples.len() == self.chunk_size {
                    self.flush().await?;
                }
                return Ok(());
            }

            debug_assert_eq!(self.active_samples.len(), self.chunk_size);
            self.flush().await?;
        }

        // A dropped worker no longer needs its allocation, so failure to return
        // this empty buffer is harmless.
        let _ = game.recycle_tx.send(game.samples);
        Ok(())
    }

    async fn flush(&mut self) -> Result<(), ChunkWriterError> {
        if self.active_samples.is_empty() {
            return Ok(());
        }

        // allocate new vec descriptor and take the active samples
        let samples = std::mem::replace(
            &mut self.active_samples,
            Vec::with_capacity(self.chunk_size),
        );
        self.chunks_tx
            .send(TrainingChunk { samples })
            .await
            .map_err(|_| ChunkWriterError::OutputClosed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        game::{game_state::GameState, rules::Rules},
        inference::{inputs::NNInput, policy::POLICY_SIZE},
        self_play::training_data::{TrainingSample, ValueTarget},
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

    #[tokio::test]
    async fn emits_chunks_and_returns_empty_game_buffers() {
        let (completed_tx, completed_rx) = mpsc::channel(1);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(1);
        let (recycle_tx, recycle_rx) = oneshot::channel();
        let writer = ChunkWriter::new(2, completed_rx, chunks_tx);
        let writer_task = tokio::spawn(writer.run());
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
        assert_eq!(chunk.samples.len(), 2);
        let recycled = recycle_rx.await.unwrap();
        assert!(recycled.is_empty());
        assert_eq!(recycled.as_ptr(), allocation);

        drop(completed_tx);
        assert_eq!(writer_task.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn combines_complete_games_in_one_chunk() {
        let (completed_tx, completed_rx) = mpsc::channel(2);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(1);
        let writer = ChunkWriter::new(2, completed_rx, chunks_tx);
        let writer_task = tokio::spawn(writer.run());
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
        assert_eq!(chunk.samples.len(), 2);
        for receiver in recycle_receivers {
            assert!(receiver.await.unwrap().is_empty());
        }
        assert_eq!(writer_task.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn splits_large_games_into_fixed_chunks_and_a_final_partial_chunk() {
        let (completed_tx, completed_rx) = mpsc::channel(1);
        let (chunks_tx, mut chunks_rx) = mpsc::channel(3);
        let (recycle_tx, recycle_rx) = oneshot::channel();
        let writer = ChunkWriter::new(2, completed_rx, chunks_tx);
        let writer_task = tokio::spawn(writer.run());

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
            chunk_sizes.push(chunk.samples.len());
        }
        assert_eq!(chunk_sizes, [2, 2, 1]);
        assert!(recycle_rx.await.unwrap().is_empty());
        assert_eq!(writer_task.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn writer_failure_cancels_the_unprocessed_game_recycle() {
        let (completed_tx, completed_rx) = mpsc::channel(2);
        let (chunks_tx, chunks_rx) = mpsc::channel(1);
        drop(chunks_rx);
        let writer = ChunkWriter::new(1, completed_rx, chunks_tx);

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
        let writer_task = tokio::spawn(writer.run());

        assert_eq!(
            writer_task.await.unwrap(),
            Err(ChunkWriterError::OutputClosed)
        );
        assert!(first_recycle_rx.await.unwrap().is_empty());
        assert!(second_recycle_rx.await.is_err());
    }
}
