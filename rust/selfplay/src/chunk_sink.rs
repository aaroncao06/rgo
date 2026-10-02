use std::{
    io,
    path::{Path, PathBuf},
};

use rand::{TryRng, rngs::SysError, rngs::SysRng};
use rgo_artifacts::{ChunkId, chunk_path};
use tokio::{
    fs::{self, File},
    io::AsyncWriteExt,
    sync::mpsc,
};

use super::{
    chunk_assembler::TrainingChunk,
    control::{Event, EventPublisher},
};

const PENDING_CHUNK_FILE: &str = ".pending-chunk.tmp";

#[derive(Debug)]
pub(super) enum FileChunkSinkError {
    Io(io::Error),
    RandomnessUnavailable(SysError),
}

impl From<io::Error> for FileChunkSinkError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Persists assembled chunks as immutable files in one local directory.
///
/// The output directory must already exist and have been durably provisioned
/// by the caller.
///
/// Exactly one `FileChunkSink` may write to a given output directory at a
/// time. Chunk filenames use random 128-bit IDs drawn from the operating
/// system, so deleting acknowledged chunks or restarting the sink cannot cause
/// IDs to be reused through counter recovery. The serialized sink reuses one
/// temporary path, bounding abandoned staging data to at most one chunk.
///
/// This is intentionally separate from chunk assembly. The bounded chunk
/// channel controls memory use while filesystem I/O is in progress.
pub(super) struct FileChunkSink {
    output_dir: PathBuf,
    chunks_rx: mpsc::Receiver<TrainingChunk>,
    events: EventPublisher,
}

impl FileChunkSink {
    pub(super) fn new(
        output_dir: PathBuf,
        chunks_rx: mpsc::Receiver<TrainingChunk>,
        events: EventPublisher,
    ) -> Self {
        Self {
            output_dir,
            chunks_rx,
            events,
        }
    }

    pub(super) async fn run(mut self) -> Result<(), FileChunkSinkError> {
        let metadata = fs::metadata(&self.output_dir).await?;
        if !metadata.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                "chunk output path is not a directory",
            )
            .into());
        }

        while let Some(chunk) = self.chunks_rx.recv().await {
            let chunk_id = random_chunk_id().map_err(FileChunkSinkError::RandomnessUnavailable)?;
            self.write_chunk(chunk_id, &chunk).await?;
            let mut bytes = chunk.bytes;
            bytes.clear();
            let _ = chunk.recycle_tx.send(bytes);
        }
        Ok(())
    }

    async fn write_chunk(
        &self,
        chunk_id: ChunkId,
        chunk: &TrainingChunk,
    ) -> Result<(), FileChunkSinkError> {
        let final_path = chunk_path(&self.output_dir, chunk_id);
        let temporary_path = self.output_dir.join(PENDING_CHUNK_FILE);

        let mut file = File::create(&temporary_path).await?;
        file.write_all(&chunk.bytes).await?;
        file.sync_all().await?;
        drop(file);
        fs::rename(&temporary_path, &final_path).await?;
        sync_directory(&self.output_dir).await?;
        self.events
            .emit(Event::ChunkReady {
                path: final_path,
                bytes: chunk.bytes.len(),
                records: chunk.records,
            })
            .await?;
        Ok(())
    }
}

#[cfg(unix)]
async fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path).await?.sync_all().await
}

#[cfg(not(unix))]
async fn sync_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

fn random_chunk_id() -> Result<ChunkId, SysError> {
    let mut bytes = [0; size_of::<ChunkId>()];
    SysRng.try_fill_bytes(&mut bytes)?;
    Ok(ChunkId::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use rgo_artifacts::{CHUNK_FILE_PREFIX, CHUNK_FILE_SUFFIX};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::{
        chunk_assembler::{ChunkAssembler, ChunkMode, CompletedGame},
        game::{game_state::GameState, rules::Rules},
        inference::{inputs::NNInput, policy::MAX_POLICY_SIZE},
        training_data::{
            CHUNK_CHECKSUM_SIZE, CHUNK_FORMAT_VERSION, CHUNK_HEADER_SIZE, CHUNK_MAGIC,
            TrainingSample, ValueTarget, encode_chunk, training_record_size, verify_chunk_checksum,
        },
    };
    use tokio::sync::oneshot;

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            loop {
                let id = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir()
                    .join(format!("rgo-file-writer-test-{}-{id}", std::process::id()));
                match std::fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("failed to create test directory: {error}"),
                }
            }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sample() -> TrainingSample {
        TrainingSample {
            input: NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH_9)),
            policy_target: [0.0; MAX_POLICY_SIZE],
            value_target: ValueTarget {
                win_probability: 0.5,
                score_mean: 0.0,
                score_stdev: 1.0,
                ownership: [1; crate::game::board::MAX_BOARD_POINTS],
            },
        }
    }

    #[tokio::test]
    async fn requires_a_preexisting_output_directory() {
        let parent = TestDir::new();
        let missing = parent.0.join("missing");
        let (chunks_tx, chunks_rx) = mpsc::channel(1);
        let sink = FileChunkSink::new(missing, chunks_rx, EventPublisher::discard());
        drop(chunks_tx);

        let Err(FileChunkSinkError::Io(error)) = sink.run().await else {
            panic!("missing output directory should fail sink startup");
        };
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn restart_after_deletion_does_not_reuse_chunk_id() {
        let output_dir = TestDir::new();

        let write_one_chunk = || {
            let output_dir = output_dir.0.clone();
            async move {
                let (chunks_tx, chunks_rx) = mpsc::channel(1);
                let sink = FileChunkSink::new(output_dir, chunks_rx, EventPublisher::discard());
                let sink_task = tokio::spawn(sink.run());
                let (recycle_tx, recycle_rx) = tokio::sync::oneshot::channel();
                let bytes = encode_chunk(&[sample()]);
                let allocation = bytes.as_ptr();
                chunks_tx
                    .send(TrainingChunk {
                        bytes,
                        records: 1,
                        recycle_tx,
                    })
                    .await
                    .unwrap();
                drop(chunks_tx);
                assert!(sink_task.await.unwrap().is_ok());
                let returned = recycle_rx.await.unwrap();
                assert!(returned.is_empty());
                assert_eq!(returned.as_ptr(), allocation);
            }
        };

        write_one_chunk().await;
        let mut entries = fs::read_dir(&output_dir.0).await.unwrap();
        let first_entry = entries.next_entry().await.unwrap().unwrap();
        assert!(entries.next_entry().await.unwrap().is_none());
        let first_name = first_entry.file_name();
        fs::remove_file(first_entry.path()).await.unwrap();

        write_one_chunk().await;
        let mut entries = fs::read_dir(&output_dir.0).await.unwrap();
        let second_entry = entries.next_entry().await.unwrap().unwrap();
        assert!(entries.next_entry().await.unwrap().is_none());
        let second_name = second_entry.file_name();
        assert_ne!(first_name, second_name);

        let name = second_name.to_string_lossy();
        let id = name
            .strip_prefix(CHUNK_FILE_PREFIX)
            .and_then(|name| name.strip_suffix(CHUNK_FILE_SUFFIX))
            .and_then(|id| ChunkId::from_str_radix(id, 16).ok())
            .expect("chunk filename contains a 128-bit hexadecimal ID");
        assert_eq!(chunk_path(&output_dir.0, id), second_entry.path());

        let bytes = fs::read(second_entry.path()).await.unwrap();
        assert_eq!(&bytes[..8], &CHUNK_MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            CHUNK_FORMAT_VERSION
        );
        assert_eq!(
            bytes.len(),
            CHUNK_HEADER_SIZE + training_record_size(9) + CHUNK_CHECKSUM_SIZE
        );
        assert!(verify_chunk_checksum(&bytes));
    }

    #[tokio::test]
    async fn assembled_chunks_flow_to_local_files() {
        let output_dir = TestDir::new();
        let (completed_games_tx, completed_games_rx) = mpsc::channel(1);
        let (chunks_tx, chunks_rx) = mpsc::channel(1);
        let assembler =
            ChunkAssembler::new(ChunkMode::FixedRecords(2), completed_games_rx, chunks_tx);
        let sink = FileChunkSink::new(output_dir.0.clone(), chunks_rx, EventPublisher::discard());
        let assembler_task = tokio::spawn(assembler.run());
        let sink_task = tokio::spawn(sink.run());
        let (recycle_tx, recycle_rx) = oneshot::channel();

        completed_games_tx
            .send(CompletedGame {
                samples: (0..5).map(|_| sample()).collect(),
                recycle_tx,
            })
            .await
            .unwrap();
        drop(completed_games_tx);

        assert!(recycle_rx.await.unwrap().is_empty());
        assert!(assembler_task.await.unwrap().is_ok());
        assert!(sink_task.await.unwrap().is_ok());

        let mut entries = fs::read_dir(&output_dir.0).await.unwrap();
        let mut record_counts = Vec::new();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let bytes = fs::read(entry.path()).await.unwrap();
            record_counts.push(u32::from_le_bytes(bytes[16..20].try_into().unwrap()));
        }
        record_counts.sort_unstable();
        assert_eq!(record_counts, [1, 2, 2]);
    }
}
