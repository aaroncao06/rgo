use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc as std_mpsc,
    },
    thread,
    time::Duration,
};

use rgo_artifacts::{CHUNK_FILE_PREFIX, CHUNK_FILE_SUFFIX};
use serde_json::Value;

use super::*;
use crate::{
    game::{game_state::GameState, rules::Rules},
    inference::{inputs::NNInput, policy::MAX_POLICY_SIZE},
    training_data::{ValueTarget, encode_chunk, verify_chunk_checksum},
};

static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);
const TIMEOUT: Duration = Duration::from_secs(5);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        loop {
            let id = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("rgo-chunk-writer-test-{}-{id}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("failed to create test directory: {error}"),
            }
        }
    }

    fn files(&self) -> Vec<PathBuf> {
        fs::read_dir(&self.0)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect()
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn sample(dim: usize, tag: usize) -> TrainingSample {
    let mut rules = Rules::TROMP_TAYLORISH_9;
    rules.board_dim = dim;
    TrainingSample {
        input: NNInput::encode(&GameState::new(rules)),
        policy_target: [half::f16::from_f32(tag as f32); MAX_POLICY_SIZE],
        value_target: ValueTarget {
            win_probability: 0.5,
            score_mean: tag as f32,
            score_stdev: 1.0,
            ownership: [1; crate::game::board::MAX_BOARD_AREA],
        },
    }
}

fn game(samples: Vec<TrainingSample>) -> (CompletedGame, oneshot::Receiver<Vec<TrainingSample>>) {
    let (recycle_tx, recycle_rx) = oneshot::channel();
    (
        CompletedGame {
            samples,
            recycle_tx,
        },
        recycle_rx,
    )
}

// Checks that each announcement names a complete file before acknowledging it.
struct EventOutput {
    line: Vec<u8>,
    events: mpsc::UnboundedSender<Value>,
    block_first_flush: Option<(oneshot::Sender<()>, std_mpsc::Receiver<()>)>,
}

impl Write for EventOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.line.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let event: Value = serde_json::from_slice(&self.line).unwrap();
        assert_eq!(event["type"], "chunk_ready");
        let bytes = fs::read(event["path"].as_str().unwrap())?;
        assert_eq!(event["bytes"].as_u64().unwrap(), bytes.len() as u64);
        assert_eq!(
            event["records"].as_u64().unwrap(),
            u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as u64
        );
        assert!(verify_chunk_checksum(&bytes));
        if let Some((reached, release)) = self.block_first_flush.take() {
            let _ = reached.send(());
            // A failed assertion in the test drops release, so this cannot
            // strand the writer thread indefinitely during test unwinding.
            let _ = release.recv();
        }
        let _ = self.events.send(event);
        self.line.clear();
        Ok(())
    }
}

fn event_output() -> (EventOutput, mpsc::UnboundedReceiver<Value>) {
    let (events, rx) = mpsc::unbounded_channel();
    (
        EventOutput {
            line: Vec::new(),
            events,
            block_first_flush: None,
        },
        rx,
    )
}

struct RunningWriter {
    thread: thread::JoinHandle<()>,
    result: oneshot::Receiver<Result<(), ChunkWriterError>>,
}

impl RunningWriter {
    fn start(writer: ChunkWriter) -> Self {
        let (tx, result) = oneshot::channel();
        let thread = thread::spawn(move || {
            let _ = tx.send(writer.run());
        });
        Self { thread, result }
    }

    async fn finish(self) -> Result<(), ChunkWriterError> {
        let result = tokio::time::timeout(TIMEOUT, self.result)
            .await
            .expect("writer did not finish")
            .expect("writer panicked");
        tokio::task::spawn_blocking(move || self.thread.join().unwrap())
            .await
            .unwrap();
        result
    }
}

async fn next_event(rx: &mut mpsc::UnboundedReceiver<Value>) -> Value {
    tokio::time::timeout(TIMEOUT, rx.recv())
        .await
        .expect("writer did not publish")
        .expect("event output stopped")
}

async fn recycled(rx: oneshot::Receiver<Vec<TrainingSample>>) -> Vec<TrainingSample> {
    tokio::time::timeout(TIMEOUT, rx)
        .await
        .expect("writer did not recycle samples")
        .expect("writer dropped samples")
}

fn assert_chunk(event: &Value, expected: &[u8]) {
    let path = Path::new(event["path"].as_str().unwrap());
    assert_eq!(fs::read(path).unwrap(), expected);
    let name = path.file_name().unwrap().to_str().unwrap();
    let id = ChunkId::from_str_radix(
        name.strip_prefix(CHUNK_FILE_PREFIX)
            .unwrap()
            .strip_suffix(CHUNK_FILE_SUFFIX)
            .unwrap(),
        16,
    )
    .unwrap();
    assert_eq!(path, chunk_path(path.parent().unwrap(), id));
}

#[test]
fn requires_an_existing_directory() {
    let dir = TestDir::new();
    let missing = dir.0.join("missing");
    let file = dir.0.join("file");
    fs::write(&file, []).unwrap();
    for (path, kind) in [
        (missing.clone(), io::ErrorKind::NotFound),
        (file, io::ErrorKind::NotADirectory),
    ] {
        let (tx, rx) = mpsc::channel(1);
        drop(tx);
        let writer = ChunkWriter::new(ChunkMode::PerGame, path, rx, EventPublisher::discard());
        let Err(ChunkWriterError::Io(error)) = writer.run() else {
            panic!("invalid output directory must fail startup");
        };
        assert_eq!(error.kind(), kind);
    }
    assert!(!missing.exists());
}

#[test]
fn empty_games_and_an_empty_queue_do_not_publish_chunks() {
    for mode in [ChunkMode::PerGame, ChunkMode::FixedRecords(2)] {
        for submit_empty_game in [false, true] {
            let dir = TestDir::new();
            let (tx, rx) = mpsc::channel(1);
            let returned = if submit_empty_game {
                let (game, returned) = game(Vec::new());
                tx.try_send(game).unwrap();
                Some(returned)
            } else {
                None
            };
            drop(tx);
            ChunkWriter::new(mode, dir.0.clone(), rx, EventPublisher::discard())
                .run()
                .unwrap();
            if let Some(returned) = returned {
                assert!(returned.blocking_recv().unwrap().is_empty());
            }
            assert!(dir.files().is_empty());
        }
    }
}

#[tokio::test]
async fn per_game_publishes_without_waiting_for_queue_closure_or_splitting() {
    let dir = TestDir::new();
    let (tx, rx) = mpsc::channel(1);
    let (output, mut events) = event_output();
    let writer = RunningWriter::start(ChunkWriter::new(
        ChunkMode::PerGame,
        dir.0.clone(),
        rx,
        EventPublisher::with_output(output),
    ));
    for (dim, length) in [(9, 1), (13, 3), (19, 2)] {
        let samples: Vec<_> = (0..length).map(|tag| sample(dim, tag)).collect();
        let expected = encode_chunk(&samples);
        let allocation = samples.as_ptr();
        let (game, returned) = game(samples);
        tx.send(game).await.unwrap();
        let samples = recycled(returned).await;
        assert!(samples.is_empty());
        assert_eq!(samples.as_ptr(), allocation);
        assert_chunk(&next_event(&mut events).await, &expected);
    }
    // An empty game must not republish the previous game's finished encoder.
    let (empty, returned) = game(Vec::new());
    tx.send(empty).await.unwrap();
    assert!(recycled(returned).await.is_empty());
    drop(tx);
    writer.finish().await.unwrap();
    assert_eq!(dir.files().len(), 3);
    assert!(!dir.0.join(PENDING_CHUNK_FILE).exists());
}

#[tokio::test]
async fn fixed_chunks_combine_games_and_flush_a_partial_chunk() {
    let dir = TestDir::new();
    let (tx, rx) = mpsc::channel(1);
    let (output, mut events) = event_output();
    let writer = RunningWriter::start(ChunkWriter::new(
        ChunkMode::FixedRecords(3),
        dir.0.clone(),
        rx,
        EventPublisher::with_output(output),
    ));
    let mut expected_samples = Vec::new();
    for (dim, tag) in [(9, 1), (13, 2)] {
        let (game, returned) = game(vec![sample(dim, tag)]);
        tx.send(game).await.unwrap();
        assert!(recycled(returned).await.is_empty());
        expected_samples.push(sample(dim, tag));
    }
    assert!(events.try_recv().is_err());
    // The next game fills the old chunk from its tail, leaving one sample for
    // the final partial chunk. Neither dimension nor game boundaries split it.
    let (game, returned) = game(vec![sample(19, 3), sample(19, 4)]);
    tx.send(game).await.unwrap();
    assert!(recycled(returned).await.is_empty());
    expected_samples.push(sample(19, 4));
    assert_chunk(
        &next_event(&mut events).await,
        &encode_chunk(&expected_samples),
    );
    assert!(events.try_recv().is_err());
    drop(tx);
    writer.finish().await.unwrap();
    assert_chunk(
        &next_event(&mut events).await,
        &encode_chunk(&[sample(19, 3)]),
    );
    assert_eq!(dir.files().len(), 2);
}

#[tokio::test]
async fn fixed_chunks_split_a_game_preserving_existing_tail_order() {
    let dir = TestDir::new();
    let (tx, rx) = mpsc::channel(1);
    let (output, mut events) = event_output();
    let writer = RunningWriter::start(ChunkWriter::new(
        ChunkMode::FixedRecords(2),
        dir.0.clone(),
        rx,
        EventPublisher::with_output(output),
    ));
    let samples: Vec<_> = (0..5).map(|tag| sample(9, tag)).collect();
    let allocation = samples.as_ptr();
    let (game, returned) = game(samples);
    tx.send(game).await.unwrap();
    let samples = recycled(returned).await;
    assert!(samples.is_empty());
    assert_eq!(samples.as_ptr(), allocation);
    for tags in [[3, 4], [1, 2]] {
        let expected: Vec<_> = tags.map(|tag| sample(9, tag)).into();
        assert_chunk(&next_event(&mut events).await, &encode_chunk(&expected));
    }
    drop(tx);
    writer.finish().await.unwrap();
    assert_chunk(
        &next_event(&mut events).await,
        &encode_chunk(&[sample(9, 0)]),
    );
    assert_eq!(dir.files().len(), 3);
}

#[test]
fn repeated_games_reuse_the_encoder() {
    for mode in [ChunkMode::PerGame, ChunkMode::FixedRecords(3)] {
        let dir = TestDir::new();
        let (_tx, rx) = mpsc::channel(1);
        let (output, mut events) = event_output();
        let mut writer =
            ChunkWriter::new(mode, dir.0.clone(), rx, EventPublisher::with_output(output));
        for _ in 0..4 {
            let (game, returned) = game((0..3).map(|tag| sample(19, tag)).collect());
            match mode {
                ChunkMode::PerGame => writer.append_game_per_game(game).unwrap(),
                ChunkMode::FixedRecords(records) => {
                    writer.append_game_fixed(game, records).unwrap()
                }
            }
            assert!(returned.blocking_recv().unwrap().is_empty());
            let expected: Vec<_> = (0..3).map(|tag| sample(19, tag)).collect();
            assert_chunk(&events.try_recv().unwrap(), &encode_chunk(&expected));
        }
        assert_eq!(dir.files().len(), 4);
    }
}

#[tokio::test]
async fn restarting_after_deletion_does_not_reuse_chunk_ids() {
    let dir = TestDir::new();
    let mut paths = Vec::new();
    for _ in 0..2 {
        let (tx, rx) = mpsc::channel(1);
        let (output, mut events) = event_output();
        let writer = RunningWriter::start(ChunkWriter::new(
            ChunkMode::PerGame,
            dir.0.clone(),
            rx,
            EventPublisher::with_output(output),
        ));
        let (game, returned) = game(vec![sample(9, 0)]);
        tx.send(game).await.unwrap();
        drop(tx);
        recycled(returned).await;
        writer.finish().await.unwrap();
        let event = next_event(&mut events).await;
        assert_chunk(&event, &encode_chunk(&[sample(9, 0)]));
        let path = PathBuf::from(event["path"].as_str().unwrap());
        fs::remove_file(&path).unwrap();
        paths.push(path);
    }
    assert_ne!(paths[0], paths[1]);
}

#[test]
fn file_failure_recycles_consumed_game_and_cancels_queued_games() {
    for mode in [ChunkMode::PerGame, ChunkMode::FixedRecords(1)] {
        let dir = TestDir::new();
        fs::create_dir(dir.0.join(PENDING_CHUNK_FILE)).unwrap();
        let (tx, rx) = mpsc::channel(2);
        let (output, mut events) = event_output();
        let (first, returned_first) = game(vec![sample(9, 0)]);
        let (second, returned_second) = game(vec![sample(9, 1)]);
        tx.try_send(first).unwrap();
        tx.try_send(second).unwrap();
        drop(tx);
        let result =
            ChunkWriter::new(mode, dir.0.clone(), rx, EventPublisher::with_output(output)).run();
        assert!(matches!(result, Err(ChunkWriterError::Io(_))));
        assert!(returned_first.blocking_recv().unwrap().is_empty());
        assert!(returned_second.blocking_recv().is_err());
        assert!(events.try_recv().is_err());
        assert_eq!(dir.files(), [dir.0.join(PENDING_CHUNK_FILE)]);
    }
}

struct FailingOutput {
    fail_on_flush: bool,
}

impl Write for FailingOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.fail_on_flush {
            Ok(bytes.len())
        } else {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "test pipe closed",
            ))
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "test pipe closed",
        ))
    }
}

#[test]
fn notification_failure_is_reported_after_file_publication() {
    for fail_on_flush in [false, true] {
        let dir = TestDir::new();
        let (tx, rx) = mpsc::channel(2);
        let (first, returned_first) = game(vec![sample(9, 0)]);
        let (second, returned_second) = game(vec![sample(9, 1)]);
        tx.try_send(first).unwrap();
        tx.try_send(second).unwrap();
        drop(tx);
        let result = ChunkWriter::new(
            ChunkMode::PerGame,
            dir.0.clone(),
            rx,
            EventPublisher::with_output(FailingOutput { fail_on_flush }),
        )
        .run();
        let Err(ChunkWriterError::Io(error)) = result else {
            panic!("failed event write must fail the writer");
        };
        // Serialization wraps write errors; a flush error is returned directly.
        assert_eq!(
            error.kind(),
            if fail_on_flush {
                io::ErrorKind::BrokenPipe
            } else {
                io::ErrorKind::Other
            }
        );
        assert!(error.to_string().contains("test pipe closed"));
        assert!(returned_first.blocking_recv().unwrap().is_empty());
        assert!(returned_second.blocking_recv().is_err());
        let files = dir.files();
        assert_eq!(files.len(), 1);
        assert_eq!(fs::read(&files[0]).unwrap(), encode_chunk(&[sample(9, 0)]));
    }
}

#[tokio::test]
async fn blocked_publication_recycles_samples_but_backpressures_the_next_game() {
    for mode in [ChunkMode::PerGame, ChunkMode::FixedRecords(1)] {
        let dir = TestDir::new();
        let (tx, rx) = mpsc::channel(1);
        let (mut output, mut events) = event_output();
        let (reached_tx, reached_rx) = oneshot::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        output.block_first_flush = Some((reached_tx, release_rx));
        let writer = RunningWriter::start(ChunkWriter::new(
            mode,
            dir.0.clone(),
            rx,
            EventPublisher::with_output(output),
        ));
        let (first, returned_first) = game(vec![sample(9, 0)]);
        tx.send(first).await.unwrap();
        tokio::time::timeout(TIMEOUT, reached_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(recycled(returned_first).await.is_empty());
        let (second, mut returned_second) = game(vec![sample(9, 1)]);
        tx.send(second).await.unwrap();
        drop(tx);
        // The async runtime still handles timers while publication is blocked.
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(matches!(
            returned_second.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert!(events.try_recv().is_err());
        assert_eq!(dir.files().len(), 1);
        release_tx.send(()).unwrap();
        assert_chunk(
            &next_event(&mut events).await,
            &encode_chunk(&[sample(9, 0)]),
        );
        recycled(returned_second).await;
        writer.finish().await.unwrap();
        assert_chunk(
            &next_event(&mut events).await,
            &encode_chunk(&[sample(9, 1)]),
        );
    }
}

#[tokio::test]
async fn a_split_game_keeps_its_sample_vector_until_all_records_are_encoded() {
    let dir = TestDir::new();
    let (tx, rx) = mpsc::channel(1);
    let (mut output, mut events) = event_output();
    let (reached_tx, reached_rx) = oneshot::channel();
    let (release_tx, release_rx) = std_mpsc::channel();
    output.block_first_flush = Some((reached_tx, release_rx));
    let writer = RunningWriter::start(ChunkWriter::new(
        ChunkMode::FixedRecords(2),
        dir.0.clone(),
        rx,
        EventPublisher::with_output(output),
    ));
    let (game, mut returned) = game((0..5).map(|tag| sample(9, tag)).collect());
    tx.send(game).await.unwrap();
    drop(tx);
    tokio::time::timeout(TIMEOUT, reached_rx)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        returned.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    assert_eq!(dir.files().len(), 1);
    release_tx.send(()).unwrap();
    assert!(recycled(returned).await.is_empty());
    writer.finish().await.unwrap();
    for tags in [vec![3, 4], vec![1, 2], vec![0]] {
        let samples: Vec<_> = tags.into_iter().map(|tag| sample(9, tag)).collect();
        assert_chunk(&next_event(&mut events).await, &encode_chunk(&samples));
    }
}

#[test]
fn final_partial_chunk_publication_errors_are_propagated() {
    let dir = TestDir::new();
    fs::create_dir(dir.0.join(PENDING_CHUNK_FILE)).unwrap();
    let (tx, rx) = mpsc::channel(1);
    let (game, returned) = game(vec![sample(9, 0)]);
    tx.try_send(game).unwrap();
    drop(tx);
    let result = ChunkWriter::new(
        ChunkMode::FixedRecords(3),
        dir.0.clone(),
        rx,
        EventPublisher::discard(),
    )
    .run();
    assert!(matches!(result, Err(ChunkWriterError::Io(_))));
    assert!(returned.blocking_recv().unwrap().is_empty());
    assert_eq!(dir.files(), [dir.0.join(PENDING_CHUNK_FILE)]);
}
