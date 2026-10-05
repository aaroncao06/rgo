use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc as std_mpsc,
    },
    thread,
    time::Duration,
};

use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rgo_artifacts::chunk::verify_chunk_checksum;
use rgo_artifacts::{CHUNK_FILE_PREFIX, CHUNK_FILE_SUFFIX};
use serde_json::Value;
use tokio::sync::oneshot;

use super::*;
use crate::{
    game::board::Loc,
    game::{game_state::GameState, rules::Rules},
    inference::{inputs::NNInput, policy::MAX_POLICY_SIZE},
    training_data::{
        SelfPlayRecord,
        test_support::{TestSample, ValueTarget, encode_chunk},
    },
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

// Build real completed games and independent snapshots of each pre-move input.
fn game(dim: usize, tags: &[usize]) -> (CompletedGame, Vec<TestSample>) {
    assert!(tags.len() >= 2);
    let rules = Rules {
        board_dim: dim,
        ..Rules::TROMP_TAYLORISH_9
    };
    let state = GameState::new(rules);
    let moves: Vec<_> = (0..tags.len())
        .map(|index| {
            if index + 2 >= tags.len() {
                Loc::PASS
            } else {
                state.board().loc(index, 0).unwrap()
            }
        })
        .collect();
    game_with_moves(rules, &moves, tags)
}

fn game_with_moves(
    rules: Rules,
    moves: &[Loc],
    tags: &[usize],
) -> (CompletedGame, Vec<TestSample>) {
    assert_eq!(moves.len(), tags.len());
    let mut state = GameState::new(rules);
    let mut records = Vec::new();
    let mut samples = Vec::new();
    for (&selected_move, &tag) in moves.iter().zip(tags) {
        let policy_target = [half::f16::from_f32(tag as f32); MAX_POLICY_SIZE];
        samples.push(TestSample {
            input: NNInput::encode(&state),
            policy_target,
            value_target: ValueTarget {
                win_target: 0.5,
                final_score: 0.0,
                ownership: [1; crate::game::board::MAX_BOARD_AREA],
            },
        });
        records.push(SelfPlayRecord {
            player: state.next_player(),
            selected_move,
            policy_target,
        });
        assert!(state.play(selected_move));
    }
    let white_score = state.final_score_white_minus_black();
    let game = CompletedGame::new(&state, records);
    for (sample, record) in samples.iter_mut().zip(&game.records) {
        let score = if record.player == Player::White {
            white_score
        } else {
            -white_score
        };
        sample.value_target.final_score = score;
        sample.value_target.win_target = if score > 0.0 {
            1.0
        } else if score < 0.0 {
            0.0
        } else {
            0.5
        };
        // Derive expected labels from the finished board independently of the
        // writer's perspective-conversion helper.
        for loc in state.board().locs() {
            let color = game.final_ownership[loc.index()];
            let own_color = match record.player {
                Player::Black => Color::Black,
                Player::White => Color::White,
            };
            sample.value_target.ownership[loc_to_spatial(state.board(), loc)] =
                if color == Color::Empty {
                    1
                } else if color == own_color {
                    2
                } else {
                    0
                };
        }
    }
    (game, samples)
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
fn an_empty_queue_does_not_publish_chunks() {
    for mode in [ChunkMode::PerGame, ChunkMode::FixedRecords(2)] {
        let dir = TestDir::new();
        let (tx, rx) = mpsc::channel(1);
        drop(tx);
        ChunkWriter::new(mode, dir.0.clone(), rx, EventPublisher::discard())
            .run()
            .unwrap();
        assert!(dir.files().is_empty());
    }
}

#[tokio::test]
async fn per_game_publishes_small_games_without_waiting_for_queue_closure() {
    let dir = TestDir::new();
    let (tx, rx) = mpsc::channel(1);
    let (output, mut events) = event_output();
    let writer = RunningWriter::start(ChunkWriter::new(
        ChunkMode::PerGame,
        dir.0.clone(),
        rx,
        EventPublisher::with_output(output),
    ));
    for (dim, tags) in [(9, vec![0, 1]), (13, vec![0, 1, 2]), (19, vec![0, 1, 2, 3])] {
        let (game, samples) = game(dim, &tags);
        let expected = encode_chunk(&samples);
        tx.send(game).await.unwrap();
        assert_chunk(&next_event(&mut events).await, &expected);
    }
    drop(tx);
    writer.finish().await.unwrap();
    assert_eq!(dir.files().len(), 3);
    assert!(!dir.0.join(PENDING_CHUNK_FILE).exists());
}

#[tokio::test]
async fn per_game_splits_at_the_limit_and_carries_tails_until_next_game_or_shutdown() {
    let rules = Rules {
        board_dim: 19,
        ..Rules::TROMP_TAYLORISH_9
    };
    let mut state = GameState::new(rules);
    let mut rng = SmallRng::seed_from_u64(7);
    let mut moves = Vec::new();
    // Generate legal captures and passes so the fixture exceeds the board area.
    // Each tested prefix is completed with two passes below.
    for _ in 0..2 * MAX_CHUNK_RECORDS + 1 {
        let legal: Vec<_> = state
            .board()
            .locs()
            .filter(|&loc| state.is_legal(loc))
            .collect();
        let selected_move = if legal.is_empty() {
            Loc::PASS
        } else {
            legal[rng.random_range(0..legal.len())]
        };
        assert!(state.play(selected_move));
        assert!(!state.is_finished(), "long-game fixture ended early");
        moves.push(selected_move);
    }

    let dir = TestDir::new();
    let (tx, rx) = mpsc::channel(1);
    let (output, mut events) = event_output();
    let writer = RunningWriter::start(ChunkWriter::new(
        ChunkMode::PerGame,
        dir.0.clone(),
        rx,
        EventPublisher::with_output(output),
    ));
    let mut expected_files = 0;
    let mut pending_samples = Vec::new();
    for (dim, records) in [
        (19, MAX_CHUNK_RECORDS - 1),
        (19, MAX_CHUNK_RECORDS),
        (19, MAX_CHUNK_RECORDS + 2),
        (19, 2 * MAX_CHUNK_RECORDS),
        (13, 3),
        (19, MAX_CHUNK_RECORDS + 2),
    ] {
        let tags: Vec<_> = (0..records).collect();
        let (game, samples) = if dim == 19 {
            let mut game_moves = moves[..records - 2].to_vec();
            game_moves.extend([Loc::PASS, Loc::PASS]);
            game_with_moves(rules, &game_moves, &tags)
        } else {
            game(dim, &tags)
        };
        let capacity = (pending_samples.len() + records).min(MAX_CHUNK_RECORDS);
        pending_samples.extend(samples);
        tx.send(game).await.unwrap();
        let mut published = 0;
        while pending_samples.len() >= capacity {
            assert_chunk(
                &next_event(&mut events).await,
                &encode_chunk(&pending_samples[..capacity]),
            );
            pending_samples.drain(..capacity);
            expected_files += 1;
            published += 1;
        }
        assert!(published > 0, "every completed game must publish a chunk");
        assert!(events.try_recv().is_err());
    }
    assert_eq!(pending_samples.len(), 2);
    drop(tx);
    writer.finish().await.unwrap();
    assert_chunk(
        &next_event(&mut events).await,
        &encode_chunk(&pending_samples),
    );
    expected_files += 1;
    assert!(events.try_recv().is_err());
    assert_eq!(dir.files().len(), expected_files);
    assert!(!dir.0.join(PENDING_CHUNK_FILE).exists());
}

#[tokio::test]
async fn fixed_chunks_combine_games_and_flush_a_partial_chunk() {
    let dir = TestDir::new();
    let (tx, rx) = mpsc::channel(1);
    let (output, mut events) = event_output();
    let writer = RunningWriter::start(ChunkWriter::new(
        ChunkMode::FixedRecords(5),
        dir.0.clone(),
        rx,
        EventPublisher::with_output(output),
    ));
    let mut expected_samples = Vec::new();
    for (dim, tags) in [(9, [0, 1]), (13, [2, 3])] {
        let (game, samples) = game(dim, &tags);
        tx.send(game).await.unwrap();
        expected_samples.extend(samples);
    }
    assert!(events.try_recv().is_err());
    // The next game's first move fills the old chunk; its remaining moves
    // become the final partial chunk, preserving chronological replay order.
    let (game, samples) = game(19, &[4, 5, 6]);
    tx.send(game).await.unwrap();
    expected_samples.extend(samples);
    assert_chunk(
        &next_event(&mut events).await,
        &encode_chunk(&expected_samples[..5]),
    );
    assert!(events.try_recv().is_err());
    drop(tx);
    writer.finish().await.unwrap();
    assert_chunk(
        &next_event(&mut events).await,
        &encode_chunk(&expected_samples[5..]),
    );
    assert_eq!(dir.files().len(), 2);
}

#[tokio::test]
async fn fixed_chunks_split_a_game_in_chronological_order() {
    let dir = TestDir::new();
    let (tx, rx) = mpsc::channel(1);
    let (output, mut events) = event_output();
    let writer = RunningWriter::start(ChunkWriter::new(
        ChunkMode::FixedRecords(2),
        dir.0.clone(),
        rx,
        EventPublisher::with_output(output),
    ));
    let (game, samples) = game(9, &[0, 1, 2, 3, 4]);
    tx.send(game).await.unwrap();
    for chunk in samples[..4].chunks(2) {
        assert_chunk(&next_event(&mut events).await, &encode_chunk(chunk));
    }
    drop(tx);
    writer.finish().await.unwrap();
    assert_chunk(&next_event(&mut events).await, &encode_chunk(&samples[4..]));
    assert_eq!(dir.files().len(), 3);
}

#[test]
fn replay_encodes_ko_history_passes_and_rule_features_for_all_board_dims() {
    for board_dim in [9, 13, 19] {
        let rules = Rules {
            board_dim,
            komi: board_dim as f32 + 0.5,
            multi_stone_suicide_legal: board_dim != 13,
        };
        let state = GameState::new(rules);
        let moves: Vec<_> = [
            (4, 3),
            (4, 4),
            (3, 4),
            (4, 6),
            (5, 4),
            (3, 5),
            (0, 0),
            (5, 5),
            (4, 5),
        ]
        .into_iter()
        .map(|(x, y)| state.board().loc(x, y).unwrap())
        .chain([Loc::PASS, Loc::PASS])
        .collect();
        let tags: Vec<_> = (0..moves.len()).collect();
        let (game, samples) = game_with_moves(rules, &moves, &tags);
        let recapture = crate::inference::policy::loc_to_spatial(
            state.board(),
            state.board().loc(4, 4).unwrap(),
        );
        assert_eq!(
            samples[9].input.spatial[2 * crate::game::board::MAX_BOARD_AREA + recapture],
            1
        );
        assert_eq!(samples[10].input.global[1], 1.0);
        assert_eq!(samples[0].input.global[0], -rules.komi);
        assert_eq!(samples[1].input.global[0], rules.komi);
        let dir = TestDir::new();
        let (tx, rx) = mpsc::channel(1);
        let (output, mut events) = event_output();
        tx.try_send(game).unwrap();
        drop(tx);
        ChunkWriter::new(
            ChunkMode::PerGame,
            dir.0.clone(),
            rx,
            EventPublisher::with_output(output),
        )
        .run()
        .unwrap();
        assert_chunk(&events.try_recv().unwrap(), &encode_chunk(&samples));
    }
}

#[test]
fn repeated_games_reuse_the_encoder() {
    for mode in [ChunkMode::PerGame, ChunkMode::FixedRecords(3)] {
        let dir = TestDir::new();
        let (_tx, rx) = mpsc::channel(1);
        let (output, mut events) = event_output();
        let mut writer =
            ChunkWriter::new(mode, dir.0.clone(), rx, EventPublisher::with_output(output));
        for dim in [19, 9, 13, 19] {
            let (game, samples) = game(dim, &[0, 1, 2]);
            writer.append_game(game).unwrap();
            assert_chunk(&events.try_recv().unwrap(), &encode_chunk(&samples));
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
        let (game, samples) = game(9, &[0, 1]);
        tx.send(game).await.unwrap();
        drop(tx);
        writer.finish().await.unwrap();
        let event = next_event(&mut events).await;
        assert_chunk(&event, &encode_chunk(&samples));
        let path = PathBuf::from(event["path"].as_str().unwrap());
        fs::remove_file(&path).unwrap();
        paths.push(path);
    }
    assert_ne!(paths[0], paths[1]);
}

#[test]
fn file_failure_closes_the_queue_and_cancels_queued_games() {
    for mode in [ChunkMode::PerGame, ChunkMode::FixedRecords(2)] {
        let dir = TestDir::new();
        fs::create_dir(dir.0.join(PENDING_CHUNK_FILE)).unwrap();
        let (tx, rx) = mpsc::channel(2);
        let (output, mut events) = event_output();
        let (first, _) = game(9, &[0, 1]);
        let (second, _) = game(9, &[2, 3]);
        tx.try_send(first).unwrap();
        tx.try_send(second).unwrap();
        let result =
            ChunkWriter::new(mode, dir.0.clone(), rx, EventPublisher::with_output(output)).run();
        assert!(matches!(result, Err(ChunkWriterError::Io(_))));
        assert!(tx.is_closed());
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
        let (first, first_samples) = game(9, &[0, 1]);
        let (second, _) = game(9, &[2, 3]);
        tx.try_send(first).unwrap();
        tx.try_send(second).unwrap();
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
        assert!(tx.is_closed());
        let files = dir.files();
        assert_eq!(files.len(), 1);
        assert_eq!(fs::read(&files[0]).unwrap(), encode_chunk(&first_samples));
    }
}

#[tokio::test]
async fn blocked_publication_backpressures_producers_when_the_queue_is_full() {
    for mode in [ChunkMode::PerGame, ChunkMode::FixedRecords(2)] {
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
        let (first, first_samples) = game(9, &[0, 1]);
        tx.send(first).await.unwrap();
        tokio::time::timeout(TIMEOUT, reached_rx)
            .await
            .unwrap()
            .unwrap();
        let (second, second_samples) = game(9, &[2, 3]);
        tx.send(second).await.unwrap();
        assert!(matches!(
            tx.try_send(game(9, &[4, 5]).0),
            Err(mpsc::error::TrySendError::Full(_))
        ));
        drop(tx);
        // The async runtime still handles timers while publication is blocked.
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(events.try_recv().is_err());
        assert_eq!(dir.files().len(), 1);
        release_tx.send(()).unwrap();
        assert_chunk(
            &next_event(&mut events).await,
            &encode_chunk(&first_samples),
        );
        writer.finish().await.unwrap();
        assert_chunk(
            &next_event(&mut events).await,
            &encode_chunk(&second_samples),
        );
    }
}

#[tokio::test]
async fn a_split_game_resumes_encoding_after_blocked_publication() {
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
    let (game, samples) = game(9, &[0, 1, 2, 3, 4]);
    tx.send(game).await.unwrap();
    drop(tx);
    tokio::time::timeout(TIMEOUT, reached_rx)
        .await
        .unwrap()
        .unwrap();
    assert!(events.try_recv().is_err());
    assert_eq!(dir.files().len(), 1);
    release_tx.send(()).unwrap();
    writer.finish().await.unwrap();
    for chunk in samples.chunks(2) {
        assert_chunk(&next_event(&mut events).await, &encode_chunk(chunk));
    }
}

#[test]
fn final_partial_chunk_publication_errors_are_propagated() {
    let dir = TestDir::new();
    fs::create_dir(dir.0.join(PENDING_CHUNK_FILE)).unwrap();
    let (tx, rx) = mpsc::channel(1);
    let (game, _) = game(9, &[0, 1]);
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
    assert_eq!(dir.files(), [dir.0.join(PENDING_CHUNK_FILE)]);
}

#[test]
fn ownership_labels_use_each_players_perspective_and_neutral_padding() {
    for dim in [9, 13, 19] {
        let board = Board::new(dim);
        let mut final_ownership = [Color::Wall; BOARD_STORAGE_LEN];
        for loc in board.locs() {
            final_ownership[loc.index()] = Color::Empty;
        }
        let black = board.loc(0, 0).unwrap();
        let white = board.loc(dim - 1, dim - 1).unwrap();
        final_ownership[black.index()] = Color::Black;
        final_ownership[white.index()] = Color::White;
        for player in [Player::Black, Player::White] {
            let mut expected = [1; MAX_BOARD_AREA];
            expected[loc_to_spatial(&board, black)] = if player == Player::Black { 2 } else { 0 };
            expected[loc_to_spatial(&board, white)] = if player == Player::White { 2 } else { 0 };
            assert_eq!(
                ownership_for_player(&board, &final_ownership, player),
                expected
            );
        }
    }
}

#[test]
fn chunk_values_use_final_area_and_komi_in_each_players_perspective() {
    use crate::inference::inputs::{NUM_GLOBAL_FEATURES, NUM_SPATIAL_FEATURES};
    use rgo_artifacts::chunk::{CHUNK_HEADER_SIZE, training_record_size};

    for board_dim in [9, 13, 19] {
        let board = Board::new(board_dim);
        let center = board.loc(board_dim / 2, board_dim / 2).unwrap();
        let area = (board_dim * board_dim) as f32;
        for (moves, komi, white_score) in [
            (vec![Loc::PASS, Loc::PASS], -7.5, -7.5),
            (vec![Loc::PASS, Loc::PASS], 0.0, 0.0),
            (vec![Loc::PASS, Loc::PASS], 7.5, 7.5),
            (vec![center, Loc::PASS, Loc::PASS], 7.5, 7.5 - area),
            (
                vec![Loc::PASS, center, Loc::PASS, Loc::PASS],
                7.5,
                7.5 + area,
            ),
        ] {
            let rules = Rules {
                board_dim,
                komi,
                ..Rules::TROMP_TAYLORISH_9
            };
            let tags: Vec<_> = (0..moves.len()).collect();
            let (game, samples) = game_with_moves(rules, &moves, &tags);
            let dir = TestDir::new();
            let (_tx, rx) = mpsc::channel(1);
            let (output, mut events) = event_output();
            let mut writer = ChunkWriter::new(
                ChunkMode::PerGame,
                dir.0.clone(),
                rx,
                EventPublisher::with_output(output),
            );
            writer.append_game(game).unwrap();
            let event = events.try_recv().unwrap();
            assert_chunk(&event, &encode_chunk(&samples));
            let bytes = fs::read(event["path"].as_str().unwrap()).unwrap();
            let values_offset = 1
                + NUM_SPATIAL_FEATURES * (board_dim * board_dim).div_ceil(8)
                + NUM_GLOBAL_FEATURES * size_of::<f32>()
                + (board_dim * board_dim + 1) * size_of::<half::f16>();
            for index in 0..moves.len() {
                let offset =
                    CHUNK_HEADER_SIZE + index * training_record_size(board_dim) + values_offset;
                let win = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                let score = f32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
                let expected_score = if index % 2 == 0 {
                    -white_score
                } else {
                    white_score
                };
                let expected_win = if expected_score > 0.0 {
                    1.0
                } else if expected_score < 0.0 {
                    0.0
                } else {
                    0.5
                };
                assert_eq!(score, expected_score);
                assert_eq!(win, expected_win);
            }
        }
    }
}
