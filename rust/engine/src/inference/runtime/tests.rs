use super::*;
use crate::game::{board::Loc, rules::Rules};
use std::{
    path::PathBuf,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread,
};

fn onnx_runtime_config() -> ModelRuntimeConfig {
    ModelRuntimeConfig {
        model_dir: PathBuf::from("unused-models"),
        executors: [1, 4]
            .map(|max_batch_size| ExecutorConfig {
                device: crate::inference::onnx::InferenceDevice::Cpu { intra_threads: 1 },
                max_batch_size,
            })
            .into(),
        cache_capacity: 64,
        num_cache_shards: 2,
    }
}

struct TestModelDir(PathBuf);

impl TestModelDir {
    fn new() -> Self {
        static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
        let dir = loop {
            let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("rgo-model-test-{}-{id}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => break Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("failed to create model fixture directory: {error}"),
            }
        };
        let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        for (version, filename) in [
            (42, "v0.onnx"),
            (43, "wrong_version.onnx"),
            (44, "wrong_shape.onnx"),
        ] {
            std::fs::copy(
                fixtures.join(filename),
                dir.0.join(format!("{version}.onnx")),
            )
            .unwrap();
        }
        dir
    }

    fn config(&self) -> ModelRuntimeConfig {
        ModelRuntimeConfig {
            model_dir: self.0.clone(),
            ..onnx_runtime_config()
        }
    }
}

impl Drop for TestModelDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[tokio::test]
async fn load_starts_configured_onnx_executors_and_evaluates() {
    let models = TestModelDir::new();
    let handle = ModelRuntime::load(42, 2, &models.config()).unwrap();
    assert_eq!(handle.0.executor_threads.len(), 2);
    let state = GameState::new(Rules::TROMP_TAYLORISH_9);
    let mut client = InferenceClient::new(handle, None);
    assert_eq!(client.model_version(), 42);
    let output = client.evaluate(&state, true).await.unwrap();
    assert!(output.is_processed());
    assert!(output.has_ownership());
    let mut expected = NNOutput::from_raw(
        vec![0.0; state.board().dim().pow(2) + 1].into_boxed_slice(),
        -3.75,
        -3.75,
        -3.75,
    );
    expected.process_in_place(
        state.next_player(),
        &legal_mask(&state),
        state.board().dim(),
    );
    assert_eq!(output.white_win_prob(), expected.white_win_prob());
    assert_eq!(output.white_score_mean(), expected.white_score_mean());
}

#[test]
fn load_returns_checkpoint_errors_through_startup() {
    let models = TestModelDir::new();
    for version in [43, 44, 45] {
        assert!(matches!(
            ModelRuntime::load(version, 2, &models.config()),
            Err(ModelStartupError::Backend(_))
        ));
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn load_rejects_unsupported_cuda_before_starting_executors() {
    let models = TestModelDir::new();
    let mut config = models.config();
    config.executors.push(ExecutorConfig {
        device: crate::inference::onnx::InferenceDevice::Cuda { device_id: 0 },
        max_batch_size: 8,
    });
    assert!(matches!(
        ModelRuntime::load(42, 2, &config),
        Err(ModelStartupError::Backend(_))
    ));
    // Rejection leaves the loader usable for a supported configuration.
    drop(ModelRuntime::load(42, 2, &models.config()).unwrap());
}

#[test]
fn load_rejects_empty_executor_config() {
    let mut config = onnx_runtime_config();
    config.executors.clear();
    assert!(matches!(
        ModelRuntime::load(0, 1, &config),
        Err(ModelStartupError::Backend(_))
    ));
}

#[test]
fn load_validates_all_batch_limits_before_starting_executors() {
    let mut config = onnx_runtime_config();
    config.executors[1].max_batch_size = 0;
    assert!(matches!(
        ModelRuntime::load(0, 1, &config),
        Err(ModelStartupError::Backend(_))
    ));
}

#[test]
fn load_validates_cache_dimensions_before_allocating() {
    for (capacity, shards) in [(0, 1), (3, 1), (64, 0), (64, 3), (64, 128)] {
        let mut config = onnx_runtime_config();
        config.cache_capacity = capacity;
        config.num_cache_shards = shards;
        assert!(matches!(
            ModelRuntime::load(0, 1, &config),
            Err(ModelStartupError::Backend(_))
        ));
    }
}

fn test_input() -> NNInput {
    NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH_9))
}

fn test_output(board_dim: usize) -> Arc<NNOutput> {
    Arc::new(NNOutput::from_raw(
        vec![0.0; board_dim * board_dim + 1].into_boxed_slice(),
        0.0,
        0.0,
        0.0,
    ))
}

#[tokio::test]
async fn executors_respect_individual_batch_limits_on_a_shared_queue() {
    struct RecordingBackend {
        limit: usize,
        first_batch: bool,
        started: Arc<std::sync::Barrier>,
        batches: Arc<Mutex<Vec<(usize, usize)>>>,
    }
    impl InferenceBackend for RecordingBackend {
        fn evaluate_batch(
            &mut self,
            inputs: &dyn InputBatch,
            outputs: &mut Vec<Arc<NNOutput>>,
        ) -> Result<(), InferenceError> {
            self.batches
                .lock()
                .unwrap()
                .push((self.limit, inputs.len()));
            if self.first_batch {
                self.first_batch = false;
                // Hold both executors' first batches until the queue is filled.
                self.started.wait();
            }
            inputs.for_each_input(&mut |input| outputs.push(test_output(input.board_dim)));
            Ok(())
        }
    }

    let started = Arc::new(std::sync::Barrier::new(3));
    let batches = Arc::new(Mutex::new(Vec::new()));
    let factories = [1, 4].map(|limit| {
        test_backend_factory(
            RecordingBackend {
                limit,
                first_batch: true,
                started: started.clone(),
                batches: batches.clone(),
            },
            limit,
        )
    });
    let handle = start_test_runtime(0, factories.into(), 16, 8, 1).unwrap();
    let slots: Vec<_> = (0..16).map(|_| Arc::new(EvalSlot::new())).collect();
    for slot in &slots {
        slot.queue(test_input());
        handle
            .submit_request(slot.clone(), Rules::TROMP_TAYLORISH_9.board_dim)
            .unwrap();
    }
    started.wait();
    for slot in slots {
        slot.wait_for_result().await.unwrap();
    }
    let batches = batches.lock().unwrap();
    assert_eq!(batches.iter().map(|(_, size)| size).sum::<usize>(), 16);
    for limit in [1, 4] {
        assert!(batches.iter().any(|&(actual, _)| actual == limit));
    }
    assert!(
        batches
            .iter()
            .all(|&(limit, size)| size > 0 && size <= limit)
    );
}

fn loc(x: usize, y: usize) -> Loc {
    crate::game::board::Board::new(9)
        .loc(x, y)
        .expect("test coordinates must be on the board")
}

#[tokio::test]
async fn one_loaded_model_evaluates_all_supported_board_dims() {
    let models = TestModelDir::new();
    let handle = ModelRuntime::load(42, 1, &models.config()).unwrap();
    let mut client = InferenceClient::new(handle, Some(0));
    for dim in crate::inference::SUPPORTED_BOARD_DIMS
        .into_iter()
        .filter(|&dim| dim <= crate::game::board::MAX_BOARD_DIM)
    {
        let state = GameState::new(Rules {
            board_dim: dim,
            ..Rules::default()
        });
        let output = client.evaluate(&state, true).await.unwrap();
        let legal = legal_mask(&state);
        for (&probability, allowed) in output.policy_probs().iter().zip(legal) {
            assert_eq!(probability > 0.0, allowed);
        }
        assert!(
            (output
                .policy_probs()
                .iter()
                .copied()
                .map(f64::from)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );
        assert!(output.has_ownership());
        assert!(Arc::ptr_eq(
            &output,
            &client.evaluate(&state, true).await.unwrap()
        ));
    }
}

#[tokio::test]
async fn an_executor_drains_requests_through_the_dynamic_onnx_model() {
    let sizes = [9, 13, 19, 9, 13, 19];
    let states: Vec<_> = sizes
        .into_iter()
        .map(|board_dim| {
            GameState::new(Rules {
                board_dim,
                ..Rules::default()
            })
        })
        .collect();
    let slots: Vec<_> = states
        .iter()
        .map(|state| {
            let slot = Arc::new(EvalSlot::new());
            let mut input = NNInput::encode(state);
            input.include_ownership = true;
            slot.queue(input);
            slot
        })
        .collect();
    let queue = Arc::new(BatchQueue::new(slots.len()));
    for (slot, state) in slots.iter().zip(&states) {
        queue
            .submit_request(slot.clone(), state.board().dim())
            .unwrap();
    }
    queue.close();
    let executor = thread::spawn(move || {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/v0.onnx");
        let backend = OnnxBackend::load(
            &path,
            crate::inference::onnx::InferenceDevice::Cpu { intra_threads: 1 },
        )
        .unwrap();
        InferenceExecutor::new(queue, backend, 4).run();
    });
    for (slot, state) in slots.iter().zip(&states) {
        let mut output = slot.wait_for_result().await.unwrap();
        let output = Arc::get_mut(&mut output).unwrap();
        let legal = legal_mask(state);
        output.process_in_place(state.next_player(), &legal, state.board().dim());
        for (&probability, allowed) in output.policy_probs().iter().zip(legal) {
            assert_eq!(probability > 0.0, allowed);
        }
        assert!(output.has_ownership());
        assert!(
            (output
                .policy_probs()
                .iter()
                .copied()
                .map(f64::from)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );
    }
    executor.join().unwrap();
}

struct TestBackend {
    batch_sizes: Arc<Mutex<Vec<usize>>>,
    fail: bool,
}

struct OwnershipBackend(Arc<Mutex<Vec<bool>>>);

struct EchoSpatialOwnershipBackend(Arc<Mutex<usize>>);

impl InferenceBackend for EchoSpatialOwnershipBackend {
    fn evaluate_batch(
        &mut self,
        inputs: &dyn InputBatch,
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        *self.0.lock().unwrap() += inputs.len();
        inputs.for_each_input(&mut |input| {
            assert!(input.include_ownership);
            let plane = input
                .spatial
                .first_chunk::<{ crate::game::board::MAX_BOARD_AREA }>()
                .unwrap();
            let ownership = crate::inference::policy::active_rows(plane, input.board_dim)
                .flatten()
                .copied()
                .map(f32::from)
                .collect();
            outputs.push(Arc::new(
                NNOutput::from_raw(
                    vec![0.0; input.board_dim * input.board_dim + 1].into_boxed_slice(),
                    0.0,
                    0.0,
                    0.0,
                )
                .with_ownership_logits(ownership),
            ));
        });
        Ok(())
    }
}

impl InferenceBackend for OwnershipBackend {
    fn evaluate_batch(
        &mut self,
        inputs: &dyn InputBatch,
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        inputs.for_each_input(&mut |input| {
            self.0.lock().unwrap().push(input.include_ownership);
            let mut output = NNOutput::from_raw(
                vec![0.0; input.board_dim * input.board_dim + 1].into_boxed_slice(),
                0.0,
                0.0,
                0.0,
            );
            if input.include_ownership {
                // Deliberately different predictions, as randomized inference
                // could produce. An ownership upgrade must preserve the old ones.
                let mut policy = vec![0.0; input.board_dim * input.board_dim + 1];
                policy[0] = 5.0;
                output = NNOutput::from_raw(policy.into(), 2.0, 3.0, 1.0)
                    .with_ownership_logits(vec![1.0; input.board_dim * input.board_dim].into());
            }
            outputs.push(Arc::new(output));
        });
        Ok(())
    }
}

#[tokio::test]
async fn root_requests_upgrade_cached_outputs_without_mutating_interior_outputs() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let handle = start_test_runtime(
        17,
        vec![test_backend_factory(OwnershipBackend(requests.clone()), 1)],
        1,
        16,
        1,
    )
    .unwrap();
    let mut client = InferenceClient::new(handle, None);
    assert_eq!(client.model_version(), 17);
    let state = GameState::new(Rules::TROMP_TAYLORISH_9);
    let interior = client.evaluate(&state, false).await.unwrap();
    assert!(!interior.has_ownership());
    let root = client.evaluate(&state, true).await.unwrap();
    assert!(
        root.white_ownership()
            .unwrap()
            .iter()
            .all(|&value| (value + 1.0_f32.tanh()).abs() < 1e-6)
    );
    assert!(!interior.has_ownership());
    assert!(!Arc::ptr_eq(&interior, &root));
    assert_eq!(root.policy_probs(), interior.policy_probs());
    assert_eq!(root.white_win_prob(), interior.white_win_prob());
    assert_eq!(root.white_score_mean(), interior.white_score_mean());
    assert_eq!(root.white_score_mean_sq(), interior.white_score_mean_sq());
    assert!(Arc::ptr_eq(
        &root,
        &client.evaluate(&state, true).await.unwrap()
    ));
    assert!(Arc::ptr_eq(
        &root,
        &client.evaluate(&state, false).await.unwrap()
    ));
    assert_eq!(*requests.lock().unwrap(), [false, true]);
}

#[tokio::test]
async fn randomized_symmetry_is_restored_before_caching() {
    let requests = Arc::new(Mutex::new(0));
    let handle = start_test_runtime(
        0,
        vec![test_backend_factory(
            EchoSpatialOwnershipBackend(requests.clone()),
            1,
        )],
        1,
        16,
        1,
    )
    .unwrap();

    // Choose a seed whose first symmetry is nontrivial.
    let seed = (0..1000)
        .find(|&candidate| {
            let mut client = InferenceClient::new(handle.clone(), Some(candidate));
            client.next_symmetry() == Some(Symmetry::TransposeFlipY)
        })
        .unwrap();

    let black_stone = loc(1, 2);
    let mut state = GameState::new(Rules::TROMP_TAYLORISH_9);
    assert!(state.play(black_stone));
    assert!(state.play(loc(3, 4)));

    let mut client = InferenceClient::new(handle, Some(seed));
    let first = client.evaluate(&state, true).await.unwrap();
    let ownership = first.white_ownership().unwrap();
    for (board_index, &actual) in ownership.iter().enumerate() {
        let expected =
            if board_index == crate::inference::policy::loc_to_policy(state.board(), black_stone) {
                -1.0_f32.tanh()
            } else {
                0.0
            };
        assert!((actual - expected).abs() < 1e-6);
    }

    let second = client.evaluate(&state, true).await.unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(*requests.lock().unwrap(), 1);
}

#[test]
fn client_can_disable_randomized_symmetry() {
    let handle = start_test_runtime(
        0,
        vec![test_backend_factory(
            TestBackend {
                batch_sizes: Arc::new(Mutex::new(Vec::new())),
                fail: false,
            },
            1,
        )],
        1,
        8,
        1,
    )
    .unwrap();
    let mut client = InferenceClient::new(handle, None);
    assert_eq!(client.next_symmetry(), None);
    let mut unbound = InferenceClient::unbound(None);
    assert_eq!(unbound.next_symmetry(), None);
}

#[test]
fn rebinding_a_client_preserves_its_slot_and_symmetry_sequence() {
    let old_model = start_test_runtime(
        0,
        vec![test_backend_factory(
            TestBackend {
                batch_sizes: Arc::new(Mutex::new(Vec::new())),
                fail: false,
            },
            1,
        )],
        1,
        16,
        1,
    )
    .unwrap();
    let mut client = InferenceClient::unbound(Some(42));
    assert!(!client.has_model());
    client.install_model(old_model);
    let mut uninterrupted = InferenceClient::new(
        start_test_runtime(
            2,
            vec![test_backend_factory(
                TestBackend {
                    batch_sizes: Arc::new(Mutex::new(Vec::new())),
                    fail: false,
                },
                1,
            )],
            1,
            16,
            1,
        )
        .unwrap(),
        Some(42),
    );
    for _ in 0..5 {
        assert_eq!(client.next_symmetry(), uninterrupted.next_symmetry());
    }
    let slot = client.slot.clone();
    client.release_model();
    assert!(!client.has_model());
    let new_model = start_test_runtime(
        1,
        vec![test_backend_factory(
            TestBackend {
                batch_sizes: Arc::new(Mutex::new(Vec::new())),
                fail: false,
            },
            1,
        )],
        1,
        16,
        1,
    )
    .unwrap();
    client.install_model(new_model);
    assert!(Arc::ptr_eq(&slot, &client.slot));
    for _ in 0..20 {
        assert_eq!(client.next_symmetry(), uninterrupted.next_symmetry());
    }
}

#[cfg(debug_assertions)]
#[tokio::test]
#[should_panic(expected = "backend omitted requested ownership output")]
async fn missing_requested_ownership_violates_backend_contract() {
    let handle = start_test_runtime(
        0,
        vec![test_backend_factory(
            TestBackend {
                batch_sizes: Arc::new(Mutex::new(Vec::new())),
                fail: false,
            },
            1,
        )],
        1,
        16,
        1,
    )
    .unwrap();
    let mut client = InferenceClient::new(handle, None);
    let state = GameState::new(Rules::TROMP_TAYLORISH_9);
    let _ = client.evaluate(&state, true).await;
}

impl InferenceBackend for TestBackend {
    fn evaluate_batch(
        &mut self,
        inputs: &dyn InputBatch,
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        debug_assert!(outputs.is_empty());
        self.batch_sizes.lock().unwrap().push(inputs.len());

        if self.fail {
            return Err(InferenceError::ExecutionFailed);
        }

        inputs.for_each_input(&mut |input| {
            outputs.push(test_output(input.board_dim));
        });
        Ok(())
    }
}

#[tokio::test]
async fn inference_executor_processes_batches_and_completes_every_slot() {
    let queue = Arc::new(BatchQueue::new(3));
    let slots: Vec<_> = (0..3).map(|_| Arc::new(EvalSlot::new())).collect();
    for slot in &slots {
        slot.queue(test_input());
        queue
            .submit_request(slot.clone(), Rules::TROMP_TAYLORISH_9.board_dim)
            .unwrap();
    }
    queue.close();

    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let backend = TestBackend {
        batch_sizes: batch_sizes.clone(),
        fail: false,
    };
    let executor = InferenceExecutor::new(queue, backend, 2);
    let executor_thread = thread::spawn(move || executor.run());

    for slot in &slots {
        let output = slot.wait_for_result().await.unwrap();
        assert!(!output.is_processed());
        assert!(slot.is_idle());
    }
    executor_thread.join().unwrap();

    assert_eq!(*batch_sizes.lock().unwrap(), [2, 1]);
}

#[tokio::test]
async fn inference_executor_returns_backend_errors_to_every_slot() {
    let queue = Arc::new(BatchQueue::new(3));
    let slots: Vec<_> = (0..3).map(|_| Arc::new(EvalSlot::new())).collect();
    for slot in &slots {
        slot.queue(test_input());
        queue
            .submit_request(slot.clone(), Rules::TROMP_TAYLORISH_9.board_dim)
            .unwrap();
    }
    queue.close();

    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let backend = TestBackend {
        batch_sizes: batch_sizes.clone(),
        fail: true,
    };
    let executor = InferenceExecutor::new(queue, backend, 2);
    let executor_thread = thread::spawn(move || executor.run());

    for slot in &slots {
        assert!(matches!(
            slot.wait_for_result().await,
            Err(InferenceError::ExecutionFailed)
        ));
        assert!(slot.is_idle());
    }
    executor_thread.join().unwrap();

    assert_eq!(*batch_sizes.lock().unwrap(), [2, 1]);
}

#[tokio::test]
async fn model_runtime_evaluates_through_a_client() {
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let backend = TestBackend {
        batch_sizes: batch_sizes.clone(),
        fail: false,
    };
    let model_handle =
        start_test_runtime(0, vec![test_backend_factory(backend, 4)], 1, 8, 2).unwrap();
    let queue = model_handle.0.queue.clone();
    let mut client = InferenceClient::new(model_handle, None);

    let game_state = GameState::new(Rules::TROMP_TAYLORISH_9);
    let output = client.evaluate(&game_state, false).await.unwrap();

    assert!(output.is_processed());
    assert_eq!(*batch_sizes.lock().unwrap(), [1]);

    drop(client);
    assert!(queue.is_closed());
}

#[tokio::test]
async fn repeated_evaluation_uses_the_model_cache() {
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let backend = TestBackend {
        batch_sizes: batch_sizes.clone(),
        fail: false,
    };
    let model_handle =
        start_test_runtime(0, vec![test_backend_factory(backend, 4)], 1, 8, 2).unwrap();
    let mut client = InferenceClient::new(model_handle, None);
    let game_state = GameState::new(Rules::TROMP_TAYLORISH_9);

    let first = client.evaluate(&game_state, false).await.unwrap();
    let second = client.evaluate(&game_state, false).await.unwrap();

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(*batch_sizes.lock().unwrap(), [1]);
}

#[test]
fn model_runtime_shuts_down_after_the_last_handle_is_dropped() {
    let backend = TestBackend {
        batch_sizes: Arc::new(Mutex::new(Vec::new())),
        fail: false,
    };
    let first_handle =
        start_test_runtime(0, vec![test_backend_factory(backend, 4)], 1, 8, 2).unwrap();
    let second_handle = first_handle.clone();
    let queue = first_handle.0.queue.clone();

    drop(first_handle);
    assert!(!queue.is_closed());

    drop(second_handle);
    assert!(queue.is_closed());
}

#[tokio::test]
async fn multiple_clients_share_one_model_runtime() {
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let backend = TestBackend {
        batch_sizes: batch_sizes.clone(),
        fail: false,
    };
    let model_handle =
        start_test_runtime(0, vec![test_backend_factory(backend, 2)], 2, 8, 2).unwrap();
    let mut first_client = InferenceClient::new(model_handle.clone(), None);
    let mut second_client = InferenceClient::new(model_handle, None);
    let first_game = GameState::new(Rules::TROMP_TAYLORISH_9);
    let mut second_game = GameState::new(Rules::TROMP_TAYLORISH_9);
    assert!(second_game.play(loc(4, 4)));

    let (first_result, second_result) = tokio::join!(
        first_client.evaluate(&first_game, false),
        second_client.evaluate(&second_game, false)
    );

    assert!(first_result.unwrap().is_processed());
    assert!(second_result.unwrap().is_processed());
    assert_eq!(batch_sizes.lock().unwrap().iter().sum::<usize>(), 2);
}

#[tokio::test]
async fn unsupported_board_dims_return_errors_and_leave_the_client_reusable() {
    let models = TestModelDir::new();
    let handle = ModelRuntime::load(42, 1, &models.config()).unwrap();
    let mut client = InferenceClient::new(handle, Some(0));
    for dim in 1..9 {
        let state = GameState::new(Rules {
            board_dim: dim,
            ..Rules::default()
        });
        assert!(matches!(client.evaluate(&state, true).await,
            Err(InferenceError::UnsupportedBoardDim(actual)) if actual == dim));
        assert!(client.slot.is_idle());
    }
    client
        .evaluate(&GameState::new(Rules::default()), true)
        .await
        .unwrap();
}
