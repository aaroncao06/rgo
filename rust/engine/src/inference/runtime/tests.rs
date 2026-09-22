use super::*;
use crate::{
    game::{board::Loc, rules::Rules},
    inference::policy::POLICY_SIZE,
};
use std::{sync::Mutex, thread};

fn test_input() -> NNInput {
    NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH))
}

fn test_output() -> Arc<NNOutput> {
    Arc::new(NNOutput::from_raw([0.0; POLICY_SIZE], 0.0, 0.0, 0.0))
}

fn loc(x: usize, y: usize) -> Loc {
    Loc::new(x, y).expect("test coordinates must be on the board")
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
        inputs: &[NNInput],
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        *self.0.lock().unwrap() += inputs.len();
        for input in inputs {
            assert!(input.include_ownership);
            let mut ownership = [0.0; crate::inference::policy::BOARD_POLICY_SIZE];
            ownership
                .copy_from_slice(&input.spatial[..crate::inference::policy::BOARD_POLICY_SIZE]);
            outputs.push(Arc::new(
                NNOutput::from_raw([0.0; POLICY_SIZE], 0.0, 0.0, 0.0)
                    .with_ownership_logits(ownership),
            ));
        }
        Ok(())
    }
}

impl InferenceBackend for OwnershipBackend {
    fn evaluate_batch(
        &mut self,
        inputs: &[NNInput],
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        for input in inputs {
            self.0.lock().unwrap().push(input.include_ownership);
            let mut output = NNOutput::from_raw([0.0; POLICY_SIZE], 0.0, 0.0, 0.0);
            if input.include_ownership {
                // Deliberately different predictions, as randomized inference
                // could produce. An ownership upgrade must preserve the old ones.
                let mut policy = [0.0; POLICY_SIZE];
                policy[0] = 5.0;
                output = NNOutput::from_raw(policy, 2.0, 3.0, 1.0)
                    .with_ownership_logits([1.0; crate::inference::policy::BOARD_POLICY_SIZE]);
            }
            outputs.push(Arc::new(output));
        }
        Ok(())
    }
}

#[tokio::test]
async fn root_requests_upgrade_cached_outputs_without_mutating_interior_outputs() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let handle = ModelRuntime::start(17, vec![OwnershipBackend(requests.clone())], 1, 1, 16, 1);
    let mut client = InferenceClient::new(handle, false);
    assert_eq!(client.model_version(), 17);
    let state = GameState::new(Rules::TROMP_TAYLORISH);
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
    let handle = ModelRuntime::start(
        0,
        vec![EchoSpatialOwnershipBackend(requests.clone())],
        1,
        1,
        16,
        1,
    );

    // Force a nontrivial draw while leaving production seeding nondeterministic.
    let seed = (0..1000)
        .find(|&candidate| {
            let mut client = InferenceClient::with_symmetry_seed(handle.clone(), candidate);
            client.next_symmetry() == Some(Symmetry::TransposeFlipY)
        })
        .unwrap();

    let black_stone = loc(1, 2);
    let mut state = GameState::new(Rules::TROMP_TAYLORISH);
    assert!(state.play(black_stone));
    assert!(state.play(loc(7, 6)));

    let mut client = InferenceClient::with_symmetry_seed(handle, seed);
    let first = client.evaluate(&state, true).await.unwrap();
    let ownership = first.white_ownership().unwrap();
    for (board_index, &actual) in ownership.iter().enumerate() {
        let expected = if board_index == crate::inference::policy::loc_to_policy(black_stone) {
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
    let handle = ModelRuntime::start(
        0,
        vec![TestBackend {
            batch_sizes: Arc::new(Mutex::new(Vec::new())),
            fail: false,
        }],
        1,
        1,
        8,
        1,
    );
    let mut client = InferenceClient::new(handle, false);
    assert_eq!(client.next_symmetry(), None);
}

#[tokio::test]
#[should_panic(expected = "backend omitted requested ownership output")]
async fn missing_requested_ownership_violates_backend_contract() {
    let handle = ModelRuntime::start(
        0,
        vec![TestBackend {
            batch_sizes: Arc::new(Mutex::new(Vec::new())),
            fail: false,
        }],
        1,
        1,
        16,
        1,
    );
    let mut client = InferenceClient::new(handle, false);
    let state = GameState::new(Rules::TROMP_TAYLORISH);
    let _ = client.evaluate(&state, true).await;
}

impl InferenceBackend for TestBackend {
    fn evaluate_batch(
        &mut self,
        inputs: &[NNInput],
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        debug_assert!(outputs.is_empty());
        self.batch_sizes.lock().unwrap().push(inputs.len());

        if self.fail {
            return Err(InferenceError::ExecutionFailed);
        }

        for _input in inputs {
            outputs.push(test_output());
        }
        Ok(())
    }
}

#[tokio::test]
async fn inference_executor_processes_batches_and_completes_every_slot() {
    let queue = Arc::new(BatchQueue::new(3));
    let slots: Vec<_> = (0..3).map(|_| Arc::new(EvalSlot::new())).collect();
    for slot in &slots {
        slot.queue(test_input());
        queue.submit_request(slot.clone()).unwrap();
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
        queue.submit_request(slot.clone()).unwrap();
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
    let model_handle = ModelRuntime::start(0, vec![backend], 4, 1, 8, 2);
    let queue = model_handle.0.queue.clone();
    let mut client = InferenceClient::new(model_handle, false);

    let game_state = GameState::new(Rules::TROMP_TAYLORISH);
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
    let model_handle = ModelRuntime::start(0, vec![backend], 4, 1, 8, 2);
    let mut client = InferenceClient::new(model_handle, false);
    let game_state = GameState::new(Rules::TROMP_TAYLORISH);

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
    let first_handle = ModelRuntime::start(0, vec![backend], 4, 1, 8, 2);
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
    let model_handle = ModelRuntime::start(0, vec![backend], 2, 2, 8, 2);
    let mut first_client = InferenceClient::new(model_handle.clone(), false);
    let mut second_client = InferenceClient::new(model_handle, false);
    let first_game = GameState::new(Rules::TROMP_TAYLORISH);
    let mut second_game = GameState::new(Rules::TROMP_TAYLORISH);
    assert!(second_game.play(loc(4, 4)));

    let (first_result, second_result) = tokio::join!(
        first_client.evaluate(&first_game, false),
        second_client.evaluate(&second_game, false)
    );

    assert!(first_result.unwrap().is_processed());
    assert!(second_result.unwrap().is_processed());
    assert_eq!(batch_sizes.lock().unwrap().iter().sum::<usize>(), 2);
}
