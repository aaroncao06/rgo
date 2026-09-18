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
    let model_handle = ModelRuntime::start(vec![backend], 4, 1, 8, 2);
    let queue = model_handle.0.queue.clone();
    let mut client = InferenceClient::new(model_handle);

    let game_state = GameState::new(Rules::TROMP_TAYLORISH);
    let output = client.evaluate(&game_state).await.unwrap();

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
    let model_handle = ModelRuntime::start(vec![backend], 4, 1, 8, 2);
    let mut client = InferenceClient::new(model_handle);
    let game_state = GameState::new(Rules::TROMP_TAYLORISH);

    let first = client.evaluate(&game_state).await.unwrap();
    let second = client.evaluate(&game_state).await.unwrap();

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(*batch_sizes.lock().unwrap(), [1]);
}

#[test]
fn model_runtime_shuts_down_after_the_last_handle_is_dropped() {
    let backend = TestBackend {
        batch_sizes: Arc::new(Mutex::new(Vec::new())),
        fail: false,
    };
    let first_handle = ModelRuntime::start(vec![backend], 4, 1, 8, 2);
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
    let model_handle = ModelRuntime::start(vec![backend], 2, 2, 8, 2);
    let mut first_client = InferenceClient::new(model_handle.clone());
    let mut second_client = InferenceClient::new(model_handle);
    let first_game = GameState::new(Rules::TROMP_TAYLORISH);
    let mut second_game = GameState::new(Rules::TROMP_TAYLORISH);
    assert!(second_game.play(loc(4, 4)));

    let (first_result, second_result) = tokio::join!(
        first_client.evaluate(&first_game),
        second_client.evaluate(&second_game)
    );

    assert!(first_result.unwrap().is_processed());
    assert!(second_result.unwrap().is_processed());
    assert_eq!(batch_sizes.lock().unwrap().iter().sum::<usize>(), 2);
}
