use super::*;
use crate::{
    game::{game_state::GameState, rules::Rules},
    inference::policy::POLICY_SIZE,
};
use std::{sync::mpsc, thread, time::Duration};

fn test_input() -> NNInput {
    NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH))
}

fn test_output() -> Arc<NNOutput> {
    Arc::new(NNOutput::from_raw([0.0; POLICY_SIZE], 0.0, 0.0, 0.0))
}

#[tokio::test]
async fn eval_slot_transitions_through_a_complete_request() {
    let slot = EvalSlot::new();
    slot.queue(test_input());
    assert!(matches!(&*slot.state.lock().unwrap(), SlotState::Queued(_)));

    let _input = slot.take_input();
    assert!(matches!(&*slot.state.lock().unwrap(), SlotState::Running));

    let expected = test_output();
    slot.complete(Ok(Arc::clone(&expected)));
    assert!(matches!(
        &*slot.state.lock().unwrap(),
        SlotState::Completed(_)
    ));

    let actual = slot.wait_for_result().await.unwrap();
    assert!(Arc::ptr_eq(&actual, &expected));
    assert!(matches!(&*slot.state.lock().unwrap(), SlotState::Idle));
}

#[tokio::test]
async fn eval_slot_can_wait_while_still_queued() {
    let slot = Arc::new(EvalSlot::new());
    slot.queue(test_input());

    let waiting_slot = Arc::clone(&slot);
    let waiter = tokio::spawn(async move { waiting_slot.wait_for_result().await });
    tokio::task::yield_now().await;

    let _input = slot.take_input();
    let expected = test_output();
    slot.complete(Ok(Arc::clone(&expected)));

    let actual = waiter.await.unwrap().unwrap();
    assert!(Arc::ptr_eq(&actual, &expected));
    assert!(matches!(&*slot.state.lock().unwrap(), SlotState::Idle));
}

#[tokio::test]
async fn eval_slot_propagates_inference_errors() {
    let slot = EvalSlot::new();
    slot.queue(test_input());
    let _input = slot.take_input();
    slot.complete(Err(InferenceError::ExecutionFailed));

    assert!(matches!(
        slot.wait_for_result().await,
        Err(InferenceError::ExecutionFailed)
    ));
    assert!(matches!(&*slot.state.lock().unwrap(), SlotState::Idle));
}

#[test]
fn receive_batch_is_fifo_and_respects_max_batch_size() {
    let queue = BatchQueue::new(3);
    let first = Arc::new(EvalSlot::new());
    let second = Arc::new(EvalSlot::new());
    let third = Arc::new(EvalSlot::new());

    queue.submit_request(Arc::clone(&first)).unwrap();
    queue.submit_request(Arc::clone(&second)).unwrap();
    queue.submit_request(Arc::clone(&third)).unwrap();

    let mut batch = Vec::new();
    assert!(queue.receive_batch(2, &mut batch));
    assert_eq!(batch.len(), 2);
    assert!(Arc::ptr_eq(&batch[0], &first));
    assert!(Arc::ptr_eq(&batch[1], &second));

    assert!(queue.receive_batch(2, &mut batch));
    assert_eq!(batch.len(), 1);
    assert!(Arc::ptr_eq(&batch[0], &third));
}

#[test]
fn closed_queue_drains_requests_then_stops() {
    let queue = BatchQueue::new(1);
    let request = Arc::new(EvalSlot::new());
    queue.submit_request(Arc::clone(&request)).unwrap();
    queue.close();

    assert!(matches!(
        queue.submit_request(Arc::new(EvalSlot::new())),
        Err(InferenceError::RuntimeClosed)
    ));

    let mut batch = Vec::new();
    assert!(queue.receive_batch(1, &mut batch));
    assert_eq!(batch.len(), 1);
    assert!(Arc::ptr_eq(&batch[0], &request));

    assert!(!queue.receive_batch(1, &mut batch));
    assert!(batch.is_empty());
}

#[test]
fn submission_wakes_a_waiting_receiver() {
    let queue = Arc::new(BatchQueue::new(1));
    let request = Arc::new(EvalSlot::new());
    let receiver_queue = Arc::clone(&queue);
    let (started_tx, started_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();

    let receiver = thread::spawn(move || {
        started_tx.send(()).unwrap();
        let mut batch = Vec::new();
        let received = receiver_queue.receive_batch(1, &mut batch);
        result_tx.send((received, batch)).unwrap();
    });

    started_rx.recv().unwrap();
    queue.submit_request(Arc::clone(&request)).unwrap();

    let (received, batch) = result_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("receiver did not wake after submission");
    assert!(received);
    assert_eq!(batch.len(), 1);
    assert!(Arc::ptr_eq(&batch[0], &request));
    receiver.join().unwrap();
}

#[test]
fn close_wakes_a_waiting_receiver() {
    let queue = Arc::new(BatchQueue::new(1));
    let receiver_queue = Arc::clone(&queue);
    let (started_tx, started_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();

    let receiver = thread::spawn(move || {
        started_tx.send(()).unwrap();
        let mut batch = Vec::new();
        let received = receiver_queue.receive_batch(1, &mut batch);
        result_tx.send(received).unwrap();
    });

    started_rx.recv().unwrap();
    queue.close();

    assert!(
        !result_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("receiver did not wake after closure")
    );
    receiver.join().unwrap();
}
