use crate::inference::inputs::NNInputs;
use crate::inference::{backend::InferenceError, outputs::NNOutput};
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex},
};
use tokio::sync::Notify;

enum SlotState {
    Idle,
    Queued(NNInputs),
    Running,
    Completed(Result<Arc<NNOutput>, InferenceError>),
}
// pointers to evalslots are sent to the executor
struct EvalSlot {
    state: Mutex<SlotState>,
    ready: Notify,
}

struct BatchQueue {
    inner: Mutex<QueueInner>,
    state_changed: Condvar, // signals either that it is non empty or that it is closed
    capacity: usize,
}
struct QueueInner {
    requests: VecDeque<Arc<EvalSlot>>,
    closed: bool, // executor exits thread when queue is closed and empty, instead of waiting
}
impl EvalSlot {
    fn new() -> Self {
        Self {
            state: Mutex::new(SlotState::Idle),
            ready: Notify::new(),
        }
    }
    fn queue(&self, inputs: NNInputs) {
        let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(&*slot_state, SlotState::Idle),
            "can only queue in idle slots"
        );
        *slot_state = SlotState::Queued(inputs);
    }
    fn take_input(&self) -> NNInputs {
        let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
        //update state and return the input
        match std::mem::replace(&mut *slot_state, SlotState::Running) {
            SlotState::Queued(inputs) => inputs,
            _ => panic!("can only take input from a queued slot"),
        }
    }
    fn complete(&self, result: Result<Arc<NNOutput>, InferenceError>) {
        // fill slot with the result
        let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(&*slot_state, SlotState::Running),
            "can only put results in running slots"
        );
        *slot_state = SlotState::Completed(result);
        drop(slot_state);
        self.ready.notify_one();
    }
    async fn wait_for_result(&self) -> Result<Arc<NNOutput>, InferenceError> {
        // can wait on active tasks (not idle)
        loop {
            let notified = self.ready.notified();
            {
                // scope so that the mutex gets dropped before await
                let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
                match &*slot_state {
                    SlotState::Completed(_) => {
                        let previous = std::mem::replace(&mut *slot_state, SlotState::Idle);
                        let SlotState::Completed(result) = previous else {
                            unreachable!()
                        };
                        return result;
                    }
                    SlotState::Queued(_) | SlotState::Running => {}
                    SlotState::Idle => panic!("cant wait on an idle slot"),
                }
            }
            notified.await;
        }
    }
}
impl BatchQueue {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(QueueInner {
                requests: VecDeque::with_capacity(capacity),
                closed: false,
            }),
            state_changed: Condvar::new(),
            capacity,
        }
    }
    fn submit_request(&self, request: Arc<EvalSlot>) -> Result<(), InferenceError> {
        let mut queue_inner = self.inner.lock().expect("batch queue mutex poisoned");
        if queue_inner.closed {
            return Err(InferenceError::RuntimeClosed);
        }
        // stay debug since this should never be an issue, our queue should be exactly the size fo the number of workers, it is cheap
        debug_assert!(
            queue_inner.requests.len() < self.capacity,
            "batch queue capacity exceeded"
        );

        queue_inner.requests.push_back(request);
        drop(queue_inner);

        self.state_changed.notify_one();
        Ok(())
    }
    fn receive_batch(&self, max_batch_size: usize, batch: &mut Vec<Arc<EvalSlot>>) -> bool {
        // take up to max_batch_size requests and put them in slots
        debug_assert!(max_batch_size > 0);
        batch.clear(); //outside the mutex

        let mut queue_inner = self.inner.lock().expect("batch queue mutex poisoned");
        while queue_inner.requests.is_empty() && !queue_inner.closed {
            queue_inner = self
                .state_changed
                .wait(queue_inner)
                .expect("batch queue mutex poisoned");
        }
        if queue_inner.requests.is_empty() {
            debug_assert!(queue_inner.closed);
            return false;
        }
        while batch.len() < max_batch_size {
            let Some(request) = queue_inner.requests.pop_front() else {
                break;
            };
            batch.push(request);
        }

        let requests_remain = !queue_inner.requests.is_empty();
        drop(queue_inner);
        if requests_remain {
            self.state_changed.notify_one();
        }
        true
    }
    fn close(&self) {
        // close the queue, useful for switching out model versions
        let mut queue_inner = self.inner.lock().expect("batch queue mutex poisoned");
        queue_inner.closed = true;
        drop(queue_inner);
        self.state_changed.notify_all();
    }
}

// search workers own, submits requests to the shared queue
struct InferenceClient {}

// pulls from the queue
struct InferenceRuntime {}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, mpsc},
        thread,
        time::Duration,
    };

    use crate::game::{game_state::GameState, rules::Rules};
    use crate::inference::policy::POLICY_SIZE;

    use super::*;

    fn test_inputs() -> NNInputs {
        NNInputs::encode(&GameState::new(Rules::TROMP_TAYLORISH))
    }

    fn test_output() -> Arc<NNOutput> {
        Arc::new(NNOutput::from_raw([0.0; POLICY_SIZE], 0.0, 0.0, 0.0))
    }

    #[tokio::test]
    async fn eval_slot_transitions_through_a_complete_request() {
        let slot = EvalSlot::new();
        slot.queue(test_inputs());
        assert!(matches!(&*slot.state.lock().unwrap(), SlotState::Queued(_)));

        let _inputs = slot.take_input();
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
        slot.queue(test_inputs());

        let waiting_slot = Arc::clone(&slot);
        let waiter = tokio::spawn(async move { waiting_slot.wait_for_result().await });
        tokio::task::yield_now().await;

        let _inputs = slot.take_input();
        let expected = test_output();
        slot.complete(Ok(Arc::clone(&expected)));

        let actual = waiter.await.unwrap().unwrap();
        assert!(Arc::ptr_eq(&actual, &expected));
        assert!(matches!(&*slot.state.lock().unwrap(), SlotState::Idle));
    }

    #[tokio::test]
    async fn eval_slot_propagates_inference_errors() {
        let slot = EvalSlot::new();
        slot.queue(test_inputs());
        let _inputs = slot.take_input();
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
}
