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
    Ready(Result<Arc<NNOutput>, InferenceError>),
}
// pointers to evalslots are sent to the executor
struct EvalSlot {
    state: Mutex<SlotState>,
    ready: Notify,
}

struct BatchQueue {
    state: Mutex<QueueState>,
    state_changed: Condvar, // signals either that it is non empty or that it is closed
    capacity: usize,
}
struct QueueState {
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
}
impl BatchQueue {
    fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(QueueState {
                requests: VecDeque::with_capacity(capacity),
                closed: false,
            }),
            state_changed: Condvar::new(),
            capacity,
        }
    }
    fn submit_request(&self, request: Arc<EvalSlot>) -> Result<(), InferenceError> {
        let mut queue_state = self.state.lock().expect("batch queue mutex poisoned");
        if queue_state.closed {
            return Err(InferenceError::RuntimeClosed);
        }
        // stay debug since this should never be an issue, our queue should be exactly the size fo the number of workers, it is cheap
        debug_assert!(
            queue_state.requests.len() < self.capacity,
            "batch queue capacity exceeded"
        );

        queue_state.requests.push_back(request);
        drop(queue_state);

        self.state_changed.notify_one();
        Ok(())
    }
    fn receive_batch(&self, max_batch_size: usize, batch: &mut Vec<Arc<EvalSlot>>) -> bool {
        // take up to max_batch_size requests and put them in slots
        debug_assert!(max_batch_size > 0);
        batch.clear(); //outside the mutex

        let mut queue_state = self.state.lock().expect("batch queue mutex poisoned");
        while queue_state.requests.is_empty() && !queue_state.closed {
            queue_state = self
                .state_changed
                .wait(queue_state)
                .expect("batch queue mutex poisoned");
        }
        if queue_state.requests.is_empty() {
            debug_assert!(queue_state.closed);
            return false;
        }
        while batch.len() < max_batch_size {
            let Some(request) = queue_state.requests.pop_front() else {
                break;
            };
            batch.push(request);
        }

        let requests_remain = !queue_state.requests.is_empty();
        drop(queue_state);
        if requests_remain {
            self.state_changed.notify_one();
        }
        true
    }
    fn close(&self) {
        // close the queue, useful for switching out model versions
        let mut queue_state = self.state.lock().expect("batch queue mutex poisoned");
        queue_state.closed = true;
        drop(queue_state);
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

    use super::*;

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
