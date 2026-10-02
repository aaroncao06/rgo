//! Reusable request slots and the shared batch queue.

use crate::inference::{
    SUPPORTED_BOARD_SIZES, backend::InferenceError, inputs::NNInput, outputs::NNOutput,
};
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex},
};
use tokio::sync::Notify;

enum SlotState {
    Idle,
    Queued(NNInput),
    Running,
    Completed(Result<Arc<NNOutput>, InferenceError>),
}
// pointers to evalslots are sent to the executor
pub(super) struct EvalSlot {
    state: Mutex<SlotState>,
    ready: Notify,
}

pub(super) struct BatchQueue {
    inner: Mutex<QueueInner>,
    state_changed: Condvar, // signals either that it is non empty or that it is closed
    capacity: usize,
}
struct QueueInner {
    requests: [VecDeque<Arc<EvalSlot>>; SUPPORTED_BOARD_SIZES.len()],
    // Each nonempty size appears once. Rotate after dispatch to avoid starvation.
    ready_sizes: VecDeque<usize>,
    closed: bool, // executor exits thread when queue is closed and empty, instead of waiting
}

impl EvalSlot {
    pub(super) fn is_idle(&self) -> bool {
        matches!(&*self.state.lock().unwrap(), SlotState::Idle)
    }

    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(SlotState::Idle),
            ready: Notify::new(),
        }
    }
    pub(super) fn queue(&self, input: NNInput) {
        let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(&*slot_state, SlotState::Idle),
            "can only queue in idle slots"
        );
        *slot_state = SlotState::Queued(input);
    }
    pub(super) fn take_input(&self) -> NNInput {
        let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
        //update state and return the input
        match std::mem::replace(&mut *slot_state, SlotState::Running) {
            SlotState::Queued(input) => input,
            _ => panic!("can only take input from a queued slot"),
        }
    }
    pub(super) fn complete(&self, result: Result<Arc<NNOutput>, InferenceError>) {
        // fill slot with the result, moves the pointer so that the worker can process it
        let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(&*slot_state, SlotState::Running),
            "can only put results in running slots"
        );
        *slot_state = SlotState::Completed(result);
        drop(slot_state);
        self.ready.notify_one();
    }
    pub(super) fn cancel_queued(&self) {
        //only called after submit_request fails
        let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(&*slot_state, SlotState::Queued(_)),
            "only a request that failed submission can be reset"
        );
        *slot_state = SlotState::Idle;
    }
    pub(super) async fn wait_for_result(&self) -> Result<Arc<NNOutput>, InferenceError> {
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
    #[cfg(test)]
    pub(super) fn is_closed(&self) -> bool {
        self.inner.lock().unwrap().closed
    }

    pub(super) fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(QueueInner {
                requests: std::array::from_fn(|_| VecDeque::new()),
                ready_sizes: VecDeque::with_capacity(SUPPORTED_BOARD_SIZES.len()),
                closed: false,
            }),
            state_changed: Condvar::new(),
            capacity,
        }
    }
    pub(super) fn submit_request(
        &self,
        request: Arc<EvalSlot>,
        board_size: usize,
    ) -> Result<(), InferenceError> {
        let size_index = SUPPORTED_BOARD_SIZES
            .iter()
            .position(|&size| size == board_size)
            .ok_or(InferenceError::UnsupportedBoardSize(board_size))?;
        let mut queue_inner = self.inner.lock().expect("batch queue mutex poisoned");
        if queue_inner.closed {
            return Err(InferenceError::RuntimeClosed);
        }
        // stay debug since this should never be an issue, our queue should be exactly the size fo the number of workers, it is cheap
        debug_assert!(
            queue_inner.requests.iter().map(|q| q.len()).sum::<usize>() < self.capacity,
            "batch queue capacity exceeded"
        );

        // not currently in ready sizes so push it now
        if queue_inner.requests[size_index].is_empty() {
            queue_inner.ready_sizes.push_back(size_index);
        }
        queue_inner.requests[size_index].push_back(request);
        drop(queue_inner);

        self.state_changed.notify_one();
        Ok(())
    }
    pub(super) fn receive_batch(
        &self,
        max_batch_size: usize,
        batch: &mut Vec<Arc<EvalSlot>>,
    ) -> bool {
        // take up to max_batch_size requests and put them in slots. return whether it succeeded
        debug_assert!(max_batch_size > 0);
        batch.clear(); //outside the mutex

        let mut queue_inner = self.inner.lock().expect("batch queue mutex poisoned");
        while queue_inner.ready_sizes.is_empty() && !queue_inner.closed {
            queue_inner = self
                .state_changed
                .wait(queue_inner)
                .expect("batch queue mutex poisoned");
        }
        if queue_inner.ready_sizes.is_empty() {
            debug_assert!(queue_inner.closed);
            return false;
        }
        let size_index = queue_inner
            .ready_sizes
            .pop_front()
            .expect("a size is ready");
        while batch.len() < max_batch_size {
            let Some(request) = queue_inner.requests[size_index].pop_front() else {
                break;
            };
            batch.push(request);
        }

        if !queue_inner.requests[size_index].is_empty() {
            queue_inner.ready_sizes.push_back(size_index);
        }
        let requests_remain = !queue_inner.ready_sizes.is_empty();
        drop(queue_inner);
        if requests_remain {
            self.state_changed.notify_one();
        }
        true
    }
    pub(super) fn close(&self) {
        // close the queue, useful for switching out model versions
        // signal executors who are waiting on the queue to end their loop
        let mut queue_inner = self.inner.lock().expect("batch queue mutex poisoned");
        queue_inner.closed = true;
        drop(queue_inner);
        self.state_changed.notify_all();
    }
}

#[cfg(test)]
mod tests;
