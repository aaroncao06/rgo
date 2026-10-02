//! Reusable request slots and the shared batch queue.

use crate::game::game_state::GameState;
use crate::inference::{
    SUPPORTED_BOARD_DIMS, backend::InferenceError, inputs::NNInput, outputs::NNOutput,
    symmetry::Symmetry,
};
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex},
};
use tokio::sync::Notify;

enum SlotState {
    Idle,
    Queued,
    Running,
    Completed(Result<Arc<NNOutput>, InferenceError>),
}
struct SlotData {
    input: NNInput,
    state: SlotState,
}
// pointers to evalslots are sent to the executor
pub(super) struct EvalSlot {
    data: Mutex<SlotData>,
    ready: Notify,
}

pub(super) struct BatchQueue {
    inner: Mutex<QueueInner>,
    state_changed: Condvar, // signals either that it is non empty or that it is closed
    capacity: usize,
}
struct QueueInner {
    requests: [VecDeque<Arc<EvalSlot>>; SUPPORTED_BOARD_DIMS.len()],
    // Each nonempty board dimension appears once. Rotate after dispatch to avoid starvation.
    ready_dims: VecDeque<usize>,
    closed: bool, // executor exits thread when queue is closed and empty, instead of waiting
}

impl EvalSlot {
    pub(super) fn is_idle(&self) -> bool {
        matches!(&self.data.lock().unwrap().state, SlotState::Idle)
    }

    pub(super) fn new() -> Self {
        Self {
            data: Mutex::new(SlotData {
                input: NNInput::empty(),
                state: SlotState::Idle,
            }),
            ready: Notify::new(),
        }
    }
    pub(super) fn encode_and_queue(
        &self,
        game_state: &GameState,
        include_ownership: bool,
        symmetry: Option<Symmetry>,
    ) {
        let mut data = self.data.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(&data.state, SlotState::Idle),
            "can only queue in idle slots"
        );
        data.input.encode_in_place(game_state);
        if let Some(symmetry) = symmetry {
            data.input.apply_symmetry_in_place(symmetry);
        }
        data.input.include_ownership = include_ownership;
        data.state = SlotState::Queued;
    }
    pub(super) fn start(&self) -> usize {
        let mut data = self.data.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(data.state, SlotState::Queued),
            "can only start a queued slot"
        );
        data.state = SlotState::Running;
        data.input.board_dim
    }
    pub(super) fn visit_input(&self, visit: &mut dyn FnMut(&NNInput)) {
        let data = self.data.lock().expect("eval slot mutex poisoned");
        debug_assert!(matches!(data.state, SlotState::Running));
        visit(&data.input);
    }
    pub(super) fn complete(&self, result: Result<Arc<NNOutput>, InferenceError>) {
        // fill slot with the result, moves the pointer so that the worker can process it
        let mut data = self.data.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(&data.state, SlotState::Running),
            "can only put results in running slots"
        );
        data.state = SlotState::Completed(result);
        drop(data);
        self.ready.notify_one();
    }
    pub(super) fn cancel_queued(&self) {
        //only called after submit_request fails
        let mut data = self.data.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(&data.state, SlotState::Queued),
            "only a request that failed submission can be reset"
        );
        data.state = SlotState::Idle;
    }
    pub(super) async fn wait_for_result(&self) -> Result<Arc<NNOutput>, InferenceError> {
        // can wait on active tasks (not idle)
        loop {
            let notified = self.ready.notified();
            {
                // scope so that the mutex gets dropped before await
                let mut data = self.data.lock().expect("eval slot mutex poisoned");
                match &data.state {
                    SlotState::Completed(_) => {
                        let previous = std::mem::replace(&mut data.state, SlotState::Idle);
                        let SlotState::Completed(result) = previous else {
                            unreachable!()
                        };
                        return result;
                    }
                    SlotState::Queued | SlotState::Running => {}
                    SlotState::Idle => panic!("cant wait on an idle slot"),
                }
            }
            notified.await;
        }
    }
}
impl BatchQueue {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(QueueInner {
                requests: std::array::from_fn(|_| VecDeque::new()),
                ready_dims: VecDeque::with_capacity(SUPPORTED_BOARD_DIMS.len()),
                closed: false,
            }),
            state_changed: Condvar::new(),
            capacity,
        }
    }
    pub(super) fn submit_request(
        &self,
        request: Arc<EvalSlot>,
        board_dim: usize,
    ) -> Result<(), InferenceError> {
        let dim_index = SUPPORTED_BOARD_DIMS
            .iter()
            .position(|&dim| dim == board_dim)
            .ok_or(InferenceError::UnsupportedBoardDim(board_dim))?;
        let mut queue_inner = self.inner.lock().expect("batch queue mutex poisoned");
        if queue_inner.closed {
            return Err(InferenceError::RuntimeClosed);
        }
        // stay debug since this should never be an issue, our queue should be exactly the size fo the number of workers, it is cheap
        debug_assert!(
            queue_inner.requests.iter().map(|q| q.len()).sum::<usize>() < self.capacity,
            "batch queue capacity exceeded"
        );

        // An empty queue is not in ready_dims yet.
        if queue_inner.requests[dim_index].is_empty() {
            queue_inner.ready_dims.push_back(dim_index);
        }
        queue_inner.requests[dim_index].push_back(request);
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
        while queue_inner.ready_dims.is_empty() && !queue_inner.closed {
            queue_inner = self
                .state_changed
                .wait(queue_inner)
                .expect("batch queue mutex poisoned");
        }
        if queue_inner.ready_dims.is_empty() {
            debug_assert!(queue_inner.closed);
            return false;
        }
        let dim_index = queue_inner
            .ready_dims
            .pop_front()
            .expect("a board dimension is ready");
        while batch.len() < max_batch_size {
            let Some(request) = queue_inner.requests[dim_index].pop_front() else {
                break;
            };
            batch.push(request);
        }

        if !queue_inner.requests[dim_index].is_empty() {
            queue_inner.ready_dims.push_back(dim_index);
        }
        let requests_remain = !queue_inner.ready_dims.is_empty();
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
impl BatchQueue {
    pub(super) fn is_closed(&self) -> bool {
        self.inner.lock().unwrap().closed
    }
}

#[cfg(test)]
impl EvalSlot {
    pub(super) fn queue(&self, input: NNInput) {
        let mut data = self.data.lock().unwrap();
        assert!(matches!(data.state, SlotState::Idle));
        data.input = input;
        data.state = SlotState::Queued;
    }
}

#[cfg(test)]
mod tests;
