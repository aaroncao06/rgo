//! Model lifetime, client evaluation, and executor orchestration.
//! Cache and queue implementations stay private to this runtime.

use crate::game::game_state::GameState;
use crate::inference::{
    backend::{InferenceBackend, InferenceError},
    inputs::NNInput,
    outputs::NNOutput,
    policy::legal_mask,
};
use std::{sync::Arc, thread::JoinHandle};

mod cache;
mod queue;
use cache::{EvaluationCache, EvaluationKey};
use queue::{BatchQueue, EvalSlot};

pub(crate) struct ModelRuntime {
    // each model runtime owns its own queue and cache and executors. makes it easier to switch out and make new ones
    queue: Arc<BatchQueue>, // model runtime owns this, should be responsible for dropping everything
    cache: EvaluationCache,
    executor_threads: Vec<JoinHandle<()>>,
}

// wrapper so that client doesnt access executor threads and allow easy switching. api for queueing and caching
#[derive(Clone)]
pub(crate) struct ModelHandle(Arc<ModelRuntime>);

/// A worker's client for one model runtime, with one reusable request slot.
/// Only one evaluation may be outstanding for a client.
pub(crate) struct InferenceClient {
    model_handle: ModelHandle,
    slot: Arc<EvalSlot>,
}

// pulls from the queue
struct InferenceExecutor<B: InferenceBackend> {
    queue: Arc<BatchQueue>,
    backend: B,
    max_batch_size: usize,
}
impl ModelHandle {
    fn submit_request(&self, request: Arc<EvalSlot>) -> Result<(), InferenceError> {
        self.0.queue.submit_request(request)
    }
    fn lookup(&self, key: EvaluationKey) -> Option<Arc<NNOutput>> {
        self.0.cache.lookup(key)
    }
    fn insert(&self, key: EvaluationKey, output: Arc<NNOutput>) {
        self.0.cache.insert(key, output);
    }
}
impl InferenceClient {
    pub(crate) fn new(model_handle: ModelHandle) -> Self {
        Self {
            model_handle,
            slot: Arc::new(EvalSlot::new()),
        }
    }
    /// Return a cached processed output or submit and await a raw evaluation.
    ///
    /// This operation is not cancellation-safe for client reuse once submitted:
    /// dropping the future does not cancel the queued/running request or reset
    /// its slot. Await completion before reusing this client, or discard the
    /// client if its evaluation future is cancelled.
    pub(crate) async fn evaluate(
        &mut self,
        game_state: &GameState,
    ) -> Result<Arc<NNOutput>, InferenceError> {
        //first check cache
        let key = EvaluationKey::new(game_state);
        if let Some(output) = self.model_handle.lookup(key) {
            return Ok(output);
        }

        let input = NNInput::encode(game_state);
        let next_player = game_state.next_player();
        let legal_mask = legal_mask(game_state);

        self.slot.queue(input);
        // send a clone of the arc pointer
        if let Err(error) = self.model_handle.submit_request(self.slot.clone()) {
            self.slot.cancel_queued();
            return Err(error);
        }

        // executor moves its Arc<NNOutput> into the slot so it doesnt retain a copy, cache gets its copy after the mutation
        let mut output = self.slot.wait_for_result().await?;
        Arc::get_mut(&mut output)
            .expect("raw NN output must be exclusively owned")
            .process_in_place(next_player, &legal_mask);

        self.model_handle.insert(key, output.clone());

        Ok(output)
    }
}

impl<B: InferenceBackend> InferenceExecutor<B> {
    fn new(queue: Arc<BatchQueue>, backend: B, max_batch_size: usize) -> Self {
        assert!(max_batch_size > 0, "need positive batch size");
        Self {
            queue,
            backend,
            max_batch_size,
        }
    }
    fn run(mut self) {
        let mut requests: Vec<Arc<EvalSlot>> = Vec::with_capacity(self.max_batch_size);
        let mut inputs: Vec<NNInput> = Vec::with_capacity(self.max_batch_size);
        let mut outputs: Vec<Arc<NNOutput>> = Vec::with_capacity(self.max_batch_size);
        while self.queue.receive_batch(self.max_batch_size, &mut requests) {
            inputs.clear();
            outputs.clear(); // backend expects it to be cleared beforehand
            //gather inputs
            for slot in &requests {
                inputs.push(slot.take_input());
            }
            //run backend
            match self.backend.evaluate_batch(&inputs, &mut outputs) {
                Ok(()) => {
                    // evaluation successful, go through request,output pairs to send results back
                    debug_assert_eq!(
                        requests.len(),
                        outputs.len(),
                        "backend returned wrong number of outputs"
                    );
                    for (request, output) in requests.drain(..).zip(outputs.drain(..)) {
                        request.complete(Ok(output));
                    }
                }
                Err(error) => {
                    // send back errors
                    for request in requests.drain(..) {
                        request.complete(Err(error.clone()));
                    }
                }
            }
        }
    }
}

impl ModelRuntime {
    /// Start an immutable model runtime and its executor threads.
    /// `queue_capacity` must cover the maximum number of clients that can submit
    /// concurrently. The queue relies on this caller-established bound; it does
    /// not implement capacity backpressure. Dropping the last handle closes and
    /// drains the queue, then joins the executor threads.
    pub(crate) fn start<B>(
        backends: Vec<B>,
        max_batch_size: usize,
        queue_capacity: usize, // max number of inference clients, each with one outstanding request
        cache_capacity: usize,
        num_cache_shards: usize,
    ) -> ModelHandle
    where
        B: InferenceBackend + Send + 'static, // send backends to the different executor threads. static is a requirement to move into the thread (backend owns everything it needs)
    {
        assert!(!backends.is_empty(), "need at least one inference backend");
        assert!(max_batch_size > 0, "need positive batch size");
        assert!(queue_capacity > 0, "need positive queue capacity");

        let queue = Arc::new(BatchQueue::new(queue_capacity));
        let cache = EvaluationCache::new(cache_capacity, num_cache_shards);

        let mut executor_threads = Vec::with_capacity(backends.len());
        for backend in backends {
            let executor = InferenceExecutor::new(queue.clone(), backend, max_batch_size);
            executor_threads.push(std::thread::spawn(move || {
                executor.run();
            }));
        }
        ModelHandle(Arc::new(Self {
            queue,
            cache,
            executor_threads,
        }))
    }
}
//destructor that closes the queue and executor threads
impl Drop for ModelRuntime {
    fn drop(&mut self) {
        self.queue.close();
        for executor_thread in self.executor_threads.drain(..) {
            let _ = executor_thread.join();
        }
    }
}

#[cfg(test)]
mod tests;
