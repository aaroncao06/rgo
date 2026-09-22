//! Model lifetime, client evaluation, and executor orchestration.
//! Cache and queue implementations stay private to this runtime.

use crate::game::game_state::GameState;
use crate::inference::{
    backend::{InferenceBackend, InferenceError},
    inputs::NNInput,
    outputs::NNOutput,
    policy::legal_mask,
    symmetry::Symmetry,
};
use rand::{RngExt, SeedableRng, rngs::SmallRng, rngs::SysRng};
use std::{sync::Arc, thread::JoinHandle};

mod cache;
mod queue;
use cache::{EvaluationCache, EvaluationKey};
use queue::{BatchQueue, EvalSlot};

pub(crate) type ModelVersion = u64;

pub(crate) struct ModelRuntime {
    // each model runtime owns its own queue and cache and executors. makes it easier to switch out and make new ones
    model_version: ModelVersion,
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
    symmetry_rng: Option<SmallRng>,
}

// pulls from the queue
struct InferenceExecutor<B: InferenceBackend> {
    queue: Arc<BatchQueue>,
    backend: B,
    max_batch_size: usize,
}
impl ModelHandle {
    fn model_version(&self) -> ModelVersion {
        self.0.model_version
    }
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
    pub(crate) fn model_version(&self) -> ModelVersion {
        self.model_handle.model_version()
    }
    pub(crate) fn new(model_handle: ModelHandle, randomize_symmetry: bool) -> Self {
        let symmetry_rng = randomize_symmetry.then(|| {
            SmallRng::try_from_rng(&mut SysRng)
                .expect("system RNG unavailable for inference symmetry")
        });
        Self {
            model_handle,
            slot: Arc::new(EvalSlot::new()),
            symmetry_rng,
        }
    }
    #[cfg(test)]
    pub(crate) fn with_symmetry_seed(model_handle: ModelHandle, symmetry_seed: u64) -> Self {
        Self {
            model_handle,
            slot: Arc::new(EvalSlot::new()),
            symmetry_rng: Some(SmallRng::seed_from_u64(symmetry_seed)),
        }
    }
    pub(crate) fn reseed_symmetry(&mut self, symmetry_seed: u64) {
        if let Some(rng) = self.symmetry_rng.as_mut() {
            *rng = SmallRng::seed_from_u64(symmetry_seed);
        }
    }
    fn next_symmetry(&mut self) -> Option<Symmetry> {
        self.symmetry_rng
            .as_mut()
            .map(|rng| Symmetry::from_index(rng.random_range(0..Symmetry::ALL.len())))
    }
    /// Return a cached processed output or submit and await a raw evaluation.
    /// A cached output without ownership cannot satisfy a request that includes it.
    ///
    /// This operation is not cancellation-safe for client reuse once submitted:
    /// dropping the future does not cancel the queued/running request or reset
    /// its slot. Await completion before reusing this client, or discard the
    /// client if its evaluation future is cancelled.
    pub(crate) async fn evaluate(
        &mut self,
        game_state: &GameState,
        include_ownership: bool,
    ) -> Result<Arc<NNOutput>, InferenceError> {
        // Randomized and non-randomized clients intentionally share this cache.
        // Every entry is restored to canonical coordinates before insertion, so
        // either kind of evaluation is a valid prediction for the same position;
        // sharing avoids splitting the cache merely for exact reproducibility.
        let key = EvaluationKey::new(game_state);
        let cached = match self.model_handle.lookup(key) {
            Some(output) if !include_ownership || output.has_ownership() => return Ok(output),
            cached => cached,
        };

        // Match KataGo's cache boundary: randomize only cache misses, restore
        // outputs to canonical coordinates, then cache under the original state.
        let symmetry = self.next_symmetry();
        let mut input = NNInput::encode(game_state);
        if let Some(symmetry) = symmetry {
            input.apply_symmetry_in_place(symmetry);
        }
        input.include_ownership = include_ownership;
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
        assert!(
            !include_ownership || output.has_ownership(),
            "backend omitted requested ownership output"
        );
        let fresh = Arc::get_mut(&mut output).expect("raw NN output must be exclusively owned");
        if let Some(symmetry) = symmetry {
            fresh.restore_symmetry_in_place(symmetry);
        }
        if let Some(cached) = cached {
            // Preserve the original predictions when only ownership was missing.
            // Future randomized symmetries may produce different policy/value outputs.
            fresh.process_with_cached_values_in_place(next_player, &cached);
        } else {
            fresh.process_in_place(next_player, &legal_mask);
        }

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
        model_version: ModelVersion,
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
            model_version,
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
