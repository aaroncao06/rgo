//! Model lifetime, client evaluation, and executor orchestration.
//! Cache and queue implementations stay private to this runtime.

use crate::game::game_state::GameState;
use crate::inference::{
    backend::{InferenceBackend, InferenceError, InputBatch},
    inputs::NNInput,
    onnx::OnnxBackend,
    outputs::NNOutput,
    policy::legal_mask,
    symmetry::Symmetry,
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
pub use rgo_artifacts::ModelVersion;
use rgo_artifacts::model_path;
use std::{
    sync::{Arc, mpsc},
    thread::JoinHandle,
};

mod cache;
mod params;
mod queue;
use cache::{EvaluationCache, EvaluationKey};
use params::batch_sizes;
pub use params::{ExecutorConfig, ModelRuntimeConfig};
use queue::{BatchQueue, EvalSlot};

pub struct ModelRuntime {
    // each model runtime owns its own queue and cache and executors. makes it easier to switch out and make new ones
    model_version: ModelVersion,
    queue: Arc<BatchQueue>, // model runtime owns this, should be responsible for dropping everything
    cache: EvaluationCache,
    executor_threads: Vec<JoinHandle<()>>,
}

#[derive(Debug)]
pub enum ModelStartupError<E> {
    Backend(E),
    ExecutorPanicked,
}

/// Configuration validation runs before spawning executors. File loading,
/// tensor-contract validation, and device initialization run on executor threads;
/// failures share the same startup error path.
pub type ModelLoadError = ModelStartupError<ort::Error>;

// wrapper so that client doesnt access executor threads and allow easy switching. api for queueing and caching
#[derive(Clone)]
pub struct ModelHandle(Arc<ModelRuntime>);

/// A worker's reusable request slot and symmetry RNG, bound to a model when active.
/// Only one evaluation may be outstanding for a client.
pub struct InferenceClient {
    model_handle: Option<ModelHandle>,
    slot: Arc<EvalSlot>,
    symmetry_rng: Option<SmallRng>,
}

// pulls from the queue
struct InferenceExecutor<B: InferenceBackend> {
    queue: Arc<BatchQueue>,
    backend: B,
    batch_sizes: [usize; super::SUPPORTED_BOARD_DIMS.len()],
}
impl ModelHandle {
    pub fn is_last_handle(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }

    fn model_version(&self) -> ModelVersion {
        self.0.model_version
    }
    fn submit_request(
        &self,
        request: Arc<EvalSlot>,
        board_dim: usize,
    ) -> Result<(), InferenceError> {
        self.0.queue.submit_request(request, board_dim)
    }
    fn lookup(&self, key: EvaluationKey) -> Option<Arc<NNOutput>> {
        self.0.cache.lookup(key)
    }
    fn insert(&self, key: EvaluationKey, output: Arc<NNOutput>) {
        self.0.cache.insert(key, output);
    }
}
impl InferenceClient {
    pub fn model_version(&self) -> ModelVersion {
        self.model_handle().model_version()
    }
    pub fn new(model_handle: ModelHandle, symmetry_seed: Option<u64>) -> Self {
        let mut client = Self::unbound(symmetry_seed);
        client.install_model(model_handle);
        client
    }
    pub fn unbound(symmetry_seed: Option<u64>) -> Self {
        Self {
            model_handle: None,
            slot: Arc::new(EvalSlot::new()),
            symmetry_rng: symmetry_seed.map(SmallRng::seed_from_u64),
        }
    }
    fn model_handle(&self) -> &ModelHandle {
        self.model_handle
            .as_ref()
            .expect("inference requires an installed model")
    }
    pub fn has_model(&self) -> bool {
        self.model_handle.is_some()
    }
    pub fn release_model(&mut self) {
        debug_assert!(
            self.slot.is_idle(),
            "cannot release a model during inference"
        );
        drop(
            self.model_handle
                .take()
                .expect("cannot release an unbound inference client"),
        );
    }
    pub fn install_model(&mut self, model_handle: ModelHandle) {
        debug_assert!(self.model_handle.is_none(), "model already installed");
        debug_assert!(
            self.slot.is_idle(),
            "cannot install a model during inference"
        );
        self.model_handle = Some(model_handle);
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
    pub async fn evaluate(
        &mut self,
        game_state: &GameState,
        include_ownership: bool,
    ) -> Result<Arc<NNOutput>, InferenceError> {
        // Randomized and non-randomized clients intentionally share this cache.
        // Every entry is restored to canonical coordinates before insertion, so
        // either kind of evaluation is a valid prediction for the same position;
        // sharing avoids splitting the cache merely for exact reproducibility.
        let key = EvaluationKey::new(game_state);
        let cached = match self.model_handle().lookup(key) {
            Some(output) if !include_ownership || output.has_ownership() => return Ok(output),
            cached => cached,
        };

        // Match KataGo's cache boundary: randomize only cache misses, restore
        // outputs to canonical coordinates, then cache under the original state.
        let symmetry = self.next_symmetry();
        let next_player = game_state.next_player();
        let legal_mask = legal_mask(game_state);

        self.slot
            .encode_and_queue(game_state, include_ownership, symmetry);
        // send a clone of the arc pointer
        if let Err(error) = self
            .model_handle()
            .submit_request(self.slot.clone(), game_state.board().dim())
        {
            self.slot.cancel_queued();
            return Err(error);
        }

        // executor moves its Arc<NNOutput> into the slot so it doesnt retain a copy, cache gets its copy after the mutation
        let mut output = self.slot.wait_for_result().await?;
        debug_assert!(
            !include_ownership || output.has_ownership(),
            "backend omitted requested ownership output"
        );
        let fresh = Arc::get_mut(&mut output).expect("raw NN output must be exclusively owned");
        if let Some(symmetry) = symmetry {
            fresh.restore_symmetry_in_place(symmetry, game_state.board().dim());
        }
        if let Some(cached) = cached {
            // Preserve the original predictions when only ownership was missing.
            // Future randomized symmetries may produce different policy/value outputs.
            fresh.process_with_cached_values_in_place(
                next_player,
                &cached,
                game_state.board().dim(),
            );
        } else {
            fresh.process_in_place(next_player, &legal_mask, game_state.board().dim());
        }

        self.model_handle().insert(key, output.clone());

        Ok(output)
    }
}

impl<B: InferenceBackend> InferenceExecutor<B> {
    fn new(queue: Arc<BatchQueue>, backend: B, base_batch_size: usize) -> Self {
        Self {
            queue,
            backend,
            batch_sizes: batch_sizes(base_batch_size).expect("invalid executor batch limit"),
        }
    }
    fn run(mut self) {
        let mut requests: Vec<Arc<EvalSlot>> = Vec::new();
        let mut outputs: Vec<Arc<NNOutput>> = Vec::new();
        while self.queue.receive_batch(&self.batch_sizes, &mut requests) {
            outputs.clear(); // backend expects it to be cleared beforehand
            let mut board_dim = 0;
            for slot in &requests {
                let dim = slot.start();
                debug_assert!(board_dim == 0 || board_dim == dim);
                board_dim = dim;
            }
            let inputs = SlotBatch {
                requests: &requests,
                board_dim,
            };
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

struct SlotBatch<'a> {
    requests: &'a [Arc<EvalSlot>],
    board_dim: usize,
}

impl InputBatch for SlotBatch<'_> {
    fn len(&self) -> usize {
        self.requests.len()
    }
    fn board_dim(&self) -> usize {
        self.board_dim
    }
    fn for_each_input(&self, visit: &mut dyn FnMut(&NNInput)) {
        for slot in self.requests {
            slot.visit_input(visit);
        }
    }
}

impl ModelRuntime {
    /// Load an immutable model runtime, constructing each ONNX backend on its
    /// executor thread. Returns only after every backend has loaded and validated
    /// the checkpoint. Failed startup closes the queue and joins all executors.
    /// Loads `<config.model_dir>/<model_version>.onnx`; versions identify immutable
    /// published models, not process-local runtime generations.
    ///
    /// Dropping the last handle closes and drains the queue, then joins the
    /// executor threads.
    pub fn load(
        model_version: ModelVersion,
        config: &ModelRuntimeConfig,
    ) -> Result<ModelHandle, ModelLoadError> {
        config
            .validate()
            .map_err(|message| ModelStartupError::Backend(ort::Error::new(message)))?;

        let checkpoint_path = model_path(&config.model_dir, model_version);
        let queue = Arc::new(BatchQueue::new());
        let cache = EvaluationCache::new(config.cache_capacity, config.num_cache_shards);
        let mut executor_threads = Vec::with_capacity(config.executors.len());
        let (startup_tx, startup_rx) = mpsc::channel();
        for executor in config.executors.iter().copied() {
            let path = checkpoint_path.clone();
            let queue = queue.clone();
            let startup_tx = startup_tx.clone();
            executor_threads.push(std::thread::spawn(move || {
                match OnnxBackend::load(&path, executor.device) {
                    Ok(backend) => {
                        let _ = startup_tx.send(Ok(()));
                        drop(startup_tx);
                        InferenceExecutor::new(queue, backend, executor.base_batch_size).run();
                    }
                    Err(error) => {
                        let _ = startup_tx.send(Err(error));
                    }
                }
            }));
        }
        drop(startup_tx);
        let startup_result = (0..executor_threads.len()).try_for_each(|_| {
            startup_rx
                .recv()
                .map_err(|_| ModelStartupError::ExecutorPanicked)?
                .map_err(ModelStartupError::Backend)
        });
        if let Err(error) = startup_result {
            queue.close();
            for thread in executor_threads {
                let _ = thread.join();
            }
            return Err(error);
        }
        Ok(ModelHandle(Arc::new(Self {
            model_version,
            queue,
            cache,
            executor_threads,
        })))
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

#[cfg(any(test, feature = "test-support"))]
mod test_support;
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub use test_support::{start_test_runtime, test_backend_factory};

#[cfg(test)]
mod tests;
