use crate::game::board::Loc;
use crate::game::game_state::GameState;
use crate::game::hash::{Hash128, komi_hash, pass_hash, player_hash, suicide_hash, superko_hash};
use crate::inference::backend::InferenceBackend;
use crate::inference::inputs::NNInput;
use crate::inference::policy::legal_mask;
use crate::inference::{backend::InferenceError, outputs::NNOutput};
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex},
    thread::JoinHandle,
};
use tokio::sync::Notify;
// code for the runtime (queue+cache), and what interacts with it (clients + executors)

enum SlotState {
    Idle,
    Queued(NNInput),
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

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EvaluationKey(Hash128); // for the eval cache
// Striped eval cache between hash(input * mask * player) and an Arc of the processed output.

struct CacheEntry {
    key: EvaluationKey,
    output: Arc<NNOutput>,
}
struct CacheShard {
    entries: Box<[Option<CacheEntry>]>, //dyn array of entries
}

struct EvaluationCache {
    //indexing: xxxx{shard bits}{entry bits}
    shards: Box<[Mutex<CacheShard>]>, //dyn array of shards
    shard_bits: u32,                  // upper
    entry_bits: u32,                  // lower
}
pub(crate) struct ModelRuntime {
    // each model runtime owns its own queue and cache and executors. makes it easier to switch out and make new ones
    queue: Arc<BatchQueue>, // model runtime owns this, should be responsible for dropping everything
    cache: EvaluationCache,
    executor_threads: Vec<JoinHandle<()>>,
}

// wrapper so that client doesnt access executor threads and allow easy switching. api for queueing and caching
#[derive(Clone)]
pub(crate) struct ModelHandle(Arc<ModelRuntime>);

// search workers own, submits requests to the shared queue
pub(crate) struct InferenceClient {
    model_handle: ModelHandle,
    slot: Arc<EvalSlot>,
}

// pulls from the queue
struct InferenceExecutor<T: InferenceBackend> {
    queue: Arc<BatchQueue>,
    backend: T,
    max_batch_size: usize,
}
impl CacheShard {
    fn new(entries_per_shard: usize) -> Self {
        let mut entries = Vec::with_capacity(entries_per_shard);
        entries.resize_with(entries_per_shard, || None);
        Self {
            entries: entries.into_boxed_slice(),
        }
    }
}
impl EvaluationCache {
    fn new(capacity: usize, num_shards: usize) -> Self {
        assert!(
            capacity.is_power_of_two(),
            "cache capacity must be a power of two"
        );
        assert!(
            num_shards.is_power_of_two(),
            "cache shard count must be a power of two"
        );
        assert!(
            num_shards <= capacity,
            "cache cannot have more shards than entries"
        );

        let entries_per_shard = capacity / num_shards;
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(Mutex::new(CacheShard::new(entries_per_shard)));
        }
        let shards = shards.into_boxed_slice();
        Self {
            shards,
            shard_bits: num_shards.ilog2(),
            entry_bits: entries_per_shard.ilog2(),
        }
    }
    fn indices(&self, key: EvaluationKey) -> (usize, usize) {
        let bits = key.0 as usize;
        let entry_mask = (1_usize << self.entry_bits) - 1;
        let shard_mask = (1_usize << self.shard_bits) - 1;
        let entry_index = bits & entry_mask;
        let shard_index = (bits >> self.entry_bits) & shard_mask;
        (shard_index, entry_index)
    }
    fn lookup(&self, key: EvaluationKey) -> Option<Arc<NNOutput>> {
        let (shard_index, entry_index) = self.indices(key);
        //take guard, check if it matches our key
        let shard = self.shards[shard_index]
            .lock()
            .expect("evaluation cache mutex poisoned");
        match &shard.entries[entry_index] {
            Some(entry) if entry.key == key => Some(entry.output.clone()),
            _ => None,
        }
    }
    fn insert(&self, key: EvaluationKey, output: Arc<NNOutput>) {
        debug_assert!(
            output.is_processed(),
            "evaluation cache only takes processed outputs"
        );
        let (shard_index, entry_index) = self.indices(key);
        let old_entry = {
            let mut shard = self.shards[shard_index]
                .lock()
                .expect("evaluation cache mutex poisoned");
            shard.entries[entry_index].replace(CacheEntry { key, output })
            //mutex drops
        };
        drop(old_entry); // outside of the mutex
    }
}

impl EvaluationKey {
    fn new(game_state: &GameState) -> Self {
        let mut key = game_state.board().position_hash();
        key ^= player_hash(game_state.next_player());
        for loc in Loc::board_iter() {
            if game_state.is_superko_banned(loc) {
                key ^= superko_hash(loc);
            }
        }
        key ^= pass_hash(game_state.consecutive_ending_passes());

        //state hash done, hash rules now
        key ^= komi_hash(game_state.rules().komi); //randomized komi will cause more cache misses for similar positions. should be discrete

        if game_state.rules().multi_stone_suicide_legal {
            key ^= suicide_hash();
        }
        Self(key)
    }
}
impl EvalSlot {
    fn new() -> Self {
        Self {
            state: Mutex::new(SlotState::Idle),
            ready: Notify::new(),
        }
    }
    fn queue(&self, input: NNInput) {
        let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(&*slot_state, SlotState::Idle),
            "can only queue in idle slots"
        );
        *slot_state = SlotState::Queued(input);
    }
    fn take_input(&self) -> NNInput {
        let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
        //update state and return the input
        match std::mem::replace(&mut *slot_state, SlotState::Running) {
            SlotState::Queued(input) => input,
            _ => panic!("can only take input from a queued slot"),
        }
    }
    fn complete(&self, result: Result<Arc<NNOutput>, InferenceError>) {
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
    fn cancel_queued(&self) {
        //only called after submit_request fails
        let mut slot_state = self.state.lock().expect("eval slot mutex poisoned");
        debug_assert!(
            matches!(&*slot_state, SlotState::Queued(_)),
            "only a request that failed submission can be reset"
        );
        *slot_state = SlotState::Idle;
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
        // take up to max_batch_size requests and put them in slots. return whether it succeeded
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
        // signal executors who are waiting on the queue to end their loop
        let mut queue_inner = self.inner.lock().expect("batch queue mutex poisoned");
        queue_inner.closed = true;
        drop(queue_inner);
        self.state_changed.notify_all();
    }
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

impl<T: InferenceBackend> InferenceExecutor<T> {
    fn new(queue: Arc<BatchQueue>, backend: T, max_batch_size: usize) -> Self {
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
    pub(crate) fn start<T>(
        backends: Vec<T>,
        max_batch_size: usize,
        queue_capacity: usize, // max number of inference clients, each with one outstanding request
        cache_capacity: usize,
        num_cache_shards: usize,
    ) -> ModelHandle
    where
        T: InferenceBackend + Send + 'static, // send backends to the different executor threads. static is a requirement to move into the thread (backend owns everything it needs)
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
mod tests {
    use std::{
        sync::{Arc, Mutex, mpsc},
        thread,
        time::Duration,
    };

    use crate::game::{game_state::GameState, rules::Rules};
    use crate::inference::policy::POLICY_SIZE;

    use super::*;

    fn test_input() -> NNInput {
        NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH))
    }

    fn test_output() -> Arc<NNOutput> {
        Arc::new(NNOutput::from_raw([0.0; POLICY_SIZE], 0.0, 0.0, 0.0))
    }

    fn test_processed_output() -> Arc<NNOutput> {
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let mut output = test_output();
        Arc::get_mut(&mut output)
            .unwrap()
            .process_in_place(game_state.next_player(), &legal_mask(&game_state));
        output
    }

    fn loc(x: usize, y: usize) -> Loc {
        Loc::new(x, y).expect("test coordinates must be on the board")
    }

    #[test]
    fn evaluation_key_is_deterministic_and_changes_with_position_and_passes() {
        let initial = GameState::new(Rules::TROMP_TAYLORISH);
        let equivalent = GameState::new(Rules::TROMP_TAYLORISH);
        assert_eq!(
            EvaluationKey::new(&initial),
            EvaluationKey::new(&equivalent)
        );

        let mut moved = GameState::new(Rules::TROMP_TAYLORISH);
        assert!(moved.play(loc(4, 4)));
        assert_ne!(EvaluationKey::new(&initial), EvaluationKey::new(&moved));

        let mut passed_twice = GameState::new(Rules::TROMP_TAYLORISH);
        assert!(passed_twice.play(Loc::PASS));
        assert!(passed_twice.play(Loc::PASS));
        assert_eq!(
            initial.board().position_hash(),
            passed_twice.board().position_hash()
        );
        assert_eq!(initial.next_player(), passed_twice.next_player());
        assert_ne!(
            EvaluationKey::new(&initial),
            EvaluationKey::new(&passed_twice)
        );
    }

    #[test]
    fn evaluation_key_changes_with_komi_and_suicide_rule() {
        let baseline = GameState::new(Rules::TROMP_TAYLORISH);
        let different_komi = GameState::new(Rules {
            komi: 6.5,
            multi_stone_suicide_legal: true,
        });
        let different_suicide_rule = GameState::new(Rules::OGS_CHINESE);

        assert_ne!(
            EvaluationKey::new(&baseline),
            EvaluationKey::new(&different_komi)
        );
        assert_ne!(
            EvaluationKey::new(&baseline),
            EvaluationKey::new(&different_suicide_rule)
        );
    }

    #[test]
    fn evaluation_key_includes_superko_history_for_the_same_position() {
        let recapture = loc(4, 4);
        let capture = loc(4, 5);

        let mut with_repetition = GameState::new(Rules::TROMP_TAYLORISH);
        for move_loc in [
            loc(4, 3),
            recapture,
            loc(3, 4),
            loc(4, 6),
            loc(5, 4),
            loc(3, 5),
            loc(0, 0),
            loc(5, 5),
            capture,
        ] {
            assert!(with_repetition.play(move_loc));
        }

        let mut without_repetition = GameState::new(Rules::TROMP_TAYLORISH);
        for move_loc in [
            loc(4, 3),
            loc(4, 6),
            loc(3, 4),
            loc(3, 5),
            loc(5, 4),
            loc(5, 5),
            loc(0, 0),
            Loc::PASS,
            capture,
        ] {
            assert!(without_repetition.play(move_loc));
        }

        assert_eq!(
            with_repetition.board().position_hash(),
            without_repetition.board().position_hash()
        );
        assert_eq!(
            with_repetition.next_player(),
            without_repetition.next_player()
        );
        assert_eq!(
            with_repetition.consecutive_ending_passes(),
            without_repetition.consecutive_ending_passes()
        );
        assert!(with_repetition.is_superko_banned(recapture));
        assert!(!without_repetition.is_superko_banned(recapture));
        assert_ne!(
            EvaluationKey::new(&with_repetition),
            EvaluationKey::new(&without_repetition)
        );
    }

    #[test]
    fn evaluation_cache_misses_then_returns_the_inserted_output() {
        let cache = EvaluationCache::new(8, 2);
        let key = EvaluationKey(3);
        let output = test_processed_output();

        assert!(cache.lookup(key).is_none());
        cache.insert(key, output.clone());

        let cached = cache.lookup(key).unwrap();
        assert!(Arc::ptr_eq(&cached, &output));
    }

    #[test]
    fn evaluation_cache_replaces_collisions_without_invalidating_old_outputs() {
        let cache = EvaluationCache::new(8, 2);
        let first_key = EvaluationKey(3);
        let colliding_key = EvaluationKey(11);
        assert_eq!(cache.indices(first_key), cache.indices(colliding_key));

        let first_output = test_processed_output();
        cache.insert(first_key, first_output.clone());
        let retained = cache.lookup(first_key).unwrap();

        let second_output = test_processed_output();
        cache.insert(colliding_key, second_output.clone());

        assert!(cache.lookup(first_key).is_none());
        let cached = cache.lookup(colliding_key).unwrap();
        assert!(Arc::ptr_eq(&cached, &second_output));
        assert!(Arc::ptr_eq(&retained, &first_output));
        assert!(retained.is_processed());
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
            assert!(matches!(&*slot.state.lock().unwrap(), SlotState::Idle));
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
            assert!(matches!(&*slot.state.lock().unwrap(), SlotState::Idle));
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
        assert!(queue.inner.lock().unwrap().closed);
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
        assert!(!queue.inner.lock().unwrap().closed);

        drop(second_handle);
        assert!(queue.inner.lock().unwrap().closed);
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
