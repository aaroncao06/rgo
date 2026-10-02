//! Local self-play task lifecycle. A caller publishes its latest model
//! version; this module loads and replaces model runtimes at move boundaries.

use std::thread;

use tokio::{
    sync::{mpsc, watch},
    task::JoinHandle,
};

use super::{
    chunk_assembler::{ChunkAssembler, ChunkAssemblerError, CompletedGame, TrainingChunk},
    chunk_sink::{FileChunkSink, FileChunkSinkError},
    config::SelfPlayConfig,
    control::EventPublisher,
    worker::{SelfPlayError, WorkerModelControl},
};
use crate::inference::runtime::{
    ModelHandle, ModelLoadError, ModelRuntime, ModelRuntimeConfig, ModelVersion,
};

mod worker_group;
use worker_group::{WorkerEvent, WorkerGroup, WorkerSpec};

#[derive(Debug)]
pub(super) enum SelfPlayRunError {
    Worker(SelfPlayError),
    Assembler(ChunkAssemblerError),
    Sink(FileChunkSinkError),
    Task(tokio::task::JoinError),
    ModelLoad(ModelLoadError),
    ModelSourceClosed,
    FinishSourceClosed,
    LatestModelCleared,
    OldModelStillInUse,
    WorkerEventsClosed,
}

/// Owns one local self-play pipeline and one file sink for its output directory.
/// Each of the `worker_threads` OS threads runs `workers_per_thread` local tasks.
pub(super) struct SelfPlayOrchestrator {
    runtime_config: ModelRuntimeConfig,
    worker_groups: Vec<WorkerGroup>,
    assembler: ChunkAssembler,
    sink: FileChunkSink,
    pause_tx: watch::Sender<bool>,
    paused_rx: mpsc::Receiver<usize>,
    resume_txs: Vec<mpsc::Sender<ModelHandle>>,
}

impl SelfPlayOrchestrator {
    pub(super) fn new(config: SelfPlayConfig, events: EventPublisher) -> Self {
        let SelfPlayConfig {
            search: search_params,
            self_play: self_play_params,
            inference: runtime_config,
            worker_threads,
            workers_per_thread,
            chunk: chunk_mode,
            output_dir,
        } = config;
        let worker_count = worker_threads
            .checked_mul(workers_per_thread)
            .expect("self-play worker count overflow");
        // Each worker can submit at most one finished game before awaiting its
        // recycled sample buffer. A single encoded chunk may wait for the sink.
        let (completed_games_tx, completed_games_rx) = mpsc::channel::<CompletedGame>(worker_count);
        let (chunks_tx, chunks_rx) = mpsc::channel::<TrainingChunk>(1);
        let (pause_tx, pause_rx) = watch::channel(true);
        let (paused_tx, paused_rx) = mpsc::channel(worker_count);
        let mut resume_txs = Vec::with_capacity(worker_count);
        let mut worker_groups = Vec::with_capacity(worker_threads);
        for thread_index in 0..worker_threads {
            let mut group = WorkerGroup::new();
            for local_index in 0..workers_per_thread {
                let worker_index = thread_index * workers_per_thread + local_index;
                let (resume_tx, resume_rx) = mpsc::channel(1);
                resume_txs.push(resume_tx);
                group.push(WorkerSpec {
                    search_params,
                    self_play_params: self_play_params.clone(),
                    completed_games_tx: completed_games_tx.clone(),
                    model_control: WorkerModelControl {
                        worker_index,
                        pause_rx: pause_rx.clone(),
                        paused_tx: paused_tx.clone(),
                        resume_rx,
                    },
                });
            }
            worker_groups.push(group);
        }
        // Only workers retain these endpoints; constructor-owned copies must
        // not keep the game/acknowledgment channels or pause receiver alive.
        drop(completed_games_tx);
        drop(paused_tx);
        drop(pause_rx);

        Self {
            runtime_config,
            worker_groups,
            assembler: ChunkAssembler::new(chunk_mode, completed_games_rx, chunks_tx),
            sink: FileChunkSink::new(output_dir, chunks_rx, events),
            pause_tx,
            paused_rx,
            resume_txs,
        }
    }

    /// Run until a finish-games request stops new games. The checkpoint source
    /// may start at `None`, but must not clear its version once one is published.
    /// Keep both control-channel senders alive while workers are active. Send
    /// `true` to request a graceful finish; dropping either sender during
    /// coordination is an error, not a shutdown signal. The final chunk drain
    /// no longer uses these channels.
    /// Active games and model updates complete before the chunk pipeline drains.
    /// On error, return promptly; the caller must terminate the self-play
    /// process because worker-group threads may still be running.
    pub(super) async fn run(
        self,
        latest_model_version_rx: watch::Receiver<Option<ModelVersion>>,
        finish_games_rx: watch::Receiver<bool>,
    ) -> Result<(), SelfPlayRunError> {
        // Each configured worker can have one outstanding inference request.
        let queue_capacity = self.resume_txs.len();
        self.run_inner(
            None,
            latest_model_version_rx,
            finish_games_rx,
            |version, runtime_config| ModelRuntime::load(version, queue_capacity, runtime_config),
        )
        .await
    }

    #[cfg(test)]
    async fn run_n_games_per_worker<F>(
        self,
        games_per_worker: usize,
        latest_model_version_rx: watch::Receiver<Option<ModelVersion>>,
        mut load_model: F,
    ) -> Result<(), SelfPlayRunError>
    where
        F: FnMut(ModelVersion) -> Result<ModelHandle, ModelLoadError>,
    {
        let (_finish_tx, finish_rx) = watch::channel(false);
        self.run_inner(
            Some(games_per_worker),
            latest_model_version_rx,
            finish_rx,
            |version, _| load_model(version),
        )
        .await
    }

    // The production run() API always loads ONNX through ModelRuntime::load.
    // Private lifecycle tests inject controlled failures and update timing here.
    async fn run_inner<F>(
        self,
        max_games_per_worker: Option<usize>,
        mut latest_model_version_rx: watch::Receiver<Option<ModelVersion>>,
        mut finish_games_rx: watch::Receiver<bool>,
        mut load_model: F, // can pass in dummy loaders for testing
    ) -> Result<(), SelfPlayRunError>
    where
        F: FnMut(ModelVersion, &ModelRuntimeConfig) -> Result<ModelHandle, ModelLoadError>,
    {
        if !wait_for_first_model(&mut latest_model_version_rx, &mut finish_games_rx).await? {
            return Ok(());
        }
        let mut pipeline = RunningPipeline::start(
            self,
            max_games_per_worker,
            latest_model_version_rx,
            finish_games_rx,
        );
        pipeline.drive(&mut load_model).await?;
        pipeline.drain().await
    }
}

// No game can start without a model. Waiting before spawning workers also
// lets a finish request exit without leaving them blocked on their first model.
async fn wait_for_first_model(
    latest_model_version_rx: &mut watch::Receiver<Option<ModelVersion>>,
    finish_games_rx: &mut watch::Receiver<bool>,
) -> Result<bool, SelfPlayRunError> {
    loop {
        if *finish_games_rx.borrow() {
            return Ok(false);
        }
        if latest_model_version_rx.borrow().is_some() {
            return Ok(!*finish_games_rx.borrow());
        }
        tokio::select! {
            biased;
            change = finish_games_rx.changed() => {
                change.map_err(|_| SelfPlayRunError::FinishSourceClosed)?;
            }
            change = latest_model_version_rx.changed() => {
                change.map_err(|_| SelfPlayRunError::ModelSourceClosed)?;
            }
        }
    }
}

/// The tasks, channels, and counters that live from pipeline startup through
/// model handoffs and the final drain.
struct RunningPipeline {
    runtime_config: ModelRuntimeConfig,
    pause_tx: watch::Sender<bool>,
    latest_model_version_rx: watch::Receiver<Option<ModelVersion>>,
    finish_games_rx: watch::Receiver<bool>,
    paused_rx: mpsc::Receiver<usize>,
    resume_txs: Vec<mpsc::Sender<ModelHandle>>,
    events_rx: mpsc::UnboundedReceiver<WorkerEvent>,
    group_threads: Vec<thread::JoinHandle<()>>,
    assembler_task: Option<JoinHandle<Result<(), ChunkAssemblerError>>>,
    sink_task: Option<JoinHandle<Result<(), FileChunkSinkError>>>,
    active_workers: usize,
    model: Option<ModelHandle>,
}

impl RunningPipeline {
    fn start(
        orchestrator: SelfPlayOrchestrator,
        max_games_per_worker: Option<usize>,
        latest_model_version_rx: watch::Receiver<Option<ModelVersion>>,
        finish_games_rx: watch::Receiver<bool>,
    ) -> Self {
        let SelfPlayOrchestrator {
            runtime_config,
            worker_groups,
            assembler,
            sink,
            pause_tx,
            paused_rx,
            resume_txs,
        } = orchestrator;
        let sink_task = tokio::spawn(sink.run());
        let assembler_task = tokio::spawn(assembler.run());
        let active_workers = resume_txs.len();
        // Each worker reports completion once, so this unbounded channel still
        // holds at most active_workers events.
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let group_threads = worker_groups
            .into_iter()
            .map(|group| {
                group.spawn(
                    max_games_per_worker,
                    finish_games_rx.clone(),
                    events_tx.clone(),
                )
            })
            .collect();
        drop(events_tx);

        Self {
            runtime_config,
            pause_tx,
            latest_model_version_rx,
            finish_games_rx,
            paused_rx,
            resume_txs,
            events_rx,
            group_threads,
            assembler_task: Some(assembler_task),
            sink_task: Some(sink_task),
            active_workers,
            model: None,
        }
    }

    async fn drive<F>(&mut self, load_model: &mut F) -> Result<(), SelfPlayRunError>
    where
        F: FnMut(ModelVersion, &ModelRuntimeConfig) -> Result<ModelHandle, ModelLoadError>,
    {
        // Treat the channel's current value as the first publication.
        self.latest_model_version_rx.mark_changed();
        while self.active_workers > 0 {
            tokio::select! {
                biased;
                update = self.latest_model_version_rx.changed() => {
                    update.map_err(|_| SelfPlayRunError::ModelSourceClosed)?;
                    if self.latest_model_version_rx.borrow().is_none() {
                        // Startup already observed a checkpoint. Its published
                        // version must never be cleared, even before the first load.
                        return Err(SelfPlayRunError::LatestModelCleared);
                    }
                    if !self.replace_model(load_model).await? {
                        // replace model returns false when it already checks active worker == 0
                        break;
                    }
                }
                change = self.finish_games_rx.changed() => {
                    // finish games signals are processed by workers, who send events to decrement active workers
                    change.map_err(|_| SelfPlayRunError::FinishSourceClosed)?;
                }
                event = self.events_rx.recv() => {
                    record_worker_event(event, &mut self.active_workers)?;
                }
                result = async { self.assembler_task.as_mut().unwrap().await }, if self.assembler_task.is_some() => {
                    self.assembler_task.take();
                    check_assembler_result(result)?;
                }
                result = async { self.sink_task.as_mut().unwrap().await }, if self.sink_task.is_some() => {
                    self.sink_task.take();
                    check_sink_result(result)?;
                }
            }
        }
        Ok(())
    }

    /// Pause at move boundaries, retire the old executors, then publish the
    /// newest checkpoint. Keep observing failures while workers are pausing.
    /// Returns `true` if a new model was installed and workers should continue,
    /// or `false` if all workers finished before the replacement was needed.
    async fn replace_model<F>(&mut self, load_model: &mut F) -> Result<bool, SelfPlayRunError>
    where
        F: FnMut(ModelVersion, &ModelRuntimeConfig) -> Result<ModelHandle, ModelLoadError>,
    {
        let _ = self.pause_tx.send(true);
        let mut paused_workers = 0;
        let mut paused_open = true;
        while paused_workers < self.active_workers {
            tokio::select! {
                ack = self.paused_rx.recv(), if paused_open => {
                    if ack.is_some() {
                        paused_workers += 1;
                    } else {
                        // Workers may finish without acknowledging this pause.
                        // Their completion events still need to be counted.
                        paused_open = false;
                    }
                }
                change = self.latest_model_version_rx.changed() => {
                    change.map_err(|_| SelfPlayRunError::ModelSourceClosed)?;
                }
                change = self.finish_games_rx.changed() => {
                    change.map_err(|_| SelfPlayRunError::FinishSourceClosed)?;
                }
                event = self.events_rx.recv() => {
                    record_worker_event(event, &mut self.active_workers)?;
                }
                result = async { self.assembler_task.as_mut().unwrap().await }, if self.assembler_task.is_some() => {
                    self.assembler_task.take();
                    check_assembler_result(result)?;
                }
                result = async { self.sink_task.as_mut().unwrap().await }, if self.sink_task.is_some() => {
                    self.sink_task.take();
                    check_sink_result(result)?;
                }
            }
        }
        if self.active_workers == 0 {
            return Ok(false);
        }
        if self
            .model
            .as_ref()
            .is_some_and(|model| !model.is_last_handle())
        {
            // is last handle checks if all of the workers have dropped their arcs
            return Err(SelfPlayRunError::OldModelStillInUse);
        }

        // All workers released the old model at move boundaries. Its final
        // drop joins the old executor threads before loading the new backend.
        drop(self.model.take());
        // A newer checkpoint may have appeared during the pause. borrow and update avoids the race
        let version = (*self.latest_model_version_rx.borrow_and_update())
            .ok_or(SelfPlayRunError::LatestModelCleared)?;
        let new_model =
            load_model(version, &self.runtime_config).map_err(SelfPlayRunError::ModelLoad)?;
        let _ = self.pause_tx.send(false);
        for resume_tx in &self.resume_txs {
            if !resume_tx.is_closed() {
                let _ = resume_tx.send(new_model.clone()).await;
            }
        }
        self.model = Some(new_model);
        Ok(true)
    }

    async fn drain(mut self) -> Result<(), SelfPlayRunError> {
        debug_assert_eq!(self.active_workers, 0, "drain requires finished workers");
        tokio::task::spawn_blocking(move || {
            for thread in self.group_threads {
                // Normal builds abort on panic. Test builds unwind instead.
                thread.join().expect("worker-group thread panicked");
            }
        })
        .await
        .map_err(SelfPlayRunError::Task)?;
        self.resume_txs.clear();
        drop(self.model.take());

        // Wait for both stages on success, but return promptly on any failure.
        // If both results are ready, prefer the sink's original error over a
        // downstream assembler error caused by its channel closing.
        while self.assembler_task.is_some() || self.sink_task.is_some() {
            tokio::select! {
                biased;
                result = async { self.sink_task.as_mut().unwrap().await }, if self.sink_task.is_some() => {
                    self.sink_task.take();
                    check_sink_result(result)?;
                }
                result = async { self.assembler_task.as_mut().unwrap().await }, if self.assembler_task.is_some() => {
                    self.assembler_task.take();
                    check_assembler_result(result)?;
                }
            }
        }
        Ok(())
    }
}

fn check_assembler_result(
    result: Result<Result<(), ChunkAssemblerError>, tokio::task::JoinError>,
) -> Result<(), SelfPlayRunError> {
    result
        .map_err(SelfPlayRunError::Task)?
        .map_err(SelfPlayRunError::Assembler)
}

fn check_sink_result(
    result: Result<Result<(), FileChunkSinkError>, tokio::task::JoinError>,
) -> Result<(), SelfPlayRunError> {
    result
        .map_err(SelfPlayRunError::Task)?
        .map_err(SelfPlayRunError::Sink)
}

fn record_worker_event(
    event: Option<WorkerEvent>,
    active_workers: &mut usize,
) -> Result<(), SelfPlayRunError> {
    match event {
        Some(WorkerEvent::Finished(result)) => {
            *active_workers -= 1;
            result
                .map_err(SelfPlayRunError::Task)?
                .map_err(SelfPlayRunError::Worker)
        }
        None => Err(SelfPlayRunError::WorkerEventsClosed), // this should only be called when active workers are still open
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    use super::super::chunk_assembler::ChunkMode;
    use super::super::params::{SearchBudgetPolicy, SelfPlayParams};
    use super::*;
    use crate::{
        inference::{
            backend::{InferenceBackend, InferenceError},
            inputs::NNInput,
            outputs::NNOutput,
            runtime::{start_test_runtime, test_backend_factory},
        },
        search::{params::SearchParams, worker::SearchBudget},
        training_data::verify_chunk_checksum,
    };

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    fn cpu_runtime_config() -> ModelRuntimeConfig {
        ModelRuntimeConfig {
            model_dir: PathBuf::from("unused-models"),
            executors: vec![crate::inference::runtime::ExecutorConfig {
                device: crate::inference::onnx::InferenceDevice::Cpu { intra_threads: 1 },
                max_batch_size: 4,
            }],
            cache_capacity: 64,
            num_cache_shards: 1,
        }
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            loop {
                let id = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir()
                    .join(format!("rgo-orchestrator-test-{}-{id}", std::process::id()));
                match std::fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("failed to create test directory: {error}"),
                }
            }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct PassBackend;

    impl InferenceBackend for PassBackend {
        fn evaluate_batch(
            &mut self,
            inputs: &[NNInput],
            outputs: &mut Vec<Arc<NNOutput>>,
        ) -> Result<(), InferenceError> {
            for input in inputs {
                let board_area = input.board_dim * input.board_dim;
                let mut logits = vec![-100.0; board_area + 1].into_boxed_slice();
                logits[board_area] = 100.0;
                let mut output = NNOutput::from_raw(logits, 0.0, 0.0, 0.0);
                if input.include_ownership {
                    output = output.with_ownership_logits(
                        vec![0.0; input.board_dim * input.board_dim].into_boxed_slice(),
                    );
                }
                outputs.push(Arc::new(output));
            }
            Ok(())
        }
    }

    struct FailingBackend;

    impl InferenceBackend for FailingBackend {
        fn evaluate_batch(
            &mut self,
            _inputs: &[NNInput],
            _outputs: &mut Vec<Arc<NNOutput>>,
        ) -> Result<(), InferenceError> {
            Err(InferenceError::ExecutionFailed)
        }
    }

    struct DropTrackingBackend(Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for DropTrackingBackend {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl InferenceBackend for DropTrackingBackend {
        fn evaluate_batch(
            &mut self,
            inputs: &[NNInput],
            outputs: &mut Vec<Arc<NNOutput>>,
        ) -> Result<(), InferenceError> {
            PassBackend.evaluate_batch(inputs, outputs)
        }
    }

    struct BlockingPassBackend {
        first_eval_tx: Option<tokio::sync::oneshot::Sender<()>>,
        release_rx: std::sync::mpsc::Receiver<()>,
        drops: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Drop for BlockingPassBackend {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl InferenceBackend for BlockingPassBackend {
        fn evaluate_batch(
            &mut self,
            inputs: &[NNInput],
            outputs: &mut Vec<Arc<NNOutput>>,
        ) -> Result<(), InferenceError> {
            if let Some(first_eval_tx) = self.first_eval_tx.take() {
                let _ = first_eval_tx.send(());
                self.release_rx.recv().unwrap();
            }
            PassBackend.evaluate_batch(inputs, outputs)
        }
    }

    fn orchestrator(
        output_dir: PathBuf,
        mode: ChunkMode,
        worker_threads: usize,
    ) -> SelfPlayOrchestrator {
        orchestrator_with_layout(output_dir, mode, worker_threads, 1)
    }

    fn orchestrator_with_layout(
        output_dir: PathBuf,
        mode: ChunkMode,
        worker_threads: usize,
        workers_per_thread: usize,
    ) -> SelfPlayOrchestrator {
        let mut search_params = SearchParams::KATAGO_SELFPLAY8_MAIN_B18;
        search_params.root_noise_enabled = false;
        search_params.root_ending_bonus_points = 0.0;
        search_params.chosen_move_temperature_early = 0.0;
        search_params.chosen_move_temperature = 0.0;
        let params = SelfPlayParams {
            search_budget_policy: SearchBudgetPolicy::fixed(SearchBudget::new(0, 0)),
            ..SelfPlayParams::default()
        };
        SelfPlayOrchestrator::new(
            SelfPlayConfig {
                search: search_params,
                self_play: params,
                inference: cpu_runtime_config(),
                worker_threads,
                workers_per_thread,
                chunk: mode,
                output_dir,
            },
            EventPublisher::discard(),
        )
    }

    async fn run_with_pass_model(
        orchestrator: SelfPlayOrchestrator,
        games_per_worker: usize,
        worker_count: usize,
    ) -> Result<(), SelfPlayRunError> {
        let (_latest_tx, latest_rx) = watch::channel(Some(0));
        orchestrator
            .run_n_games_per_worker(games_per_worker, latest_rx, |_| {
                Ok(start_test_runtime(
                    0,
                    vec![test_backend_factory(PassBackend, 8)],
                    worker_count,
                    64,
                    1,
                )
                .unwrap())
            })
            .await
    }

    fn written_record_counts(dir: &TestDir) -> Vec<u32> {
        let mut counts = Vec::new();
        for entry in std::fs::read_dir(&dir.0).unwrap() {
            let bytes = std::fs::read(entry.unwrap().path()).unwrap();
            assert!(verify_chunk_checksum(&bytes));
            counts.push(u32::from_le_bytes(bytes[16..20].try_into().unwrap()));
        }
        counts
    }

    #[tokio::test]
    async fn runs_workers_through_per_game_file_publication() {
        let dir = TestDir::new();
        run_with_pass_model(orchestrator(dir.0.clone(), ChunkMode::PerGame, 2), 2, 2)
            .await
            .unwrap();
        assert_eq!(written_record_counts(&dir), [2, 2, 2, 2]);
    }

    #[tokio::test]
    async fn runs_multiple_workers_per_local_thread() {
        let dir = TestDir::new();
        run_with_pass_model(
            orchestrator_with_layout(dir.0.clone(), ChunkMode::PerGame, 2, 2),
            1,
            4,
        )
        .await
        .unwrap();
        assert_eq!(written_record_counts(&dir), [2, 2, 2, 2]);
    }

    #[tokio::test]
    async fn flushes_final_fixed_record_chunk_before_returning() {
        let dir = TestDir::new();
        run_with_pass_model(
            orchestrator(dir.0.clone(), ChunkMode::FixedRecords(10), 1),
            1,
            1,
        )
        .await
        .unwrap();
        assert_eq!(written_record_counts(&dir), [2]);
    }

    #[tokio::test]
    async fn reports_sink_startup_failure() {
        let dir = TestDir::new();
        let missing = dir.0.join("missing");
        let result = run_with_pass_model(orchestrator(missing, ChunkMode::PerGame, 1), 0, 1).await;
        assert!(
            matches!(
                result,
                Err(SelfPlayRunError::Sink(FileChunkSinkError::Io(_)))
            ),
            "unexpected result: {result:?}"
        );
    }

    #[tokio::test]
    async fn continuous_run_returns_on_worker_failure() {
        let dir = TestDir::new();
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 1);
        let (_latest_tx, latest_rx) = watch::channel(Some(0));
        let (_finish_tx, finish_rx) = watch::channel(false);

        let result = orchestrator
            .run_inner(None, latest_rx, finish_rx, |_, _| {
                Ok(
                    start_test_runtime(0, vec![test_backend_factory(FailingBackend, 8)], 1, 64, 1)
                        .unwrap(),
                )
            })
            .await;
        assert!(
            matches!(result, Err(SelfPlayRunError::Worker(_))),
            "unexpected result: {result:?}"
        );
    }

    #[tokio::test]
    async fn continuous_run_returns_on_sink_failure() {
        let dir = TestDir::new();
        let missing = dir.0.join("missing");
        let orchestrator = orchestrator(missing, ChunkMode::PerGame, 1);
        let (_latest_tx, latest_rx) = watch::channel(Some(0));
        let (_finish_tx, finish_rx) = watch::channel(false);

        let result = orchestrator
            .run_inner(None, latest_rx, finish_rx, |_, _| {
                Ok(
                    start_test_runtime(0, vec![test_backend_factory(PassBackend, 8)], 1, 64, 1)
                        .unwrap(),
                )
            })
            .await;
        assert!(matches!(result, Err(SelfPlayRunError::Sink(_))));
    }

    #[tokio::test]
    async fn initial_load_failure_returns_promptly() {
        let dir = TestDir::new();
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 2);
        let (_latest_tx, latest_rx) = watch::channel(Some(99));

        let result = orchestrator
            .run_n_games_per_worker(1, latest_rx, |_| {
                Err(ModelLoadError::Backend(ort::Error::new(
                    "initial model load failed",
                )))
            })
            .await;
        assert!(matches!(result, Err(SelfPlayRunError::ModelLoad(_))));
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn waits_for_first_checkpoint_publication() {
        let dir = TestDir::new();
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 1);
        let (latest_tx, latest_rx) = watch::channel(None);
        let run = orchestrator.run_n_games_per_worker(1, latest_rx, |version| {
            assert_eq!(version, 42);
            Ok(start_test_runtime(
                version,
                vec![test_backend_factory(PassBackend, 8)],
                1,
                64,
                1,
            )
            .unwrap())
        });
        let publish = async {
            tokio::task::yield_now().await;
            latest_tx.send(Some(42)).unwrap();
            latest_tx
        };
        let (result, _latest_tx) = tokio::join!(run, publish);
        result.unwrap();
        assert_eq!(written_record_counts(&dir), [2]);
    }

    #[tokio::test]
    async fn rejects_checkpoint_cleared_between_startup_gate_and_first_load() {
        let dir = TestDir::new();
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 1);
        let (latest_tx, mut latest_rx) = watch::channel(Some(0));
        let (_finish_tx, mut finish_rx) = watch::channel(false);
        assert!(
            wait_for_first_model(&mut latest_rx, &mut finish_rx)
                .await
                .unwrap()
        );
        latest_tx.send(None).unwrap();

        // Zero games lets the group exit without a model, so cleanup does not
        // depend on the rejected checkpoint being loaded.
        let mut pipeline = RunningPipeline::start(orchestrator, Some(0), latest_rx, finish_rx);
        let result = pipeline
            .drive(&mut |_, _| panic!("a cleared checkpoint must not be loaded"))
            .await;
        while pipeline.active_workers > 0 {
            record_worker_event(
                pipeline.events_rx.recv().await,
                &mut pipeline.active_workers,
            )
            .unwrap();
        }
        pipeline.drain().await.unwrap();
        assert!(matches!(result, Err(SelfPlayRunError::LatestModelCleared)));
    }

    #[tokio::test]
    async fn finish_before_first_checkpoint_starts_no_workers() {
        let dir = TestDir::new();
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 1);
        let (_latest_tx, latest_rx) = watch::channel(None);
        let (finish_tx, finish_rx) = watch::channel(false);
        let run = orchestrator.run(latest_rx, finish_rx);
        let request_finish = async {
            tokio::task::yield_now().await;
            finish_tx.send(true).unwrap();
        };
        let (result, ()) = tokio::join!(run, request_finish);
        result.unwrap();
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn closed_finish_source_before_first_checkpoint_is_an_error() {
        let dir = TestDir::new();
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 1);
        let (_latest_tx, latest_rx) = watch::channel(None);
        let (finish_tx, finish_rx) = watch::channel(false);
        drop(finish_tx);

        assert!(matches!(
            orchestrator.run(latest_rx, finish_rx).await,
            Err(SelfPlayRunError::FinishSourceClosed)
        ));
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn closed_checkpoint_source_during_model_pause_is_an_error() {
        let dir = TestDir::new();
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 1);
        let mut pause_rx = orchestrator.pause_tx.subscribe();
        let (latest_tx, latest_rx) = watch::channel(Some(0));
        let (finish_tx, finish_rx) = watch::channel(false);
        let (first_eval_tx, first_eval_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut first_eval_tx = Some(first_eval_tx);
        let mut release_rx = Some(release_rx);
        let run = async {
            let result = orchestrator
                .run_inner(Some(1), latest_rx, finish_rx, |version, _| {
                    assert_eq!(version, 0);
                    Ok(start_test_runtime(
                        0,
                        vec![test_backend_factory(
                            BlockingPassBackend {
                                first_eval_tx: first_eval_tx.take(),
                                release_rx: release_rx.take().unwrap(),
                                drops: drops.clone(),
                            },
                            8,
                        )],
                        1,
                        64,
                        1,
                    )
                    .unwrap())
                })
                .await;
            // The error must return while inference is still blocking the
            // worker's pause acknowledgment, before allowing it to finish.
            release_tx.send(()).unwrap();
            result
        };
        let close_source = async {
            first_eval_rx.await.unwrap();
            assert!(!*pause_rx.borrow_and_update());
            latest_tx.send(Some(1)).unwrap();
            pause_rx.changed().await.unwrap();
            assert!(*pause_rx.borrow());
            drop(latest_tx);
        };
        let (result, ()) = tokio::join!(run, close_source);
        drop(finish_tx);
        assert!(matches!(result, Err(SelfPlayRunError::ModelSourceClosed)));
    }

    #[tokio::test]
    async fn replaces_runtime_only_after_old_executor_exits() {
        let dir = TestDir::new();
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 1);
        let (latest_tx, latest_rx) = watch::channel(Some(0));

        orchestrator
            .run_n_games_per_worker(1, latest_rx, |version| match version {
                0 => {
                    let model = start_test_runtime(
                        0,
                        vec![test_backend_factory(DropTrackingBackend(drops.clone()), 8)],
                        1,
                        64,
                        1,
                    )
                    .unwrap();
                    latest_tx.send(Some(1)).unwrap();
                    Ok(model)
                }
                1 => {
                    assert_eq!(drops.load(Ordering::SeqCst), 1);
                    Ok(
                        start_test_runtime(1, vec![test_backend_factory(PassBackend, 8)], 1, 64, 1)
                            .unwrap(),
                    )
                }
                _ => panic!("unexpected model version"),
            })
            .await
            .unwrap();
        assert_eq!(written_record_counts(&dir), [2]);
    }

    #[tokio::test]
    async fn waits_for_every_worker_before_replacing_runtime() {
        let dir = TestDir::new();
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let orchestrator = orchestrator_with_layout(dir.0.clone(), ChunkMode::PerGame, 2, 2);
        let (latest_tx, latest_rx) = watch::channel(Some(0));

        orchestrator
            .run_n_games_per_worker(1, latest_rx, |version| match version {
                0 => {
                    let model = start_test_runtime(
                        0,
                        vec![test_backend_factory(DropTrackingBackend(drops.clone()), 8)],
                        4,
                        64,
                        1,
                    )
                    .unwrap();
                    latest_tx.send(Some(1)).unwrap();
                    Ok(model)
                }
                1 => {
                    assert_eq!(drops.load(Ordering::SeqCst), 1);
                    Ok(
                        start_test_runtime(1, vec![test_backend_factory(PassBackend, 8)], 4, 64, 1)
                            .unwrap(),
                    )
                }
                _ => panic!("unexpected model version"),
            })
            .await
            .unwrap();
        assert_eq!(written_record_counts(&dir), [2, 2, 2, 2]);
    }

    #[tokio::test]
    async fn model_update_failure_returns_promptly() {
        let dir = TestDir::new();
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 2);
        let (latest_tx, latest_rx) = watch::channel(Some(0));

        let result = orchestrator
            .run_n_games_per_worker(1, latest_rx, |version| {
                if version == 0 {
                    let model =
                        start_test_runtime(0, vec![test_backend_factory(PassBackend, 8)], 2, 64, 1)
                            .unwrap();
                    latest_tx.send(Some(99)).unwrap();
                    Ok(model)
                } else {
                    Err(ModelLoadError::Backend(ort::Error::new(
                        "model load failed",
                    )))
                }
            })
            .await;
        assert!(matches!(result, Err(SelfPlayRunError::ModelLoad(_))));
    }

    #[tokio::test]
    async fn refuses_to_start_new_runtime_while_old_handle_is_held() {
        let dir = TestDir::new();
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 1);
        let mut extra_handle = None;
        let (latest_tx, latest_rx) = watch::channel(Some(0));

        let result = orchestrator
            .run_n_games_per_worker(1, latest_rx, |version| {
                assert_eq!(version, 0);
                let model =
                    start_test_runtime(0, vec![test_backend_factory(PassBackend, 8)], 1, 64, 1)
                        .unwrap();
                extra_handle = Some(model.clone());
                latest_tx.send(Some(1)).unwrap();
                Ok(model)
            })
            .await;
        assert!(matches!(result, Err(SelfPlayRunError::OldModelStillInUse)));
        drop(extra_handle);
    }

    #[tokio::test]
    async fn finish_games_request_still_completes_an_in_progress_model_update() {
        let dir = TestDir::new();
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (first_eval_tx, first_eval_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let orchestrator = orchestrator(dir.0.clone(), ChunkMode::PerGame, 1);
        let mut pause_rx = orchestrator.pause_tx.subscribe();
        let (latest_tx, latest_rx) = watch::channel(Some(41));
        let (finish_tx, finish_rx) = watch::channel(false);
        let mut first_eval_tx = Some(first_eval_tx);
        let mut release_rx = Some(release_rx);
        let run = orchestrator.run_inner(None, latest_rx, finish_rx, |version, _| match version {
            41 => Ok(start_test_runtime(
                version,
                vec![test_backend_factory(
                    BlockingPassBackend {
                        first_eval_tx: first_eval_tx.take(),
                        release_rx: release_rx.take().unwrap(),
                        drops: drops.clone(),
                    },
                    8,
                )],
                1,
                64,
                1,
            )
            .unwrap()),
            103 => {
                assert_eq!(drops.load(Ordering::SeqCst), 1);
                Ok(start_test_runtime(
                    version,
                    vec![test_backend_factory(PassBackend, 8)],
                    1,
                    64,
                    1,
                )
                .unwrap())
            }
            _ => panic!("unexpected model version"),
        });
        let request_update = async {
            first_eval_rx.await.unwrap();
            assert!(!*pause_rx.borrow_and_update());
            latest_tx.send(Some(99)).unwrap();
            pause_rx.changed().await.unwrap();
            assert!(*pause_rx.borrow());
            latest_tx.send(Some(103)).unwrap();
            finish_tx.send(true).unwrap();
            release_tx.send(()).unwrap();
        };
        let (result, ()) = tokio::join!(run, request_update);
        result.unwrap();
        // One two-move game was finalized, rather than two partial games.
        assert_eq!(written_record_counts(&dir), [2]);
    }
}
