//! Test fixtures for exercising clients and orchestration with synthetic backends.
//! Production model loading is implemented directly in `ModelRuntime::load`.

use super::*;

pub fn start_test_runtime<B, F, E>(
    model_version: ModelVersion,
    backend_factories: Vec<(F, usize)>,
    queue_capacity: usize, // max number of inference clients, each with one outstanding request
    cache_capacity: usize,
    num_cache_shards: usize,
) -> Result<ModelHandle, ModelStartupError<E>>
where
    B: InferenceBackend + 'static,
    F: FnOnce() -> Result<B, E> + Send + 'static,
    E: Send + 'static,
{
    assert!(
        !backend_factories.is_empty(),
        "need at least one inference backend"
    );
    assert!(
        backend_factories.iter().all(|(_, limit)| *limit > 0),
        "need positive batch size for every executor"
    );
    assert!(queue_capacity > 0, "need positive queue capacity");

    let queue = Arc::new(BatchQueue::new(queue_capacity));
    let cache = EvaluationCache::new(cache_capacity, num_cache_shards);

    let mut executor_threads = Vec::with_capacity(backend_factories.len());
    let (startup_tx, startup_rx) = mpsc::channel();
    for (factory, max_batch_size) in backend_factories {
        let queue = queue.clone();
        let startup_tx = startup_tx.clone();
        executor_threads.push(std::thread::spawn(move || match factory() {
            Ok(backend) => {
                let _ = startup_tx.send(Ok(()));
                drop(startup_tx);
                InferenceExecutor::new(queue, backend, max_batch_size).run();
            }
            Err(error) => {
                let _ = startup_tx.send(Err(error));
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
    Ok(ModelHandle(Arc::new(ModelRuntime {
        model_version,
        queue,
        cache,
        executor_threads,
    })))
}

pub fn test_backend_factory<B: InferenceBackend + Send + 'static>(
    backend: B,
    max_batch_size: usize,
) -> (impl FnOnce() -> Result<B, &'static str> + Send, usize) {
    (move || Ok(backend), max_batch_size)
}
