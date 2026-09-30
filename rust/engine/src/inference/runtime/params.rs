use crate::inference::onnx::InferenceDevice;
use std::path::PathBuf;

/// Device and batch limit for one backend/executor thread.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExecutorConfig {
    pub(crate) device: InferenceDevice,
    pub(crate) max_batch_size: usize,
}

/// Executors share the runtime's queue and cache.
#[derive(Debug, Clone)]
pub(crate) struct ModelRuntimeConfig {
    /// Published ONNX models are stored as `<model_dir>/<version>.onnx`.
    pub(crate) model_dir: PathBuf,
    pub(crate) executors: Vec<ExecutorConfig>,
    pub(crate) cache_capacity: usize,
    pub(crate) num_cache_shards: usize,
}
