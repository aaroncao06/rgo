use crate::inference::onnx::InferenceDevice;
use std::path::PathBuf;

/// Device and batch limit for one backend/executor thread.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExecutorConfig {
    pub(crate) device: InferenceDevice,
    pub(crate) max_batch_size: usize,
}

/// Executors share the runtime's queue and cache.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelRuntimeConfig {
    /// Published ONNX models are stored as `<model_dir>/<version>.onnx`.
    pub(crate) model_dir: PathBuf,
    pub(crate) executors: Vec<ExecutorConfig>,
    pub(crate) cache_capacity: usize,
    pub(crate) num_cache_shards: usize,
}

impl ExecutorConfig {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if self.max_batch_size == 0 {
            return Err("executor max_batch_size must be positive");
        }
        self.device.validate()
    }
}

impl ModelRuntimeConfig {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if self.executors.is_empty() {
            return Err("inference needs at least one executor");
        }
        super::cache::EvaluationCache::validate_dimensions(
            self.cache_capacity,
            self.num_cache_shards,
        )?;
        for executor in &self.executors {
            executor.validate()?;
        }
        Ok(())
    }
}
