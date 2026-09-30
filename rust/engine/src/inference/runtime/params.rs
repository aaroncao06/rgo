use crate::inference::onnx::InferenceDevice;
use std::path::PathBuf;

/// Device and batch limit for one backend/executor thread.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutorConfig {
    pub device: InferenceDevice,
    pub max_batch_size: usize,
}

/// Executors share the runtime's queue and cache.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRuntimeConfig {
    /// Published ONNX models are stored as `<model_dir>/<version>.onnx`.
    pub model_dir: PathBuf,
    pub executors: Vec<ExecutorConfig>,
    pub cache_capacity: usize,
    pub num_cache_shards: usize,
}

impl ExecutorConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_batch_size == 0 {
            return Err("executor max_batch_size must be positive");
        }
        self.device.validate()
    }
}

impl ModelRuntimeConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
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
