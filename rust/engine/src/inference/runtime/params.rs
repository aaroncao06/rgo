use crate::inference::{SUPPORTED_BOARD_DIMS, onnx::InferenceDevice};
use std::path::PathBuf;

/// Device and session settings for one backend/executor thread.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutorConfig {
    pub device: InferenceDevice,
    /// ONNX Runtime CPU threads per operator, including CPU fallback work.
    /// Defaults to one to avoid multiplying thread pools across executors.
    #[serde(default = "default_intra_threads")]
    pub intra_threads: usize,
    /// Maximum positions per 19×19 batch. Smaller boards scale by active area:
    /// floor(base_batch_size * 361 / board_dim²). This bounds input volume,
    /// not model compute, which can scale differently with board dimensions.
    pub base_batch_size: usize,
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
        if self.intra_threads == 0 {
            return Err("executor intra_threads must be positive");
        }
        if self.intra_threads > i32::MAX as usize {
            return Err("executor intra_threads exceeds ONNX Runtime's i32 limit");
        }
        batch_sizes(self.base_batch_size)?;
        self.device.validate()
    }
}

fn default_intra_threads() -> usize {
    1
}

/// Limits in SUPPORTED_BOARD_DIMS order, computed before batch dispatch.
pub(super) fn batch_sizes(
    base_batch_size: usize,
) -> Result<[usize; SUPPORTED_BOARD_DIMS.len()], &'static str> {
    if base_batch_size == 0 {
        return Err("executor base_batch_size must be positive");
    }
    let max_batch_area = base_batch_size
        .checked_mul(19 * 19)
        .ok_or("executor base_batch_size overflows the 19×19 batch area")?;
    Ok(SUPPORTED_BOARD_DIMS.map(|dim| max_batch_area / dim.pow(2)))
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
