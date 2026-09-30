//! Process configuration, separate from per-game and search algorithm parameters.

use std::{
    error::Error,
    fmt, fs, io,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use super::{chunk_assembler::ChunkMode, params::SelfPlayParams};
use crate::{inference::runtime::ModelRuntimeConfig, search::params::SearchParams};

/// Operational settings are explicit; algorithm settings inherit Rust defaults.
/// Relative directory paths are relative to the process's working directory.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SelfPlayConfig {
    pub(super) worker_threads: usize,
    pub(super) workers_per_thread: usize,
    pub(super) output_dir: PathBuf,
    pub(super) chunk: ChunkMode,
    pub(super) inference: ModelRuntimeConfig,
    #[serde(default)]
    pub(super) self_play: SelfPlayParams,
    #[serde(default)]
    pub(super) search: SearchParams,
}

#[derive(Debug)]
pub(super) enum ConfigError {
    Read(io::Error),
    Parse(toml::de::Error),
    Invalid(&'static str),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => write!(f, "could not read self-play configuration: {error}"),
            Self::Parse(error) => write!(f, "could not parse self-play configuration: {error}"),
            Self::Invalid(message) => write!(f, "invalid self-play configuration: {message}"),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read(error) => Some(error),
            Self::Parse(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl SelfPlayConfig {
    pub(super) fn load(path: &Path) -> Result<Self, ConfigError> {
        Self::from_toml(&fs::read_to_string(path).map_err(ConfigError::Read)?)
    }

    fn from_toml(source: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(source).map_err(ConfigError::Parse)?;
        config.validate().map_err(ConfigError::Invalid)?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), &'static str> {
        if self.worker_threads == 0 || self.workers_per_thread == 0 {
            return Err("worker_threads and workers_per_thread must be positive");
        }
        if self
            .worker_threads
            .checked_mul(self.workers_per_thread)
            .is_none()
        {
            return Err("self-play worker count overflows usize");
        }
        if self.output_dir.as_os_str().is_empty() || self.inference.model_dir.as_os_str().is_empty()
        {
            return Err("output_dir and model_dir must not be empty");
        }
        if let ChunkMode::FixedRecords(records) = self.chunk {
            if records == 0 || u32::try_from(records).is_err() {
                return Err("fixed chunk record count must fit a positive u32");
            }
        }
        self.inference.validate()?;
        if !self.self_play.rules.komi.is_finite() {
            return Err("komi must be finite");
        }
        self.self_play.search_budget_policy.validate()?;
        self.search.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
