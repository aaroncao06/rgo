//! Whole-chunk FIFO replay storage and shuffled group sampling.
//!
//! The store owns its directory and serializes mutations through one handle.
//! File I/O and SQLite calls are synchronous; a future async server should run
//! them on a storage worker. Transport, scheduling, and neural compute belong
//! outside this module.

use self::chunk_reader::{ChunkError, validate_chunk};
use rgo_artifacts::ChunkId;
use rusqlite::Connection;
use std::{collections::VecDeque, error::Error, fmt, fs::File, io, path::PathBuf};

mod chunk_reader;
mod sampling;
mod shard;
mod storage;

#[derive(Debug, Clone, Copy)]
pub struct ReplayConfig {
    /// Upper bound for the configured window target, in records.
    pub capacity_records: usize,
    /// Target retained records, rounded up to whole newest chunks.
    pub window_records: usize,
    /// Soft input-record target per group, rounded up to whole chunks.
    pub sample_group_records: usize,
    /// Maximum fresh, disjoint input groups contributing to one shard.
    pub sample_groups_per_shard: usize,
    /// Target output rows per shard; fewer if selected input is insufficient.
    pub shard_records: usize,
}

impl ReplayConfig {
    fn validate(&self) -> Result<()> {
        if self.capacity_records == 0 || self.capacity_records as u128 > i64::MAX as u128 {
            return Err(ReplayError::InvalidConfig(
                "capacity_records must fit a positive i64",
            ));
        }
        if self.window_records == 0 || self.window_records > self.capacity_records {
            return Err(ReplayError::InvalidConfig(
                "window_records must be positive and at most capacity_records",
            ));
        }
        if self.sample_group_records == 0 {
            return Err(ReplayError::InvalidConfig(
                "sample_group_records must be positive",
            ));
        }
        if self.sample_groups_per_shard == 0 {
            return Err(ReplayError::InvalidConfig(
                "sample_groups_per_shard must be positive",
            ));
        }
        if self.shard_records == 0 {
            return Err(ReplayError::InvalidConfig("shard_records must be positive"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayStats {
    pub retained_records: usize,
    pub retained_chunks: usize,
    pub window_records: usize,
    pub capacity_records: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestOutcome {
    Added { records: usize },
    AlreadyAccepted { records: usize },
}

// Catalog metadata only; tensor bytes and record offsets live in the sampled group.
struct IndexedChunk {
    id: ChunkId,
    byte_len: usize,
    records: usize,
}

impl IndexedChunk {
    fn from_chunk(id: ChunkId, bytes: &[u8]) -> Result<Self> {
        validate_chunk(bytes)?;
        Ok(Self {
            id,
            byte_len: bytes.len(),
            records: u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize,
        })
    }
}

/// A persistent replay store with exclusive ownership of its storage directory.
///
/// Accepted chunks are hard-linked into `chunks/`; intake filenames are removed
/// after durable registration. SQLite records acceptance and FIFO order. Retry
/// receipts survive eviction and restart, so old retries never reintroduce
/// records. Those small receipts grow with accepted chunks; the active record
/// window and chunk files are bounded separately.
///
/// Retention rounds the window target up to whole chunks, so overshoot is less
/// than one retained chunk's record count. All records in retained files are
/// eligible for sampling. Intake also stages one new chunk before eviction;
/// a storage mutation failure blocks further use until reopening. Evicted
/// records cannot return when the window subsequently grows. Callers must not
/// mutate this store's files or database.
///
/// The root directory must already exist and be durably provisioned by the
/// caller. Recovery runs only during open: failed opens return no usable handle.
/// After a failed mutation, drop the handle and reopen before retrying; the
/// failed operation may have committed, so use the same chunk ID on retry.
///
/// ```no_run
/// use std::path::Path;
/// use rand::{SeedableRng, rngs::SmallRng};
/// use crate::replay::{ReplayConfig, ReplayStore};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Provision and durably persist the "replay" directory before opening it.
/// let mut replay = ReplayStore::open(Path::new("replay"), ReplayConfig {
///     capacity_records: 5_000_000,
///     window_records: 250_000,
///     sample_group_records: 200_000,
///     sample_groups_per_shard: 4,
///     shard_records: 100_000,
/// })?;
/// replay.ingest_file(1, Path::new("selfplay/chunk.rgo"))?;
/// // The caller chooses and maintains the RNG; this example is reproducible.
/// let mut rng = SmallRng::seed_from_u64(7);
/// replay.sample_file(&mut rng, Path::new("training/shard.npz"))?;
/// # Ok(())
/// # }
/// ```
pub struct ReplayStore {
    config: ReplayConfig,
    chunk_dir: PathBuf,
    db: Connection,
    chunks: VecDeque<IndexedChunk>,
    retained_records: usize,
    needs_reopen: bool,
    _lock: File,
}

impl ReplayStore {
    /// Report current counts, or require reopening after a failed mutation.
    pub fn stats(&self) -> Result<ReplayStats> {
        self.ensure_usable()?;
        Ok(ReplayStats {
            retained_records: self.retained_records,
            retained_chunks: self.chunks.len(),
            window_records: self.config.window_records,
            capacity_records: self.config.capacity_records,
        })
    }

    fn ensure_usable(&self) -> Result<()> {
        if self.needs_reopen {
            return Err(ReplayError::ReopenRequired);
        }
        Ok(())
    }
}

type Result<T> = std::result::Result<T, ReplayError>;

#[derive(Debug)]
pub enum ReplayError {
    Io(io::Error),
    Database(rusqlite::Error),
    InvalidChunk(ChunkError),
    InvalidConfig(&'static str),
    CorruptCatalog(&'static str),
    AlreadyOpen,
    ReopenRequired,
    CorruptStore { id: ChunkId },
    EmptyReplay,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "replay I/O failed: {error}"),
            Self::Database(error) => write!(f, "replay database failed: {error}"),
            Self::InvalidChunk(error) => write!(f, "invalid replay chunk: {error}"),
            Self::InvalidConfig(message) => write!(f, "invalid replay configuration: {message}"),
            Self::CorruptCatalog(message) => write!(f, "invalid replay catalog: {message}"),
            Self::AlreadyOpen => f.write_str("replay store is already open"),
            Self::ReopenRequired => {
                f.write_str("replay store must be dropped and reopened after a failed mutation")
            }
            Self::CorruptStore { id } => {
                write!(f, "stored chunk {id:032x} does not match its catalog entry")
            }
            Self::EmptyReplay => f.write_str("cannot sample an empty replay store"),
        }
    }
}

impl Error for ReplayError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Database(error) => Some(error),
            Self::InvalidChunk(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for ReplayError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
impl From<rusqlite::Error> for ReplayError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error)
    }
}
impl From<ChunkError> for ReplayError {
    fn from(error: ChunkError) -> Self {
        Self::InvalidChunk(error)
    }
}

#[cfg(test)]
mod tests;
