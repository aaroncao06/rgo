//! Chunk ingestion, whole-file FIFO retention, and durable catalog recovery.

use rgo_artifacts::{ChunkId, chunk_id_from_filename, chunk_path};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::{
    collections::{HashSet, VecDeque},
    fs::{self, File},
    io::{self, Read},
    path::Path,
};

use super::{IndexedChunk, IngestOutcome, ReplayConfig, ReplayError, ReplayStore, Result};

impl ReplayStore {
    /// Open or initialize a store in an existing, durably provisioned root.
    /// Load chunk metadata and complete cleanup before returning a handle.
    /// The supplied configuration determines the active window on each open.
    pub fn open(root: &Path, config: ReplayConfig) -> Result<Self> {
        config.validate()?;
        let lock = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("replay.lock"))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => return Err(ReplayError::AlreadyOpen),
            Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
        }
        let chunk_dir = root.join("chunks");
        fs::create_dir_all(&chunk_dir)?;
        sync_directory(root)?;
        let mut db = Connection::open(root.join("replay.sqlite"))?;
        // SQLite owns catalog durability and recovery. EXTRA syncs journal
        // deletion; fullfsync flushes journal/database writes on macOS.
        // We accept residual catalog-corruption risk if power loss follows a
        // directory-sync failure SQLite ignores. Future startup repair could
        // validate surviving chunks and rebuild the catalog, accepting loss of
        // arrival order and receipts for evicted chunks. No rebuild exists yet.
        // Set fullfsync first: preparing synchronous loads the schema and can
        // recover a hot journal, which must fully sync the restored database.
        db.execute_batch(
            "PRAGMA fullfsync = ON;
             PRAGMA synchronous = EXTRA;
             PRAGMA journal_mode = DELETE;",
        )?;
        let transaction = db.transaction()?;
        transaction.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                 seq INTEGER PRIMARY KEY,
                 id BLOB NOT NULL UNIQUE CHECK(length(id) = 16),
                 record_count INTEGER NOT NULL CHECK(record_count > 0),
                 byte_len INTEGER NOT NULL CHECK(byte_len > 0),
                 active INTEGER NOT NULL CHECK(active IN (0, 1))
             );
             CREATE INDEX IF NOT EXISTS active_chunks ON chunks(seq)
                 WHERE active = 1;",
        )?;
        transaction.commit()?;
        let mut store = Self {
            config,
            chunk_dir,
            db,
            chunks: VecDeque::new(),
            retained_records: 0,
            needs_reopen: false,
            _lock: lock,
        };
        store.load_index()?;
        store.set_window(config.window_records)?;
        store.remove_orphans()?;
        Ok(store)
    }

    /// Take ownership of an immutable, durably published regular file on the same
    /// filesystem, outside `chunks/`. Hard-link it into replay, commit acceptance,
    /// then remove the intake filename before returning acknowledgment.
    /// The first accepted ID wins. Retries discard the intake file without
    /// reading it and return the original receipt, including after eviction.
    pub fn ingest_file(&mut self, id: ChunkId, source: &Path) -> Result<IngestOutcome> {
        self.ensure_usable()?;
        let source_dir = source
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let previous: Option<usize> = self
            .db
            .query_row(
                "SELECT record_count FROM chunks WHERE id = ?1",
                [id.to_be_bytes().as_slice()],
                |row| Ok(row.get::<_, u32>(0)? as usize),
            )
            .optional()?;
        // Never let intake cleanup unlink a canonical replay filename, even
        // through a directory alias. Intake is separate from replay storage.
        if fs::canonicalize(source_dir)? == fs::canonicalize(&self.chunk_dir)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "intake files must be outside the replay chunk directory",
            )
            .into());
        }

        let outcome = if let Some(records) = previous {
            IngestOutcome::AlreadyAccepted { records }
        } else {
            // Hard-linking a symlink preserves the link rather than its target.
            if !fs::symlink_metadata(source)?.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "new intake chunks must be regular files",
                )
                .into());
            }
            // Windows' FlushFileBuffers requires a handle with write access.
            let mut file = File::options()
                .read(true)
                .write(cfg!(windows))
                .open(source)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            let indexed = IndexedChunk::from_chunk(id, &bytes)?;
            drop(bytes);
            let records = indexed.records;
            // Any error after mutation begins requires reopening.
            self.needs_reopen = true;

            // Keep intake until the catalog commits. Recovery can remove an
            // uncommitted replay link without losing the producer's only copy.
            file.sync_all()?;
            fs::hard_link(source, chunk_path(&self.chunk_dir, id))?;
            sync_directory(&self.chunk_dir)?;

            let total = self.retained_records as u64 + records as u64;
            let evicted = {
                let transaction = self.db.transaction()?;
                transaction.execute(
                    "INSERT INTO chunks (id, record_count, byte_len, active)
                     VALUES (?1, ?2, ?3, 1)",
                    params![
                        id.to_be_bytes().as_slice(),
                        records as i64,
                        indexed.byte_len as i64
                    ],
                )?;
                let evicted = evict_chunks(
                    &transaction,
                    &self.chunks,
                    total,
                    self.config.window_records,
                )?;
                transaction.commit()?;
                evicted
            };
            self.chunks.push_back(indexed);
            self.finish_eviction(total, evicted)?;
            IngestOutcome::Added { records }
        };
        // Also finish intake cleanup on retries of a committed acceptance.
        // Sync even when the file is already absent: a prior unlink's directory
        // sync may have failed before acknowledgment.
        self.needs_reopen = true;
        remove_if_exists(source)?;
        sync_directory(source_dir)?;
        self.needs_reopen = false;
        Ok(outcome)
    }

    /// Change the soft window target, bounded by capacity. Shrinking evicts
    /// whole oldest chunks; growing permits future arrivals to fill the gap.
    pub fn set_window(&mut self, records: usize) -> Result<()> {
        self.ensure_usable()?;
        if records == 0 || records > self.config.capacity_records {
            return Err(ReplayError::InvalidConfig(
                "window_records must be positive and at most capacity_records",
            ));
        }
        self.needs_reopen = true;
        let transaction = self.db.transaction()?;
        let evicted = evict_chunks(
            &transaction,
            &self.chunks,
            self.retained_records as u64,
            records,
        )?;
        transaction.commit()?;
        self.config.window_records = records;
        self.finish_eviction(self.retained_records as u64, evicted)?;
        self.needs_reopen = false;
        Ok(())
    }

    // The catalog has already committed. If deletion is interrupted, reopening
    // loads the active inventory and removes the remaining inactive files.
    fn finish_eviction(&mut self, mut total: u64, evicted: usize) -> Result<()> {
        for _ in 0..evicted {
            let chunk = self.chunks.pop_front().unwrap();
            remove_if_exists(&chunk_path(&self.chunk_dir, chunk.id))?;
            total -= chunk.records as u64;
        }
        if evicted > 0 {
            sync_directory(&self.chunk_dir)?;
        }
        self.retained_records = total as usize;
        Ok(())
    }

    fn load_index(&mut self) -> Result<()> {
        let mut statement = self.db.prepare(
            "SELECT id, record_count, byte_len FROM chunks
             WHERE active = 1 ORDER BY seq",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let id = decode_id(&row.get::<_, Vec<u8>>(0)?)?;
            let count = row.get::<_, u32>(1)? as usize;
            let byte_len = usize::try_from(row.get::<_, i64>(2)?)
                .map_err(|_| ReplayError::CorruptCatalog("invalid chunk length"))?;
            // Confirm availability without reading or revalidating accepted bytes.
            let metadata = fs::metadata(chunk_path(&self.chunk_dir, id))?;
            if !metadata.is_file() || metadata.len() != byte_len as u64 {
                return Err(ReplayError::CorruptStore { id });
            }
            let indexed = IndexedChunk {
                id,
                byte_len,
                records: count,
            };
            self.retained_records = self
                .retained_records
                .checked_add(indexed.records)
                .ok_or(ReplayError::CorruptStore { id })?;
            self.chunks.push_back(indexed);
        }
        Ok(())
    }

    fn remove_orphans(&self) -> Result<()> {
        let active: HashSet<_> = self.chunks.iter().map(|chunk| chunk.id).collect();
        for entry in fs::read_dir(&self.chunk_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(id) = chunk_id_from_filename(&name) else {
                continue;
            };
            if !active.contains(&id) {
                remove_if_exists(&entry.path())?;
            }
        }
        sync_directory(&self.chunk_dir)?;
        Ok(())
    }
}

fn evict_chunks(
    transaction: &Transaction<'_>,
    chunks: &VecDeque<IndexedChunk>,
    mut total: u64,
    window: usize,
) -> Result<usize> {
    let mut evicted = 0;
    for chunk in chunks {
        let remaining = total - chunk.records as u64;
        // Keep the oldest boundary chunk whole until newer files cover the target.
        if remaining < window as u64 {
            break;
        }
        total = remaining;
        evicted += 1;
    }
    if evicted > 0 {
        let mut update = transaction.prepare("UPDATE chunks SET active = 0 WHERE id = ?1")?;
        for chunk in chunks.iter().take(evicted) {
            update.execute([chunk.id.to_be_bytes().as_slice()])?;
        }
    }
    Ok(evicted)
}

fn decode_id(bytes: &[u8]) -> Result<ChunkId> {
    let bytes: [u8; 16] = bytes
        .try_into()
        .map_err(|_| ReplayError::CorruptCatalog("invalid chunk ID"))?;
    Ok(ChunkId::from_be_bytes(bytes))
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    #[cfg(all(test, unix))]
    super::tests::before_remove(path)?;
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

pub(crate) fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(all(test, unix))]
    super::tests::before_directory_sync(path)?;
    rgo_fs::sync_directory(path)
}
