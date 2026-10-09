use rgo_artifacts::chunk::{CHUNK_HEADER_SIZE, chunk_header, training_record_size};
use rgo_artifacts::chunk_path;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    io::{self, Cursor, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use rand::{SeedableRng, rngs::SmallRng};
use tempfile::TempDir;

use super::chunk_reader::Record;
use super::storage::sync_directory;
use super::*;

#[cfg(unix)]
mod faults {
    use std::cell::RefCell;

    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Operation {
        Sync,
        Remove,
    }

    #[derive(Default)]
    struct State {
        failures: VecDeque<(Operation, PathBuf)>,
        synced: Vec<PathBuf>,
        skip_matches: usize,
    }

    thread_local! {
        static STATE: RefCell<State> = RefCell::new(State::default());
    }

    pub struct Guard;

    impl Drop for Guard {
        fn drop(&mut self) {
            STATE.with(|state| *state.borrow_mut() = State::default());
        }
    }

    pub fn inject(failures: impl IntoIterator<Item = (Operation, PathBuf)>) -> Guard {
        STATE.with(|state| {
            *state.borrow_mut() = State {
                failures: failures.into_iter().collect(),
                ..State::default()
            };
        });
        Guard
    }

    pub fn inject_after(operation: Operation, path: PathBuf, skip_matches: usize) -> Guard {
        let guard = inject([(operation, path)]);
        STATE.with(|state| state.borrow_mut().skip_matches = skip_matches);
        guard
    }

    pub fn before(operation: Operation, path: &Path) -> io::Result<()> {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            if operation == Operation::Sync {
                state.synced.push(path.to_owned());
            }
            if state.failures.front() == Some(&(operation, path.to_owned())) {
                if state.skip_matches > 0 {
                    state.skip_matches -= 1;
                    return Ok(());
                }
                state.failures.pop_front();
                return Err(io::Error::other("injected filesystem failure"));
            }
            Ok(())
        })
    }

    pub fn synced() -> Vec<PathBuf> {
        STATE.with(|state| state.borrow().synced.clone())
    }
}

#[cfg(unix)]
pub(super) fn before_directory_sync(path: &Path) -> io::Result<()> {
    faults::before(faults::Operation::Sync, path)
}

#[cfg(unix)]
pub(super) fn before_remove(path: &Path) -> io::Result<()> {
    faults::before(faults::Operation::Remove, path)
}

fn config(window: usize) -> ReplayConfig {
    ReplayConfig {
        capacity_records: 100,
        window_records: window,
        sample_group_records: 20,
        sample_groups_per_shard: 2,
        shard_records: 2048,
    }
}

fn intake_path(store: &ReplayStore, id: ChunkId) -> PathBuf {
    store
        .chunk_dir
        .parent()
        .unwrap()
        .join(format!("intake-{id:032x}.rgo"))
}

impl ReplayStore {
    // Byte fixtures still exercise the production file-ownership handoff.
    fn ingest(&mut self, id: ChunkId, bytes: &[u8]) -> Result<IngestOutcome> {
        self.ensure_usable()?;
        let source = intake_path(self, id);
        fs::write(&source, bytes)?;
        sync_directory(source.parent().unwrap())?;
        self.ingest_file(id, &source)
    }
}

fn chunk(samples: &[(usize, u8)]) -> Vec<u8> {
    let mut bytes = chunk_header(samples.len()).to_vec();
    for &(dim, tag) in samples {
        let mut record = vec![tag; training_record_size(dim)];
        record[0] = dim as u8;
        bytes.extend_from_slice(&record);
    }
    let checksum = Sha256::digest(&bytes);
    bytes.extend_from_slice(&checksum);
    bytes
}

fn samples(store: &mut ReplayStore, shard_records: usize, seed: u64) -> Vec<u8> {
    // Test-only access lets the same fixtures exercise different configurations.
    assert!(shard_records > 0);
    store.config.shard_records = shard_records;
    let mut bytes = Cursor::new(Vec::new());
    store
        .sample_into(&mut SmallRng::seed_from_u64(seed), &mut bytes)
        .unwrap();
    bytes.into_inner()
}

// Inspect retained storage directly so retention tests don't consume sample passes.
fn tags(store: &ReplayStore) -> BTreeSet<u8> {
    let mut result = BTreeSet::new();
    for chunk in &store.chunks {
        let bytes = fs::read(chunk_path(&store.chunk_dir, chunk.id)).unwrap();
        let mut offset = CHUNK_HEADER_SIZE;
        for _ in 0..chunk.records {
            let record = Record::at(&bytes, offset);
            result.insert(record.spatial_bytes()[0]);
            offset += record.bytes().len();
        }
    }
    result
}

fn array<T: npyz::Deserialize>(bytes: &[u8], name: &str) -> (Vec<u64>, Vec<T>) {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
    let entry = archive
        .by_name(&npyz::npz::file_name_from_array_name(name))
        .unwrap();
    let array = npyz::NpyFile::new(entry).unwrap();
    let shape = array.shape().to_vec();
    (shape, array.into_vec().unwrap())
}

fn shard_tags(bytes: &[u8]) -> Vec<u8> {
    let (_, dims) = array::<u8>(bytes, "board_sizes");
    let mut result = Vec::new();
    for dim in dims {
        let (shape, values) = array::<u8>(bytes, &format!("{dim}/spatial"));
        let row_len = (shape[1] * shape[2]) as usize;
        result.extend(values.chunks_exact(row_len).map(|row| row[0]));
    }
    result
}

fn assert_requires_reopen(store: &mut ReplayStore, root: &Path) {
    assert!(matches!(store.stats(), Err(ReplayError::ReopenRequired)));
    assert!(matches!(
        store.ingest(99, &chunk(&[(9, 99)])),
        Err(ReplayError::ReopenRequired)
    ));
    assert!(matches!(
        store.ingest_file(99, &root.join("missing-source")),
        Err(ReplayError::ReopenRequired)
    ));
    assert!(matches!(
        store.set_window(0),
        Err(ReplayError::ReopenRequired)
    ));
    let mut output = Cursor::new(Vec::new());
    let mut rng = SmallRng::seed_from_u64(7);
    assert!(matches!(
        store.sample_into(&mut rng, &mut output),
        Err(ReplayError::ReopenRequired)
    ));
    assert!(output.get_ref().is_empty());
    let destination = root.join("blocked-shard.rgo");
    assert!(matches!(
        store.sample_file(&mut rng, &destination),
        Err(ReplayError::ReopenRequired)
    ));
    assert!(!destination.exists());
    assert!(!chunk_path(&root.join("chunks"), 99).exists());
}

mod recovery;
mod sampling;
mod shard;
mod storage;
