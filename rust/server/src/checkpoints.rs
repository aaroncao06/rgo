//! Durable handoff of immutable, indexed trainer checkpoints.

use crate::replay::storage::sync_directory;
use serde::Deserialize;
use std::{
    ffi::OsStr,
    fs::{self, File},
    io::{self, ErrorKind},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CheckpointConfig {
    pub prune_old_checkpoints: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            prune_old_checkpoints: true,
        }
    }
}

/// Own an immutable checkpoint inventory accessed by one storage worker.
/// Callers must not change this directory behind the cached latest index.
pub struct CheckpointStore {
    directory: PathBuf,
    config: CheckpointConfig,
    latest_index: Option<i64>,
    failed: bool,
}

impl CheckpointStore {
    /// Recover final filenames once and finish pruning before serving requests.
    pub fn open(directory: PathBuf, config: CheckpointConfig) -> io::Result<Self> {
        let mut latest_index = None;
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            if let Some(index) = checkpoint_index(&entry.file_name()) {
                latest_index = latest_index.max(Some(index));
            }
        }
        let store = Self {
            directory,
            config,
            latest_index,
            failed: false,
        };
        if let Some(path) = store.latest()? {
            require_regular_file(&path)?;
            // Establish the recovered entry before deleting its predecessor.
            sync_directory(&store.directory)?;
        }
        if config.prune_old_checkpoints {
            store.cleanup(latest_index)?;
        }
        Ok(store)
    }

    /// Whether a storage failure has made this handle unusable; the server must stop.
    pub fn failed(&self) -> bool {
        self.failed
    }

    pub fn latest(&self) -> io::Result<Option<PathBuf>> {
        Ok(self
            .latest_index()?
            .map(|index| self.directory.join(format!("{index}.pt"))))
    }

    pub fn latest_index(&self) -> io::Result<Option<i64>> {
        self.ensure_usable()?;
        Ok(self.latest_index)
    }

    /// Hard-link a published local checkpoint, then remove its producer name.
    pub fn accept(&mut self, source: &Path) -> io::Result<()> {
        self.ensure_usable()?;
        if !source.is_absolute() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "checkpoint path must be absolute",
            ));
        }
        let filename = source.file_name().ok_or_else(|| {
            io::Error::new(ErrorKind::InvalidInput, "checkpoint path has no filename")
        })?;
        let index = checkpoint_index(filename).ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidInput,
                "invalid canonical checkpoint filename",
            )
        })?;
        let source_dir = source.parent().unwrap();
        if fs::canonicalize(source_dir)? == fs::canonicalize(&self.directory)? {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "producer checkpoint must be outside server checkpoint storage",
            ));
        }
        // One trainer lineage publishes immutable contents under each index,
        // and its client submits new indices in order. An older/equal index is
        // a lost-ack retry whose source may be gone; no byte comparison is needed.
        if self.latest_index.is_none_or(|current| index > current) {
            let destination = self.directory.join(filename);
            require_regular_file(source)?;
            let file = File::options()
                .read(true)
                .write(cfg!(windows))
                .open(source)?;
            if file.metadata()?.len() == 0 {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "checkpoint must not be empty",
                ));
            }
            // Storage failures are fatal for this handle; startup reconciles
            // any partial effects instead of retrying on a live store.
            self.failed = true;
            file.sync_all()?;
            drop(file);
            fs::hard_link(source, &destination)?;
            sync_directory(&self.directory)?;
            if self.config.prune_old_checkpoints {
                self.cleanup(Some(index))?;
            }
            self.latest_index = Some(index);
        }
        self.failed = true;
        finish_source(source, source_dir)?;
        self.failed = false;
        Ok(())
    }

    fn ensure_usable(&self) -> io::Result<()> {
        if self.failed {
            return Err(io::Error::other(
                "checkpoint storage mutation failed; restart the server before retrying",
            ));
        }
        Ok(())
    }

    fn cleanup(&self, latest_index: Option<i64>) -> io::Result<()> {
        let mut removed = false;
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let name = entry.file_name();
            let old = checkpoint_index(&name).is_some_and(|index| Some(index) != latest_index);
            if old {
                fs::remove_file(entry.path())?;
                removed = true;
            }
        }
        if removed {
            sync_directory(&self.directory)?;
        }
        Ok(())
    }
}

fn finish_source(source: &Path, source_dir: &Path) -> io::Result<()> {
    match fs::remove_file(source) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    sync_directory(source_dir)
}

fn checkpoint_index(filename: &OsStr) -> Option<i64> {
    let name = filename.to_str()?;
    let index: i64 = name.strip_suffix(".pt")?.parse().ok()?;
    (index >= -1 && name == format!("{index}.pt")).then_some(index)
}

fn require_regular_file(path: &Path) -> io::Result<()> {
    if !fs::symlink_metadata(path)?.is_file() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "checkpoint must be a regular file",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(root: &Path, index: i64, bytes: &[u8]) -> PathBuf {
        let directory = root.join("trainer");
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join(format!("{index}.pt"));
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn accepts_indexed_files_and_recovers_the_newest_checkpoint() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("checkpoints");
        fs::create_dir(&directory).unwrap();
        let mut store = CheckpointStore::open(
            directory.clone(),
            CheckpointConfig {
                prune_old_checkpoints: false,
            },
        )
        .unwrap();
        assert_eq!(store.latest().unwrap(), None);

        let initial = source(temp.path(), -1, b"initial");
        store.accept(&initial).unwrap();
        assert!(!initial.exists());
        assert_eq!(fs::read(directory.join("-1.pt")).unwrap(), b"initial");

        let zero = source(temp.path(), 0, b"checkpoint zero");
        store.accept(&zero).unwrap();
        assert!(!zero.exists());
        assert!(directory.join("-1.pt").exists());
        store.accept(&zero).unwrap();

        let one = source(temp.path(), 1, b"checkpoint one");
        store.accept(&one).unwrap();
        store.accept(&zero).unwrap();
        drop(store);
        let store = CheckpointStore::open(
            directory.clone(),
            CheckpointConfig {
                prune_old_checkpoints: false,
            },
        )
        .unwrap();
        assert_eq!(store.latest().unwrap(), Some(directory.join("1.pt")));
        assert_eq!(fs::read(directory.join("1.pt")).unwrap(), b"checkpoint one");
        assert_eq!(fs::read_dir(directory).unwrap().count(), 3);
    }

    #[test]
    fn enabling_pruning_removes_history_but_keeps_the_latest() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("checkpoints");
        fs::create_dir(&directory).unwrap();
        let mut store = CheckpointStore::open(
            directory.clone(),
            CheckpointConfig {
                prune_old_checkpoints: false,
            },
        )
        .unwrap();
        store.accept(&source(temp.path(), 0, b"zero")).unwrap();
        store.accept(&source(temp.path(), 1, b"one")).unwrap();
        assert!(directory.join("0.pt").exists());
        drop(store);

        let store = CheckpointStore::open(directory.clone(), CheckpointConfig::default()).unwrap();
        assert_eq!(store.latest().unwrap(), Some(directory.join("1.pt")));
        assert!(!directory.join("0.pt").exists());
        assert_eq!(fs::read(directory.join("1.pt")).unwrap(), b"one");
    }

    #[test]
    fn rejects_invalid_producer_files() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("checkpoints");
        fs::create_dir(&directory).unwrap();
        let mut store = CheckpointStore::open(
            directory.clone(),
            CheckpointConfig {
                prune_old_checkpoints: false,
            },
        )
        .unwrap();
        assert!(store.accept(Path::new("0.pt")).is_err());
        let invalid = source(temp.path(), 0, b"checkpoint");
        let malformed = invalid.with_file_name("00.pt");
        fs::rename(&invalid, &malformed).unwrap();
        assert!(store.accept(&malformed).is_err());
        let empty = source(temp.path(), 0, b"");
        assert!(store.accept(&empty).is_err());
        fs::write(&empty, b"checkpoint").unwrap();
        store.accept(&empty).unwrap();
        assert!(store.accept(&directory.join("0.pt")).is_err());
        let prefixed = malformed.with_file_name("dummy_0.pt");
        fs::write(&prefixed, b"other checkpoint").unwrap();
        assert!(store.accept(&prefixed).is_err());
        assert!(prefixed.exists());
        assert_eq!(store.latest().unwrap(), Some(directory.join("0.pt")));
    }

    #[cfg(unix)]
    #[test]
    fn local_acceptance_keeps_the_producer_inode() {
        use std::os::unix::fs::MetadataExt;
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("checkpoints");
        fs::create_dir(&directory).unwrap();
        let input = source(temp.path(), 0, b"checkpoint");
        let original = fs::metadata(&input).unwrap();
        CheckpointStore::open(
            directory.clone(),
            CheckpointConfig {
                prune_old_checkpoints: false,
            },
        )
        .unwrap()
        .accept(&input)
        .unwrap();
        let accepted = fs::metadata(directory.join("0.pt")).unwrap();
        assert_eq!(
            (original.dev(), original.ino()),
            (accepted.dev(), accepted.ino())
        );
        assert!(!input.exists());
    }

    #[cfg(unix)]
    #[test]
    fn failed_publication_requires_restart_and_recovers_the_newer_file() {
        use crate::replay::tests::faults::{self, Operation};
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("checkpoints");
        fs::create_dir(&directory).unwrap();
        let mut store =
            CheckpointStore::open(directory.clone(), CheckpointConfig::default()).unwrap();
        let zero = source(temp.path(), 0, b"checkpoint zero");
        store.accept(&zero).unwrap();
        let one = source(temp.path(), 1, b"checkpoint one");
        let fault = faults::inject([(Operation::Sync, directory.clone())]);
        assert!(store.accept(&one).is_err());
        assert!(one.exists());
        assert!(directory.join("0.pt").exists());
        assert!(directory.join("1.pt").exists());
        assert!(store.failed());
        drop(fault);

        // Clearing the injected fault does not make this handle usable again.
        assert!(store.accept(&zero).is_err());
        assert!(store.latest().is_err());
        assert!(directory.join("0.pt").exists());
        drop(store);
        let mut store =
            CheckpointStore::open(directory.clone(), CheckpointConfig::default()).unwrap();
        store.accept(&zero).unwrap();
        assert_eq!(store.latest().unwrap(), Some(directory.join("1.pt")));
        store.accept(&one).unwrap();
        assert!(!one.exists());
        assert_eq!(fs::read(directory.join("1.pt")).unwrap(), b"checkpoint one");
    }

    #[cfg(unix)]
    #[test]
    fn startup_syncs_newer_entry_before_removing_older_one() {
        use crate::replay::tests::faults::{self, Operation};
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("checkpoints");
        fs::create_dir(&directory).unwrap();
        let mut store = CheckpointStore::open(
            directory.clone(),
            CheckpointConfig {
                prune_old_checkpoints: false,
            },
        )
        .unwrap();
        let zero = source(temp.path(), 0, b"checkpoint zero");
        store.accept(&zero).unwrap();
        let one = source(temp.path(), 1, b"checkpoint one");
        let fault = faults::inject([(Operation::Sync, directory.clone())]);
        assert!(store.accept(&one).is_err());
        drop(store);
        drop(fault);

        let fault = faults::inject([(Operation::Sync, directory.clone())]);
        assert!(CheckpointStore::open(directory.clone(), CheckpointConfig::default()).is_err());
        assert!(directory.join("0.pt").exists());
        drop(fault);
        let mut store =
            CheckpointStore::open(directory.clone(), CheckpointConfig::default()).unwrap();
        assert!(!directory.join("0.pt").exists());
        store.accept(&one).unwrap();
        assert!(!one.exists());
    }

    #[cfg(unix)]
    #[test]
    fn source_cleanup_sync_failure_requires_restart() {
        use crate::replay::tests::faults::{self, Operation};
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("checkpoints");
        fs::create_dir(&directory).unwrap();
        let config = CheckpointConfig {
            prune_old_checkpoints: false,
        };
        let mut store = CheckpointStore::open(directory.clone(), config).unwrap();
        let input = source(temp.path(), 0, b"checkpoint");
        let fault = faults::inject([(Operation::Sync, input.parent().unwrap().to_owned())]);
        assert!(store.accept(&input).is_err());
        assert!(!input.exists());
        assert!(directory.join("0.pt").exists());
        assert!(store.failed());
        drop(fault);
        assert!(store.accept(&input).is_err());
        drop(store);
        let mut store = CheckpointStore::open(directory, config).unwrap();
        store.accept(&input).unwrap();
        assert!(!store.failed());
    }
}
