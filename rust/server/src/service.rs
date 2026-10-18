//! Serial storage operations behind bounded local control requests.

use crate::{
    Error,
    replay::{IngestOutcome, ReplayConfig, ReplayStore, storage::sync_directory},
};
use rand::{SeedableRng, rngs::SmallRng};
use rgo_artifacts::{
    ModelVersion, chunk_id_from_filename, model_path, model_version_from_filename,
};
use rgo_protocol::server::{Request, Response, valid_name};
use std::{
    fs, io,
    path::{Path, PathBuf},
};
use tokio::sync::watch;

/// Replay and artifact state owned by the server's storage thread.
pub struct StorageState {
    replay: ReplayStore,
    models: PathBuf,
    shards: PathBuf,
    // Shared model-version snapshot for connection tasks. Recovered from final
    // filenames on startup and updated by producer notifications afterward.
    latest_model_version: watch::Sender<Option<ModelVersion>>,
    rng: SmallRng,
}

impl StorageState {
    pub fn open(data_dir: &Path, replay_config: ReplayConfig) -> Result<Self, Error> {
        let root = rgo_fs::prepare_directory_with_sync(data_dir, sync_directory)?;
        for name in ["replay", "models", "shards"] {
            rgo_fs::prepare_directory_with_sync(&root.join(name), sync_directory)?;
        }
        let replay = ReplayStore::open(&root.join("replay"), replay_config)?;
        let models = root.join("models");
        let latest_model_version = scan_latest_model(&models)?;
        Ok(Self {
            replay,
            models,
            latest_model_version: watch::Sender::new(latest_model_version),
            shards: root.join("shards"),
            // Repeatable local sampling for fixed replay contents and request order.
            rng: SmallRng::seed_from_u64(0),
        })
    }

    /// Stop the storage worker when replay requires a fresh handle.
    pub fn needs_restart(&self) -> bool {
        self.replay.needs_reopen()
    }

    pub fn subscribe_latest_model_version(&self) -> watch::Receiver<Option<ModelVersion>> {
        self.latest_model_version.subscribe()
    }

    pub fn model_directory(&self) -> &Path {
        &self.models
    }

    pub fn handle(&mut self, request: Request) -> Result<Response, Error> {
        match request {
            Request::Hello { .. } | Request::LatestModel {} => {
                Err("request must be handled by the connection task".into())
            }
            Request::AnnounceModel { version } => {
                self.announce_model(version)?;
                Ok(Response::Model {
                    path: (*self.latest_model_version.borrow())
                        .map(|version| model_path(&self.models, version)),
                })
            }
            Request::AcceptChunk { path } => {
                if !path.is_absolute() {
                    return Err("chunk path must be absolute".into());
                }
                let id = path
                    .file_name()
                    .and_then(chunk_id_from_filename)
                    .ok_or("invalid canonical chunk filename")?;
                let outcome = self.replay.ingest_file(id, &path)?;
                let (records, already_accepted) = match outcome {
                    IngestOutcome::Added { records } => (records, false),
                    IngestOutcome::AlreadyAccepted { records } => (records, true),
                };
                Ok(Response::ChunkAccepted {
                    records,
                    already_accepted,
                })
            }
            Request::SampleShard { filename } => {
                let stem = filename
                    .strip_suffix(".npz")
                    .ok_or("shard filename must end in .npz")?;
                if !valid_name(stem) {
                    return Err("invalid shard filename".into());
                }
                let path = self.shards.join(filename);
                match fs::symlink_metadata(&path) {
                    Ok(metadata) if metadata.is_file() => {
                        // A lost reply can be retried using the same filename.
                        // Repeat the publication barrier if an earlier sync failed.
                        sync_directory(&self.shards)?;
                    }
                    Ok(_) => return Err("shard destination is not a regular file".into()),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {
                        self.replay.sample_file(&mut self.rng, &path)?;
                    }
                    Err(error) => return Err(error.into()),
                }
                Ok(Response::Shard { path })
            }
            Request::Stats {} => {
                let stats = self.replay.stats()?;
                Ok(Response::Stats {
                    retained_records: stats.retained_records,
                    retained_chunks: stats.retained_chunks,
                    window_records: stats.window_records,
                })
            }
        }
    }

    fn announce_model(&mut self, version: u64) -> Result<(), Error> {
        if self
            .latest_model_version
            .borrow()
            .is_some_and(|current| version <= current)
        {
            return Ok(());
        }
        // The producer syncs the model, renames it to its final filename, and
        // syncs the directory before notifying us. No separate marker is needed.
        require_regular_model_file(&model_path(&self.models, version))?;
        self.latest_model_version.send_replace(Some(version));
        Ok(())
    }
}

fn scan_latest_model(models: &Path) -> io::Result<Option<ModelVersion>> {
    let mut latest_model_version = None;
    for entry in fs::read_dir(models)? {
        let entry = entry?;
        if let Some(version) = model_version_from_filename(&entry.file_name()) {
            latest_model_version = latest_model_version.max(Some(version));
        }
    }
    if let Some(version) = latest_model_version {
        require_regular_model_file(&model_path(models, version))?;
    }
    Ok(latest_model_version)
}

fn require_regular_model_file(path: &Path) -> io::Result<()> {
    if !fs::symlink_metadata(path)?.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "published models must be regular files",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay_config() -> ReplayConfig {
        ReplayConfig {
            capacity_records: 100,
            window_records: 100,
            sample_group_records: 20,
            sample_groups_per_shard: 4,
            shard_records: 10,
        }
    }

    #[test]
    fn startup_recovers_final_models_and_notifications_never_regress() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("server");
        let mut state = StorageState::open(&root, replay_config()).unwrap();
        let models = state.subscribe_latest_model_version();
        assert!(state.announce_model(1).is_err());
        assert!(!state.needs_restart());
        fs::write(model_path(&state.models, 1), b"model").unwrap();
        fs::write(state.models.join("10.onnx.tmp"), b"unfinished").unwrap();
        state.announce_model(1).unwrap();
        state.announce_model(1).unwrap();
        state.announce_model(0).unwrap();
        assert_eq!(*models.borrow(), Some(1));
        // A completed export survives a lost notification. Compare numerically,
        // and ignore temporary, unrelated, and noncanonical filenames.
        for name in [
            "9.onnx",
            "10.onnx",
            "100.onnx.tmp",
            "notes.onnx",
            "0100.onnx",
        ] {
            fs::write(state.models.join(name), b"model").unwrap();
        }
        drop(state);
        let mut state = StorageState::open(&root, replay_config()).unwrap();
        assert_eq!(*state.latest_model_version.borrow(), Some(10));
        state.announce_model(9).unwrap();
        assert_eq!(*state.latest_model_version.borrow(), Some(10));
        assert!(!state.models.join("latest.json").exists());
    }

    #[cfg(unix)]
    #[test]
    fn retry_of_published_shard_repeats_a_failed_directory_sync() {
        use crate::replay::tests::faults::{self, Operation};
        let temp = tempfile::tempdir().unwrap();
        let mut state = StorageState::open(temp.path(), replay_config()).unwrap();
        let path = state.shards.join("batch.npz");
        // Stand in for a file published immediately before its directory sync
        // failed; the sampling test covers publication and NPZ contents.
        fs::write(&path, b"published shard").unwrap();
        let fault = faults::inject([(Operation::Sync, state.shards.clone())]);
        assert!(
            state
                .handle(Request::SampleShard {
                    filename: "batch.npz".into(),
                })
                .is_err()
        );
        drop(fault);
        assert!(matches!(
            state.handle(Request::SampleShard {
                filename: "batch.npz".into(),
            }),
            Ok(Response::Shard { path: returned }) if returned == path
        ));
        assert_eq!(fs::read(path).unwrap(), b"published shard");
    }

    #[test]
    fn invalid_final_models_fail_without_updating_the_cached_version() {
        let temp = tempfile::tempdir().unwrap();
        let mut state = StorageState::open(temp.path(), replay_config()).unwrap();
        let models = state.subscribe_latest_model_version();
        fs::create_dir(model_path(&state.models, 1)).unwrap();
        assert!(state.announce_model(1).is_err());
        assert_eq!(*models.borrow(), None);
        assert!(!state.needs_restart());
        drop(state);
        assert!(StorageState::open(temp.path(), replay_config()).is_err());
        // An invalid older entry does not prevent using a newer final model.
        fs::write(model_path(&temp.path().join("models"), 2), b"model").unwrap();
        let state = StorageState::open(temp.path(), replay_config()).unwrap();
        assert_eq!(*state.latest_model_version.borrow(), Some(2));
    }

    #[cfg(unix)]
    #[test]
    fn model_symlinks_are_rejected_on_notification_and_restart() {
        let temp = tempfile::tempdir().unwrap();
        let mut state = StorageState::open(temp.path(), replay_config()).unwrap();
        let target = temp.path().join("target");
        fs::write(&target, b"model").unwrap();
        std::os::unix::fs::symlink(target, model_path(&state.models, 1)).unwrap();
        assert!(state.announce_model(1).is_err());
        assert_eq!(*state.latest_model_version.borrow(), None);
        drop(state);
        assert!(StorageState::open(temp.path(), replay_config()).is_err());
        fs::write(model_path(&temp.path().join("models"), 2), b"model").unwrap();
        let state = StorageState::open(temp.path(), replay_config()).unwrap();
        assert_eq!(*state.latest_model_version.borrow(), Some(2));
    }

    #[cfg(unix)]
    #[test]
    fn storage_directory_symlinks_support_startup_models_and_reopening() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("storage");
        fs::create_dir(&root).unwrap();
        let alias = temp.path().join("alias");
        symlink(&root, &alias).unwrap();
        for name in ["replay", "models", "shards"] {
            let target = temp.path().join(format!("{name}-target"));
            fs::create_dir(&target).unwrap();
            symlink(target, root.join(name)).unwrap();
        }
        let mut state = StorageState::open(&alias, replay_config()).unwrap();
        // Different directory names must still share the replay lock.
        assert!(StorageState::open(&root, replay_config()).is_err());
        assert_eq!(
            fs::canonicalize(state.model_directory()).unwrap(),
            fs::canonicalize(temp.path().join("models-target")).unwrap()
        );
        fs::write(model_path(&state.models, 1), b"model").unwrap();
        state.announce_model(1).unwrap();
        assert_eq!(*state.latest_model_version.borrow(), Some(1));
        drop(state);
        let state = StorageState::open(&alias, replay_config()).unwrap();
        assert_eq!(*state.latest_model_version.borrow(), Some(1));
    }

    #[cfg(unix)]
    #[test]
    fn existing_server_storage_under_execute_only_ancestor_opens() {
        use crate::replay::tests::faults::{self, Operation};
        use std::os::unix::fs::PermissionsExt;
        struct Restore(std::path::PathBuf, fs::Permissions);
        impl Drop for Restore {
            fn drop(&mut self) {
                fs::set_permissions(&self.0, self.1.clone()).unwrap();
            }
        }
        let temp = tempfile::tempdir().unwrap();
        let ancestor = fs::canonicalize(temp.path()).unwrap().join("execute-only");
        let root = ancestor.join("server");
        drop(StorageState::open(&root, replay_config()).unwrap());
        let _restore = Restore(
            ancestor.clone(),
            fs::metadata(&ancestor).unwrap().permissions(),
        );
        fs::set_permissions(&ancestor, fs::Permissions::from_mode(0o111)).unwrap();
        // Detect an unnecessary sync even if the test has elevated privileges.
        let _fault = faults::inject([(Operation::Sync, ancestor)]);
        StorageState::open(&root, replay_config()).unwrap();
    }
}
