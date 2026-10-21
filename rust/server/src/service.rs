//! Serial storage operations behind bounded local control requests.

use crate::{
    Error,
    replay::{IngestOutcome, ReplayConfig, ReplayStore, storage::sync_directory},
};
use rand::{SeedableRng, rngs::SmallRng};
use rgo_artifacts::{
    ModelVersion, chunk_id_from_filename, model_path, model_version_from_filename,
};
use rgo_protocol::client_server::{Request, Response};
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
        self.replay.failed()
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
            Request::Shard { index } => {
                let path = self.shards.join(format!("{index}.npz"));
                match fs::symlink_metadata(&path) {
                    Ok(metadata) if metadata.is_file() => {
                        // A lost reply can be retried using the same file. Repeat
                        // the barrier if an earlier directory sync failed.
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
    use rgo_artifacts::chunk::{chunk_header, training_record_size};
    use sha2::{Digest, Sha256};

    fn replay_config() -> ReplayConfig {
        ReplayConfig {
            capacity_records: 100,
            window_records: 100,
            prune_old_chunks: true,
            sample_group_records: 20,
            sample_groups_per_shard: 4,
            shard_records: 10,
        }
    }

    fn add_test_chunk(state: &mut StorageState, directory: &Path, id: u128, tag: u8) {
        let mut bytes = chunk_header(1).to_vec();
        let mut record = vec![tag; training_record_size(9)];
        record[0] = 9;
        bytes.extend(record);
        bytes.extend(Sha256::digest(&bytes));
        let source = directory.join(format!("producer-{id}.rgo"));
        fs::write(&source, bytes).unwrap();
        sync_directory(directory).unwrap();
        state.replay.ingest_file(id, &source).unwrap();
    }

    fn shard_path(state: &mut StorageState, index: u64) -> PathBuf {
        match state.handle(Request::Shard { index }).unwrap() {
            Response::Shard { path } => path,
            response => panic!("unexpected response: {response:?}"),
        }
    }

    #[test]
    fn indexed_shards_reuse_published_files_across_retries_and_restart() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("server");
        let mut config = replay_config();
        config.window_records = 1;
        let mut state = StorageState::open(&root, config).unwrap();
        add_test_chunk(&mut state, temp.path(), 1, 1);

        let path = shard_path(&mut state, 0);
        assert_eq!(path, state.shards.join("0.npz"));
        let bytes = fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"PK"));
        assert_eq!(shard_path(&mut state, 0), path);
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(fs::read_dir(&state.shards).unwrap().count(), 1);

        // Requests can run ahead of training to prefetch multiple shards.
        let next_path = shard_path(&mut state, 1);
        let next_bytes = fs::read(&next_path).unwrap();
        let third_path = shard_path(&mut state, 2);
        let third_bytes = fs::read(&third_path).unwrap();

        add_test_chunk(&mut state, temp.path(), 2, 2);
        assert_eq!(state.replay.stats().unwrap().retained_chunks, 1);
        drop(state);

        let mut state = StorageState::open(&root, config).unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        // A checkpoint through shard 0 must replay later, uncheckpointed shards.
        assert_eq!(shard_path(&mut state, 1), next_path);
        assert_eq!(fs::read(&next_path).unwrap(), next_bytes);
        assert_eq!(shard_path(&mut state, 2), third_path);
        assert_eq!(fs::read(&third_path).unwrap(), third_bytes);
        let new_path = shard_path(&mut state, 3);
        assert_ne!(fs::read(new_path).unwrap(), bytes);
        assert!(path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn indexed_shard_retries_after_publication_sync_error_and_restart() {
        use crate::replay::tests::faults::{self, Operation};
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("server");
        let mut state = StorageState::open(&root, replay_config()).unwrap();
        add_test_chunk(&mut state, temp.path(), 1, 1);
        let path = state.shards.join("0.npz");
        let fault = faults::inject([(Operation::Sync, state.shards.clone())]);
        assert!(state.handle(Request::Shard { index: 0 }).is_err());
        assert!(path.exists());
        drop(fault);
        assert_eq!(shard_path(&mut state, 0), path);
        drop(state);

        let mut state = StorageState::open(&root, replay_config()).unwrap();
        assert_eq!(shard_path(&mut state, 0), path);
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
