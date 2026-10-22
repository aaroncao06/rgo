//! Serial storage operations behind bounded local control requests.

use crate::{
    Error,
    checkpoints::{CheckpointConfig, CheckpointStore},
    replay::{ReplayConfig, ReplayStore, storage::sync_directory},
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
    checkpoints: CheckpointStore,
    models: PathBuf,
    shards: PathBuf,
    prune_old_shards: bool,
    // Shared model-version snapshot for connection tasks. Recovered from final
    // filenames on startup; future server-side export will update it directly.
    latest_model_version: watch::Sender<Option<ModelVersion>>,
    rng: SmallRng,
    // Failures in artifact operations outside ReplayStore also stop the server.
    failed: bool,
}

impl StorageState {
    pub fn open(
        data_dir: &Path,
        replay_config: ReplayConfig,
        checkpoint_config: CheckpointConfig,
        prune_old_shards: bool,
    ) -> Result<Self, Error> {
        let root = rgo_fs::prepare_directory_with_sync(data_dir, sync_directory)?;
        for name in ["replay", "models", "shards", "checkpoints"] {
            rgo_fs::prepare_directory_with_sync(&root.join(name), sync_directory)?;
        }
        let replay = ReplayStore::open(&root.join("replay"), replay_config)?;
        let checkpoints = CheckpointStore::open(root.join("checkpoints"), checkpoint_config)?;
        let models = root.join("models");
        let latest_model_version = scan_latest_model(&models)?;
        let mut state = Self {
            replay,
            checkpoints,
            models,
            latest_model_version: watch::Sender::new(latest_model_version),
            shards: root.join("shards"),
            prune_old_shards,
            // Repeatable local sampling for fixed replay contents and request order.
            rng: SmallRng::seed_from_u64(0),
            failed: false,
        };
        if state.prune_old_shards {
            state.prune_checkpointed_shards()?;
        }
        Ok(state)
    }

    /// Stop the storage worker after a replay or artifact storage failure.
    pub fn failed(&self) -> bool {
        self.failed || self.replay.failed() || self.checkpoints.failed()
    }

    pub fn subscribe_latest_model_version(&self) -> watch::Receiver<Option<ModelVersion>> {
        self.latest_model_version.subscribe()
    }

    pub fn model_directory(&self) -> &Path {
        &self.models
    }

    pub fn handle(&mut self, request: Request) -> Result<Response, Error> {
        if self.failed() {
            return Err("storage operation failed; restart the server before retrying".into());
        }
        match request {
            Request::Hello { .. } | Request::GetLatestModel {} => {
                Err("request must be handled by the connection task".into())
            }
            Request::SubmitChunk { path } => {
                if !path.is_absolute() {
                    return Err("chunk path must be absolute".into());
                }
                let id = path
                    .file_name()
                    .and_then(chunk_id_from_filename)
                    .ok_or("invalid canonical chunk filename")?;
                self.replay.ingest_file(id, &path)?;
                Ok(Response::ChunkAccepted {})
            }
            Request::SubmitCheckpoint { path } => {
                self.checkpoints.accept(&path)?;
                if self.prune_old_shards {
                    self.prune_checkpointed_shards()?;
                }
                Ok(Response::CheckpointAccepted {})
            }
            Request::GetLatestCheckpoint {} => Ok(Response::Checkpoint {
                path: self.checkpoints.latest()?,
            }),
            Request::GetTrainingShard { index } => {
                if self
                    .checkpoints
                    .latest_index()?
                    .is_some_and(|latest| i128::from(index) <= i128::from(latest))
                {
                    return Err("shard is already covered by the latest checkpoint".into());
                }
                let path = self.shards.join(format!("{index}.npz"));
                // Any storage error below is fatal. Clear only after success
                // or the normal no-data result.
                self.failed = true;
                let ready = match fs::symlink_metadata(&path) {
                    Ok(metadata) if metadata.is_file() => {
                        // A lost reply reuses the published file. Repeat the
                        // directory barrier before returning its path.
                        sync_directory(&self.shards)?;
                        true
                    }
                    Ok(_) => return Err("shard destination is not a regular file".into()),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {
                        self.replay.sample_shard(&mut self.rng, &path)?
                    }
                    Err(error) => return Err(error.into()),
                };
                self.failed = false;
                Ok(Response::TrainingShard {
                    path: ready.then_some(path),
                })
            }
            Request::GetReplayStats {} => {
                let stats = self.replay.stats()?;
                Ok(Response::Stats {
                    retained_records: stats.retained_records,
                    retained_chunks: stats.retained_chunks,
                    window_records: stats.window_records,
                })
            }
        }
    }

    /// Accepted checkpoints include every shard through their index. Finish
    /// cleanup before acknowledgment, and repeat it on startup after interruption.
    fn prune_checkpointed_shards(&mut self) -> Result<(), Error> {
        let Some(latest) = self.checkpoints.latest_index()?.filter(|&index| index >= 0) else {
            return Ok(());
        };
        self.failed = true;
        for entry in fs::read_dir(&self.shards)? {
            let entry = entry?;
            let filename = entry.file_name();
            let Some(name) = filename.to_str() else {
                continue;
            };
            let Some(index) = name
                .strip_suffix(".npz")
                .and_then(|stem| stem.parse::<u64>().ok())
            else {
                continue;
            };
            if index <= latest as u64 && name == format!("{index}.npz") {
                fs::remove_file(entry.path())?;
            }
        }
        // Also complete a previous deletion's barrier when no files remain.
        sync_directory(&self.shards)?;
        self.failed = false;
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
    use crate::replay::ReplayError;
    use rgo_artifacts::chunk::{chunk_header, training_record_size};
    use sha2::{Digest, Sha256};

    fn replay_config() -> ReplayConfig {
        ReplayConfig {
            capacity_records: 100,
            window_records: 100,
            prune_old_chunks: true,
            sample_group_records: 20,
            sample_groups_per_shard: 4,
            records_per_shard: 1,
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
        match state.handle(Request::GetTrainingShard { index }).unwrap() {
            Response::TrainingShard { path: Some(path) } => path,
            response => panic!("unexpected response: {response:?}"),
        }
    }

    #[test]
    fn checkpoints_prune_only_covered_shards_and_reject_requests_for_them() {
        for prune_old_checkpoints in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path().join("server");
            let mut state = StorageState::open(
                &root,
                replay_config(),
                CheckpointConfig {
                    prune_old_checkpoints,
                },
                true,
            )
            .unwrap();
            add_test_chunk(&mut state, temp.path(), 1, 1);
            let paths: Vec<_> = (0..3).map(|index| shard_path(&mut state, index)).collect();
            let prefetched = fs::read(&paths[2]).unwrap();
            for name in [
                "01.npz",
                "-1.npz",
                "notes.npz",
                "0.npz.tmp",
                "18446744073709551615.npz",
            ] {
                fs::write(state.shards.join(name), b"keep").unwrap();
            }
            for index in [-1, 0, 1] {
                let source = temp.path().join(format!("{index}.pt"));
                fs::write(&source, b"").unwrap();
                assert!(
                    state
                        .handle(Request::SubmitCheckpoint {
                            path: source.clone()
                        })
                        .is_err()
                );
                assert!(!state.failed());
                assert!(paths[(index + 1) as usize].exists());
                fs::write(&source, b"checkpoint").unwrap();
                assert!(matches!(
                    state
                        .handle(Request::SubmitCheckpoint {
                            path: source.clone()
                        })
                        .unwrap(),
                    Response::CheckpointAccepted {}
                ));
                // Retry a lost acknowledgment after producer-name removal.
                assert!(!source.exists());
                state
                    .handle(Request::SubmitCheckpoint { path: source })
                    .unwrap();
                for shard in 0..=index {
                    assert!(!paths[shard as usize].exists());
                    assert!(
                        state
                            .handle(Request::GetTrainingShard {
                                index: shard as u64
                            })
                            .is_err()
                    );
                    assert!(!state.failed());
                }
            }
            // An older checkpoint retry must not move the cleanup boundary back.
            state
                .handle(Request::SubmitCheckpoint {
                    path: temp.path().join("0.pt"),
                })
                .unwrap();
            assert!(
                state
                    .handle(Request::GetTrainingShard { index: 1 })
                    .is_err()
            );
            assert_eq!(shard_path(&mut state, 2), paths[2]);
            assert_eq!(fs::read(&paths[2]).unwrap(), prefetched);
            for name in [
                "01.npz",
                "-1.npz",
                "notes.npz",
                "0.npz.tmp",
                "18446744073709551615.npz",
            ] {
                assert_eq!(fs::read(state.shards.join(name)).unwrap(), b"keep");
            }
            assert_eq!(
                root.join("checkpoints/-1.pt").exists(),
                !prune_old_checkpoints
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn startup_finishes_partial_shard_cleanup_after_syncing_the_checkpoint() {
        use crate::replay::tests::faults::{self, Operation};
        let temp = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(temp.path()).unwrap().join("server");
        let mut state =
            StorageState::open(&root, replay_config(), CheckpointConfig::default(), true).unwrap();
        add_test_chunk(&mut state, temp.path(), 1, 1);
        let paths: Vec<_> = (0..3).map(|index| shard_path(&mut state, index)).collect();
        let source = temp.path().join("1.pt");
        fs::write(&source, b"checkpoint").unwrap();
        // Stop after durable acceptance and partway through shard deletion.
        state.checkpoints.accept(&source).unwrap();
        fs::remove_file(&paths[0]).unwrap();
        drop(state);
        let fault = faults::inject([(Operation::Sync, root.join("checkpoints"))]);
        assert!(
            StorageState::open(&root, replay_config(), CheckpointConfig::default(), true).is_err()
        );
        assert!(paths[1].exists());
        assert!(paths[2].exists());
        drop(fault);
        let mut state =
            StorageState::open(&root, replay_config(), CheckpointConfig::default(), true).unwrap();
        assert!(!paths[0].exists());
        assert!(!paths[1].exists());
        assert_eq!(shard_path(&mut state, 2), paths[2]);
        assert!(
            state
                .handle(Request::GetTrainingShard { index: 1 })
                .is_err()
        );
    }

    #[test]
    fn shard_retention_survives_acceptance_and_restart_until_pruning_is_enabled() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("server");
        let mut state =
            StorageState::open(&root, replay_config(), CheckpointConfig::default(), false).unwrap();
        add_test_chunk(&mut state, temp.path(), 1, 1);
        let covered = shard_path(&mut state, 0);
        let prefetched = shard_path(&mut state, 1);
        let bytes = fs::read(&covered).unwrap();
        let source = temp.path().join("0.pt");
        fs::write(&source, b"checkpoint").unwrap();
        state
            .handle(Request::SubmitCheckpoint { path: source })
            .unwrap();
        assert_eq!(fs::read(&covered).unwrap(), bytes);
        assert!(
            state
                .handle(Request::GetTrainingShard { index: 0 })
                .is_err()
        );
        assert!(!state.failed());
        drop(state);

        let state =
            StorageState::open(&root, replay_config(), CheckpointConfig::default(), false).unwrap();
        assert_eq!(fs::read(&covered).unwrap(), bytes);
        assert!(prefetched.exists());
        drop(state);

        let mut state =
            StorageState::open(&root, replay_config(), CheckpointConfig::default(), true).unwrap();
        assert!(!covered.exists());
        assert_eq!(shard_path(&mut state, 1), prefetched);
    }

    #[cfg(unix)]
    #[test]
    fn shard_cleanup_errors_are_fatal_and_restart_completes_cleanup() {
        use crate::replay::tests::faults::{self, Operation};
        for fail_sync in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path().join("server");
            let mut state =
                StorageState::open(&root, replay_config(), CheckpointConfig::default(), true)
                    .unwrap();
            let covered = state.shards.join("0.npz");
            if fail_sync {
                fs::write(&covered, b"shard").unwrap();
            } else {
                // A canonical filename occupied by a directory cannot be unlinked.
                fs::create_dir(&covered).unwrap();
            }
            let source = temp.path().join("0.pt");
            fs::write(&source, b"checkpoint").unwrap();
            let fault = faults::inject(if fail_sync {
                vec![(Operation::Sync, state.shards.clone())]
            } else {
                vec![]
            });
            assert!(
                state
                    .handle(Request::SubmitCheckpoint {
                        path: source.clone()
                    })
                    .is_err()
            );
            assert!(state.failed());
            assert!(!source.exists());
            assert!(root.join("checkpoints/0.pt").exists());
            assert!(state.handle(Request::GetLatestCheckpoint {}).is_err());
            drop(fault);
            drop(state);
            if !fail_sync {
                fs::remove_dir(&covered).unwrap();
            }
            let fault = faults::inject([]);
            let mut state =
                StorageState::open(&root, replay_config(), CheckpointConfig::default(), true)
                    .unwrap();
            assert!(!covered.exists());
            assert!(faults::synced().contains(&state.shards));
            state
                .handle(Request::SubmitCheckpoint { path: source })
                .unwrap();
            assert!(!state.failed());
            drop(fault);
        }
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_sync_failures_require_restart_and_recover() {
        use crate::replay::tests::faults::{self, Operation};
        for source_cleanup in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let root = fs::canonicalize(temp.path()).unwrap().join("server");
            let mut state =
                StorageState::open(&root, replay_config(), CheckpointConfig::default(), true)
                    .unwrap();
            let source = root.parent().unwrap().join("0.pt");
            fs::write(&source, b"checkpoint").unwrap();
            let shard = state.shards.join("0.npz");
            fs::write(&shard, b"shard").unwrap();
            let directory = if source_cleanup {
                source.parent().unwrap().to_owned()
            } else {
                root.join("checkpoints")
            };
            let fault = faults::inject([(Operation::Sync, directory)]);
            assert!(
                state
                    .handle(Request::SubmitCheckpoint {
                        path: source.clone()
                    })
                    .is_err()
            );
            assert!(state.failed());
            assert!(!state.replay.failed());
            assert!(root.join("checkpoints/0.pt").exists());
            assert!(shard.exists());
            assert_eq!(source.exists(), !source_cleanup);
            drop(fault);
            assert!(state.handle(Request::GetLatestCheckpoint {}).is_err());
            assert!(state.handle(Request::GetReplayStats {}).is_err());
            drop(state);

            let mut state =
                StorageState::open(&root, replay_config(), CheckpointConfig::default(), true)
                    .unwrap();
            assert!(!shard.exists());
            assert!(matches!(
                state
                    .handle(Request::SubmitCheckpoint { path: source })
                    .unwrap(),
                Response::CheckpointAccepted {}
            ));
            assert!(matches!(
                state.handle(Request::GetLatestCheckpoint {}).unwrap(),
                Response::Checkpoint { path: Some(path) } if path == root.join("checkpoints/0.pt")
            ));
            assert!(!state.failed());
        }
    }

    #[test]
    fn full_shards_wait_for_replay_and_existing_shards_remain_available() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("server");
        let config = ReplayConfig {
            records_per_shard: 3,
            ..replay_config()
        };
        let mut state =
            StorageState::open(&root, config, CheckpointConfig::default(), true).unwrap();
        for id in 0..3 {
            assert!(matches!(
                state
                    .handle(Request::GetTrainingShard { index: 0 })
                    .unwrap(),
                Response::TrainingShard { path: None }
            ));
            assert!(!state.failed());
            assert_eq!(fs::read_dir(&state.shards).unwrap().count(), 0);
            add_test_chunk(&mut state, temp.path(), id, id as u8);
        }
        let path = shard_path(&mut state, 0);
        let bytes = fs::read(&path).unwrap();
        state.replay.set_window(1).unwrap();
        assert_eq!(state.replay.stats().unwrap().retained_records, 1);
        assert_eq!(shard_path(&mut state, 0), path);
        assert!(matches!(
            state
                .handle(Request::GetTrainingShard { index: 1 })
                .unwrap(),
            Response::TrainingShard { path: None }
        ));
        drop(state);
        let mut state =
            StorageState::open(&root, config, CheckpointConfig::default(), true).unwrap();
        assert_eq!(shard_path(&mut state, 0), path);
        assert_eq!(fs::read(path).unwrap(), bytes);
        assert!(matches!(
            state.handle(Request::GetReplayStats {}).unwrap(),
            Response::Stats {
                retained_records: 1,
                ..
            }
        ));
    }

    #[test]
    fn missing_shard_directory_requires_restart_before_sampling() {
        let temp = tempfile::tempdir().unwrap();
        let mut state = StorageState::open(
            temp.path(),
            replay_config(),
            CheckpointConfig::default(),
            true,
        )
        .unwrap();
        add_test_chunk(&mut state, temp.path(), 1, 1);
        fs::remove_dir(&state.shards).unwrap();
        let error = state
            .handle(Request::GetTrainingShard { index: 0 })
            .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<ReplayError>(),
            Some(ReplayError::Io(error)) if error.kind() == io::ErrorKind::NotFound
        ));
        assert!(state.failed());
        assert!(state.handle(Request::GetReplayStats {}).is_err());
    }

    #[test]
    fn indexed_shards_reuse_published_files_across_retries_and_restart() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("server");
        let mut config = replay_config();
        config.window_records = 1;
        let mut state =
            StorageState::open(&root, config, CheckpointConfig::default(), true).unwrap();
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

        let mut state =
            StorageState::open(&root, config, CheckpointConfig::default(), true).unwrap();
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
    fn new_and_existing_shard_sync_errors_require_restart() {
        use crate::replay::tests::faults::{self, Operation};
        for existing in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path().join("server");
            let mut state =
                StorageState::open(&root, replay_config(), CheckpointConfig::default(), true)
                    .unwrap();
            add_test_chunk(&mut state, temp.path(), 1, 1);
            let path = state.shards.join("0.npz");
            if existing {
                assert_eq!(shard_path(&mut state, 0), path);
            }
            let fault = faults::inject([(Operation::Sync, state.shards.clone())]);
            assert!(
                state
                    .handle(Request::GetTrainingShard { index: 0 })
                    .is_err()
            );
            assert!(state.failed());
            let bytes = fs::read(&path).unwrap();
            drop(fault);
            assert!(
                state
                    .handle(Request::GetTrainingShard { index: 0 })
                    .is_err()
            );
            drop(state);

            let mut state =
                StorageState::open(&root, replay_config(), CheckpointConfig::default(), true)
                    .unwrap();
            assert_eq!(shard_path(&mut state, 0), path);
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
    }

    #[test]
    fn startup_recovers_highest_final_model() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("server");
        let state =
            StorageState::open(&root, replay_config(), CheckpointConfig::default(), true).unwrap();
        let models = state.subscribe_latest_model_version();
        assert_eq!(*models.borrow(), None);
        fs::write(model_path(&state.models, 1), b"model").unwrap();
        fs::write(state.models.join("10.onnx.tmp"), b"unfinished").unwrap();
        // Compare numerically, and ignore temporary, unrelated, and
        // noncanonical filenames.
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
        let state =
            StorageState::open(&root, replay_config(), CheckpointConfig::default(), true).unwrap();
        assert_eq!(*state.latest_model_version.borrow(), Some(10));
        assert!(!state.models.join("latest.json").exists());
    }

    #[test]
    fn invalid_highest_final_model_fails_startup() {
        let temp = tempfile::tempdir().unwrap();
        let state = StorageState::open(
            temp.path(),
            replay_config(),
            CheckpointConfig::default(),
            true,
        )
        .unwrap();
        fs::create_dir(model_path(&state.models, 1)).unwrap();
        drop(state);
        assert!(
            StorageState::open(
                temp.path(),
                replay_config(),
                CheckpointConfig::default(),
                true
            )
            .is_err()
        );
        // An invalid older entry does not prevent using a newer final model.
        fs::write(model_path(&temp.path().join("models"), 2), b"model").unwrap();
        let state = StorageState::open(
            temp.path(),
            replay_config(),
            CheckpointConfig::default(),
            true,
        )
        .unwrap();
        assert_eq!(*state.latest_model_version.borrow(), Some(2));
    }

    #[cfg(unix)]
    #[test]
    fn model_symlinks_are_rejected_on_restart() {
        let temp = tempfile::tempdir().unwrap();
        let state = StorageState::open(
            temp.path(),
            replay_config(),
            CheckpointConfig::default(),
            true,
        )
        .unwrap();
        let target = temp.path().join("target");
        fs::write(&target, b"model").unwrap();
        std::os::unix::fs::symlink(target, model_path(&state.models, 1)).unwrap();
        drop(state);
        assert!(
            StorageState::open(
                temp.path(),
                replay_config(),
                CheckpointConfig::default(),
                true
            )
            .is_err()
        );
        fs::write(model_path(&temp.path().join("models"), 2), b"model").unwrap();
        let state = StorageState::open(
            temp.path(),
            replay_config(),
            CheckpointConfig::default(),
            true,
        )
        .unwrap();
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
        let state =
            StorageState::open(&alias, replay_config(), CheckpointConfig::default(), true).unwrap();
        // Different directory names must still share the replay lock.
        assert!(
            StorageState::open(&root, replay_config(), CheckpointConfig::default(), true).is_err()
        );
        assert_eq!(
            fs::canonicalize(state.model_directory()).unwrap(),
            fs::canonicalize(temp.path().join("models-target")).unwrap()
        );
        fs::write(model_path(&state.models, 1), b"model").unwrap();
        drop(state);
        let state =
            StorageState::open(&alias, replay_config(), CheckpointConfig::default(), true).unwrap();
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
        drop(
            StorageState::open(&root, replay_config(), CheckpointConfig::default(), true).unwrap(),
        );
        let _restore = Restore(
            ancestor.clone(),
            fs::metadata(&ancestor).unwrap().permissions(),
        );
        fs::set_permissions(&ancestor, fs::Permissions::from_mode(0o111)).unwrap();
        // Detect an unnecessary sync even if the test has elevated privileges.
        let _fault = faults::inject([(Operation::Sync, ancestor)]);
        StorageState::open(&root, replay_config(), CheckpointConfig::default(), true).unwrap();
    }
}
