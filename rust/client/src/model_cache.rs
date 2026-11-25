//! Client-owned links to immutable models within the selected run.

use rgo_artifacts::{ModelVersion, model_path, model_version_from_filename};
use rgo_fs::sync_directory;
use std::{
    fs::{self, File},
    io::{self, ErrorKind},
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::oneshot;

pub struct ModelCache {
    directory: PathBuf,
    // A detached filesystem call must still exclude a restarted client.
    _client_lock: File,
    // Fields drop in order. Notify the supervisor after releasing the lock.
    _closed: oneshot::Sender<()>,
}

impl ModelCache {
    pub fn new(directory: PathBuf, client_lock: File) -> (Arc<Self>, oneshot::Receiver<()>) {
        let (closed, wait_for_close) = oneshot::channel();
        let cache = Arc::new(Self {
            directory,
            _client_lock: client_lock,
            _closed: closed,
        });
        (cache, wait_for_close)
    }

    pub async fn acquire(
        self: &Arc<Self>,
        source: PathBuf,
        version: ModelVersion,
    ) -> io::Result<Option<PathBuf>> {
        self.run_blocking(move |directory| acquire(directory, &source, version))
            .await
    }

    async fn run_blocking<T: Send + 'static>(
        self: &Arc<Self>,
        operation: impl FnOnce(&Path) -> io::Result<T> + Send + 'static,
    ) -> io::Result<T> {
        let cache = Arc::clone(self);
        tokio::task::spawn_blocking(move || operation(&cache.directory)).await?
    }
}

/// Reuse or acquire the version before giving the worker its local path.
/// Published models are regular, self-contained `.onnx` files with embedded weights.
/// A missing source may have been pruned since the server returned its path.
fn acquire(directory: &Path, source: &Path, version: ModelVersion) -> io::Result<Option<PathBuf>> {
    let local = model_path(directory, version);
    match fs::symlink_metadata(&local) {
        Ok(metadata) if metadata.is_file() => {}
        Ok(_) => {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("cached model {} must be a regular file", local.display()),
            ));
        }
        Err(error) if error.kind() == ErrorKind::NotFound => match fs::hard_link(source, &local) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        },
        Err(error) => return Err(error),
    }
    // Make the replacement durable before removing older links.
    sync_directory(directory)?;
    prune_older(directory, version)?;
    Ok(Some(local))
}

fn prune_older(directory: &Path, current: ModelVersion) -> io::Result<()> {
    const ERROR_SHARING_VIOLATION: i32 = 32;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if model_version_from_filename(&entry.file_name()).is_some_and(|version| version < current)
        {
            match fs::remove_file(entry.path()) {
                // A subsequent acquisition sweeps older links again.
                Err(error)
                    if cfg!(windows) && error.raw_os_error() == Some(ERROR_SHARING_VIOLATION) => {}
                result => result?,
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[tokio::test]
async fn timed_out_cache_io_keeps_its_lock_until_the_call_returns() {
    use std::{fs::TryLockError, sync::mpsc, time::Duration};
    use tokio::time;

    let temp = tempfile::tempdir().unwrap();
    let lock_path = temp.path().join(".client.lock");
    let lock = File::create(&lock_path).unwrap();
    lock.try_lock().unwrap();
    let (cache, mut cache_closed) = ModelCache::new(temp.path().into(), lock);
    let (started, started_rx) = oneshot::channel();
    let (release, release_rx) = mpsc::channel();
    let mut operation = Box::pin(cache.run_blocking(move |_| {
        started.send(()).unwrap();
        release_rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(io::Error::other)
    }));
    tokio::select! {
        started = started_rx => started.unwrap(),
        result = &mut operation => panic!("blocking call returned early: {result:?}"),
    }
    assert!(
        time::timeout(Duration::from_millis(10), operation)
            .await
            .is_err()
    );
    drop(cache);
    assert!(matches!(
        cache_closed.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));

    let contender = File::options()
        .read(true)
        .write(true)
        .open(&lock_path)
        .unwrap();
    assert!(matches!(
        contender.try_lock(),
        Err(TryLockError::WouldBlock)
    ));
    release.send(()).unwrap();
    assert!(
        time::timeout(Duration::from_secs(2), cache_closed)
            .await
            .unwrap()
            .is_err()
    );
    contender.try_lock().unwrap();
}

#[cfg(all(test, windows))]
#[test]
fn retries_a_busy_model_link_on_the_next_sweep() {
    use std::os::windows::fs::OpenOptionsExt;

    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path();
    let busy = model_path(directory, 1);
    let other = model_path(directory, 2);
    let current = model_path(directory, 3);
    for path in [&busy, &other, &current] {
        fs::write(path, b"model fixture").unwrap();
    }
    let handle = fs::OpenOptions::new()
        .read(true)
        .share_mode(0)
        .open(&busy)
        .unwrap();
    prune_older(directory, 3).unwrap();
    assert!(!other.exists());
    drop(handle);
    assert!(busy.is_file());

    prune_older(directory, 3).unwrap();
    assert!(!busy.exists());
    assert!(current.is_file());
}
