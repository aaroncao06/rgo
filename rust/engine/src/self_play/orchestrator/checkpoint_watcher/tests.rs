use super::*;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
use tokio::time::timeout;

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
        loop {
            let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("rgo-checkpoint-watch-{}-{id}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("failed to create watcher test directory: {error}"),
            }
        }
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

async fn wait_for_version(rx: &mut watch::Receiver<Option<ModelVersion>>, expected: ModelVersion) {
    timeout(Duration::from_secs(5), async {
        loop {
            if *rx.borrow_and_update() == Some(expected) {
                return;
            }
            rx.changed().await.unwrap();
        }
    })
    .await
    .expect("checkpoint publication timed out");
}

#[tokio::test]
async fn scans_only_canonical_regular_model_files_and_chooses_numeric_maximum() {
    let dir = TestDir::new();
    for name in [
        "2.onnx",
        "10.onnx",
        "099.onnx",
        "+99.onnx",
        "100.onnx.tmp",
        "model.onnx",
        "18446744073709551616.onnx",
    ] {
        std::fs::write(dir.0.join(name), b"fixture").unwrap();
    }
    std::fs::create_dir(dir.0.join("200.onnx")).unwrap();
    assert_eq!(newest_version(&dir.0).await.unwrap(), Some(10));
}

#[tokio::test]
async fn discovers_existing_model_and_waits_for_atomic_publications() {
    let dir = TestDir::new();
    std::fs::write(dir.0.join("41.onnx"), b"first").unwrap();
    let (tx, mut rx) = watch::channel(None);
    let task = tokio::spawn(run(dir.0.clone(), tx));
    wait_for_version(&mut rx, 41).await;

    // Establishes that watching is registered before these filesystem changes.
    std::fs::write(dir.0.join("103.onnx.tmp"), b"complete model").unwrap();
    std::fs::rename(dir.0.join("103.onnx.tmp"), dir.0.join("103.onnx")).unwrap();
    wait_for_version(&mut rx, 103).await;

    drop(rx);
    timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn does_not_clear_or_downgrade_the_published_version() {
    let dir = TestDir::new();
    std::fs::write(dir.0.join("10.onnx"), b"first").unwrap();
    let (tx, mut rx) = watch::channel(None);
    let task = tokio::spawn(run(dir.0.clone(), tx));
    wait_for_version(&mut rx, 10).await;
    std::fs::remove_file(dir.0.join("10.onnx")).unwrap();
    std::fs::write(dir.0.join("2.onnx"), b"older").unwrap();
    // Even after notifications arrive, no lower/empty publication is permitted.
    assert!(
        timeout(Duration::from_millis(500), rx.changed())
            .await
            .is_err()
    );
    assert_eq!(*rx.borrow(), Some(10));
    std::fs::write(dir.0.join("12.onnx.tmp"), b"newer").unwrap();
    std::fs::rename(dir.0.join("12.onnx.tmp"), dir.0.join("12.onnx")).unwrap();
    timeout(Duration::from_secs(5), rx.changed())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(*rx.borrow_and_update(), Some(12));
    drop(rx);
    timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn missing_directory_returns_a_watcher_error() {
    let dir = TestDir::new();
    let (tx, rx) = watch::channel(None);
    assert!(matches!(
        run(dir.0.join("missing"), tx).await,
        Err(CheckpointWatchError::Notification(_))
    ));
    assert_eq!(*rx.borrow(), None);
}

#[tokio::test]
async fn exits_when_the_consumer_closes_even_with_an_empty_directory() {
    let dir = TestDir::new();
    let (tx, rx) = watch::channel(None);
    drop(rx);
    timeout(Duration::from_secs(5), run(dir.0.clone(), tx))
        .await
        .unwrap()
        .unwrap();
}
