//! Local discovery of immutable, atomically published `<version>.onnx` files.

use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use notify::{Event, RecursiveMode, Watcher};
use tokio::{fs, sync::watch};

use crate::inference::runtime::ModelVersion;

#[derive(Debug)]
pub(in crate::self_play) enum CheckpointWatchError {
    Filesystem(io::Error),
    Notification(Arc<notify::Error>),
    EventsClosed,
}

/// The directory must already exist and remain in place while watching.
/// Writers must rename completed temporary files to their final version names;
/// a filesystem event alone cannot distinguish an incomplete direct write.
pub(super) async fn run(
    model_dir: PathBuf,
    latest_version_tx: watch::Sender<Option<ModelVersion>>,
) -> Result<(), CheckpointWatchError> {
    // Coalesce event bursts into one rescan, without an unbounded event queue.
    // Once an error is observed, retain it rather than overwrite it with a later
    // successful notification before the async task can inspect it.
    let (changes_tx, mut changes_rx) = watch::channel(Ok::<(), Arc<notify::Error>>(()));
    let mut failed = false;
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<Event>| {
        if failed || event.as_ref().is_ok_and(|event| event.kind.is_access()) {
            return;
        }
        failed = event.is_err();
        let _ = changes_tx.send_replace(event.map(|_| ()).map_err(Arc::new));
    })
    .map_err(|error| CheckpointWatchError::Notification(Arc::new(error)))?;
    // Register before the initial scan so publications during it aren't missed.
    watcher
        .watch(&model_dir, RecursiveMode::NonRecursive)
        .map_err(|error| CheckpointWatchError::Notification(Arc::new(error)))?;

    loop {
        changes_rx
            .borrow_and_update()
            .clone()
            .map_err(CheckpointWatchError::Notification)?;
        if let Some(version) = newest_version(&model_dir)
            .await
            .map_err(CheckpointWatchError::Filesystem)?
        {
            latest_version_tx.send_if_modified(|latest| {
                if latest.is_none_or(|current| version > current) {
                    *latest = Some(version);
                    true
                } else {
                    false
                }
            });
        }
        // Rescanning also handles coalesced events and backend rescan requests.
        // Never clear or downgrade the publication when old files are removed.
        tokio::select! {
            _ = latest_version_tx.closed() => return Ok(()),
            change = changes_rx.changed() => {
                change.map_err(|_| CheckpointWatchError::EventsClosed)?;
            }
        }
    }
}

async fn newest_version(model_dir: &Path) -> io::Result<Option<ModelVersion>> {
    let mut entries = fs::read_dir(model_dir).await?;
    let mut newest = None;
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name();
        let Some(stem) = name.to_str().and_then(|name| name.strip_suffix(".onnx")) else {
            continue;
        };
        let Ok(version) = stem.parse::<ModelVersion>() else {
            continue;
        };
        // Match the loader's canonical decimal filename (not `007.onnx`, etc.).
        if stem != version.to_string() {
            continue;
        }
        match entry.file_type().await {
            Ok(kind) if kind.is_file() => {}
            // Retention cleanup can remove an entry during this scan.
            Ok(_) => continue,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        }
        newest = Some(newest.map_or(version, |current: ModelVersion| current.max(version)));
    }
    Ok(newest)
}

#[cfg(test)]
mod tests;
