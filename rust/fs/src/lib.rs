//! Directory creation and synchronization.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

/// Persist directory-entry changes on Unix. Other platforms retain the existing
/// no-op behavior; this does not provide a directory durability guarantee there.
pub fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        fs::File::open(path)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// Create missing directories and synchronize their parent entries.
/// Existing directories are assumed to be provisioned, including after an
/// interrupted setup; no provisioning recovery state is maintained.
pub fn prepare_directory(path: &Path) -> io::Result<PathBuf> {
    prepare_directory_with_sync(path, sync_directory)
}

/// The supplied directory barrier also allows callers to inject I/O failures.
pub fn prepare_directory_with_sync(
    path: &Path,
    mut sync: impl FnMut(&Path) -> io::Result<()>,
) -> io::Result<PathBuf> {
    let absolute = std::path::absolute(path)?;
    create_directory(&absolute, &mut sync)?;
    fs::canonicalize(absolute)
}

fn create_directory(path: &Path, sync: &mut impl FnMut(&Path) -> io::Result<()>) -> io::Result<()> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => return Ok(()),
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "storage path must be a directory",
            ));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    if let Some(parent) = path.parent() {
        create_directory(parent, sync)?;
    }
    match fs::create_dir(path) {
        Ok(()) => {}
        // Another process may have created the directory in the meantime.
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists && path.is_dir() => {}
        Err(error) => return Err(error),
    }
    if let Some(parent) = path.parent() {
        sync(parent)?;
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn parent_components_resolve_existing_missing_and_symlink_paths() {
        let temp = tempfile::tempdir().unwrap();
        let base = fs::canonicalize(temp.path()).unwrap();
        fs::create_dir(base.join("existing")).unwrap();
        for path in [
            base.join("existing/.."),
            base.join("existing/../."),
            PathBuf::from("/"),
        ] {
            let expected = fs::canonicalize(&path).unwrap();
            assert_eq!(
                prepare_directory_with_sync(&path, |_| {
                    panic!("existing targets need no provisioning barriers")
                })
                .unwrap(),
                expected
            );
        }
        assert_eq!(
            prepare_directory(&base.join("new/../output")).unwrap(),
            base.join("output")
        );
        assert!(base.join("new").is_dir());
        assert_eq!(prepare_directory(&base.join("new-end/..")).unwrap(), base);
        assert!(base.join("new-end").is_dir());

        fs::create_dir_all(base.join("elsewhere/child")).unwrap();
        std::os::unix::fs::symlink(base.join("elsewhere/child"), base.join("alias")).unwrap();
        assert_eq!(
            prepare_directory(&base.join("alias/..")).unwrap(),
            base.join("elsewhere")
        );
        assert_eq!(
            prepare_directory(&base.join("alias/../output")).unwrap(),
            base.join("elsewhere/output")
        );
    }

    #[test]
    fn concurrent_creation_under_a_shared_parent_succeeds() {
        use std::sync::{Arc, Barrier};
        let temp = tempfile::tempdir().unwrap();
        let shared = fs::canonicalize(temp.path()).unwrap().join("shared");
        let barrier = Arc::new(Barrier::new(2));
        let threads: Vec<_> = ["a", "b"]
            .into_iter()
            .map(|leaf| {
                let path = shared.join(leaf);
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    prepare_directory(&path)
                })
            })
            .collect();
        for (thread, leaf) in threads.into_iter().zip(["a", "b"]) {
            assert_eq!(thread.join().unwrap().unwrap(), shared.join(leaf));
        }
    }

    #[test]
    fn new_directories_sync_their_parents_and_report_sync_errors() {
        let temp = tempfile::tempdir().unwrap();
        let base = fs::canonicalize(temp.path()).unwrap();
        let path = base.join("nested/output");
        let mut synced = Vec::new();
        prepare_directory_with_sync(&path, |parent| {
            synced.push(parent.to_path_buf());
            sync_directory(parent)
        })
        .unwrap();
        assert_eq!(synced, [base.clone(), base.join("nested")]);

        let error = prepare_directory_with_sync(&base.join("failed"), |_| {
            Err(io::Error::other("failed directory sync"))
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "failed directory sync");
        assert!(prepare_directory(&base.join("failed")).is_ok());

        fs::write(base.join("file"), b"data").unwrap();
        assert!(prepare_directory(&base.join("file")).is_err());
    }
}
