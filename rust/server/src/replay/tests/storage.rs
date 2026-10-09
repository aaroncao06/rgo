use super::*;

#[test]
fn fifo_eviction_keeps_whole_boundary_and_oversized_chunks() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(4)).unwrap();
    store
        .ingest(30, &chunk(&[(9, 1), (13, 2), (9, 3)]))
        .unwrap();
    store
        .ingest(10, &chunk(&[(19, 4), (9, 5), (13, 6)]))
        .unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 6);
    assert_eq!(store.stats().unwrap().retained_chunks, 2);
    assert_eq!(tags(&store), BTreeSet::from([1, 2, 3, 4, 5, 6]));
    assert!(chunk_path(&root.path().join("chunks"), 30).exists());

    store
        .ingest(20, &chunk(&[(9, 7), (9, 8), (13, 9), (19, 10), (9, 11)]))
        .unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 5);
    assert_eq!(store.stats().unwrap().retained_chunks, 1);
    assert_eq!(tags(&store), BTreeSet::from([7, 8, 9, 10, 11]));
    for id in [30, 10] {
        assert!(!chunk_path(&root.path().join("chunks"), id).exists());
    }
    store.ingest(40, &chunk(&[(9, 12)])).unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 6);
    assert_eq!(tags(&store), BTreeSet::from([7, 8, 9, 10, 11, 12]));
    store
        .ingest(50, &chunk(&[(13, 13), (19, 14), (9, 15)]))
        .unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 4);
    assert_eq!(tags(&store), BTreeSet::from([12, 13, 14, 15]));
    assert!(!chunk_path(&root.path().join("chunks"), 20).exists());
    let expected = samples(&mut store, 200, 91);
    drop(store);
    let mut reopened = ReplayStore::open(root.path(), config(4)).unwrap();
    assert_eq!(reopened.stats().unwrap().retained_records, 4);
    assert_eq!(samples(&mut reopened, 200, 91), expected);
}

#[test]
fn window_changes_persist_eviction_and_growth_only_uses_new_arrivals() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(6)).unwrap();
    for id in 1..=3 {
        store
            .ingest(id, &chunk(&[(9, (2 * id - 1) as u8), (9, (2 * id) as u8)]))
            .unwrap();
    }
    store.set_window(3).unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 4);
    assert_eq!(tags(&store), BTreeSet::from([3, 4, 5, 6]));
    store.set_window(2).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([5, 6]));
    store.set_window(6).unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 2);
    assert_eq!(tags(&store), BTreeSet::from([5, 6]));
    drop(store);
    let mut store = ReplayStore::open(root.path(), config(6)).unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 2);
    store.ingest(4, &chunk(&[(13, 7), (19, 8)])).unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 4);
    assert_eq!(tags(&store), BTreeSet::from([5, 6, 7, 8]));
    let previous = store.stats().unwrap();
    for invalid in [0, 101] {
        assert!(matches!(
            store.set_window(invalid),
            Err(ReplayError::InvalidConfig(_))
        ));
        assert_eq!(store.stats().unwrap(), previous);
    }
    drop(store);
    let shrunk_on_open = ReplayStore::open(root.path(), config(1)).unwrap();
    assert_eq!(shrunk_on_open.stats().unwrap().retained_records, 2);
    assert_eq!(tags(&shrunk_on_open), BTreeSet::from([7, 8]));
}

#[test]
fn retry_receipts_survive_restart_and_eviction_without_reintroducing_data() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    let first = chunk(&[(9, 1)]);
    assert_eq!(
        store.ingest(1, &first).unwrap(),
        IngestOutcome::Added { records: 1 }
    );
    assert_eq!(
        store.ingest(1, &first).unwrap(),
        IngestOutcome::AlreadyAccepted { records: 1 }
    );
    store.ingest(2, &chunk(&[(13, 2)])).unwrap();
    drop(store);
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    assert_eq!(
        store.ingest(1, &first).unwrap(),
        IngestOutcome::AlreadyAccepted { records: 1 }
    );
    assert_eq!(tags(&store), BTreeSet::from([2]));
    assert!(!chunk_path(&root.path().join("chunks"), 1).exists());
    // An existing ID wins even when the later contents and record count differ.
    assert_eq!(
        store.ingest(1, &chunk(&[(9, 3), (19, 4)])).unwrap(),
        IngestOutcome::AlreadyAccepted { records: 1 }
    );
    assert!(!intake_path(&store, 1).exists());
    // Duplicate contents are discarded without parsing or checksum validation.
    assert_eq!(
        store.ingest(2, b"invalid duplicate").unwrap(),
        IngestOutcome::AlreadyAccepted { records: 1 }
    );
    assert!(!intake_path(&store, 2).exists());
    assert_eq!(tags(&store), BTreeSet::from([2]));
}

#[test]
fn local_ingestion_transfers_the_existing_file_to_replay() {
    let root = TempDir::new().unwrap();
    let source = root.path().join("selfplay.rgo");
    let bytes = chunk(&[(9, 3), (19, 4)]);
    fs::write(&source, &bytes).unwrap();
    fs::create_dir(root.path().join("server")).unwrap();
    sync_directory(root.path()).unwrap();
    let mut store = ReplayStore::open(&root.path().join("server"), config(100)).unwrap();
    #[cfg(unix)]
    let original = fs::metadata(&source).unwrap();
    store.ingest_file(ChunkId::MAX, &source).unwrap();
    let stored = chunk_path(&store.chunk_dir, ChunkId::MAX);
    assert!(!source.exists());
    assert_eq!(fs::read(&stored).unwrap(), bytes);
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let accepted = fs::metadata(&stored).unwrap();
        assert_eq!(
            (accepted.dev(), accepted.ino()),
            (original.dev(), original.ino())
        );
        assert_eq!(accepted.nlink(), 1);
    }
    assert_eq!(
        store.ingest_file(ChunkId::MAX, &source).unwrap(),
        IngestOutcome::AlreadyAccepted { records: 2 }
    );
    assert_eq!(tags(&store), BTreeSet::from([3, 4]));
    drop(store);
    let store = ReplayStore::open(&root.path().join("server"), config(100)).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([3, 4]));
}

#[cfg(unix)]
#[test]
fn symlink_intake_is_rejected_without_registering_or_consuming_it() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    let original = root.path().join("original.rgo");
    let intake = root.path().join("intake.rgo");
    let bytes = chunk(&[(9, 1)]);
    fs::write(&original, &bytes).unwrap();
    // Moving this relative symlink into chunks/ would make it dangling.
    std::os::unix::fs::symlink("original.rgo", &intake).unwrap();
    assert!(matches!(
        store.ingest_file(1, &intake),
        Err(ReplayError::Io(error)) if error.kind() == io::ErrorKind::InvalidInput
    ));
    assert!(intake.is_symlink());
    assert_eq!(fs::read(&original).unwrap(), bytes);
    assert!(!chunk_path(&store.chunk_dir, 1).exists());
    assert_eq!(store.stats().unwrap().retained_records, 0);
    assert_eq!(
        store.ingest_file(1, &original).unwrap(),
        IngestOutcome::Added { records: 1 }
    );
    // A known-ID retry still discards its now-dangling link without reading it.
    assert_eq!(
        store.ingest_file(1, &intake).unwrap(),
        IngestOutcome::AlreadyAccepted { records: 1 }
    );
    assert!(!intake.is_symlink());
    assert_eq!(shard_tags(&samples(&mut store, 1, 7)), vec![1]);
}

#[test]
fn missing_file_retries_survive_restart_and_eviction() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    let source = intake_path(&store, 1);
    store.ingest(1, &chunk(&[(9, 1), (19, 2)])).unwrap();
    store.ingest(2, &chunk(&[(9, 3)])).unwrap();
    assert!(!source.exists());
    drop(store);
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    assert_eq!(
        store.ingest_file(1, &source).unwrap(),
        IngestOutcome::AlreadyAccepted { records: 2 }
    );
    assert_eq!(tags(&store), BTreeSet::from([3]));
    assert!(!chunk_path(&store.chunk_dir, 1).exists());
    assert!(matches!(
        store.ingest_file(99, &source),
        Err(ReplayError::Io(error)) if error.kind() == io::ErrorKind::NotFound
    ));
    assert_eq!(store.stats().unwrap().retained_records, 1);
}

#[test]
fn canonical_files_cannot_be_used_as_intake() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    let stored = chunk_path(&store.chunk_dir, 1);
    for id in [1, 2] {
        assert!(matches!(
            store.ingest_file(id, &stored),
            Err(ReplayError::Io(error)) if error.kind() == io::ErrorKind::InvalidInput
        ));
        assert!(stored.exists());
        assert_eq!(tags(&store), BTreeSet::from([1]));
    }
    #[cfg(unix)]
    {
        let alias = root.path().join("chunk-alias");
        std::os::unix::fs::symlink(&store.chunk_dir, &alias).unwrap();
        assert!(matches!(
            store.ingest_file(1, &chunk_path(&alias, 1)),
            Err(ReplayError::Io(error)) if error.kind() == io::ErrorKind::InvalidInput
        ));
        assert!(stored.exists());
    }
}

#[test]
fn malformed_inputs_do_not_mutate_replay() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    let mut bad = chunk(&[(9, 1)]);
    bad[CHUNK_HEADER_SIZE] ^= 1;
    let malformed = root.path().join("malformed.rgo");
    fs::write(&malformed, &bad).unwrap();
    assert!(matches!(
        store.ingest_file(2, &malformed),
        Err(ReplayError::InvalidChunk(_))
    ));
    assert_eq!(fs::read(&malformed).unwrap(), bad);
    assert_eq!(store.stats().unwrap().retained_records, 0);
    assert_eq!(fs::read_dir(root.path().join("chunks")).unwrap().count(), 0);
}

#[test]
fn trusted_chunks_above_the_old_byte_limit_can_be_ingested_sampled_and_reopened() {
    let root = TempDir::new().unwrap();
    let source = root.path().join("selfplay.rgo");
    let bytes = chunk(&vec![(19, 1); 5000]);
    assert!(bytes.len() > 4 * 1024 * 1024);
    fs::write(&source, &bytes).unwrap();
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    assert_eq!(
        store.ingest_file(3, &source).unwrap(),
        IngestOutcome::Added { records: 5000 }
    );
    assert_eq!(store.stats().unwrap().retained_records, 5000);
    assert_eq!(tags(&store), BTreeSet::from([1]));
    assert_eq!(shard_tags(&samples(&mut store, 16, 7)), vec![1; 16]);
    assert!(!source.exists());
    assert_eq!(fs::read(chunk_path(&store.chunk_dir, 3)).unwrap(), bytes);
    drop(store);
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 5000);
    assert_eq!(tags(&store), BTreeSet::from([1]));
    assert_eq!(shard_tags(&samples(&mut store, 64, 7)), vec![1; 64]);
}

#[test]
fn replay_root_must_already_exist() {
    let base = TempDir::new().unwrap();
    let root = base.path().join("unprovisioned/replay");
    assert!(matches!(
        ReplayStore::open(&root, config(1)),
        Err(ReplayError::Io(error)) if error.kind() == io::ErrorKind::NotFound
    ));
    assert!(!base.path().join("unprovisioned").exists());
}

#[test]
fn accepted_files_are_not_revalidated_during_sampling_or_reopen() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    let mut bytes = chunk(&[(9, 1)]);
    store.ingest(1, &bytes).unwrap();
    // Deliberately damage only the footer to distinguish ingestion validation
    // from later indexed reads. Production callers must keep accepted files immutable.
    *bytes.last_mut().unwrap() ^= 1;
    assert!(matches!(
        store.ingest(2, &bytes),
        Err(ReplayError::InvalidChunk(ChunkError::ChecksumMismatch))
    ));
    fs::write(chunk_path(&store.chunk_dir, 1), &bytes).unwrap();
    assert_eq!(shard_tags(&samples(&mut store, 8, 1)), vec![1]);
    drop(store);
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    assert_eq!(shard_tags(&samples(&mut store, 8, 1)), vec![1]);
}

#[test]
fn invalid_configuration_is_rejected_before_creating_storage() {
    let root = TempDir::new().unwrap();
    for settings in [
        ReplayConfig {
            capacity_records: 0,
            ..config(1)
        },
        ReplayConfig {
            window_records: 0,
            ..config(1)
        },
        ReplayConfig {
            window_records: 101,
            ..config(1)
        },
        ReplayConfig {
            sample_group_records: 0,
            ..config(1)
        },
        ReplayConfig {
            sample_groups_per_shard: 0,
            ..config(1)
        },
        ReplayConfig {
            shard_records: 0,
            ..config(1)
        },
    ] {
        let path = root.path().join("bad");
        assert!(matches!(
            ReplayStore::open(&path, settings),
            Err(ReplayError::InvalidConfig(_))
        ));
        assert!(!path.exists());
    }
}
