use super::super::storage::commit_catalog;
use super::*;

#[test]
fn catalog_uses_durable_sync_settings_on_initial_open_and_reopen() {
    let root = TempDir::new().unwrap();
    for _ in 0..2 {
        let store = ReplayStore::open(root.path(), config(100)).unwrap();
        // This connection-local setting must be restored on every open.
        let synchronous: u32 = store
            .db
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .unwrap();
        assert_eq!(synchronous, 3, "catalog commits require synchronous=EXTRA");
        let fullfsync: bool = store
            .db
            .query_row("PRAGMA fullfsync", [], |row| row.get(0))
            .unwrap();
        assert!(fullfsync, "macOS catalog writes require F_FULLFSYNC");
        let cache_spill: bool = store
            .db
            .query_row("PRAGMA cache_spill", [], |row| row.get(0))
            .unwrap();
        assert!(
            !cache_spill,
            "database writes must wait for the journal barrier"
        );
        let journal_mode: String = store
            .db
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(journal_mode, "delete");
        // Atomic-write builds may defer creating the disk journal until COMMIT,
        // which would invalidate the pre-commit directory barrier.
        let deferred_journal: bool = store
            .db
            .query_row(
                "SELECT sqlite_compileoption_used('ENABLE_ATOMIC_WRITE') OR
                        sqlite_compileoption_used('ENABLE_BATCH_ATOMIC_WRITE')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            !deferred_journal,
            "catalog requires an immediately created journal"
        );
    }
}

#[cfg(unix)]
#[test]
fn failed_schema_journal_barrier_rolls_back_initialization() {
    let root = TempDir::new().unwrap();
    {
        // The first root sync provisions chunks/; the next precedes schema COMMIT.
        let _fault = faults::inject_after(faults::Operation::Sync, root.path().to_owned(), 1);
        assert!(matches!(
            ReplayStore::open(root.path(), config(100)),
            Err(ReplayError::Io(_))
        ));
    }
    let db = Connection::open(root.path().join("replay.sqlite")).unwrap();
    let tables: u32 = db
        .query_row(
            "SELECT COUNT(*) FROM sqlite_schema WHERE name = 'chunks'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(tables, 0);
    drop(db);
    // A failed open releases the lock and can safely repeat initialization.
    let store = ReplayStore::open(root.path(), config(100)).unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 0);
}

#[cfg(unix)]
#[test]
fn dirty_catalog_pages_wait_for_the_journal_barrier_even_with_a_small_cache() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    let transaction = store.db.transaction().unwrap();
    transaction
        .execute_batch(
            "CREATE TABLE barrier_fixture (id INTEGER PRIMARY KEY, payload BLOB);
             WITH RECURSIVE rows(n) AS (
                 VALUES(1) UNION ALL SELECT n + 1 FROM rows WHERE n < 200
             ) INSERT INTO barrier_fixture SELECT n, randomblob(4096) FROM rows;",
        )
        .unwrap();
    commit_catalog(transaction, root.path()).unwrap();
    store.db.execute_batch("PRAGMA cache_size = 2").unwrap();
    let original = fs::read(root.path().join("replay.sqlite")).unwrap();
    let transaction = store.db.transaction().unwrap();
    transaction
        .execute("UPDATE barrier_fixture SET payload = zeroblob(4096)", [])
        .unwrap();
    assert!(root.path().join("replay.sqlite-journal").is_file());
    // This UPDATE dirties far more pages than the cache target. They must stay
    // in RAM, rather than spill before we can persist journal creation.
    assert!(
        fs::read(root.path().join("replay.sqlite")).unwrap() == original,
        "database pages changed before the journal-creation barrier"
    );
    let _fault = faults::inject([(faults::Operation::Sync, root.path().to_owned())]);
    assert!(matches!(
        commit_catalog(transaction, root.path()),
        Err(ReplayError::Io(_))
    ));
    assert!(
        fs::read(root.path().join("replay.sqlite")).unwrap() == original,
        "aborting before the journal-creation barrier changed database pages"
    );
    assert!(store.db.is_autocommit());
}

#[cfg(unix)]
#[test]
fn failed_ingestion_journal_barrier_preserves_prior_catalog_and_intake() {
    let root = TempDir::new().unwrap();
    let intake = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    let original = fs::read(root.path().join("replay.sqlite")).unwrap();
    let source = intake.path().join("chunk.rgo");
    fs::write(&source, chunk(&[(19, 2)])).unwrap();
    sync_directory(intake.path()).unwrap();
    {
        let _fault = faults::inject([(faults::Operation::Sync, root.path().to_owned())]);
        assert!(matches!(
            store.ingest_file(2, &source),
            Err(ReplayError::Io(_))
        ));
        assert_requires_reopen(&mut store, root.path());
    }
    assert_eq!(
        fs::read(root.path().join("replay.sqlite")).unwrap(),
        original
    );
    assert!(source.exists());
    assert!(chunk_path(&store.chunk_dir, 1).exists());
    assert!(chunk_path(&store.chunk_dir, 2).exists());
    drop(store);
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([1]));
    assert!(!chunk_path(&store.chunk_dir, 2).exists());
    assert_eq!(
        store.ingest_file(2, &source).unwrap(),
        IngestOutcome::Added { records: 1 }
    );
    assert_eq!(tags(&store), BTreeSet::from([2]));
    assert!(!source.exists());
}

#[test]
fn hot_journal_recovery_restores_catalog_before_loading_index() {
    const CHILD_ROOT: &str = "RGO_TEST_HOT_JOURNAL_ROOT";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        let db = rusqlite::Connection::open(Path::new(&root).join("replay.sqlite")).unwrap();
        db.execute_batch(
            "PRAGMA fullfsync = ON;
             PRAGMA synchronous = EXTRA;
             PRAGMA cache_size = 2;
             BEGIN IMMEDIATE;
             UPDATE chunks SET active = 0;
             UPDATE crash_fixture SET payload = zeroblob(4096);",
        )
        .unwrap();
        // Spill uncommitted pages to disk, then exit without transaction or
        // connection cleanup. Reopening must roll back the catalog changes.
        std::process::exit(86);
    }

    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(3)).unwrap();
    for id in 1..=3 {
        store.ingest(id, &chunk(&[(9, id as u8)])).unwrap();
    }
    store
        .db
        .execute_batch(
            "CREATE TABLE crash_fixture (id INTEGER PRIMARY KEY, payload BLOB);
             WITH RECURSIVE rows(n) AS (
                 VALUES(1) UNION ALL SELECT n + 1 FROM rows WHERE n < 200
             ) INSERT INTO crash_fixture SELECT n, randomblob(4096) FROM rows;",
        )
        .unwrap();
    drop(store);

    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "replay::tests::recovery::hot_journal_recovery_restores_catalog_before_loading_index",
            "--test-threads=1",
        ])
        .env(CHILD_ROOT, root.path())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(86));
    let journal = root.path().join("replay.sqlite-journal");
    let bytes = fs::read(&journal).unwrap();
    // Confirm a real, populated hot journal, rather than an empty pending file.
    assert!(bytes.len() > 4096);
    assert_eq!(
        &bytes[..8],
        &[0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7]
    );

    let store = ReplayStore::open(root.path(), config(3)).unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 3);
    assert_eq!(tags(&store), BTreeSet::from([1, 2, 3]));
    assert_eq!(fs::read_dir(&store.chunk_dir).unwrap().count(), 3);
    assert!(!journal.exists());
}

#[test]
fn exclusive_store_lock_releases_on_drop() {
    let root = TempDir::new().unwrap();
    let store = ReplayStore::open(root.path(), config(100)).unwrap();
    assert!(matches!(
        ReplayStore::open(root.path(), config(100)),
        Err(ReplayError::AlreadyOpen)
    ));
    drop(store);
    assert!(ReplayStore::open(root.path(), config(100)).is_ok());
}

#[test]
fn recovery_removes_uncommitted_files_and_finishes_committed_evictions() {
    let root = TempDir::new().unwrap();
    let store = ReplayStore::open(root.path(), config(100)).unwrap();
    let dir = root.path().join("chunks");
    fs::write(chunk_path(&dir, 90), chunk(&[(9, 1)])).unwrap();
    drop(store);
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    assert!(!chunk_path(&dir, 90).exists());
    let input = chunk(&[(9, 1)]);
    store.ingest(1, &input).unwrap();
    // Simulate a crash after the retention transaction commits, before unlink.
    store
        .db
        .execute(
            "UPDATE chunks SET active = 0 WHERE id = ?1",
            [1_u128.to_be_bytes().as_slice()],
        )
        .unwrap();
    drop(store);
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 0);
    assert!(!chunk_path(&dir, 1).exists());
    assert_eq!(
        store.ingest(1, &input).unwrap(),
        IngestOutcome::AlreadyAccepted { records: 1 }
    );
}

#[test]
fn failed_catalog_write_requires_reopen_and_failed_index_load_returns_no_handle() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(3)).unwrap();
    for id in 1..=3 {
        store.ingest(id, &chunk(&[(9, id as u8)])).unwrap();
    }
    store.db.execute_batch("CREATE TRIGGER reject_insert BEFORE INSERT ON chunks BEGIN SELECT RAISE(FAIL, 'injected failure'); END;").unwrap();
    // Simulate an unavailable file halfway through an index rebuild.
    let unavailable = chunk_path(&root.path().join("chunks"), 2);
    let saved = root.path().join("saved-chunk");
    fs::rename(&unavailable, &saved).unwrap();
    assert!(matches!(
        store.ingest(4, &chunk(&[(9, 4)])),
        Err(ReplayError::Database(_))
    ));
    let intake = intake_path(&store, 4);
    assert_eq!(fs::read(&intake).unwrap(), chunk(&[(9, 4)]));
    assert_requires_reopen(&mut store, root.path());
    drop(store);
    assert!(matches!(
        ReplayStore::open(root.path(), config(3)),
        Err(ReplayError::Io(_))
    ));
    fs::rename(&saved, &unavailable).unwrap();
    let mut store = ReplayStore::open(root.path(), config(3)).unwrap();
    assert_eq!(store.stats().unwrap().retained_records, 3);
    assert_eq!(tags(&store), BTreeSet::from([1, 2, 3]));
    assert!(!chunk_path(&root.path().join("chunks"), 4).exists());
    assert_eq!(fs::read(&intake).unwrap(), chunk(&[(9, 4)]));
    store
        .db
        .execute_batch("DROP TRIGGER reject_insert;")
        .unwrap();
    assert_eq!(
        store.ingest_file(4, &intake).unwrap(),
        IngestOutcome::Added { records: 1 }
    );
    assert!(!intake.exists());
    assert_eq!(tags(&store), BTreeSet::from([2, 3, 4]));
    let catalog_records: i64 = store
        .db
        .query_row(
            "SELECT SUM(record_count) FROM chunks WHERE active = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        catalog_records as usize,
        store.stats().unwrap().retained_records
    );
    assert_eq!(catalog_records, 3);
}

#[test]
fn failed_catalog_commit_rolls_back_acceptance_and_eviction() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    // A deferred constraint permits the INSERT and retention UPDATE, but
    // rejects COMMIT. This exercises transaction rollback after both changes.
    store
        .db
        .execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE commit_guard (id INTEGER PRIMARY KEY);
             CREATE TABLE commit_failure (
                 id INTEGER REFERENCES commit_guard(id) DEFERRABLE INITIALLY DEFERRED
             );
             CREATE TRIGGER reject_commit AFTER INSERT ON chunks BEGIN
                 INSERT INTO commit_failure VALUES (1);
             END;",
        )
        .unwrap();
    assert!(matches!(
        store.ingest(2, &chunk(&[(19, 2)])),
        Err(ReplayError::Database(_))
    ));
    assert_requires_reopen(&mut store, root.path());
    let intake = intake_path(&store, 2);
    assert!(intake.exists());
    assert!(chunk_path(&store.chunk_dir, 1).exists());
    drop(store);

    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([1]));
    assert!(!chunk_path(&store.chunk_dir, 2).exists());
    store
        .db
        .execute_batch("DROP TRIGGER reject_commit;")
        .unwrap();
    assert_eq!(
        store.ingest_file(2, &intake).unwrap(),
        IngestOutcome::Added { records: 1 }
    );
    assert_eq!(tags(&store), BTreeSet::from([2]));
}

#[cfg(unix)]
#[test]
fn failed_orphan_cleanup_prevents_reopening_until_cleanup_succeeds() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    store.db.execute_batch("CREATE TRIGGER reject_insert BEFORE INSERT ON chunks BEGIN SELECT RAISE(FAIL, 'injected failure'); END;").unwrap();
    assert!(matches!(
        store.ingest(2, &chunk(&[(9, 2)])),
        Err(ReplayError::Database(_))
    ));
    let dir = root.path().join("chunks");
    let orphan = chunk_path(&dir, 2);
    assert!(orphan.exists());
    assert_requires_reopen(&mut store, root.path());
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
    drop(store);
    let _fault = faults::inject([(faults::Operation::Remove, orphan.clone())]);
    assert!(matches!(
        ReplayStore::open(root.path(), config(1)),
        Err(ReplayError::Io(_))
    ));
    assert!(orphan.exists());
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    assert!(!orphan.exists());
    assert_eq!(tags(&store), BTreeSet::from([1]));
    store
        .db
        .execute_batch("DROP TRIGGER reject_insert;")
        .unwrap();
    store.ingest(2, &chunk(&[(9, 2)])).unwrap();
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
}

#[test]
fn failed_eviction_requires_reopening_and_preserves_committed_receipts() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    let blocked = chunk_path(&root.path().join("chunks"), 1);
    fs::remove_file(&blocked).unwrap();
    fs::create_dir(&blocked).unwrap();
    let second = chunk(&[(9, 2)]);
    assert!(matches!(store.ingest(2, &second), Err(ReplayError::Io(_))));
    assert_requires_reopen(&mut store, root.path());
    drop(store);
    assert!(matches!(
        ReplayStore::open(root.path(), config(1)),
        Err(ReplayError::Io(_))
    ));
    fs::remove_dir(&blocked).unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([2]));
    assert_eq!(
        store.ingest(2, &second).unwrap(),
        IngestOutcome::AlreadyAccepted { records: 1 }
    );
    store.ingest(3, &chunk(&[(9, 3)])).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([3]));
    assert_eq!(fs::read_dir(root.path().join("chunks")).unwrap().count(), 1);
}

#[test]
fn failed_window_mutation_requires_reopen() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(3)).unwrap();
    for id in 1..=3 {
        store.ingest(id, &chunk(&[(9, id as u8)])).unwrap();
    }
    store.db.execute_batch("CREATE TRIGGER reject_update BEFORE UPDATE ON chunks BEGIN SELECT RAISE(FAIL, 'injected failure'); END;").unwrap();
    assert!(matches!(store.set_window(1), Err(ReplayError::Database(_))));
    assert_requires_reopen(&mut store, root.path());
    drop(store);
    let mut store = ReplayStore::open(root.path(), config(3)).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([1, 2, 3]));
    store
        .db
        .execute_batch("DROP TRIGGER reject_update;")
        .unwrap();
    store.set_window(1).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([3]));
}

#[cfg(unix)]
#[test]
fn window_changes_without_eviction_skip_directory_syncs() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(3)).unwrap();
    store
        .ingest(1, &chunk(&[(9, 1), (13, 2), (19, 3)]))
        .unwrap();
    {
        let _fault = faults::inject([(faults::Operation::Sync, root.path().to_owned())]);
        // Growing, repeating, and shrinking below one whole chunk need no SQL writes.
        for target in [6, 6, 3, 1] {
            store.set_window(target).unwrap();
            assert_eq!(store.stats().unwrap().retained_records, 3);
        }
        assert!(faults::synced().is_empty());
    }
    // The final target still applies to subsequent arrivals.
    store.ingest(2, &chunk(&[(9, 4)])).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([4]));
}

#[cfg(unix)]
#[test]
fn interrupted_batch_eviction_cleanup_recovers_without_new_evictions() {
    // Fail partway through deletion, before its directory barrier, or on either
    // side of the cleanup catalog commit. All receipts must survive each case.
    for stage in 0..4 {
        let root = TempDir::new().unwrap();
        let mut store = ReplayStore::open(root.path(), config(4)).unwrap();
        for id in 1..=4 {
            store.ingest(id, &chunk(&[(9, id as u8)])).unwrap();
        }
        {
            let (operation, path, skip) = match stage {
                0 => (
                    faults::Operation::Remove,
                    chunk_path(&store.chunk_dir, 2),
                    0,
                ),
                1 => (faults::Operation::Sync, store.chunk_dir.clone(), 0),
                // The eviction decision has its own two root barriers first.
                _ => (faults::Operation::Sync, root.path().to_owned(), stage),
            };
            let _fault = faults::inject_after(operation, path, skip);
            assert!(matches!(store.set_window(1), Err(ReplayError::Io(_))));
            assert_requires_reopen(&mut store, root.path());
        }
        let pending: u32 = store
            .db
            .query_row(
                "SELECT count(*) FROM chunks WHERE active = 0 AND has_file = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(pending, if stage == 3 { 0 } else { 3 });
        drop(store);

        // Growing on reopen evicts nothing, but must still finish pending cleanup.
        let mut store = ReplayStore::open(root.path(), config(4)).unwrap();
        assert_eq!(tags(&store), BTreeSet::from([4]));
        assert_eq!(fs::read_dir(&store.chunk_dir).unwrap().count(), 1);
        let pending: u32 = store
            .db
            .query_row(
                "SELECT count(*) FROM chunks WHERE active = 0 AND has_file = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(pending, 0);
        for id in 1..=3 {
            assert_eq!(
                store.ingest(id, &chunk(&[(9, id as u8)])).unwrap(),
                IngestOutcome::AlreadyAccepted { records: 1 },
            );
        }
        assert_eq!(tags(&store), BTreeSet::from([4]));
    }
}

#[cfg(unix)]
#[test]
fn failed_publication_syncs_require_reopen_and_do_not_accumulate_files() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    let dir = root.path().join("chunks");
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    for id in 2..=4 {
        let _fault = faults::inject([(faults::Operation::Sync, dir.clone())]);
        assert!(matches!(
            store.ingest(id, &chunk(&[(9, id as u8)])),
            Err(ReplayError::Io(_))
        ));
        assert_requires_reopen(&mut store, root.path());
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        let intake = intake_path(&store, id);
        let bytes = chunk(&[(9, id as u8)]);
        assert_eq!(fs::read(&intake).unwrap(), bytes);
        drop(store);
        store = ReplayStore::open(root.path(), config(1)).unwrap();
        assert!(!chunk_path(&dir, id).exists());
        assert_eq!(fs::read(&intake).unwrap(), bytes);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        assert_eq!(tags(&store), BTreeSet::from([1]));
    }
    assert_eq!(
        store.ingest(4, &chunk(&[(9, 4)])).unwrap(),
        IngestOutcome::Added { records: 1 }
    );
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
    assert_eq!(tags(&store), BTreeSet::from([4]));
}

#[cfg(unix)]
#[test]
fn failed_intake_cleanup_retries_committed_acceptance_without_reintroducing_data() {
    for operation in [faults::Operation::Remove, faults::Operation::Sync] {
        let root = TempDir::new().unwrap();
        let intake = TempDir::new().unwrap();
        let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
        let source = intake.path().join("chunk.rgo");
        let bytes = chunk(&[(9, 1), (19, 2)]);
        fs::write(&source, &bytes).unwrap();
        sync_directory(intake.path()).unwrap();
        {
            let target = match operation {
                faults::Operation::Remove => source.clone(),
                faults::Operation::Sync => intake.path().to_owned(),
            };
            let _fault = faults::inject([(operation, target)]);
            assert!(matches!(
                store.ingest_file(1, &source),
                Err(ReplayError::Io(_))
            ));
            assert_requires_reopen(&mut store, root.path());
        }
        assert_eq!(source.exists(), operation == faults::Operation::Remove);
        assert_eq!(fs::read(chunk_path(&store.chunk_dir, 1)).unwrap(), bytes);
        drop(store);
        let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
        assert_eq!(store.stats().unwrap().retained_records, 2);
        assert_eq!(tags(&store), BTreeSet::from([1, 2]));
        if operation == faults::Operation::Sync {
            // Retrying a missing filename must finish the failed directory
            // sync before it can acknowledge cleanup.
            {
                let _fault = faults::inject([(faults::Operation::Sync, intake.path().to_owned())]);
                assert!(matches!(
                    store.ingest_file(1, &source),
                    Err(ReplayError::Io(_))
                ));
                assert_requires_reopen(&mut store, root.path());
            }
            drop(store);
            store = ReplayStore::open(root.path(), config(1)).unwrap();
        }
        assert_eq!(
            store.ingest_file(1, &source).unwrap(),
            IngestOutcome::AlreadyAccepted { records: 2 }
        );
        assert!(!source.exists());
        assert_eq!(store.stats().unwrap().retained_records, 2);
        assert_eq!(tags(&store), BTreeSet::from([1, 2]));
    }
}

#[cfg(unix)]
#[test]
fn failed_postcommit_catalog_directory_sync_prevents_intake_and_eviction_cleanup() {
    let root = TempDir::new().unwrap();
    let intake = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    let source = intake.path().join("chunk.rgo");
    fs::write(&source, chunk(&[(19, 2)])).unwrap();
    sync_directory(intake.path()).unwrap();
    {
        let _fault = faults::inject_after(faults::Operation::Sync, root.path().to_owned(), 1);
        assert!(matches!(
            store.ingest_file(2, &source),
            Err(ReplayError::Io(_))
        ));
        assert_requires_reopen(&mut store, root.path());
    }
    // Registration committed, but neither intake nor the old file can be
    // deleted until the explicit catalog directory barrier succeeds.
    assert!(source.exists());
    assert!(chunk_path(&store.chunk_dir, 1).exists());
    assert!(chunk_path(&store.chunk_dir, 2).exists());
    drop(store);
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([2]));
    assert!(!chunk_path(&store.chunk_dir, 1).exists());
    assert_eq!(
        store.ingest_file(2, &source).unwrap(),
        IngestOutcome::AlreadyAccepted { records: 1 }
    );
    assert!(!source.exists());
}

#[test]
fn failed_link_publication_preserves_intake_and_requires_reopen() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    let blocked = chunk_path(&store.chunk_dir, 1);
    fs::create_dir(&blocked).unwrap();
    let bytes = chunk(&[(9, 1)]);
    assert!(matches!(store.ingest(1, &bytes), Err(ReplayError::Io(_))));
    let source = intake_path(&store, 1);
    assert_eq!(fs::read(&source).unwrap(), bytes);
    assert_requires_reopen(&mut store, root.path());
    drop(store);
    assert!(matches!(
        ReplayStore::open(root.path(), config(1)),
        Err(ReplayError::Io(_))
    ));
    fs::remove_dir(blocked).unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    store.ingest_file(1, &source).unwrap();
    assert!(!source.exists());
    assert_eq!(tags(&store), BTreeSet::from([1]));
}

#[cfg(unix)]
#[test]
fn existing_replay_under_execute_only_ancestor_does_not_sync_ancestors() {
    use std::os::unix::fs::PermissionsExt;

    struct RestorePermissions(PathBuf, fs::Permissions);
    impl Drop for RestorePermissions {
        fn drop(&mut self) {
            fs::set_permissions(&self.0, self.1.clone()).unwrap();
        }
    }

    let base = TempDir::new().unwrap();
    let ancestor = base.path().join("execute-only");
    let root = ancestor.join("replay");
    fs::create_dir(&ancestor).unwrap();
    fs::create_dir(&root).unwrap();
    sync_directory(&ancestor).unwrap();
    sync_directory(base.path()).unwrap();
    let _permissions = RestorePermissions(
        ancestor.clone(),
        fs::metadata(&ancestor).unwrap().permissions(),
    );
    fs::set_permissions(&ancestor, fs::Permissions::from_mode(0o111)).unwrap();
    // Also catch unnecessary ancestor syncs when tests run as a privileged user.
    let _fault = faults::inject([(faults::Operation::Sync, ancestor)]);
    let mut store = ReplayStore::open(&root, config(1)).unwrap();
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    assert!(
        faults::synced()
            .iter()
            .all(|path| path == &root || path == &root.join("chunks"))
    );
    drop(store);
    let store = ReplayStore::open(&root, config(1)).unwrap();
    assert_eq!(tags(&store), BTreeSet::from([1]));
}
