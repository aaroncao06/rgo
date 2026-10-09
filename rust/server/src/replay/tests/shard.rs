use super::*;

#[test]
fn mixed_board_npz_preserves_fields_and_samples_without_replacement() {
    let root = TempDir::new().unwrap();
    let mut settings = config(100);
    settings.sample_group_records = 1;
    settings.sample_groups_per_shard = 3;
    let mut store = ReplayStore::open(root.path(), settings).unwrap();
    let inputs = [
        chunk(&[(9, 1), (13, 2)]),
        chunk(&[(13, 3), (9, 4), (19, 5)]),
        chunk(&[(19, 6)]),
    ];
    for (id, bytes) in inputs.iter().enumerate() {
        store.ingest(id as u128, bytes).unwrap();
    }
    let before = store.stats().unwrap();
    let output = samples(&mut store, 1000, 7);
    assert_eq!(
        shard_tags(&output).into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([1, 2, 3, 4, 5, 6])
    );
    assert_eq!(shard_tags(&output).len(), 6);
    assert_eq!(array::<u32>(&output, "format_version"), (vec![], vec![1]));
    assert_eq!(
        array::<u8>(&output, "board_sizes"),
        (vec![3], vec![9, 13, 19])
    );
    for dim in [9usize, 13, 19] {
        let (shape, spatial) = array::<u8>(&output, &format!("{dim}/spatial"));
        let n = shape[0] as usize;
        let packed = (dim * dim).div_ceil(8);
        assert_eq!(shape, vec![n as u64, 3, packed as u64]);
        let (global_shape, global) = array::<f32>(&output, &format!("{dim}/global"));
        let (policy_shape, policy) = array::<npyz::half::f16>(&output, &format!("{dim}/policy"));
        let (value_shape, value) = array::<f32>(&output, &format!("{dim}/value"));
        let (ownership_shape, ownership) = array::<u8>(&output, &format!("{dim}/ownership"));
        assert_eq!(global_shape, vec![n as u64, 2]);
        assert_eq!(policy_shape, vec![n as u64, (dim * dim + 1) as u64]);
        assert_eq!(value_shape, vec![n as u64, 2]);
        assert_eq!(
            ownership_shape,
            vec![n as u64, (2 * dim * dim).div_ceil(8) as u64]
        );
        for row in 0..n {
            let tag = spatial[row * 3 * packed];
            assert!(
                spatial[row * 3 * packed..(row + 1) * 3 * packed]
                    .iter()
                    .all(|&x| x == tag)
            );
            let float_bits = u32::from_le_bytes([tag; 4]);
            assert!(
                global[row * 2..row * 2 + 2]
                    .iter()
                    .all(|x| x.to_bits() == float_bits)
            );
            assert!(
                value[row * 2..row * 2 + 2]
                    .iter()
                    .all(|x| x.to_bits() == float_bits)
            );
            assert!(
                policy[row * (dim * dim + 1)..(row + 1) * (dim * dim + 1)]
                    .iter()
                    .all(|x| x.to_bits() == u16::from_le_bytes([tag; 2]))
            );
            let width = (2 * dim * dim).div_ceil(8);
            assert!(
                ownership[row * width..(row + 1) * width]
                    .iter()
                    .all(|&x| x == tag)
            );
        }
    }
    let subset = samples(&mut store, 3, 7);
    let selected = shard_tags(&subset);
    assert_eq!(selected.len(), 3);
    assert_eq!(selected.iter().copied().collect::<BTreeSet<_>>().len(), 3);
    for (id, bytes) in inputs.iter().enumerate() {
        assert_eq!(
            fs::read(chunk_path(&store.chunk_dir, id as u128)).unwrap(),
            *bytes
        );
    }
    assert_eq!(store.stats().unwrap(), before);
    // Optional cross-language verification fixture, outside the repository.
    if let Some(path) = std::env::var_os("RGO_NPZ_VERIFY_PATH") {
        fs::write(path, output).unwrap();
    }
}

#[cfg(unix)]
#[test]
fn failed_shard_directory_sync_leaves_published_file_and_replay_usable() {
    let root = TempDir::new().unwrap();
    let shard_dir = TempDir::new().unwrap();
    let mut settings = config(100);
    settings.sample_group_records = 1;
    settings.shard_records = 1;
    let mut store = ReplayStore::open(root.path(), settings).unwrap();
    for id in 1..=2 {
        store.ingest(id, &chunk(&[(9, id as u8)])).unwrap();
    }
    let before = store.stats().unwrap();
    let destination = shard_dir.path().join("shard.npz");
    let _fault = faults::inject([(faults::Operation::Sync, shard_dir.path().to_owned())]);
    assert!(matches!(
        store.sample_file(&mut SmallRng::seed_from_u64(1), &destination),
        Err(ReplayError::Io(_))
    ));
    assert_eq!(shard_tags(&fs::read(&destination).unwrap()).len(), 1);
    assert_eq!(store.stats().unwrap(), before);
    // Caller removes the possibly published destination before retrying.
    fs::remove_file(&destination).unwrap();
    assert_eq!(
        store
            .sample_file(&mut SmallRng::seed_from_u64(2), &destination)
            .unwrap(),
        1
    );
    assert_eq!(store.stats().unwrap(), before);
}

#[test]
fn local_shards_publish_atomically_without_overwriting_existing_artifacts() {
    let root = TempDir::new().unwrap();
    fs::create_dir(root.path().join("replay")).unwrap();
    sync_directory(root.path()).unwrap();
    let mut store = ReplayStore::open(&root.path().join("replay"), config(100)).unwrap();
    store.ingest(1, &chunk(&[(9, 1), (13, 2)])).unwrap();
    let output_dir = root.path().join("shards");
    fs::create_dir(&output_dir).unwrap();
    let destination = output_dir.join("shard.npz");
    assert_eq!(
        store
            .sample_file(&mut SmallRng::seed_from_u64(12), &destination)
            .unwrap(),
        2
    );
    let bytes = fs::read(&destination).unwrap();
    assert_eq!(shard_tags(&bytes), vec![1, 2]);
    let before = store.stats().unwrap();
    assert!(
        store
            .sample_file(&mut SmallRng::seed_from_u64(1), &destination)
            .is_err()
    );
    assert_eq!(store.stats().unwrap(), before);
    assert_eq!(fs::read(&destination).unwrap(), bytes);
    assert_eq!(fs::read_dir(&output_dir).unwrap().count(), 1);
    let failed = output_dir.join("failed.npz");
    let source = chunk_path(&store.chunk_dir, 1);
    let unavailable = root.path().join("unavailable");
    fs::rename(&source, &unavailable).unwrap();
    assert!(
        store
            .sample_file(&mut SmallRng::seed_from_u64(1), &failed)
            .is_err()
    );
    assert!(!failed.exists());
    assert_eq!(fs::read_dir(&output_dir).unwrap().count(), 1);
    fs::rename(unavailable, source).unwrap();
    assert_eq!(store.stats().unwrap(), before);
}

#[test]
fn failed_shard_delivery_preserves_replay() {
    struct FailingWriter(usize);
    impl Seek for FailingWriter {
        fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
            Ok(0)
        }
    }
    impl Write for FailingWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.0 == 0 {
                return Err(io::ErrorKind::BrokenPipe.into());
            }
            let written = bytes.len().min(self.0);
            self.0 -= written;
            Ok(written)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    let before = store.stats().unwrap();
    assert!(matches!(
        store.sample_into(&mut SmallRng::seed_from_u64(1), &mut FailingWriter(7)),
        Err(ReplayError::Io(_))
    ));
    assert_eq!(store.stats().unwrap(), before);
    assert_eq!(tags(&store), BTreeSet::from([1]));
}
