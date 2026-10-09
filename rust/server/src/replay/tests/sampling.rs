use super::*;
use std::collections::BTreeMap;

#[test]
fn a_small_window_retains_all_and_is_eligible_again_next_request() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(1)).unwrap();
    let input = chunk(&[(9, 1)]);
    store.ingest(1, &input).unwrap();
    let before = store.stats().unwrap();
    for seed in [7, 91] {
        assert_eq!(shard_tags(&samples(&mut store, 8, seed)), vec![1]);
    }
    assert_eq!(store.stats().unwrap(), before);
    assert_eq!(fs::read(chunk_path(&store.chunk_dir, 1)).unwrap(), input);
}

#[test]
fn soft_groups_keep_whole_chunks_when_retaining_all() {
    let root = TempDir::new().unwrap();
    let mut settings = config(100);
    settings.sample_group_records = 4;
    settings.sample_groups_per_shard = 1;
    let mut store = ReplayStore::open(root.path(), settings).unwrap();
    store.ingest(1, &chunk(&[(9, 1), (9, 2)])).unwrap();
    store.ingest(2, &chunk(&[(9, 3), (9, 4), (9, 5)])).unwrap();
    store.ingest(3, &chunk(&[(9, 6)])).unwrap();
    for seed in 0..20 {
        let rows = shard_tags(&samples(&mut store, 100, seed));
        assert!((4..=6).contains(&rows.len()));
        let rows: BTreeSet<_> = rows.into_iter().collect();
        for tags in [&[1, 2][..], &[3, 4, 5][..], &[6][..]] {
            let included = tags.iter().filter(|tag| rows.contains(tag)).count();
            assert!(included == 0 || included == tags.len());
        }
    }
}

#[test]
fn multiple_disjoint_groups_share_a_nondivisible_shard_quota() {
    let root = TempDir::new().unwrap();
    let mut settings = config(100);
    settings.sample_group_records = 4;
    settings.sample_groups_per_shard = 2;
    let mut store = ReplayStore::open(root.path(), settings).unwrap();
    for id in 0..6 {
        let rows: Vec<_> = (1..=4).map(|row| (9, id * 4 + row)).collect();
        store.ingest(u128::from(id), &chunk(&rows)).unwrap();
    }
    for seed in 0..20 {
        let rows = shard_tags(&samples(&mut store, 5, seed));
        assert_eq!(rows.len(), 5);
        assert_eq!(rows.iter().collect::<BTreeSet<_>>().len(), 5);
        let mut counts = BTreeMap::new();
        for tag in rows {
            *counts.entry((tag - 1) / 4).or_insert(0) += 1;
        }
        let mut counts: Vec<_> = counts.into_values().collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![2, 3]);
    }
}

#[test]
fn uneven_groups_round_quotas_and_insufficient_input_retains_all() {
    let root = TempDir::new().unwrap();
    let mut settings = config(100);
    settings.sample_group_records = 1;
    settings.sample_groups_per_shard = 4;
    let mut store = ReplayStore::open(root.path(), settings).unwrap();
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    store
        .ingest(2, &chunk(&[(9, 2), (9, 3), (9, 4), (9, 5), (9, 6)]))
        .unwrap();
    let mut kept_small = false;
    let mut omitted_small = false;
    for seed in 0..20 {
        let rows = shard_tags(&samples(&mut store, 5, seed));
        assert_eq!(rows.len(), 5);
        kept_small |= rows.contains(&1);
        omitted_small |= !rows.contains(&1);
        assert_eq!(rows.iter().collect::<BTreeSet<_>>().len(), 5);
        let mut all = shard_tags(&samples(&mut store, 100, seed));
        all.sort_unstable();
        assert_eq!(all, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(shard_tags(&samples(&mut store, 1, seed)).len(), 1);
    }
    // A sub-row proportional quota may round to zero; exact totals still hold.
    assert!(kept_small && omitted_small);
}

#[test]
fn uneven_groups_receive_proportional_quotas_without_repeating_rows() {
    let root = TempDir::new().unwrap();
    let mut settings = config(1000);
    settings.capacity_records = 1000;
    settings.sample_group_records = 1;
    settings.sample_groups_per_shard = 3;
    let mut store = ReplayStore::open(root.path(), settings).unwrap();
    for (dim, records) in [(9, 200), (13, 200), (19, 20)] {
        let rows: Vec<_> = (1..=records).map(|tag| (dim, tag as u8)).collect();
        store.ingest(dim as u128, &chunk(&rows)).unwrap();
    }
    let before = store.stats().unwrap();
    for seed in 0..20 {
        let output = samples(&mut store, 100, seed);
        assert_eq!(shard_tags(&output).len(), 100);
        for (dim, expected) in [(9, 47..=49), (13, 47..=49), (19, 4..=6)] {
            let (shape, values) = array::<u8>(&output, &format!("{dim}/spatial"));
            let count = shape[0] as usize;
            assert!(expected.contains(&count), "board {dim}: {count} rows");
            let width = (shape[1] * shape[2]) as usize;
            let tags: BTreeSet<_> = values.chunks_exact(width).map(|row| row[0]).collect();
            assert_eq!(tags.len(), count);
        }
    }
    assert_eq!(store.stats().unwrap(), before);
}

#[test]
fn each_request_uses_the_current_window_including_arrivals_and_evictions() {
    let root = TempDir::new().unwrap();
    let mut settings = config(100);
    settings.sample_group_records = 1;
    let mut store = ReplayStore::open(root.path(), settings).unwrap();
    for id in 1..=3 {
        store.ingest(id, &chunk(&[(9, id as u8)])).unwrap();
    }
    assert_eq!(shard_tags(&samples(&mut store, 100, 1)).len(), 2);
    store.ingest(4, &chunk(&[(9, 4)])).unwrap();
    assert!((0..20).any(|seed| shard_tags(&samples(&mut store, 100, seed)).contains(&4)));
    store.set_window(2).unwrap();
    for seed in 0..10 {
        let mut rows = shard_tags(&samples(&mut store, 100, seed));
        rows.sort_unstable();
        assert_eq!(rows, vec![3, 4]);
    }
}

#[test]
fn subsampling_is_uniform_within_the_loaded_group_and_rows_are_shuffled() {
    let root = TempDir::new().unwrap();
    let mut settings = config(100);
    settings.shard_records = 2;
    let mut store = ReplayStore::open(root.path(), settings).unwrap();
    store
        .ingest(1, &chunk(&[(9, 1), (9, 2), (9, 3), (9, 4), (9, 5)]))
        .unwrap();
    let mut rng = SmallRng::seed_from_u64(17);
    let mut counts = [0; 6];
    for _ in 0..400 {
        let mut output = Cursor::new(Vec::new());
        assert_eq!(store.sample_into(&mut rng, &mut output).unwrap(), 2);
        let rows = shard_tags(output.get_ref());
        assert_ne!(rows[0], rows[1]);
        for row in rows {
            counts[row as usize] += 1;
        }
    }
    for count in &counts[1..] {
        assert!((115..=205).contains(count), "biased selection: {counts:?}");
    }
    let rows = shard_tags(&samples(&mut store, 100, 91));
    assert!(rows.windows(2).any(|pair| pair[0] > pair[1]));
}

#[test]
fn failed_final_flush_leaves_replay_usable() {
    struct FailFlush(Cursor<Vec<u8>>);
    impl Write for FailFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.write(bytes)
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::ErrorKind::BrokenPipe.into())
        }
    }
    impl Seek for FailFlush {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.0.seek(position)
        }
    }
    let root = TempDir::new().unwrap();
    let mut settings = config(100);
    settings.sample_group_records = 1;
    let mut store = ReplayStore::open(root.path(), settings).unwrap();
    for id in 1..=2 {
        store.ingest(id, &chunk(&[(9, id as u8)])).unwrap();
    }
    let before = store.stats().unwrap();
    assert!(matches!(
        store.sample_into(
            &mut SmallRng::seed_from_u64(1),
            &mut FailFlush(Cursor::new(Vec::new()))
        ),
        Err(ReplayError::Io(_))
    ));
    assert_eq!(store.stats().unwrap(), before);
    assert_eq!(shard_tags(&samples(&mut store, 2, 1)).len(), 2);
}

#[test]
fn failed_source_read_leaves_replay_usable_and_writes_nothing() {
    let root = TempDir::new().unwrap();
    let mut store = ReplayStore::open(root.path(), config(100)).unwrap();
    store.ingest(1, &chunk(&[(9, 1)])).unwrap();
    let before = store.stats().unwrap();
    let source = chunk_path(&store.chunk_dir, 1);
    let unavailable = root.path().join("unavailable");
    fs::rename(&source, &unavailable).unwrap();
    let mut output = Cursor::new(Vec::new());
    assert!(matches!(
        store.sample_into(&mut SmallRng::seed_from_u64(1), &mut output),
        Err(ReplayError::Io(_))
    ));
    assert_eq!(store.stats().unwrap(), before);
    assert!(output.get_ref().is_empty());
    fs::rename(unavailable, source).unwrap();
    assert_eq!(shard_tags(&samples(&mut store, 1, 1)), vec![1]);
}

#[test]
fn configured_shard_target_handles_empty_and_insufficient_replay() {
    let root = TempDir::new().unwrap();
    let mut settings = config(100);
    settings.shard_records = 3;
    let mut store = ReplayStore::open(root.path(), settings).unwrap();
    let mut output = Cursor::new(Vec::new());
    let mut rng = SmallRng::seed_from_u64(3);
    assert!(matches!(
        store.sample_into(&mut rng, &mut output),
        Err(ReplayError::EmptyReplay)
    ));
    assert!(output.get_ref().is_empty());
    store.ingest(1, &chunk(&[(9, 1), (13, 2)])).unwrap();
    assert_eq!(store.sample_into(&mut rng, &mut output).unwrap(), 2);
    assert_eq!(shard_tags(output.get_ref()).len(), 2);
    store.ingest(2, &chunk(&[(19, 3), (19, 4)])).unwrap();
    for _ in 0..3 {
        let mut output = Cursor::new(Vec::new());
        assert_eq!(store.sample_into(&mut rng, &mut output).unwrap(), 3);
        assert_eq!(shard_tags(output.get_ref()).len(), 3);
    }
}
