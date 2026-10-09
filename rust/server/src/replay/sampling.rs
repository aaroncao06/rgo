//! Fresh chunk groups, in-memory row selection, and shard publication.

use rand::{Rng, seq::SliceRandom};
use rgo_artifacts::{
    chunk::{CHUNK_HEADER_SIZE, training_record_size},
    chunk_path,
};
use std::{
    fs::File,
    io::{Read, Seek, Write},
    path::Path,
};

use super::{
    IndexedChunk, ReplayError, ReplayStore, Result, chunk_reader::Record, shard,
    storage::sync_directory,
};

impl ReplayStore {
    /// Shuffle the current window and form up to `sample_groups_per_shard`
    /// disjoint groups of whole chunks, each reaching the soft input-record
    /// target if possible. Allocate the configured `shard_records` target in
    /// proportion to each group's record count, uniformly selecting without
    /// replacement within each group, and write one NPZ.
    /// Returns fewer rows if the selected groups together cannot fill the target.
    ///
    /// Each request starts fresh. Rows are unique within a shard and can recur
    /// on later requests. Group quotas use integer rounding, so this approximates
    /// window-wide mixing rather than exact uniform draws. The exclusive store
    /// handle keeps ingestion and eviction paused until the entire shard finishes.
    ///
    /// Output must be empty and seekable. Errors can leave partial bytes, which
    /// must be discarded. Accepted files are not revalidated. Memory holds one
    /// source group and its row indices, plus the accumulated selected records.
    pub fn sample_into(
        &mut self,
        rng: &mut impl Rng,
        output: &mut (impl Write + Seek),
    ) -> Result<usize> {
        self.ensure_usable()?;
        if self.retained_records == 0 {
            return Err(ReplayError::EmptyReplay);
        }
        let mut order: Vec<_> = self.chunks.iter().collect();
        order.shuffle(rng);
        let mut groups = Vec::new();
        let mut start = 0;
        // More groups than output rows cannot contribute to this shard.
        let limit = self
            .config
            .sample_groups_per_shard
            .min(self.config.shard_records);
        while start < order.len() && groups.len() < limit {
            let mut end = start;
            let mut records = 0usize;
            while end < order.len() && records < self.config.sample_group_records {
                let chunk = order[end];
                records = records
                    .checked_add(chunk.records)
                    .ok_or(ReplayError::CorruptStore { id: chunk.id })?;
                end += 1;
            }
            groups.push((&order[start..end], records));
            start = end;
        }
        let mut remaining_records: usize = groups.iter().map(|(_, records)| records).sum();
        let count = self.config.shard_records.min(remaining_records);
        let mut selected = SelectedRecords {
            bytes: Vec::new(),
            offsets: Vec::with_capacity(count),
        };
        for (chunks, records) in &groups {
            // Widen before multiplying; the last group gets the rounding remainder.
            let quota = ((count - selected.offsets.len()) as u128 * *records as u128
                / remaining_records as u128) as usize;
            sample_group(&self.chunk_dir, chunks, *records, quota, rng, &mut selected)?;
            remaining_records -= records;
        }
        selected.offsets.shuffle(rng);
        let mut boards = std::array::from_fn(|_| Vec::new());
        for offset in selected.offsets {
            let record = Record::at(&selected.bytes, offset);
            let index = match record.board_dim() {
                9 => 0,
                13 => 1,
                19 => 2,
                _ => unreachable!("board size validated at ingestion"),
            };
            boards[index].push(record);
        }
        shard::write(boards, output)?;
        Ok(count)
    }

    /// Sync and atomically publish an NPZ without replacing an existing file.
    /// Delivery files belong to the caller, outside replay retention.
    pub fn sample_file(&mut self, rng: &mut impl Rng, destination: &Path) -> Result<usize> {
        self.ensure_usable()?;
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut pending = tempfile::NamedTempFile::new_in(parent)?;
        let count = self.sample_into(rng, pending.as_file_mut())?;
        pending.as_file().sync_all()?;
        pending
            .persist_noclobber(destination)
            .map_err(|error| error.error)?;
        sync_directory(parent)?;
        Ok(count)
    }
}

struct SelectedRecords {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}

fn sample_group(
    directory: &Path,
    chunks: &[&IndexedChunk],
    record_count: usize,
    quota: usize,
    rng: &mut impl Rng,
    selected: &mut SelectedRecords,
) -> Result<()> {
    // Keep one stationary buffer for the whole group; shuffle only offsets.
    // This group's bytes and indices are freed before loading the next group.
    let byte_len = chunks.iter().map(|chunk| chunk.byte_len).sum();
    let mut bytes = vec![0; byte_len];
    let mut indices = Vec::with_capacity(record_count);
    let mut start = 0;
    for chunk in chunks {
        let end = start + chunk.byte_len;
        let chunk_bytes = &mut bytes[start..end];
        File::open(chunk_path(directory, chunk.id))?.read_exact(chunk_bytes)?;
        let mut offset = CHUNK_HEADER_SIZE;
        // Ingestion validated these immutable records; only locate boundaries.
        for _ in 0..chunk.records {
            indices.push(start + offset);
            offset += training_record_size(usize::from(chunk_bytes[offset]));
        }
        start = end;
    }
    let (indices, _) = indices.partial_shuffle(rng, quota);
    let records = indices.iter().map(|&offset| Record::at(&bytes, offset));
    let selected_bytes = records.clone().map(|record| record.bytes().len()).sum();
    selected.bytes.reserve(selected_bytes);
    for record in records {
        selected.offsets.push(selected.bytes.len());
        selected.bytes.extend_from_slice(record.bytes());
    }
    Ok(())
}
