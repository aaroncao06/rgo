//! Encoding self-play inputs and targets into packed training chunk bytes.

use crate::{
    game::board::MAX_BOARD_AREA,
    inference::{
        inputs::{NNInput, NUM_GLOBAL_FEATURES, NUM_SPATIAL_FEATURES},
        policy::active_rows,
    },
    search::worker::PolicyTarget,
};
use half::f16;
use rgo_artifacts::chunk::{
    CHUNK_CHECKSUM_SIZE, CHUNK_HEADER_SIZE, chunk_header, training_record_size,
};
use sha2::{Digest, Sha256};

// The encoder's feature layout must agree with the shared file schema.
const _: () = assert!(NUM_SPATIAL_FEATURES == rgo_artifacts::chunk::NUM_SPATIAL_FEATURES);
const _: () = assert!(NUM_GLOBAL_FEATURES == rgo_artifacts::chunk::NUM_GLOBAL_FEATURES);
const _: () = assert!(crate::game::board::MAX_BOARD_DIM == rgo_artifacts::chunk::MAX_BOARD_DIM);

/// Builds one versioned chunk directly in its final byte representation.
///
/// Records are hashed as they are encoded. Finishing needs a second hashing
/// pass only when the final record count differs from the initialized header,
/// such as after changing capacity or flushing a partial chunk.
/// Finished bytes stay in the encoder; reset it after publication to reuse the
/// allocation for the next chunk.
pub(super) struct ChunkEncoder {
    record_capacity: usize,
    record_count: usize,
    bytes: Vec<u8>,
    checksum: Sha256,
    finished: bool,
}

impl ChunkEncoder {
    pub(super) fn new(record_capacity: usize) -> Self {
        let mut encoder = Self {
            record_capacity: 0,
            record_count: 0,
            bytes: Vec::new(),
            checksum: Sha256::new(),
            finished: false,
        };
        encoder.reset(record_capacity);
        encoder
    }

    pub(super) fn reset(&mut self, record_capacity: usize) {
        debug_assert!(record_capacity > 0, "training chunks must be nonempty");
        let header = chunk_header(record_capacity);
        self.bytes.clear();
        self.bytes.reserve(CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE);
        self.bytes.extend_from_slice(&header);
        self.checksum = Sha256::new();
        self.checksum.update(header);
        self.record_capacity = record_capacity;
        self.record_count = 0;
        self.finished = false;
    }

    /// Reset an empty chunk's target, preserving records in a nonempty chunk.
    pub(super) fn set_capacity(&mut self, record_capacity: usize) {
        debug_assert!(record_capacity > 0 && record_capacity >= self.record_count);
        if self.record_count == 0 {
            self.reset(record_capacity);
        } else {
            debug_assert!(!self.finished, "reset the encoder after finishing");
            // Leave the header unchanged so the incremental checksum stays valid.
            // finish() writes the actual record count and rehashes if needed.
            self.record_capacity = record_capacity;
        }
    }

    /// Records awaiting finalization; finishing clears this count.
    pub(super) fn record_count(&self) -> usize {
        self.record_count
    }

    pub(super) fn remaining_capacity(&self) -> usize {
        self.record_capacity - self.record_count
    }

    pub(super) fn push(
        &mut self,
        input: &NNInput,
        policy_target: &PolicyTarget,
        win_target: f32,
        final_score: f32,
        ownership: &[u8; MAX_BOARD_AREA],
    ) {
        debug_assert!(
            !self.finished,
            "reset the encoder before pushing more records"
        );
        debug_assert!(
            self.record_count < self.record_capacity,
            "training chunk capacity exceeded"
        );
        let record_start = self.bytes.len();
        let board_dim = input.board_dim;
        self.bytes
            .reserve(training_record_size(board_dim) + CHUNK_CHECKSUM_SIZE);
        debug_assert!(
            !input.include_ownership,
            "ownership requests are inference metadata, not training input"
        );
        self.bytes.push(board_dim as u8);
        for plane in input.spatial.as_chunks::<MAX_BOARD_AREA>().0 {
            let cells = active_rows(plane, board_dim).flatten().map(|&value| {
                debug_assert!(value <= 1, "spatial features must be binary");
                value
            });
            extend_packed::<1>(&mut self.bytes, cells);
        }
        extend_f32s(&mut self.bytes, &input.global);
        extend_f16s(&mut self.bytes, &policy_target[..board_dim * board_dim + 1]);
        extend_f32s(&mut self.bytes, &[win_target, final_score]);
        let ownership = active_rows(ownership, board_dim).flatten().map(|&value| {
            debug_assert!(value <= 2, "invalid ownership label");
            value
        });
        extend_packed::<2>(&mut self.bytes, ownership);
        debug_assert_eq!(
            self.bytes.len() - record_start,
            training_record_size(board_dim)
        );
        self.checksum.update(&self.bytes[record_start..]);
        self.record_count += 1;
    }

    pub(super) fn finish(&mut self) -> &[u8] {
        debug_assert!(!self.finished, "chunk already finished");
        debug_assert!(self.record_count > 0, "cannot finish an empty chunk");

        let header = chunk_header(self.record_count);
        let checksum = if self.bytes[..CHUNK_HEADER_SIZE] == header {
            std::mem::take(&mut self.checksum).finalize()
        } else {
            self.bytes[..CHUNK_HEADER_SIZE].copy_from_slice(&header);
            Sha256::digest(&self.bytes)
        };
        self.bytes.extend_from_slice(&checksum);
        self.record_count = 0;
        self.finished = true;
        &self.bytes
    }
}

fn extend_f32s(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn extend_f16s(bytes: &mut Vec<u8>, values: &[f16]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn extend_packed<const BITS: u32>(bytes: &mut Vec<u8>, values: impl Iterator<Item = u8>) {
    let mut byte = 0;
    let mut shift = 0;
    for value in values {
        byte |= value << shift;
        shift += BITS;
        if shift == 8 {
            bytes.push(byte);
            byte = 0;
            shift = 0;
        }
    }
    if shift != 0 {
        bytes.push(byte);
    }
}

#[cfg(test)]
pub(super) mod test_support {
    use super::*;

    pub struct ValueTarget {
        pub win_target: f32,
        pub final_score: f32,
        pub ownership: [u8; MAX_BOARD_AREA],
    }

    // Owned snapshots exist only in tests to compare replayed records.
    pub struct TestSample {
        pub input: NNInput,
        pub policy_target: PolicyTarget,
        pub value_target: ValueTarget,
    }

    impl TestSample {
        pub fn push_into(&self, encoder: &mut ChunkEncoder) {
            encoder.push(
                &self.input,
                &self.policy_target,
                self.value_target.win_target,
                self.value_target.final_score,
                &self.value_target.ownership,
            );
        }
    }

    /// Test helper for encoding a complete sample slice. Production chunk assembly
    /// encodes records incrementally.
    pub fn encode_chunk(samples: &[TestSample]) -> Vec<u8> {
        let mut encoder = ChunkEncoder::new(samples.len());
        for sample in samples {
            sample.push_into(&mut encoder);
        }
        encoder.finish().to_vec()
    }
}

#[cfg(test)]
mod tests;
