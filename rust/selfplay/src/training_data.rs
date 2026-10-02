use crate::game::board::MAX_BOARD_AREA;
use crate::inference::{
    inputs::{NNInput, NUM_GLOBAL_FEATURES, NUM_SPATIAL_FEATURES},
    policy::{MAX_POLICY_SIZE, active_rows},
};
use sha2::{Digest, Sha256};

pub(super) const CHUNK_MAGIC: [u8; 8] = *b"RGOCHNK\0";
pub(super) const CHUNK_FORMAT_VERSION: u32 = 1;
pub(super) const CHUNK_HEADER_SIZE: usize = 8 + 4 * size_of::<u32>();
pub(super) const CHUNK_CHECKSUM_SIZE: usize = 32;
/// Each record starts with its board size, followed by compact active arrays.
pub(super) fn training_record_size(board_dim: usize) -> usize {
    let board_area = board_dim * board_dim;
    size_of::<u32>()
        + (NUM_SPATIAL_FEATURES * board_area + NUM_GLOBAL_FEATURES + board_area + 1 + 3)
            * size_of::<f32>()
        + board_area
}

#[derive(Debug, PartialEq)]
pub(super) struct ValueTarget {
    pub(super) win_probability: f32,
    pub(super) score_mean: f32,
    pub(super) score_stdev: f32,
    /// 0 = opponent, 1 = neutral, 2 = current player.
    pub(super) ownership: [u8; MAX_BOARD_AREA],
}

// In-memory arrays retain capacity. Policy uses an active prefix including pass;
// spatial inputs and ownership retain the maximum row stride.
pub(super) struct TrainingSample {
    pub(super) input: NNInput,
    pub(super) policy_target: [f32; MAX_POLICY_SIZE],
    pub(super) value_target: ValueTarget,
}

pub(super) struct EncodedChunk {
    pub(super) bytes: Vec<u8>,
    pub(super) records: usize,
}

/// Builds one versioned chunk directly in its final byte representation.
///
/// Full chunks hash each record as it is encoded. Only the final partial chunk
/// needs a second hashing pass because its record count is not known when its
/// header is initialized.
pub(super) struct ChunkEncoder {
    record_capacity: usize,
    record_count: usize,
    bytes: Vec<u8>,
    checksum: Sha256,
}

impl ChunkEncoder {
    pub(super) fn new(record_capacity: usize) -> Self {
        Self::with_buffer(record_capacity, Vec::new())
    }

    pub(super) fn with_buffer(record_capacity: usize, mut bytes: Vec<u8>) -> Self {
        assert!(record_capacity > 0, "training chunks must be nonempty");
        let header = chunk_header(record_capacity);
        bytes.clear();
        bytes.reserve(CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE);
        bytes.extend_from_slice(&header);
        let mut checksum = Sha256::new();
        checksum.update(header);
        Self {
            record_capacity,
            record_count: 0,
            bytes,
            checksum,
        }
    }

    pub(super) fn record_count(&self) -> usize {
        self.record_count
    }

    pub(super) fn remaining_capacity(&self) -> usize {
        self.record_capacity - self.record_count
    }

    pub(super) fn push(&mut self, sample: &TrainingSample) {
        debug_assert!(
            self.record_count < self.record_capacity,
            "training chunk capacity exceeded"
        );
        let record_start = self.bytes.len();
        let board_dim = sample.input.board_dim;
        self.bytes
            .reserve(training_record_size(board_dim) + CHUNK_CHECKSUM_SIZE);
        debug_assert!(
            !sample.input.include_ownership,
            "ownership requests are inference metadata, not training input"
        );
        self.bytes
            .extend_from_slice(&(board_dim as u32).to_le_bytes());
        for row in sample.input.spatial_rows() {
            extend_f32s(&mut self.bytes, row);
        }
        extend_f32s(&mut self.bytes, &sample.input.global);
        extend_f32s(
            &mut self.bytes,
            &sample.policy_target[..board_dim * board_dim + 1],
        );
        extend_f32s(
            &mut self.bytes,
            &[
                sample.value_target.win_probability,
                sample.value_target.score_mean,
                sample.value_target.score_stdev,
            ],
        );
        for row in active_rows(&sample.value_target.ownership, board_dim) {
            self.bytes.extend_from_slice(row);
        }
        debug_assert_eq!(
            self.bytes.len() - record_start,
            training_record_size(board_dim)
        );
        self.checksum.update(&self.bytes[record_start..]);
        self.record_count += 1;
    }

    pub(super) fn finish(mut self) -> EncodedChunk {
        debug_assert!(self.record_count > 0, "cannot finish an empty chunk");

        let checksum = if self.record_count == self.record_capacity {
            self.checksum.finalize()
        } else {
            self.bytes[..CHUNK_HEADER_SIZE].copy_from_slice(&chunk_header(self.record_count));
            Sha256::digest(&self.bytes)
        };
        self.bytes.extend_from_slice(&checksum);
        EncodedChunk {
            bytes: self.bytes,
            records: self.record_count,
        }
    }
}

/// Test helper for encoding a complete sample slice. Production chunk assembly
/// encodes records incrementally.
#[cfg(test)]
pub(super) fn encode_chunk(samples: &[TrainingSample]) -> Vec<u8> {
    let mut encoder = ChunkEncoder::new(samples.len());
    for sample in samples {
        encoder.push(sample);
    }
    encoder.finish().bytes
}

pub(super) fn verify_chunk_checksum(bytes: &[u8]) -> bool {
    let Some(payload_len) = bytes.len().checked_sub(CHUNK_CHECKSUM_SIZE) else {
        return false;
    };
    let (payload, expected) = bytes.split_at(payload_len);
    Sha256::digest(payload).as_slice() == expected
}

fn extend_f32s(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn chunk_header(record_count: usize) -> [u8; CHUNK_HEADER_SIZE] {
    let record_count = u32::try_from(record_count).expect("chunk record count exceeds u32");
    let mut header = [0; CHUNK_HEADER_SIZE];
    header[..CHUNK_MAGIC.len()].copy_from_slice(&CHUNK_MAGIC);
    let mut offset = CHUNK_MAGIC.len();
    for value in [
        CHUNK_FORMAT_VERSION,
        NUM_SPATIAL_FEATURES as u32,
        record_count,
        NUM_GLOBAL_FEATURES as u32,
    ] {
        header[offset..offset + size_of::<u32>()].copy_from_slice(&value.to_le_bytes());
        offset += size_of::<u32>();
    }
    header
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{game_state::GameState, rules::Rules};

    fn sample(board_dim: usize) -> TrainingSample {
        let mut input = NNInput::encode(&GameState::new(Rules {
            board_dim,
            ..Rules::TROMP_TAYLORISH_9
        }));
        // Unique values across rows and planes catch accidental flat truncation.
        input.spatial = std::array::from_fn(|i| i as f32);
        TrainingSample {
            input,
            policy_target: std::array::from_fn(|i| i as f32 + 0.25),
            value_target: ValueTarget {
                win_probability: 0.75,
                score_mean: 3.5,
                score_stdev: 1.25,
                ownership: std::array::from_fn(|i| (i % 3) as u8),
            },
        }
    }

    #[test]
    fn chunk_encoding_stores_only_active_cells_and_pass_for_all_board_dims() {
        use crate::game::board::MAX_BOARD_DIM;
        for board_dim in 1..=MAX_BOARD_DIM {
            let sample = sample(board_dim);
            let bytes = encode_chunk(&[sample]);
            assert_eq!(&bytes[..8], &CHUNK_MAGIC);
            assert_eq!(read_u32(&bytes, 8), CHUNK_FORMAT_VERSION);
            assert_eq!(read_u32(&bytes, 12), NUM_SPATIAL_FEATURES as u32);
            assert_eq!(read_u32(&bytes, 16), 1);
            assert_eq!(read_u32(&bytes, 20), NUM_GLOBAL_FEATURES as u32);
            assert_eq!(
                bytes.len(),
                CHUNK_HEADER_SIZE + training_record_size(board_dim) + CHUNK_CHECKSUM_SIZE
            );
            assert!(verify_chunk_checksum(&bytes));

            let mut offset = CHUNK_HEADER_SIZE;
            assert_eq!(read_u32(&bytes, offset), board_dim as u32);
            offset += 4;
            for plane in 0..NUM_SPATIAL_FEATURES {
                for y in 0..board_dim {
                    for x in 0..board_dim {
                        assert_eq!(
                            read_f32(&bytes, offset),
                            (plane * MAX_BOARD_AREA + y * MAX_BOARD_DIM + x) as f32
                        );
                        offset += 4;
                    }
                }
            }
            for global in [-7.5, 0.0] {
                assert_eq!(read_f32(&bytes, offset), global);
                offset += 4;
            }
            for y in 0..board_dim {
                for x in 0..board_dim {
                    assert_eq!(read_f32(&bytes, offset), (y * board_dim + x) as f32 + 0.25);
                    offset += 4;
                }
            }
            assert_eq!(
                read_f32(&bytes, offset),
                (board_dim * board_dim) as f32 + 0.25
            );
            offset += 4;
            for value in [0.75, 3.5, 1.25] {
                assert_eq!(read_f32(&bytes, offset), value);
                offset += 4;
            }
            for y in 0..board_dim {
                for x in 0..board_dim {
                    assert_eq!(bytes[offset], ((y * MAX_BOARD_DIM + x) % 3) as u8);
                    offset += 1;
                }
            }
            assert_eq!(offset + CHUNK_CHECKSUM_SIZE, bytes.len());
        }
    }

    #[test]
    fn mixed_dim_records_are_self_describing_in_full_and_partial_chunks() {
        for capacity in [3, 5] {
            let mut encoder = ChunkEncoder::new(capacity);
            for board_dim in [9, 3, 5] {
                encoder.push(&sample(board_dim));
            }
            let chunk = encoder.finish();
            assert_eq!(chunk.records, 3);
            assert_eq!(read_u32(&chunk.bytes, 16), 3);
            let mut offset = CHUNK_HEADER_SIZE;
            for board_dim in [9, 3, 5] {
                assert_eq!(read_u32(&chunk.bytes, offset), board_dim);
                offset += training_record_size(board_dim as usize);
            }
            assert_eq!(offset + CHUNK_CHECKSUM_SIZE, chunk.bytes.len());
            assert!(verify_chunk_checksum(&chunk.bytes));
        }
    }

    #[test]
    fn chunk_checksum_rejects_corrupted_payload() {
        let sample = TrainingSample {
            input: NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH_9)),
            policy_target: [0.25; MAX_POLICY_SIZE],
            value_target: ValueTarget {
                win_probability: 0.75,
                score_mean: 3.5,
                score_stdev: 1.25,
                ownership: [2; MAX_BOARD_AREA],
            },
        };
        let mut bytes = encode_chunk(&[sample]);

        bytes[CHUNK_HEADER_SIZE] ^= 1;

        assert!(!verify_chunk_checksum(&bytes));
    }

    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }
}
