use crate::game::board::MAX_BOARD_AREA;
use crate::inference::{
    inputs::{NNInput, NUM_GLOBAL_FEATURES, NUM_SPATIAL_FEATURES},
    policy::active_rows,
};
use crate::search::worker::PolicyTarget;
use half::f16;
use sha2::{Digest, Sha256};

pub(super) const CHUNK_MAGIC: [u8; 8] = *b"RGOCHNK\0";
pub(super) const CHUNK_FORMAT_VERSION: u32 = 1;
pub(super) const CHUNK_HEADER_SIZE: usize = 8 + 4 * size_of::<u32>();
pub(super) const CHUNK_CHECKSUM_SIZE: usize = 32;
/// Each record stores: board dimension (u8), binary spatial planes, global
/// features (f32), policy including pass (f16), three value targets (f32), and
/// ownership labels. Only active cells are stored, in row-major order.
///
/// Each spatial plane uses one bit per cell and starts on a byte boundary;
/// ownership uses two bits per cell (0 = opponent, 1 = neutral, 2 = player).
/// Fields fill bytes from the least significant bits, with unused high bits
/// zeroed. Numeric fields remain little-endian.
pub(super) fn training_record_size(board_dim: usize) -> usize {
    let board_area = board_dim * board_dim;
    size_of::<u8>()
        + NUM_SPATIAL_FEATURES * board_area.div_ceil(8)
        + (NUM_GLOBAL_FEATURES + 3) * size_of::<f32>()
        + (board_area + 1) * size_of::<f16>()
        + (2 * board_area).div_ceil(8)
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
    pub(super) policy_target: PolicyTarget,
    pub(super) value_target: ValueTarget,
}

/// Builds one versioned chunk directly in its final byte representation.
///
/// Full chunks hash each record as it is encoded. Only the final partial chunk
/// needs a second hashing pass because its record count is not known when its
/// header is initialized.
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
        assert!(record_capacity > 0, "training chunks must be nonempty");
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

    /// Records awaiting finalization; finishing clears this count.
    pub(super) fn record_count(&self) -> usize {
        self.record_count
    }

    pub(super) fn remaining_capacity(&self) -> usize {
        self.record_capacity - self.record_count
    }

    pub(super) fn push(&mut self, sample: &TrainingSample) {
        debug_assert!(
            !self.finished,
            "reset the encoder before pushing more records"
        );
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
        self.bytes.push(board_dim as u8);
        for plane in sample.input.spatial.as_chunks::<MAX_BOARD_AREA>().0 {
            let cells = active_rows(plane, board_dim).flatten().map(|&value| {
                debug_assert!(value <= 1, "spatial features must be binary");
                value
            });
            extend_packed::<1>(&mut self.bytes, cells);
        }
        extend_f32s(&mut self.bytes, &sample.input.global);
        extend_f16s(
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
        let ownership = active_rows(&sample.value_target.ownership, board_dim)
            .flatten()
            .map(|&value| {
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

        let checksum = if self.record_count == self.record_capacity {
            std::mem::take(&mut self.checksum).finalize()
        } else {
            self.bytes[..CHUNK_HEADER_SIZE].copy_from_slice(&chunk_header(self.record_count));
            Sha256::digest(&self.bytes)
        };
        self.bytes.extend_from_slice(&checksum);
        self.record_count = 0;
        self.finished = true;
        &self.bytes
    }
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
/// Test helper for encoding a complete sample slice. Production chunk assembly
/// encodes records incrementally.
pub(super) fn encode_chunk(samples: &[TrainingSample]) -> Vec<u8> {
    let mut encoder = ChunkEncoder::new(samples.len());
    for sample in samples {
        encoder.push(sample);
    }
    encoder.finish().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::MAX_BOARD_DIM;
    use crate::game::{game_state::GameState, rules::Rules};
    use crate::inference::policy::MAX_POLICY_SIZE;

    fn sample(board_dim: usize) -> TrainingSample {
        let mut input = NNInput::encode(&GameState::new(Rules {
            board_dim,
            ..Rules::TROMP_TAYLORISH_9
        }));
        // Vary cells across rows and planes; invalid padding catches accidental
        // packing of storage cells outside the active board.
        input.spatial = std::array::from_fn(|i| {
            let cell = i % MAX_BOARD_AREA;
            if cell / MAX_BOARD_DIM < board_dim && cell % MAX_BOARD_DIM < board_dim {
                ((i * 13 + i / 5 + i / 17) % 2) as u8
            } else {
                u8::MAX
            }
        });
        input.global[1] = -0.0;
        TrainingSample {
            input,
            policy_target: std::array::from_fn(|i| {
                f16::from_f64(i as f64 / MAX_POLICY_SIZE as f64)
            }),
            value_target: ValueTarget {
                win_probability: 0.75,
                score_mean: 3.5,
                score_stdev: 1.25,
                ownership: std::array::from_fn(|i| {
                    if i / MAX_BOARD_DIM < board_dim && i % MAX_BOARD_DIM < board_dim {
                        (i % 3) as u8
                    } else {
                        3
                    }
                }),
            },
        }
    }

    #[test]
    fn chunk_encoding_stores_only_active_cells_and_pass_for_all_board_dims() {
        for board_dim in 1..=MAX_BOARD_DIM {
            let sample = sample(board_dim);
            let bytes = encode_chunk(std::slice::from_ref(&sample));
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
            assert_eq!(usize::from(bytes[offset]), board_dim);
            offset += 1;
            for plane in 0..NUM_SPATIAL_FEATURES {
                let cells = read_packed(&bytes, &mut offset, board_dim * board_dim, 1);
                for y in 0..board_dim {
                    for x in 0..board_dim {
                        assert_eq!(
                            cells[y * board_dim + x],
                            sample.input.spatial[plane * MAX_BOARD_AREA + y * MAX_BOARD_DIM + x]
                        );
                    }
                }
            }
            for value in sample.input.global {
                assert_eq!(read_f32(&bytes, offset).to_bits(), value.to_bits());
                offset += 4;
            }
            for value in &sample.policy_target[..board_dim * board_dim + 1] {
                let stored = f16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                assert_eq!(stored.to_bits(), value.to_bits());
                offset += 2;
            }
            for value in [
                sample.value_target.win_probability,
                sample.value_target.score_mean,
                sample.value_target.score_stdev,
            ] {
                assert_eq!(read_f32(&bytes, offset).to_bits(), value.to_bits());
                offset += 4;
            }
            let ownership = read_packed(&bytes, &mut offset, board_dim * board_dim, 2);
            for y in 0..board_dim {
                for x in 0..board_dim {
                    assert_eq!(
                        ownership[y * board_dim + x],
                        sample.value_target.ownership[y * MAX_BOARD_DIM + x]
                    );
                }
            }
            assert_eq!(offset + CHUNK_CHECKSUM_SIZE, bytes.len());
        }
    }

    #[test]
    fn policy_storage_uses_little_endian_binary16_and_preserves_subnormals() {
        let mut sample = sample(1);
        sample.policy_target[0] = f16::from_f64(1.0 / 3.0);
        sample.policy_target[1] = f16::from_bits(1);
        let bytes = encode_chunk(&[sample]);
        let policy_offset =
            CHUNK_HEADER_SIZE + 1 + NUM_SPATIAL_FEATURES + NUM_GLOBAL_FEATURES * size_of::<f32>();
        assert_eq!(
            &bytes[policy_offset..policy_offset + 4],
            &[0x55, 0x35, 0x01, 0x00]
        );
        assert!(verify_chunk_checksum(&bytes));
    }

    #[test]
    fn packed_fields_have_lsb_first_order_and_independent_byte_boundaries() {
        let mut sample = sample(3);
        for (plane, values) in [[1, 0, 1, 1, 0, 0, 1, 0, 1], [0; 9], [1; 9]]
            .into_iter()
            .enumerate()
        {
            for (i, value) in values.into_iter().enumerate() {
                sample.input.spatial[plane * MAX_BOARD_AREA + i / 3 * MAX_BOARD_DIM + i % 3] =
                    value;
            }
        }
        for i in 0..9 {
            sample.value_target.ownership[i / 3 * MAX_BOARD_DIM + i % 3] = (i % 3) as u8;
        }
        let bytes = encode_chunk(&[sample]);
        let spatial_offset = CHUNK_HEADER_SIZE + 1;
        assert_eq!(
            &bytes[spatial_offset..spatial_offset + 6],
            &[0x4d, 0x01, 0x00, 0x00, 0xff, 0x01]
        );
        let ownership_end = bytes.len() - CHUNK_CHECKSUM_SIZE;
        assert_eq!(
            &bytes[ownership_end - 3..ownership_end],
            &[0x24, 0x49, 0x02]
        );
    }

    #[test]
    fn packing_supports_all_inference_board_areas() {
        for board_dim in [9, 13, 19] {
            let area = board_dim * board_dim;
            let mut bytes = Vec::new();
            extend_packed::<1>(&mut bytes, (0..area).map(|i| (i % 2) as u8));
            extend_packed::<2>(&mut bytes, (0..area).map(|i| (i % 3) as u8));
            let mut offset = 0;
            assert_eq!(
                read_packed(&bytes, &mut offset, area, 1),
                (0..area).map(|i| (i % 2) as u8).collect::<Vec<_>>()
            );
            assert_eq!(
                read_packed(&bytes, &mut offset, area, 2),
                (0..area).map(|i| (i % 3) as u8).collect::<Vec<_>>()
            );
            assert_eq!(offset, bytes.len());
        }
        assert_eq!(training_record_size(9), 239);
        assert_eq!(training_record_size(13), 470);
        assert_eq!(training_record_size(19), 974);
    }

    #[test]
    fn mixed_dim_records_are_self_describing_in_full_and_partial_chunks() {
        for capacity in [3, 5] {
            let mut encoder = ChunkEncoder::new(capacity);
            for board_dim in [9, 3, 5] {
                encoder.push(&sample(board_dim));
            }
            assert_eq!(encoder.record_count(), 3);
            let bytes = encoder.finish();
            assert_eq!(read_u32(bytes, 16), 3);
            let mut offset = CHUNK_HEADER_SIZE;
            for board_dim in [9, 3, 5] {
                assert_eq!(usize::from(bytes[offset]), board_dim);
                offset += training_record_size(board_dim);
            }
            assert_eq!(offset + CHUNK_CHECKSUM_SIZE, bytes.len());
            assert!(verify_chunk_checksum(bytes));
        }
    }

    #[test]
    fn encoder_reuses_allocation_across_full_and_partial_chunks() {
        let mut encoder = ChunkEncoder::new(4);
        let mut allocation = None;
        // Start with the largest payload so subsequent resets need no growth.
        // Alternate full and partial chunks, capacities, dimensions, and data.
        for (capacity, board_dim, count) in
            [(4, 19, 4), (6, 9, 2), (3, 3, 3), (1, 19, 1), (4, 9, 3)]
        {
            encoder.reset(capacity);
            assert_eq!(encoder.record_count(), 0);
            assert_eq!(encoder.remaining_capacity(), capacity);
            let samples: Vec<_> = (0..count)
                .map(|tag| {
                    let mut sample = sample(board_dim);
                    sample.value_target.score_mean = tag as f32 + capacity as f32;
                    sample
                })
                .collect();
            for sample in &samples {
                encoder.push(sample);
            }
            assert_eq!(encoder.record_count(), count);
            assert_eq!(encoder.remaining_capacity(), capacity - count);
            let bytes = encoder.finish();
            assert_eq!(read_u32(bytes, 16), count as u32);
            assert_eq!(
                bytes.len(),
                CHUNK_HEADER_SIZE + count * training_record_size(board_dim) + CHUNK_CHECKSUM_SIZE
            );
            assert!(verify_chunk_checksum(bytes));
            assert_eq!(bytes, encode_chunk(&samples));
            assert_eq!(encoder.record_count(), 0);
            let current = (encoder.bytes.as_ptr(), encoder.bytes.capacity());
            if let Some(allocation) = allocation {
                assert_eq!(current, allocation);
            }
            allocation = Some(current);
        }
    }

    #[test]
    fn chunk_checksum_rejects_corrupted_payload() {
        let sample = TrainingSample {
            input: NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH_9)),
            policy_target: [f16::from_f32(0.25); MAX_POLICY_SIZE],
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

    fn read_packed(bytes: &[u8], offset: &mut usize, count: usize, bits: usize) -> Vec<u8> {
        let len = (count * bits).div_ceil(8);
        let packed = &bytes[*offset..*offset + len];
        let values = (0..count)
            .map(|i| (packed[i * bits / 8] >> (i * bits % 8)) & ((1 << bits) - 1))
            .collect();
        let used_bits = count * bits % 8;
        if used_bits != 0 {
            assert_eq!(
                packed[len - 1] >> used_bits,
                0,
                "nonzero trailing padding bits"
            );
        }
        *offset += len;
        values
    }
}
