use crate::inference::{
    inputs::{NNInput, NUM_GLOBAL_FEATURES, NUM_SPATIAL_FEATURES},
    policy::{BOARD_POLICY_SIZE, POLICY_SIZE},
};
use sha2::{Digest, Sha256};

pub(super) const CHUNK_MAGIC: [u8; 8] = *b"RGOCHNK\0";
pub(super) const CHUNK_FORMAT_VERSION: u32 = 1;
pub(super) const CHUNK_HEADER_SIZE: usize = 8 + 7 * size_of::<u32>();
pub(super) const CHUNK_CHECKSUM_SIZE: usize = 32;
pub(super) const TRAINING_RECORD_SIZE: usize =
    (NUM_SPATIAL_FEATURES * BOARD_POLICY_SIZE + NUM_GLOBAL_FEATURES + POLICY_SIZE + 3)
        * size_of::<f32>()
        + BOARD_POLICY_SIZE;

#[derive(Debug, PartialEq)]
pub(super) struct ValueTarget {
    pub(super) win_probability: f32,
    pub(super) score_mean: f32,
    pub(super) score_stdev: f32,
    /// 0 = opponent, 1 = neutral, 2 = current player.
    pub(super) ownership: [u8; BOARD_POLICY_SIZE],
}

// Eventually this will become a packed representation for network transfer.
// Record finalization is already the allocation boundary where packing belongs.
pub(super) struct TrainingSample {
    pub(super) input: NNInput,
    pub(super) policy_target: [f32; POLICY_SIZE],
    pub(super) value_target: ValueTarget,
}

/// Encodes a versioned chunk without relying on Rust's in-memory struct layout.
/// All numeric fields are little-endian, every record has a fixed size, and a
/// SHA-256 footer covers the complete header and record payload.
pub(super) fn encode_chunk(samples: &[TrainingSample]) -> Vec<u8> {
    let num_records = u32::try_from(samples.len()).expect("chunk record count exceeds u32");
    let mut bytes = Vec::with_capacity(
        CHUNK_HEADER_SIZE
            .checked_add(
                samples
                    .len()
                    .checked_mul(TRAINING_RECORD_SIZE)
                    .expect("encoded chunk size overflow"),
            )
            .and_then(|size| size.checked_add(CHUNK_CHECKSUM_SIZE))
            .expect("encoded chunk size overflow"),
    );

    bytes.extend_from_slice(&CHUNK_MAGIC);
    for value in [
        CHUNK_FORMAT_VERSION,
        TRAINING_RECORD_SIZE as u32,
        num_records,
        BOARD_POLICY_SIZE as u32,
        POLICY_SIZE as u32,
        NUM_SPATIAL_FEATURES as u32,
        NUM_GLOBAL_FEATURES as u32,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    for sample in samples {
        debug_assert!(
            !sample.input.include_ownership,
            "ownership requests are inference metadata, not training input"
        );
        extend_f32s(&mut bytes, &sample.input.spatial);
        extend_f32s(&mut bytes, &sample.input.global);
        extend_f32s(&mut bytes, &sample.policy_target);
        extend_f32s(
            &mut bytes,
            &[
                sample.value_target.win_probability,
                sample.value_target.score_mean,
                sample.value_target.score_stdev,
            ],
        );
        bytes.extend_from_slice(&sample.value_target.ownership);
    }

    debug_assert_eq!(
        bytes.len(),
        CHUNK_HEADER_SIZE + samples.len() * TRAINING_RECORD_SIZE
    );
    let checksum = Sha256::digest(&bytes);
    bytes.extend_from_slice(&checksum);
    bytes
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{game_state::GameState, rules::Rules};

    #[test]
    fn chunk_encoding_has_a_versioned_header_and_fixed_size_records() {
        let sample = TrainingSample {
            input: NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH)),
            policy_target: [0.25; POLICY_SIZE],
            value_target: ValueTarget {
                win_probability: 0.75,
                score_mean: 3.5,
                score_stdev: 1.25,
                ownership: [2; BOARD_POLICY_SIZE],
            },
        };

        let bytes = encode_chunk(&[sample]);

        assert_eq!(&bytes[..8], &CHUNK_MAGIC);
        assert_eq!(read_u32(&bytes, 8), CHUNK_FORMAT_VERSION);
        assert_eq!(read_u32(&bytes, 12), TRAINING_RECORD_SIZE as u32);
        assert_eq!(read_u32(&bytes, 16), 1);
        assert_eq!(read_u32(&bytes, 20), BOARD_POLICY_SIZE as u32);
        assert_eq!(read_u32(&bytes, 24), POLICY_SIZE as u32);
        assert_eq!(read_u32(&bytes, 28), NUM_SPATIAL_FEATURES as u32);
        assert_eq!(read_u32(&bytes, 32), NUM_GLOBAL_FEATURES as u32);
        assert_eq!(
            bytes.len(),
            CHUNK_HEADER_SIZE + TRAINING_RECORD_SIZE + CHUNK_CHECKSUM_SIZE
        );
        assert!(verify_chunk_checksum(&bytes));

        let policy_offset = CHUNK_HEADER_SIZE
            + (NUM_SPATIAL_FEATURES * BOARD_POLICY_SIZE + NUM_GLOBAL_FEATURES) * size_of::<f32>();
        assert_eq!(read_f32(&bytes, policy_offset), 0.25);
        let value_offset = policy_offset + POLICY_SIZE * size_of::<f32>();
        assert_eq!(read_f32(&bytes, value_offset), 0.75);
        assert_eq!(read_f32(&bytes, value_offset + 4), 3.5);
        assert_eq!(read_f32(&bytes, value_offset + 8), 1.25);
        assert_eq!(bytes[value_offset + 12], 2);
    }

    #[test]
    fn chunk_checksum_rejects_corrupted_payload() {
        let sample = TrainingSample {
            input: NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH)),
            policy_target: [0.25; POLICY_SIZE],
            value_target: ValueTarget {
                win_probability: 0.75,
                score_mean: 3.5,
                score_stdev: 1.25,
                ownership: [2; BOARD_POLICY_SIZE],
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
