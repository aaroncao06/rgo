//! Shared binary training chunk schema for self-play writers and server readers.
//!
//! A chunk contains a fixed header, variable-size records, and a SHA-256 checksum
//! over the header and records. The header stores the magic followed by four
//! little-endian u32 fields: format version, spatial feature count, record count,
//! and global feature count.

use sha2::{Digest, Sha256};

pub const CHUNK_MAGIC: [u8; 8] = *b"RGOCHNK\0";
pub const CHUNK_FORMAT_VERSION: u32 = 1;
pub const CHUNK_HEADER_SIZE: usize = 8 + 4 * size_of::<u32>();
pub const CHUNK_CHECKSUM_SIZE: usize = 32;
pub const NUM_SPATIAL_FEATURES: usize = 3;
pub const NUM_GLOBAL_FEATURES: usize = 2;
pub const MAX_BOARD_DIM: usize = 19;

/// Each record stores: board dimension (u8), binary spatial planes, global
/// features (f32), policy including pass (f16), final win/score targets (f32), and
/// ownership labels. Win targets are 0 (loss), 0.5 (draw), or 1 (win); final
/// scores and ownership are player-relative. Only active cells are stored, in
/// row-major order.
///
/// Each spatial plane uses one bit per cell and starts on a byte boundary;
/// ownership uses two bits per cell (0 = opponent, 1 = neutral, 2 = player).
/// Fields fill bytes from the least significant bits, with unused high bits
/// zeroed. Numeric fields remain little-endian.
///
/// The caller must supply a board dimension in 1..=MAX_BOARD_DIM.
pub fn training_record_size(board_dim: usize) -> usize {
    let board_area = board_dim * board_dim;
    size_of::<u8>()
        + NUM_SPATIAL_FEATURES * board_area.div_ceil(8)
        + (NUM_GLOBAL_FEATURES + 2) * size_of::<f32>()
        + (board_area + 1) * size_of::<u16>()
        + (2 * board_area).div_ceil(8)
}

/// Encode the fixed header for a chunk with the given number of records.
///
/// Panics if the record count exceeds u32.
pub fn chunk_header(record_count: usize) -> [u8; CHUNK_HEADER_SIZE] {
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

/// Check the trailing SHA-256 checksum.
///
/// This checks integrity only; readers must also validate the header and records.
pub fn verify_chunk_checksum(bytes: &[u8]) -> bool {
    let Some(payload_len) = bytes.len().checked_sub(CHUNK_CHECKSUM_SIZE) else {
        return false;
    };
    let (payload, expected) = bytes.split_at(payload_len);
    Sha256::digest(payload).as_slice() == expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_preserves_version_one_wire_layout() {
        assert_eq!(
            chunk_header(0x01020304),
            [
                b'R', b'G', b'O', b'C', b'H', b'N', b'K', 0, 1, 0, 0, 0, 3, 0, 0, 0, 4, 3, 2, 1, 2,
                0, 0, 0,
            ]
        );
    }

    #[test]
    fn record_sizes_match_packed_layout() {
        assert_eq!(training_record_size(9), 235);
        assert_eq!(training_record_size(13), 466);
        assert_eq!(training_record_size(19), 970);
    }

    #[test]
    fn checksum_detects_payload_and_checksum_corruption() {
        let mut bytes = chunk_header(1).to_vec();
        let mut record = vec![0; training_record_size(9)];
        record[0] = 9;
        bytes.extend_from_slice(&record);
        let checksum = Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        assert!(verify_chunk_checksum(&bytes));

        bytes[CHUNK_HEADER_SIZE] ^= 1;
        assert!(!verify_chunk_checksum(&bytes));
        bytes[CHUNK_HEADER_SIZE] ^= 1;
        *bytes.last_mut().unwrap() ^= 1;
        assert!(!verify_chunk_checksum(&bytes));
        assert!(!verify_chunk_checksum(&bytes[..CHUNK_CHECKSUM_SIZE - 1]));
    }
}
