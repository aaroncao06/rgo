use super::*;
use rgo_artifacts::chunk::chunk_header;
use sha2::{Digest, Sha256};

fn seal(mut payload: Vec<u8>) -> Vec<u8> {
    let checksum = Sha256::digest(&payload);
    payload.extend_from_slice(&checksum);
    payload
}

fn payload(board_dims: &[usize]) -> Vec<u8> {
    let mut bytes = chunk_header(board_dims.len()).to_vec();
    for (index, &dim) in board_dims.iter().enumerate() {
        let mut record = vec![index as u8; training_record_size(dim)];
        record[0] = dim as u8;
        bytes.extend_from_slice(&record);
    }
    bytes
}

fn validation_error(bytes: &[u8]) -> ChunkError {
    validate_chunk(bytes).unwrap_err()
}

#[test]
fn record_fields_decode_endianness_and_preserve_packed_bytes() {
    // 9x9 exercises packing across byte boundaries and policy's pass slot.
    let mut bytes = vec![99, 9];
    let spatial = [0x81, 1, 0x42].repeat(11);
    bytes.extend_from_slice(&spatial);
    for value in [7.5f32, -2.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let policy: Vec<u16> = (0..82).map(|i| 0x3000 + i).collect();
    for value in &policy {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in [0.5f32, -12.5] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let ownership = [0x24, 0x86, 2].repeat(7);
    bytes.extend_from_slice(&ownership);
    let record = Record::at(&bytes, 1);
    assert_eq!(record.spatial_bytes(), spatial);
    assert_eq!(record.global(), [7.5, -2.0]);
    assert_eq!(record.policy_bits().collect::<Vec<_>>(), policy);
    assert_eq!(record.value(), [0.5, -12.5]);
    assert_eq!(record.ownership_bytes(), ownership);
    assert_eq!(record.bytes().len(), training_record_size(9));
}

#[test]
fn validated_metadata_and_record_views_support_mixed_board_sizes() {
    let bytes = seal(payload(&[9, 13, 19]));
    let chunk = crate::replay::IndexedChunk::from_chunk(42, &bytes).unwrap();
    assert_eq!(chunk.id, 42);
    assert_eq!(chunk.byte_len, bytes.len());
    assert_eq!(chunk.records, 3);
    let mut offset = CHUNK_HEADER_SIZE;
    for (index, (dim, size)) in [(9, 235), (13, 466), (19, 970)].into_iter().enumerate() {
        let record = Record::at(&bytes, offset);
        assert_eq!(record.board_dim(), dim);
        assert_eq!(record.bytes().len(), size);
        assert_eq!(record.bytes().as_ptr(), bytes[offset..].as_ptr());
        assert!(record.bytes()[1..].iter().all(|&byte| byte == index as u8));
        offset += size;
    }
    assert_eq!(offset + CHUNK_CHECKSUM_SIZE, bytes.len());
}

#[test]
fn validation_rejects_short_files_and_invalid_headers() {
    let bytes = seal(payload(&[9]));
    for len in 0..CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE {
        assert_eq!(
            validation_error(&bytes[..len]),
            ChunkError::TooShort { actual: len }
        );
    }
    let mut invalid = payload(&[9]);
    invalid[0] ^= 1;
    assert_eq!(validation_error(&seal(invalid)), ChunkError::InvalidMagic);

    for (offset, value, expected) in [
        (8, 2_u32, ChunkError::UnsupportedVersion { version: 2 }),
        (
            12,
            4,
            ChunkError::InvalidFeatureCounts {
                spatial: 4,
                global: 2,
            },
        ),
        (
            20,
            3,
            ChunkError::InvalidFeatureCounts {
                spatial: 3,
                global: 3,
            },
        ),
        (16, 0, ChunkError::EmptyChunk),
    ] {
        let mut invalid = payload(&[9]);
        invalid[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert_eq!(validation_error(&seal(invalid)), expected);
    }
    assert_eq!(
        validation_error(&seal(chunk_header(0).to_vec())),
        ChunkError::EmptyChunk
    );
}

#[test]
fn validation_rejects_checksum_corruption() {
    let mut bytes = seal(payload(&[9]));
    bytes[CHUNK_HEADER_SIZE + 1] ^= 1;
    assert_eq!(validation_error(&bytes), ChunkError::ChecksumMismatch);
    bytes[CHUNK_HEADER_SIZE + 1] ^= 1;
    *bytes.last_mut().unwrap() ^= 1;
    assert_eq!(validation_error(&bytes), ChunkError::ChecksumMismatch);
}

#[test]
fn validation_rejects_invalid_dimensions_and_truncated_records() {
    for board_dim in (0..=u8::MAX).filter(|dim| ![9, 13, 19].contains(dim)) {
        let mut invalid = payload(&[9, 13]);
        invalid[CHUNK_HEADER_SIZE + training_record_size(9)] = board_dim;
        assert_eq!(
            validation_error(&seal(invalid)),
            ChunkError::InvalidBoardDimension {
                record_index: 1,
                board_dim
            }
        );
    }
    let complete = payload(&[9, 13]);
    let offset = CHUNK_HEADER_SIZE + training_record_size(9);
    for remaining in 0..training_record_size(13) {
        let bytes = seal(complete[..offset + remaining].to_vec());
        assert_eq!(
            validation_error(&bytes),
            ChunkError::TruncatedRecord {
                record_index: 1,
                expected: if remaining == 0 { 1 } else { 466 },
                remaining,
            }
        );
    }
}

#[test]
fn validation_rejects_record_count_mismatches_and_trailing_bytes() {
    let mut too_many = payload(&[9]);
    // A huge untrusted count must fail at the first missing record, rather
    // than allocating or iterating according to that count.
    too_many[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        validation_error(&seal(too_many)),
        ChunkError::TruncatedRecord {
            record_index: 1,
            expected: 1,
            remaining: 0
        }
    );

    let mut too_few = payload(&[9, 13]);
    too_few[16..20].copy_from_slice(&1_u32.to_le_bytes());
    assert_eq!(
        validation_error(&seal(too_few)),
        ChunkError::TrailingBytes { remaining: 466 }
    );
    let mut trailing = payload(&[9]);
    trailing.push(0);
    assert_eq!(
        validation_error(&seal(trailing)),
        ChunkError::TrailingBytes { remaining: 1 }
    );
}
