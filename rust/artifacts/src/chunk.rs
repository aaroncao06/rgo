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

/// A structurally validated chunk borrowing the caller's immutable bytes.
///
/// Parsing checks the header, checksum, board dimensions, record boundaries, and
/// exact record count. It does not decode tensors or validate their numeric
/// values or packed labels. Parsing and iteration allocate no memory.
#[derive(Debug, Clone, Copy)]
pub struct Chunk<'a> {
    records: &'a [u8],
    record_count: usize,
}

impl<'a> Chunk<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, ChunkError> {
        if bytes.len() < CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE {
            return Err(ChunkError::TooShort {
                actual: bytes.len(),
            });
        }
        if bytes[..CHUNK_MAGIC.len()] != CHUNK_MAGIC {
            return Err(ChunkError::InvalidMagic);
        }
        let read_u32 = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let version = read_u32(8);
        if version != CHUNK_FORMAT_VERSION {
            return Err(ChunkError::UnsupportedVersion { version });
        }
        let spatial = read_u32(12);
        let global = read_u32(20);
        if spatial != NUM_SPATIAL_FEATURES as u32 || global != NUM_GLOBAL_FEATURES as u32 {
            return Err(ChunkError::InvalidFeatureCounts { spatial, global });
        }
        let record_count = read_u32(16) as usize;
        if record_count == 0 {
            return Err(ChunkError::EmptyChunk);
        }
        if !verify_chunk_checksum(bytes) {
            return Err(ChunkError::ChecksumMismatch);
        }
        let records = &bytes[CHUNK_HEADER_SIZE..bytes.len() - CHUNK_CHECKSUM_SIZE];
        let mut remaining = records;
        for record_index in 0..record_count {
            let Some(&board_dim) = remaining.first() else {
                return Err(ChunkError::TruncatedRecord {
                    record_index,
                    expected: 1,
                    remaining: 0,
                });
            };
            if !(1..=MAX_BOARD_DIM).contains(&usize::from(board_dim)) {
                return Err(ChunkError::InvalidBoardDimension {
                    record_index,
                    board_dim,
                });
            }
            let expected = training_record_size(usize::from(board_dim));
            if remaining.len() < expected {
                return Err(ChunkError::TruncatedRecord {
                    record_index,
                    expected,
                    remaining: remaining.len(),
                });
            }
            remaining = &remaining[expected..];
        }
        if !remaining.is_empty() {
            return Err(ChunkError::TrailingBytes {
                remaining: remaining.len(),
            });
        }
        Ok(Self {
            records,
            record_count,
        })
    }

    pub fn record_count(&self) -> usize {
        self.record_count
    }

    /// Iterate over packed records in file order, including their file offsets.
    pub fn records(&self) -> Records<'a> {
        Records {
            remaining: self.records,
            count: self.record_count,
            offset: CHUNK_HEADER_SIZE,
        }
    }
}

/// A borrowed packed record from a validated chunk.
#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Record<'a> {
    pub fn board_dim(&self) -> usize {
        usize::from(self.bytes[0])
    }

    /// Offset of this record's board dimension byte within the original chunk.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Full packed record, including the leading board dimension byte.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

#[derive(Debug, Clone)]
pub struct Records<'a> {
    remaining: &'a [u8],
    count: usize,
    offset: usize,
}

impl<'a> Iterator for Records<'a> {
    type Item = Record<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }
        // Chunk::parse validated every boundary before constructing this iterator.
        let size = training_record_size(usize::from(self.remaining[0]));
        let (bytes, remaining) = self.remaining.split_at(size);
        let record = Record {
            bytes,
            offset: self.offset,
        };
        self.remaining = remaining;
        self.offset += size;
        self.count -= 1;
        Some(record)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl ExactSizeIterator for Records<'_> {}
impl std::iter::FusedIterator for Records<'_> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkError {
    TooShort {
        actual: usize,
    },
    InvalidMagic,
    UnsupportedVersion {
        version: u32,
    },
    InvalidFeatureCounts {
        spatial: u32,
        global: u32,
    },
    EmptyChunk,
    ChecksumMismatch,
    InvalidBoardDimension {
        record_index: usize,
        board_dim: u8,
    },
    TruncatedRecord {
        record_index: usize,
        expected: usize,
        remaining: usize,
    },
    TrailingBytes {
        remaining: usize,
    },
}

impl std::fmt::Display for ChunkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { actual } => {
                write!(f, "chunk too short for header and checksum: {actual} bytes")
            }
            Self::InvalidMagic => f.write_str("invalid chunk magic"),
            Self::UnsupportedVersion { version } => {
                write!(f, "unsupported chunk format version: {version}")
            }
            Self::InvalidFeatureCounts { spatial, global } => write!(
                f,
                "invalid chunk feature counts: {spatial} spatial, {global} global"
            ),
            Self::EmptyChunk => f.write_str("chunk has no records"),
            Self::ChecksumMismatch => f.write_str("chunk checksum mismatch"),
            Self::InvalidBoardDimension {
                record_index,
                board_dim,
            } => write!(
                f,
                "record {record_index} has invalid board dimension {board_dim}"
            ),
            Self::TruncatedRecord {
                record_index,
                expected,
                remaining,
            } => write!(
                f,
                "record {record_index} is truncated: expected {expected} bytes, have {remaining}"
            ),
            Self::TrailingBytes { remaining } => {
                write!(f, "chunk has {remaining} trailing record bytes")
            }
        }
    }
}

impl std::error::Error for ChunkError {}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn parse_error(bytes: &[u8]) -> ChunkError {
        Chunk::parse(bytes).unwrap_err()
    }

    #[test]
    fn reader_borrows_mixed_size_records_and_reports_file_offsets() {
        let bytes = seal(payload(&[1, 9, 13, 19]));
        let chunk = Chunk::parse(&bytes).unwrap();
        assert_eq!(chunk.record_count(), 4);
        let mut records = chunk.records();
        let mut offset = CHUNK_HEADER_SIZE;
        for (index, (dim, size)) in [(1, 25), (9, 235), (13, 466), (19, 970)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(records.len(), 4 - index);
            let record = records.next().unwrap();
            assert_eq!(record.board_dim(), dim);
            assert_eq!(record.offset(), offset);
            assert_eq!(record.bytes().len(), size);
            assert_eq!(record.bytes().as_ptr(), bytes[offset..].as_ptr());
            assert!(record.bytes()[1..].iter().all(|&byte| byte == index as u8));
            offset += size;
        }
        assert_eq!(offset + CHUNK_CHECKSUM_SIZE, bytes.len());
        assert_eq!(records.len(), 0);
        assert!(records.next().is_none());
        assert!(records.next().is_none());
        assert_eq!(chunk.records().count(), 4);
    }

    #[test]
    fn reader_rejects_short_files_and_invalid_headers() {
        let bytes = seal(payload(&[9]));
        for len in 0..CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE {
            assert_eq!(
                parse_error(&bytes[..len]),
                ChunkError::TooShort { actual: len }
            );
        }
        let mut invalid = payload(&[9]);
        invalid[0] ^= 1;
        assert_eq!(parse_error(&seal(invalid)), ChunkError::InvalidMagic);

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
            assert_eq!(parse_error(&seal(invalid)), expected);
        }
        assert_eq!(
            parse_error(&seal(chunk_header(0).to_vec())),
            ChunkError::EmptyChunk
        );
    }

    #[test]
    fn reader_rejects_checksum_corruption() {
        let mut bytes = seal(payload(&[9]));
        bytes[CHUNK_HEADER_SIZE + 1] ^= 1;
        assert_eq!(parse_error(&bytes), ChunkError::ChecksumMismatch);
        bytes[CHUNK_HEADER_SIZE + 1] ^= 1;
        *bytes.last_mut().unwrap() ^= 1;
        assert_eq!(parse_error(&bytes), ChunkError::ChecksumMismatch);
    }

    #[test]
    fn reader_rejects_invalid_dimensions_and_truncated_records() {
        for board_dim in [0, 20, u8::MAX] {
            let mut invalid = payload(&[9, 13]);
            invalid[CHUNK_HEADER_SIZE + training_record_size(9)] = board_dim;
            assert_eq!(
                parse_error(&seal(invalid)),
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
                parse_error(&bytes),
                ChunkError::TruncatedRecord {
                    record_index: 1,
                    expected: if remaining == 0 { 1 } else { 466 },
                    remaining,
                }
            );
        }
    }

    #[test]
    fn reader_rejects_record_count_mismatches_and_trailing_bytes() {
        let mut too_many = payload(&[9]);
        // A huge untrusted count must fail at the first missing record, rather
        // than allocating or iterating according to that count.
        too_many[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            parse_error(&seal(too_many)),
            ChunkError::TruncatedRecord {
                record_index: 1,
                expected: 1,
                remaining: 0
            }
        );

        let mut too_few = payload(&[9, 13]);
        too_few[16..20].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(
            parse_error(&seal(too_few)),
            ChunkError::TrailingBytes { remaining: 466 }
        );
        let mut trailing = payload(&[9]);
        trailing.push(0);
        assert_eq!(
            parse_error(&seal(trailing)),
            ChunkError::TrailingBytes { remaining: 1 }
        );
    }

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
