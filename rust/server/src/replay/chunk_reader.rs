//! Borrowed chunk validation and record field views for replay ingestion/export.

use rgo_artifacts::SUPPORTED_BOARD_DIMS;
use rgo_artifacts::chunk::{
    CHUNK_CHECKSUM_SIZE, CHUNK_FORMAT_VERSION, CHUNK_HEADER_SIZE, CHUNK_MAGIC, NUM_GLOBAL_FEATURES,
    NUM_SPATIAL_FEATURES, training_record_size, verify_chunk_checksum,
};

/// Validate an incoming chunk before accepting it into replay storage.
///
/// Checks the header, checksum, board dimensions, record boundaries, and exact
/// record count without allocating or decoding tensor fields. Accepted immutable
/// files are subsequently read through `Record::at` without repeating validation.
pub fn validate_chunk(bytes: &[u8]) -> Result<(), ChunkError> {
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
    let mut remaining = &bytes[CHUNK_HEADER_SIZE..bytes.len() - CHUNK_CHECKSUM_SIZE];
    for record_index in 0..record_count {
        let Some(&board_dim) = remaining.first() else {
            return Err(ChunkError::TruncatedRecord {
                record_index,
                expected: 1,
                remaining: 0,
            });
        };
        if !SUPPORTED_BOARD_DIMS.contains(&usize::from(board_dim)) {
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
    Ok(())
}

/// A borrowed, complete packed record. Chunk validation belongs at ingestion.
#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    bytes: &'a [u8],
}

impl<'a> Record<'a> {
    /// Borrow a record at an offset in already accepted, immutable chunk bytes.
    /// The offset and board dimension must already be validated. Ordinary slice
    /// bounds checks remain; an invalid offset or truncated record panics.
    /// Use `validate_chunk` at ingestion for header, checksum, and record checks.
    pub fn at(bytes: &'a [u8], offset: usize) -> Self {
        let dim = usize::from(bytes[offset]);
        let end = offset + training_record_size(dim);
        Self {
            bytes: &bytes[offset..end],
        }
    }

    pub fn board_dim(&self) -> usize {
        usize::from(self.bytes[0])
    }

    /// Full packed record, including the leading board dimension byte.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Three bit-packed planes, each starting on a byte boundary.
    pub fn spatial_bytes(&self) -> &'a [u8] {
        &self.bytes[1..self.global_offset()]
    }

    pub fn global(&self) -> [f32; NUM_GLOBAL_FEATURES] {
        std::array::from_fn(|i| self.read_f32(self.global_offset() + 4 * i))
    }

    /// Little-endian FP16 bit patterns, including the final pass target.
    pub fn policy_bits(&self) -> impl ExactSizeIterator<Item = u16> + use<'a> {
        let start = self.global_offset() + 4 * NUM_GLOBAL_FEATURES;
        self.bytes[start..self.value_offset()]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
    }

    /// Player-relative win and final score targets, in that order.
    pub fn value(&self) -> [f32; 2] {
        let start = self.value_offset();
        [self.read_f32(start), self.read_f32(start + 4)]
    }

    /// Two-bit player-relative labels, packed from least significant bits.
    pub fn ownership_bytes(&self) -> &'a [u8] {
        &self.bytes[self.value_offset() + 8..]
    }

    fn global_offset(&self) -> usize {
        1 + NUM_SPATIAL_FEATURES * (self.board_dim() * self.board_dim()).div_ceil(8)
    }

    fn value_offset(&self) -> usize {
        self.global_offset()
            + 4 * NUM_GLOBAL_FEATURES
            + 2 * (self.board_dim() * self.board_dim() + 1)
    }

    fn read_f32(&self, offset: usize) -> f32 {
        f32::from_le_bytes(self.bytes[offset..offset + 4].try_into().unwrap())
    }
}

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
mod tests;
