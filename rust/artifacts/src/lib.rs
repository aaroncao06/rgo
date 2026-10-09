//! Shared artifact identities, naming conventions, and binary schemas.
//! Callers own storage roots and file/stream delivery; this crate performs no I/O.

use std::path::{Path, PathBuf};

pub mod chunk;

/// Board sizes supported by inference, self-play, and replay ingestion.
pub const SUPPORTED_BOARD_DIMS: [usize; 3] = [9, 13, 19];

/// Identity of an immutable published model.
pub type ModelVersion = u64;
/// Identity of an immutable self-play chunk.
pub type ChunkId = u128;

/// Filename prefix for published training chunks.
pub const CHUNK_FILE_PREFIX: &str = "chunk-";
/// Filename suffix for published training chunks.
pub const CHUNK_FILE_SUFFIX: &str = ".rgo";

/// Canonical path for an immutable published ONNX model in the caller's directory.
pub fn model_path(model_dir: &Path, version: ModelVersion) -> PathBuf {
    model_dir.join(format!("{version}.onnx"))
}

/// Canonical path for an immutable training chunk in the caller's directory.
pub fn chunk_path(output_dir: &Path, chunk_id: ChunkId) -> PathBuf {
    output_dir.join(format!(
        "{CHUNK_FILE_PREFIX}{chunk_id:032x}{CHUNK_FILE_SUFFIX}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_path_uses_the_published_version_in_the_configured_directory() {
        assert_eq!(
            model_path(Path::new("models"), 42),
            Path::new("models").join("42.onnx")
        );
    }
}
