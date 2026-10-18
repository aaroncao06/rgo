//! Shared artifact identities, naming conventions, and binary schemas.
//! Callers own storage roots and file/stream delivery; this crate performs no I/O.

use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

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

/// Extract a version from a canonical published model basename.
pub fn model_version_from_filename(filename: &OsStr) -> Option<ModelVersion> {
    let name = filename.to_str()?;
    let version = name.strip_suffix(".onnx")?.parse().ok()?;
    (name == format!("{version}.onnx")).then_some(version)
}

/// Canonical path for an immutable training chunk in the caller's directory.
pub fn chunk_path(output_dir: &Path, chunk_id: ChunkId) -> PathBuf {
    output_dir.join(format!(
        "{CHUNK_FILE_PREFIX}{chunk_id:032x}{CHUNK_FILE_SUFFIX}"
    ))
}

/// Extract an ID from a canonical published chunk basename.
pub fn chunk_id_from_filename(filename: &OsStr) -> Option<ChunkId> {
    let name = filename.to_str()?;
    let hex = name
        .strip_prefix(CHUNK_FILE_PREFIX)?
        .strip_suffix(CHUNK_FILE_SUFFIX)?;
    if hex.len() != 32
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    ChunkId::from_str_radix(hex, 16).ok()
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

    #[test]
    fn model_names_round_trip_and_reject_noncanonical_names() {
        for version in [0, 42, ModelVersion::MAX] {
            let path = model_path(Path::new(""), version);
            assert_eq!(model_version_from_filename(path.as_os_str()), Some(version));
        }
        for name in [
            "042.onnx",
            "+42.onnx",
            "-1.onnx",
            "42.ONNX",
            "42.onnx.tmp",
            "../42.onnx",
            "model.onnx",
            "18446744073709551616.onnx",
        ] {
            assert!(model_version_from_filename(OsStr::new(name)).is_none());
        }
    }

    #[test]
    fn chunk_names_round_trip_and_reject_paths_or_noncanonical_names() {
        for id in [0, 42, ChunkId::MAX] {
            let path = chunk_path(Path::new(""), id);
            assert_eq!(chunk_id_from_filename(path.as_os_str()), Some(id));
        }
        for name in [
            "chunk-1.rgo",
            "chunk-0000000000000000000000000000000A.rgo",
            "../chunk-00000000000000000000000000000001.rgo",
            ".pending-chunk.tmp",
        ] {
            assert!(chunk_id_from_filename(OsStr::new(name)).is_none());
        }
    }
}
