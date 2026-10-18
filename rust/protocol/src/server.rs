//! Shared-filesystem server requests; artifact bytes stay outside the connection.

use rgo_artifacts::ModelVersion;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Client IDs and shard filename stems contain 1–64 ASCII letters, digits, `-`, or `_`.
pub fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

// client > server
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Hello {
        protocol_version: u32,
        client_id: String,
    },
    LatestModel {},
    AcceptChunk {
        path: PathBuf,
    },
    AnnounceModel {
        version: ModelVersion,
    },
    SampleShard {
        filename: String,
    },
    Stats {},
}

// server > client
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Response {
    Welcome {
        protocol_version: u32,
    },
    Model {
        path: Option<PathBuf>,
    },
    ChunkAccepted {
        records: usize,
        already_accepted: bool,
    },
    Shard {
        path: PathBuf,
    },
    Stats {
        retained_records: usize,
        retained_chunks: usize,
        window_records: usize,
    },
    Error {
        message: String,
    },
}
