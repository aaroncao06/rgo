//! Existing worker pipe protocol, shared with its supervising client.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// client > self play worker
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    ModelReady { path: PathBuf },
    Finish {},
}

// self play worker > client
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Ready {
        protocol_version: u32,
    },
    ChunkReady {
        path: PathBuf,
        // No production consumer yet; retained for possible client logging or
        // size accounting. Can be removed if those uses do not materialize.
        bytes: usize,
        records: usize,
    },
    Stopped,
    Error {
        message: String,
    },
}
