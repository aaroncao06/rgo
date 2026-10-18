//! Shared-filesystem server requests; artifact bytes stay outside the connection.

use rgo_artifacts::ModelVersion;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Client IDs contain 1–64 ASCII letters, digits, `-`, or `_`.
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
    // The trainer chooses the exact index; a request does not imply consumption.
    Shard {
        index: u64,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_shard_messages_round_trip() {
        let request = Request::Shard { index: 8 };
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(json, r#"{"type":"shard","index":8}"#);
        assert!(matches!(
            serde_json::from_str::<Request>(&json).unwrap(),
            Request::Shard { index: 8 }
        ));

        let response = Response::Shard {
            path: PathBuf::from("/shards/8.npz"),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(matches!(
            serde_json::from_str::<Response>(&json).unwrap(),
            Response::Shard { path }
                if path == PathBuf::from("/shards/8.npz")
        ));
    }
}
