//! Shared-filesystem server requests; artifact bytes stay outside the connection.

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
    GetLatestModel {},
    SubmitChunk {
        path: PathBuf,
    },
    SubmitCheckpoint {
        path: PathBuf,
    },
    GetLatestCheckpoint {},
    // The trainer chooses the exact index; a request does not imply consumption.
    GetTrainingShard {
        index: u64,
    },
    GetReplayStats {},
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
    ChunkAccepted {},
    CheckpointAccepted {},
    Checkpoint {
        path: Option<PathBuf>,
    },
    TrainingShard {
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
    fn checkpoint_messages_round_trip() {
        let request = Request::SubmitCheckpoint {
            path: "/trainer/7.pt".into(),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(matches!(
            serde_json::from_str::<Request>(&json).unwrap(),
            Request::SubmitCheckpoint { path } if path == PathBuf::from("/trainer/7.pt")
        ));
        let json = serde_json::to_string(&Request::GetLatestCheckpoint {}).unwrap();
        assert!(matches!(
            serde_json::from_str::<Request>(&json).unwrap(),
            Request::GetLatestCheckpoint {}
        ));
        let json = serde_json::to_string(&Response::CheckpointAccepted {}).unwrap();
        assert!(matches!(
            serde_json::from_str::<Response>(&json).unwrap(),
            Response::CheckpointAccepted {}
        ));
        for path in [None, Some(PathBuf::from("/checkpoints/7.pt"))] {
            let json = serde_json::to_string(&Response::Checkpoint { path: path.clone() }).unwrap();
            assert!(matches!(
                serde_json::from_str::<Response>(&json).unwrap(),
                Response::Checkpoint { path: decoded } if decoded == path
            ));
        }
    }

    #[test]
    fn indexed_shard_messages_round_trip() {
        let request = Request::GetTrainingShard { index: 8 };
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(json, r#"{"type":"get_training_shard","index":8}"#);
        assert!(matches!(
            serde_json::from_str::<Request>(&json).unwrap(),
            Request::GetTrainingShard { index: 8 }
        ));

        let response = Response::TrainingShard {
            path: PathBuf::from("/shards/8.npz"),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(matches!(
            serde_json::from_str::<Response>(&json).unwrap(),
            Response::TrainingShard { path }
                if path == PathBuf::from("/shards/8.npz")
        ));
    }
}
