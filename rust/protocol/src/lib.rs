//! Local control messages, bounded JSON framing, and process shutdown signals.

use serde::{Serialize, de::DeserializeOwned};
use std::io;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub mod selfplay;
pub mod server;
mod shutdown;

pub use shutdown::ShutdownSignals;

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_FRAME_BYTES: usize = 64 * 1024;

/// Read one complete newline-delimited message. EOF is only valid between frames.
/// Cancellation can consume part of a frame; discard the stream if cancelled.
pub async fn read<T: DeserializeOwned>(
    input: &mut (impl AsyncBufRead + Unpin),
) -> io::Result<Option<T>> {
    let mut bytes = Vec::new();
    let count = (&mut *input)
        .take((MAX_FRAME_BYTES + 1) as u64)
        .read_until(b'\n', &mut bytes)
        .await?;
    if count == 0 {
        return Ok(None);
    }
    if count > MAX_FRAME_BYTES || bytes.last() != Some(&b'\n') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incomplete or oversized control frame",
        ));
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub async fn write(
    output: &mut (impl AsyncWrite + Unpin),
    message: &impl Serialize,
) -> io::Result<()> {
    let mut bytes = serde_json::to_vec(message).map_err(io::Error::other)?;
    if bytes.len() >= MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "control frame exceeds 64 KiB",
        ));
    }
    bytes.push(b'\n');
    output.write_all(&bytes).await?;
    output.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay::Command;

    #[tokio::test]
    async fn framing_preserves_messages_and_rejects_partial_or_oversized_lines() {
        let mut input = &b"{\"type\":\"finish\"}\n{\"type\":\"model_ready\",\"version\":42}\n"[..];
        assert!(matches!(
            read::<Command>(&mut input).await.unwrap(),
            Some(Command::Finish {})
        ));
        assert!(matches!(
            read::<Command>(&mut input).await.unwrap(),
            Some(Command::ModelReady { version: 42 })
        ));
        assert!(read::<Command>(&mut input).await.unwrap().is_none());
        let mut input = &b"{\"type\":\"finish\"}"[..];
        assert!(read::<Command>(&mut input).await.is_err());
        let oversized = vec![b' '; MAX_FRAME_BYTES + 1];
        assert!(read::<Command>(&mut oversized.as_slice()).await.is_err());
        let mut input = &b"{\"type\":\"finish\",\"extra\":1}\n"[..];
        assert!(read::<Command>(&mut input).await.is_err());
    }
}
