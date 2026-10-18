use crate::config::Config;
use rgo_protocol::{
    PROTOCOL_VERSION,
    client_server::{Request, Response},
};
use std::{io, time::Duration};
use tokio::{
    io::BufReader,
    net::TcpStream,
    time::{Instant, sleep, timeout_at},
};

pub struct Connection<'a> {
    config: &'a Config,
    stream: Option<BufReader<TcpStream>>,
}

impl<'a> Connection<'a> {
    pub async fn open(config: &'a Config) -> io::Result<Self> {
        let mut connection = Self {
            config,
            stream: None,
        };
        connection.stream = Some(
            retry(
                config.request_timeout(),
                config.retry_timeout(),
                async || connection.connect().await,
            )
            .await?,
        );
        Ok(connection)
    }

    pub async fn request(&mut self, message: &Request) -> io::Result<Response> {
        retry(
            self.config.request_timeout(),
            self.config.retry_timeout(),
            async || self.exchange(message).await,
        )
        .await
    }

    async fn connect(&mut self) -> io::Result<BufReader<TcpStream>> {
        let stream = TcpStream::connect(self.config.server).await?;
        stream.set_nodelay(true)?;
        let mut stream = BufReader::new(stream);
        rgo_protocol::write(
            stream.get_mut(),
            &Request::Hello {
                protocol_version: PROTOCOL_VERSION,
                client_id: self.config.client_id.clone(),
            },
        )
        .await?;
        match read_response(&mut stream).await? {
            Response::Welcome { protocol_version } if protocol_version == PROTOCOL_VERSION => {}
            Response::Error { message } => return Err(io::Error::other(message)),
            response => {
                return Err(io::Error::other(format!(
                    "invalid server greeting: {response:?}"
                )));
            }
        }
        Ok(stream)
    }

    async fn exchange(&mut self, message: &Request) -> io::Result<Response> {
        // Keep the socket local until the whole exchange succeeds. Cancellation,
        // timeout, or error drops it, including any partially consumed frame.
        let mut stream = match self.stream.take() {
            Some(stream) => stream,
            None => self.connect().await?,
        };
        rgo_protocol::write(stream.get_mut(), message).await?;
        let response = read_response(&mut stream).await?;
        self.stream = Some(stream);
        Ok(response)
    }
}

async fn read_response(stream: &mut BufReader<TcpStream>) -> io::Result<Response> {
    match rgo_protocol::read(stream).await? {
        Some(response) => Ok(response),
        None => Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "server disconnected",
        )),
    }
}

async fn retry<T>(
    attempt_timeout: Duration,
    budget: Duration,
    mut attempt: impl AsyncFnMut() -> io::Result<T>,
) -> io::Result<T> {
    let deadline = Instant::now() + budget;
    let mut delay = Duration::from_secs(1);
    loop {
        let attempt_deadline = deadline.min(Instant::now() + attempt_timeout);
        let error = match timeout_at(attempt_deadline, attempt()).await {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(error)) => error,
            Err(_) => io::Error::new(io::ErrorKind::TimedOut, "server operation timed out"),
        };
        if !matches!(
            error.kind(),
            io::ErrorKind::ConnectionRefused
                | io::ErrorKind::ConnectionReset
                | io::ErrorKind::ConnectionAborted
                | io::ErrorKind::BrokenPipe
                | io::ErrorKind::NotConnected
                | io::ErrorKind::UnexpectedEof
                | io::ErrorKind::TimedOut
                | io::ErrorKind::Interrupted
                | io::ErrorKind::HostUnreachable
                | io::ErrorKind::NetworkUnreachable
                | io::ErrorKind::NetworkDown
        ) {
            return Err(error);
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("server retry budget expired: {error}"),
            ));
        }
        eprintln!(
            "rgo-client: {error}; retrying in {} seconds",
            delay.as_secs()
        );
        if timeout_at(deadline, sleep(delay)).await.is_err() || Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("server retry budget expired: {error}"),
            ));
        }
        delay = (delay * 2).min(Duration::from_secs(30));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn retries_back_off_and_stop_at_the_total_budget() {
        let start = Instant::now();
        let mut attempts = Vec::new();
        let error = retry(
            Duration::from_secs(5),
            Duration::from_secs(100),
            async || {
                attempts.push(Instant::now().duration_since(start).as_secs());
                Err::<(), _>(io::Error::from(io::ErrorKind::ConnectionRefused))
            },
        )
        .await
        .unwrap_err();
        assert_eq!(attempts, [0, 1, 3, 7, 15, 31, 61, 91]);
        assert_eq!(
            Instant::now().duration_since(start),
            Duration::from_secs(100)
        );
        assert!(error.to_string().contains("retry budget expired"));
    }

    #[tokio::test(start_paused = true)]
    async fn does_not_start_an_attempt_at_the_budget_deadline() {
        let mut attempts = 0;
        let error = retry(Duration::from_secs(5), Duration::from_secs(1), async || {
            attempts += 1;
            if attempts == 1 {
                Err(io::Error::from(io::ErrorKind::ConnectionRefused))
            } else {
                Ok(())
            }
        })
        .await
        .unwrap_err();
        assert_eq!(attempts, 1);
        assert!(error.to_string().contains("retry budget expired"));
    }

    #[tokio::test(start_paused = true)]
    async fn hung_attempts_are_capped_by_the_remaining_budget() {
        let start = Instant::now();
        let mut attempts = Vec::new();
        let error = retry(
            Duration::from_secs(5),
            Duration::from_secs(10),
            async || {
                attempts.push(Instant::now().duration_since(start).as_secs());
                std::future::pending::<io::Result<()>>().await
            },
        )
        .await
        .unwrap_err();
        assert_eq!(attempts, [0, 6]);
        assert_eq!(
            Instant::now().duration_since(start),
            Duration::from_secs(10)
        );
        assert!(error.to_string().contains("retry budget expired"));
    }

    #[tokio::test(start_paused = true)]
    async fn successful_operations_start_fresh_budgets() {
        let start = Instant::now();
        for _ in 0..2 {
            let mut attempts = 0;
            let result = retry(Duration::from_secs(5), Duration::from_secs(4), async || {
                attempts += 1;
                if attempts < 3 {
                    Err(io::Error::from(io::ErrorKind::ConnectionReset))
                } else {
                    Ok(42)
                }
            })
            .await
            .unwrap();
            assert_eq!(result, 42);
            assert_eq!(attempts, 3);
        }
        assert_eq!(Instant::now().duration_since(start), Duration::from_secs(6));
    }

    #[tokio::test(start_paused = true)]
    async fn fatal_errors_do_not_retry() {
        let start = Instant::now();
        for kind in [io::ErrorKind::Other, io::ErrorKind::InvalidData] {
            let mut attempts = 0;
            let error = retry(
                Duration::from_secs(5),
                Duration::from_secs(10),
                async || {
                    attempts += 1;
                    Err::<(), _>(io::Error::from(kind))
                },
            )
            .await
            .unwrap_err();
            assert_eq!(error.kind(), kind);
            assert_eq!(attempts, 1);
        }
        assert_eq!(Instant::now(), start);
    }
    #[tokio::test]
    async fn partial_response_reconnects_and_repeats_hello() {
        reconnect_after_partial_response(false).await;
    }

    #[tokio::test]
    async fn changed_directories_on_reconnect_are_reflected_in_the_model_path() {
        reconnect_after_partial_response(true).await;
    }

    async fn reconnect_after_partial_response(change_directories: bool) {
        use tokio::{io::AsyncWriteExt, net::TcpListener};
        let root = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(root.path()).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = test_config(listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (stream, _) = listener.accept().await.unwrap();
                let mut stream = BufReader::new(stream);
                assert!(matches!(
                    rgo_protocol::read::<Request>(&mut stream).await.unwrap(),
                    Some(Request::Hello {
                        protocol_version: PROTOCOL_VERSION,
                        ..
                    })
                ));
                rgo_protocol::write(
                    stream.get_mut(),
                    &Response::Welcome {
                        protocol_version: PROTOCOL_VERSION,
                    },
                )
                .await
                .unwrap();
                assert!(matches!(
                    rgo_protocol::read::<Request>(&mut stream).await.unwrap(),
                    Some(Request::LatestModel {})
                ));
                if attempt == 0 {
                    stream.get_mut().write_all(b"{\"type\":").await.unwrap();
                } else {
                    let directory = if change_directories {
                        "different"
                    } else {
                        "models"
                    };
                    rgo_protocol::write(
                        stream.get_mut(),
                        &Response::Model {
                            path: Some(root.join(directory).join("42.onnx")),
                        },
                    )
                    .await
                    .unwrap();
                }
            }
        });
        let mut connection = Connection::open(&config).await.unwrap();
        let response = connection.request(&Request::LatestModel {}).await;
        let Response::Model { path: Some(path) } = response.unwrap() else {
            panic!("missing model path")
        };
        assert_eq!(path.file_name().unwrap(), "42.onnx");
        assert_eq!(
            path.parent().unwrap().file_name().unwrap(),
            if change_directories {
                "different"
            } else {
                "models"
            }
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn shard_request_retries_after_a_lost_reply() {
        use tokio::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = test_config(listener.local_addr().unwrap());
        let shard = std::env::temp_dir().join("0.npz");
        let expected = shard.clone();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (stream, _) = listener.accept().await.unwrap();
                let mut stream = BufReader::new(stream);
                assert!(matches!(
                    rgo_protocol::read::<Request>(&mut stream).await.unwrap(),
                    Some(Request::Hello { .. })
                ));
                rgo_protocol::write(
                    stream.get_mut(),
                    &Response::Welcome {
                        protocol_version: PROTOCOL_VERSION,
                    },
                )
                .await
                .unwrap();
                assert!(matches!(
                    rgo_protocol::read::<Request>(&mut stream).await.unwrap(),
                    Some(Request::Shard { index: 0 })
                ));
                if attempt == 1 {
                    rgo_protocol::write(
                        stream.get_mut(),
                        &Response::Shard {
                            path: shard.clone(),
                        },
                    )
                    .await
                    .unwrap();
                }
            }
        });
        let mut connection = Connection::open(&config).await.unwrap();
        assert!(matches!(
            connection
                .request(&Request::Shard { index: 0 })
                .await
                .unwrap(),
            Response::Shard { path } if path == expected
        ));
        server.await.unwrap();
    }

    fn test_config(server: std::net::SocketAddr) -> Config {
        Config {
            server,
            client_id: "test".into(),
            selfplay_config: "unused.toml".into(),
            selfplay_binary: None,
            model_poll_interval_ms: 1,
            request_timeout_secs: 5,
            retry_timeout_secs: 10,
        }
    }

    #[tokio::test]
    async fn server_errors_are_responses_and_the_connection_remains_usable() {
        let root = tempfile::tempdir().unwrap();
        let path = rgo_artifacts::chunk_path(root.path(), 1);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = test_config(listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut stream = BufReader::new(stream);
            assert!(matches!(
                rgo_protocol::read::<Request>(&mut stream).await.unwrap(),
                Some(Request::Hello { .. })
            ));
            rgo_protocol::write(
                stream.get_mut(),
                &Response::Welcome {
                    protocol_version: PROTOCOL_VERSION,
                },
            )
            .await
            .unwrap();
            assert!(matches!(
                rgo_protocol::read::<Request>(&mut stream).await.unwrap(),
                Some(Request::Stats {})
            ));
            rgo_protocol::write(
                stream.get_mut(),
                &Response::Error {
                    message: "stats rejected".into(),
                },
            )
            .await
            .unwrap();
            assert!(matches!(
                rgo_protocol::read::<Request>(&mut stream).await.unwrap(),
                Some(Request::AcceptChunk { .. })
            ));
            rgo_protocol::write(
                stream.get_mut(),
                &Response::Error {
                    message: "chunk rejected".into(),
                },
            )
            .await
            .unwrap();
            assert!(matches!(
                rgo_protocol::read::<Request>(&mut stream).await.unwrap(),
                Some(Request::LatestModel {})
            ));
            rgo_protocol::write(
                stream.get_mut(),
                &Response::Model {
                    path: Some(std::env::temp_dir().join("42.onnx")),
                },
            )
            .await
            .unwrap();
        });
        let mut connection = Connection::open(&config).await.unwrap();
        assert!(
            matches!(connection.request(&Request::Stats {}).await.unwrap(),
            Response::Error { message } if message == "stats rejected")
        );
        assert!(
            matches!(connection.request(&Request::AcceptChunk { path }).await.unwrap(),
            Response::Error { message } if message == "chunk rejected")
        );
        assert!(matches!(
            connection.request(&Request::LatestModel {}).await.unwrap(),
            Response::Model { path: Some(path) } if path.file_name().unwrap() == "42.onnx"
        ));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn rejected_greeting_preserves_the_server_message_and_is_not_retried() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = test_config(listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut stream = BufReader::new(stream);
            assert!(matches!(
                rgo_protocol::read::<Request>(&mut stream).await.unwrap(),
                Some(Request::Hello { .. })
            ));
            rgo_protocol::write(
                stream.get_mut(),
                &Response::Error {
                    message: "unsupported protocol version".into(),
                },
            )
            .await
            .unwrap();
        });
        let error = match Connection::open(&config).await {
            Ok(_) => panic!("rejected greeting was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.to_string(), "unsupported protocol version");
        server.await.unwrap();
    }
}
