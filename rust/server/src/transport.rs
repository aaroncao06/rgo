//! Loopback JSON control connections; storage stays on one dedicated thread.

use crate::{Error, config::Config, service::StorageState};
use rgo_protocol::{
    PROTOCOL_VERSION, ShutdownSignals,
    server::{Request, Response, valid_name},
};
use std::{io, sync::Arc, thread, time::Duration};
use tokio::{
    io::BufReader,
    net::{TcpListener, TcpStream},
    sync::{mpsc, oneshot, watch},
    task::JoinSet,
};

/// One connection's request queued for the storage thread.
struct StorageJob {
    request: Request,
    // Return one response to the connection that submitted this job.
    // The connection keeps the receiver; this sender travels with the request.
    reply: oneshot::Sender<Response>,
}

async fn request(jobs: &mpsc::Sender<StorageJob>, request: Request) -> io::Result<Response> {
    let (reply, response) = oneshot::channel();
    // All connections share the storage queue, but each job has its own reply
    // channel. Wait here so the connection can send the result to its socket.
    jobs.send(StorageJob { request, reply })
        .await
        .map_err(|_| io::Error::other("storage worker stopped"))?;
    response
        .await
        .map_err(|_| io::Error::other("storage worker stopped"))
}

async fn connection(
    stream: TcpStream,
    jobs: mpsc::Sender<StorageJob>,
    models: watch::Receiver<Option<u64>>,
    welcome: Arc<Response>,
) -> io::Result<()> {
    stream.set_nodelay(true)?;
    let mut stream = BufReader::new(stream);
    let hello = tokio::time::timeout(
        Duration::from_secs(5),
        rgo_protocol::read::<Request>(&mut stream),
    )
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "client hello timed out"))??
    .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "missing client hello"))?;
    let Request::Hello {
        protocol_version,
        client_id,
    } = hello
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first request must be hello",
        ));
    };
    let error = if protocol_version != PROTOCOL_VERSION {
        Some("unsupported protocol version")
    } else if !valid_name(&client_id) {
        Some("client_id must contain 1–64 ASCII letters, digits, '-' or '_'")
    } else {
        None
    };
    if let Some(message) = error {
        rgo_protocol::write(
            stream.get_mut(),
            &Response::Error {
                message: message.into(),
            },
        )
        .await?;
        return Ok(());
    }
    rgo_protocol::write(stream.get_mut(), welcome.as_ref()).await?;
    while let Some(message) = rgo_protocol::read::<Request>(&mut stream).await? {
        let response = match message {
            Request::Hello { .. } => Response::Error {
                message: "hello may only be sent once".into(),
            },
            Request::LatestModel {} => Response::Model {
                version: *models.borrow(),
            },
            message => request(&jobs, message).await?,
        };
        rgo_protocol::write(stream.get_mut(), &response).await?;
    }
    Ok(())
}

/// Own storage for the server run, replying to each queued job before continuing.
fn storage_worker(
    config: Config,
    mut incoming: mpsc::Receiver<StorageJob>,
    ready: oneshot::Sender<(watch::Receiver<Option<u64>>, Response)>,
) -> Result<(), Error> {
    let mut state = StorageState::open(&config.data_dir, config.replay)?;
    ready
        .send((state.model_versions(), state.welcome()))
        .map_err(|_| "server stopped during startup")?;
    while let Some(job) = incoming.blocking_recv() {
        let response = match state.handle(job.request) {
            Ok(response) => response,
            Err(error) => Response::Error {
                message: error.to_string(),
            },
        };
        // A disconnected connection may have dropped its receiver.
        // The storage operation still finishes; only its reply is lost.
        let _ = job.reply.send(response);
        if state.needs_restart() {
            return Err("storage mutation failed; restart the server before retrying".into());
        }
    }
    Ok(())
}

pub async fn run(config: Config) -> Result<(), Error> {
    let mut signals = ShutdownSignals::new()?;
    let listener = TcpListener::bind(config.listen).await?;
    let address = listener.local_addr()?;
    let (jobs, incoming) = mpsc::channel::<StorageJob>(16);
    let (ready, startup) = oneshot::channel();
    let (finished, mut done) = oneshot::channel();
    thread::Builder::new()
        .name("server-storage".into())
        .spawn(move || {
            let _ = finished.send(storage_worker(config, incoming, ready));
        })?;
    let (models, welcome) = match startup.await {
        Ok(connection_state) => connection_state,
        Err(_) => return done.await?,
    };
    let welcome = Arc::new(welcome);
    println!("{{\"type\":\"ready\",\"address\":\"{address}\"}}");
    let mut sessions = JoinSet::new();
    let mut worker_result = None;
    let result: Result<(), Error> = loop {
        tokio::select! {
            result = &mut done => {
                worker_result = Some(result.map_err(Into::into).and_then(|result| result));
                break Ok(());
            }
            signal = signals.recv() => { break signal.map_err(Into::into); }
            accepted = listener.accept() => {
                let (stream, _) = match accepted {
                    Ok(accepted) => accepted,
                    Err(error) => break Err(error.into()),
                };
                if sessions.len() < 32 {
                    sessions.spawn(connection(
                        stream,
                        jobs.clone(),
                        models.clone(),
                        welcome.clone(),
                    ));
                }
            }
            Some(result) = sessions.join_next() => {
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => eprintln!("rgo-server: client disconnected: {error}"),
                    Err(error) => eprintln!("rgo-server: connection task failed: {error}"),
                }
            }
        }
    };
    // Disconnect clients, then complete queued storage work before releasing the
    // store lock. Durable receipts make resubmitting producer paths safe after
    // a lost acknowledgment.
    drop(listener);
    sessions.shutdown().await;
    drop(jobs);
    let storage = match worker_result {
        Some(result) => result,
        None => done.await?,
    };
    result.and(storage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn hello_and_model_polls_respond_while_the_storage_queue_is_full() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (server, _) = listener.accept().await.unwrap();
        let (jobs, mut incoming) = mpsc::channel::<StorageJob>(1);
        let (published, models) = watch::channel(None);
        // A greeting and subsequent model polls must not need queue capacity.
        let (reply, _response) = oneshot::channel();
        jobs.send(StorageJob {
            request: Request::Stats {},
            reply,
        })
        .await
        .unwrap();
        let welcome = Arc::new(Response::Welcome {
            protocol_version: PROTOCOL_VERSION,
            model_dir: "/models".into(),
            shard_dir: "/shards".into(),
        });
        let session = tokio::spawn(connection(server, jobs.clone(), models, welcome));
        let mut client = BufReader::new(client);
        rgo_protocol::write(
            client.get_mut(),
            &Request::Hello {
                protocol_version: PROTOCOL_VERSION,
                client_id: "test".into(),
            },
        )
        .await
        .unwrap();
        assert!(matches!(
            tokio::time::timeout(
                Duration::from_secs(2),
                rgo_protocol::read::<Response>(&mut client)
            )
            .await
            .unwrap()
            .unwrap(),
            Some(Response::Welcome { .. })
        ));

        for version in [None, Some(42)] {
            published.send_replace(version);
            rgo_protocol::write(client.get_mut(), &Request::LatestModel {})
                .await
                .unwrap();
            let response = tokio::time::timeout(
                Duration::from_secs(2),
                rgo_protocol::read::<Response>(&mut client),
            )
            .await
            .unwrap()
            .unwrap();
            assert!(
                matches!(response, Some(Response::Model { version: actual }) if actual == version)
            );
        }
        assert_eq!(incoming.len(), 1);
        assert!(matches!(
            incoming.recv().await.unwrap().request,
            Request::Stats {}
        ));
        drop(client);
        session.await.unwrap().unwrap();
    }
}
