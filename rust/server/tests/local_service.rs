#![cfg(unix)]

use rgo_artifacts::{
    chunk::{chunk_header, training_record_size},
    chunk_path,
};
use rgo_protocol::{
    PROTOCOL_VERSION,
    client_server::{Request, Response},
};
use sha2::{Digest, Sha256};
use std::{fs, net::SocketAddr, path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    net::TcpStream,
    process::{Child, Command},
    time::timeout,
};

async fn start(config: &Path) -> (Child, SocketAddr) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rgo-server"))
        .arg(config)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut line = String::new();
    timeout(
        Duration::from_secs(10),
        BufReader::new(child.stdout.take().unwrap()).read_line(&mut line),
    )
    .await
    .unwrap()
    .unwrap();
    let ready: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(ready["type"], "ready");
    (child, ready["address"].as_str().unwrap().parse().unwrap())
}

async fn stop(child: &mut Child) {
    assert!(
        std::process::Command::new("kill")
            .args(["-TERM", &child.id().unwrap().to_string()])
            .status()
            .unwrap()
            .success()
    );
    assert!(
        timeout(Duration::from_secs(10), child.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
}

async fn request(stream: &mut BufReader<TcpStream>, request: Request) -> Response {
    timeout(Duration::from_secs(10), async {
        rgo_protocol::write(stream.get_mut(), &request)
            .await
            .unwrap();
        rgo_protocol::read(stream).await.unwrap().unwrap()
    })
    .await
    .unwrap()
}

async fn connect(address: SocketAddr) -> BufReader<TcpStream> {
    let mut stream = BufReader::new(TcpStream::connect(address).await.unwrap());
    assert!(matches!(
        request(
            &mut stream,
            Request::Hello {
                protocol_version: PROTOCOL_VERSION,
                client_id: "local".into()
            }
        )
        .await,
        Response::Welcome { .. }
    ));
    stream
}

#[tokio::test]
async fn checkpoint_handoff_retries_and_recovers_after_restart() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("server");
    let config = temp.path().join("server.toml");
    fs::write(&config, format!("listen = '127.0.0.1:0'\ndata_dir = {root:?}\n[checkpoints]\nprune_old_checkpoints = false\n[replay]\ncapacity_records = 100\nwindow_records = 100\nsample_group_records = 20\nsample_groups_per_shard = 4\nrecords_per_shard = 10\n")).unwrap();
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    assert!(matches!(
        request(&mut stream, Request::GetLatestCheckpoint {}).await,
        Response::Checkpoint { path: None }
    ));
    let directory = fs::canonicalize(root.join("checkpoints")).unwrap();
    for index in [-1, 0] {
        let source = temp.path().join(format!("{index}.pt"));
        // Validation rejection must leave the connection and server usable.
        fs::write(&source, b"").unwrap();
        assert!(matches!(
            request(
                &mut stream,
                Request::SubmitCheckpoint {
                    path: source.clone()
                }
            )
            .await,
            Response::Error { .. }
        ));
        fs::write(&source, b"checkpoint").unwrap();
        for _ in 0..2 {
            // The repeat models a lost acceptance reply: the source is gone.
            assert!(matches!(
                request(
                    &mut stream,
                    Request::SubmitCheckpoint {
                        path: source.clone()
                    }
                )
                .await,
                Response::CheckpointAccepted {}
            ));
            assert!(!source.exists());
        }
        assert!(matches!(
            request(&mut stream, Request::GetLatestCheckpoint {}).await,
            Response::Checkpoint { path: Some(path) } if path == directory.join(format!("{index}.pt"))
        ));
    }
    assert!(directory.join("-1.pt").exists());
    drop(stream);
    stop(&mut server).await;

    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    assert!(matches!(
        request(&mut stream, Request::GetLatestCheckpoint {}).await,
        Response::Checkpoint { path: Some(path) } if path == directory.join("0.pt")
    ));
    assert!(directory.join("-1.pt").exists());
    assert_eq!(fs::read(directory.join("0.pt")).unwrap(), b"checkpoint");
    assert!(matches!(
        request(
            &mut stream,
            Request::SubmitCheckpoint {
                path: temp.path().join("0.pt")
            }
        )
        .await,
        Response::CheckpointAccepted {}
    ));
    drop(stream);
    stop(&mut server).await;
}

#[tokio::test]
async fn local_acceptance_sampling_model_publication_and_restart() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("nested/server");
    let config = temp.path().join("server.toml");
    fs::write(&config, format!("listen = '127.0.0.1:0'\ndata_dir = {:?}\n[replay]\ncapacity_records = 100\nwindow_records = 100\nsample_group_records = 20\nsample_groups_per_shard = 4\nrecords_per_shard = 3\n", root)).unwrap();
    let (mut server, address) = start(&config).await;
    for (protocol_version, client_id) in [
        (PROTOCOL_VERSION + 1, "local"),
        (PROTOCOL_VERSION, "bad/id"),
        (PROTOCOL_VERSION, ""),
    ] {
        let mut rejected = BufReader::new(TcpStream::connect(address).await.unwrap());
        assert!(matches!(
            request(
                &mut rejected,
                Request::Hello {
                    protocol_version,
                    client_id: client_id.into()
                }
            )
            .await,
            Response::Error { .. }
        ));
        assert!(
            timeout(
                Duration::from_secs(10),
                rgo_protocol::read::<Response>(&mut rejected)
            )
            .await
            .unwrap()
            .unwrap()
            .is_none()
        );
    }
    let mut stream = connect(address).await;
    for request_type in [Request::SubmitChunk {
        path: "../bad.rgo".into(),
    }] {
        assert!(matches!(
            request(&mut stream, request_type).await,
            Response::Error { .. }
        ));
    }
    assert!(matches!(
        request(&mut stream, Request::GetTrainingShard { index: 0 }).await,
        Response::TrainingShard { path: None }
    ));
    let mut bytes = chunk_header(3).to_vec();
    for dim in [9, 13, 19] {
        let mut record = vec![0; training_record_size(dim)];
        record[0] = dim as u8;
        bytes.extend(record);
    }
    bytes.extend(Sha256::digest(&bytes));
    assert!(!root.join("intake").exists());
    let output = temp.path().join("client-output");
    fs::create_dir(&output).unwrap();
    let source = chunk_path(&fs::canonicalize(output).unwrap(), 42);
    fs::write(&source, &bytes).unwrap();
    assert!(matches!(
        request(
            &mut stream,
            Request::SubmitChunk {
                path: source.clone()
            }
        )
        .await,
        Response::ChunkAccepted {}
    ));
    assert!(!source.exists());
    assert!(matches!(
        request(&mut stream, Request::SubmitChunk { path: source }).await,
        Response::ChunkAccepted {}
    ));
    assert_eq!(
        fs::read(chunk_path(&root.join("replay/chunks"), 42)).unwrap(),
        bytes
    );
    let Response::TrainingShard {
        path: Some(training_path),
    } = request(&mut stream, Request::GetTrainingShard { index: 0 }).await
    else {
        panic!("missing indexed training shard")
    };
    assert_eq!(
        training_path,
        fs::canonicalize(&root).unwrap().join("shards/0.npz")
    );
    let training_bytes = fs::read(&training_path).unwrap();
    let mut zip = zip::ZipArchive::new(fs::File::open(&training_path).unwrap()).unwrap();
    assert_eq!(zip.len(), 17);
    let mut rows = 0;
    for dim in [9, 13, 19] {
        let entry = zip.by_name(&format!("{dim}/policy.npy")).unwrap();
        rows += npyz::NpyFile::new(entry).unwrap().shape()[0];
    }
    assert_eq!(rows, 3);
    fs::write(root.join("models/1.onnx"), b"model").unwrap();
    fs::write(root.join("models/2.onnx.tmp"), b"next model").unwrap();
    let model_dir = fs::canonicalize(root.join("models")).unwrap();
    drop(stream);
    stop(&mut server).await;
    // Simulate a producer finishing publication while the server is down.
    // Restart must discover the final model without a separate marker.
    fs::File::open(root.join("models/2.onnx.tmp"))
        .unwrap()
        .sync_all()
        .unwrap();
    fs::rename(root.join("models/2.onnx.tmp"), root.join("models/2.onnx")).unwrap();
    rgo_fs::sync_directory(&root.join("models")).unwrap();
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    assert!(matches!(
        request(&mut stream, Request::GetLatestModel {}).await,
        Response::Model { path: Some(path) } if path == model_dir.join("2.onnx")
    ));
    assert!(matches!(
        request(&mut stream, Request::GetReplayStats {}).await,
        Response::Stats {
            retained_records: 3,
            retained_chunks: 1,
            ..
        }
    ));
    assert!(matches!(
        request(
            &mut stream,
            Request::GetTrainingShard { index: 0 }
        )
        .await,
        Response::TrainingShard { path: Some(path) } if path == training_path
    ));
    assert_eq!(fs::read(&training_path).unwrap(), training_bytes);
    drop(stream);
    stop(&mut server).await;
}
