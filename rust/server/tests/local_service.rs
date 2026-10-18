#![cfg(unix)]

use rgo_artifacts::{
    chunk::{chunk_header, training_record_size},
    chunk_path,
};
use rgo_protocol::{
    PROTOCOL_VERSION,
    server::{Request, Response},
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
async fn local_acceptance_sampling_model_publication_and_restart() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("nested/server");
    let config = temp.path().join("server.toml");
    fs::write(&config, format!("listen = '127.0.0.1:0'\ndata_dir = {:?}\n[replay]\ncapacity_records = 100\nwindow_records = 100\nsample_group_records = 20\nsample_groups_per_shard = 4\nshard_records = 10\n", root)).unwrap();
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
    for request_type in [
        Request::AcceptChunk {
            path: "../bad.rgo".into(),
        },
        Request::SampleShard {
            filename: "../bad.npz".into(),
        },
        Request::SampleShard {
            filename: "empty.npz".into(),
        },
        Request::AnnounceModel { version: 1 },
    ] {
        assert!(matches!(
            request(&mut stream, request_type).await,
            Response::Error { .. }
        ));
    }
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
            Request::AcceptChunk {
                path: source.clone()
            }
        )
        .await,
        Response::ChunkAccepted {
            records: 3,
            already_accepted: false
        }
    ));
    assert!(!source.exists());
    assert!(matches!(
        request(&mut stream, Request::AcceptChunk { path: source }).await,
        Response::ChunkAccepted {
            records: 3,
            already_accepted: true
        }
    ));
    assert_eq!(
        fs::read(chunk_path(&root.join("replay/chunks"), 42)).unwrap(),
        bytes
    );
    let Response::Shard { path } = request(
        &mut stream,
        Request::SampleShard {
            filename: "batch.npz".into(),
        },
    )
    .await
    else {
        panic!("missing shard")
    };
    let original = fs::read(&path).unwrap();
    let mut zip = zip::ZipArchive::new(fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(zip.len(), 17);
    for dim in [9, 13, 19] {
        assert!(zip.by_name(&format!("{dim}/policy.npy")).is_ok());
    }
    assert!(matches!(
        request(
            &mut stream,
            Request::SampleShard {
                filename: "batch.npz".into()
            }
        )
        .await,
        Response::Shard { path: retried } if retried == path
    ));
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::write(root.join("models/1.onnx"), b"model").unwrap();
    fs::write(root.join("models/2.onnx.tmp"), b"next model").unwrap();
    let model_dir = fs::canonicalize(root.join("models")).unwrap();
    assert!(matches!(
        request(&mut stream, Request::AnnounceModel { version: 1 }).await,
        Response::Model { path: Some(path) } if path == model_dir.join("1.onnx")
    ));
    drop(stream);
    stop(&mut server).await;
    // Simulate a producer finishing publication while the server is down.
    // Restart must discover the final model without a notification or marker.
    fs::File::open(root.join("models/2.onnx.tmp"))
        .unwrap()
        .sync_all()
        .unwrap();
    fs::rename(root.join("models/2.onnx.tmp"), root.join("models/2.onnx")).unwrap();
    rgo_fs::sync_directory(&root.join("models")).unwrap();
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    assert!(matches!(
        request(&mut stream, Request::LatestModel {}).await,
        Response::Model { path: Some(path) } if path == model_dir.join("2.onnx")
    ));
    assert!(matches!(
        request(&mut stream, Request::Stats {}).await,
        Response::Stats {
            retained_records: 3,
            retained_chunks: 1,
            ..
        }
    ));
    assert!(matches!(
        request(
            &mut stream,
            Request::SampleShard {
                filename: "batch.npz".into()
            }
        )
        .await,
        Response::Shard { path: retried } if retried == path
    ));
    assert_eq!(fs::read(&path).unwrap(), original);
    drop(stream);
    stop(&mut server).await;
}
