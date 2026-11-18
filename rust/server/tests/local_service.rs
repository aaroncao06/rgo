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
use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    net::TcpStream,
    process::{Child, Command},
    time::timeout,
};

async fn start(config: &Path) -> (Child, SocketAddr) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rgo-server"))
        .arg(config)
        .process_group(0)
        .current_dir(config.parent().unwrap())
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

fn write_test_python(directory: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let helper = directory.join(".venv/bin/python");
    fs::create_dir_all(helper.parent().unwrap()).unwrap();
    fs::write(
        &helper,
        r#"#!/bin/sh
set -eu
base=$(cd "$(dirname "$0")/../.." && pwd)
basename "$4" >> "$base/exports"
if [ -f "$base/pause" ]; then
    touch "$base/started"
    attempts=0
    while [ ! -f "$base/release" ]; do
        attempts=$((attempts + 1))
        [ "$attempts" -lt 1000 ] || exit 9
        sleep 0.01
    done
fi
cp "$3" "$4"
if [ -f "$base/fail" ]; then exit 7; fi
"#,
    )
    .unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o755)).unwrap();
}

fn export_test_config(directory: &Path) -> PathBuf {
    write_test_python(directory);
    let config = directory.join("server.toml");
    fs::write(
        &config,
        format!(
            r#"listen = '127.0.0.1:0'
data_dir = {:?}
[model_export]
every_shards = 2
[replay]
capacity_records = 100
window_records = 100
sample_group_records = 20
sample_groups_per_shard = 4
records_per_shard = 10
"#,
            directory.join("server")
        ),
    )
    .unwrap();
    config
}

async fn send_checkpoint(
    stream: &mut BufReader<TcpStream>,
    directory: &Path,
    index: i64,
) -> PathBuf {
    let source = directory.join(format!("{index}.pt"));
    fs::write(&source, format!("checkpoint {index}")).unwrap();
    rgo_protocol::write(
        stream.get_mut(),
        &Request::SubmitCheckpoint {
            path: source.clone(),
        },
    )
    .await
    .unwrap();
    source
}

async fn submit_checkpoint(stream: &mut BufReader<TcpStream>, directory: &Path, index: i64) {
    let source = send_checkpoint(stream, directory, index).await;
    assert!(matches!(
        timeout(
            Duration::from_secs(10),
            rgo_protocol::read::<Response>(stream)
        )
        .await
        .unwrap()
        .unwrap(),
        Some(Response::CheckpointAccepted {})
    ));
    assert!(!source.exists());
}

async fn wait_for_path(path: &Path) {
    timeout(Duration::from_secs(5), async {
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

async fn wait_for_model(stream: &mut BufReader<TcpStream>, expected: &Path) {
    timeout(Duration::from_secs(5), async {
        loop {
            let Response::Model { path } = request(stream, Request::GetLatestModel {}).await else {
                panic!("invalid model response")
            };
            if path.as_deref() == Some(expected) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn checkpoint_acceptance_exports_in_order_and_recovers_the_cadence() {
    let temp = tempfile::tempdir().unwrap();
    let base = fs::canonicalize(temp.path()).unwrap();
    let config = export_test_config(&base);
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    submit_checkpoint(&mut stream, &base, -1).await;
    let models = base.join("server/models");
    assert!(models.join("0.onnx").exists()); // Export finishes before acceptance replies.
    for index in [0, 1, 3] {
        submit_checkpoint(&mut stream, &base, index).await;
    }
    let checkpoints = base.join("server/checkpoints");
    wait_for_model(&mut stream, &models.join("4.onnx")).await;
    assert_eq!(fs::read(models.join("0.onnx")).unwrap(), b"checkpoint -1");
    assert_eq!(fs::read(models.join("4.onnx")).unwrap(), b"checkpoint 3");
    for index in [-1, 0, 1] {
        assert!(!checkpoints.join(format!("{index}.pt")).exists());
    }
    assert!(checkpoints.join("3.pt").exists());
    submit_checkpoint(&mut stream, &base, 4).await;
    // Acceptance prunes immediately even though this checkpoint is below the
    // next export threshold.
    assert!(!checkpoints.join("3.pt").exists());
    assert!(checkpoints.join("4.pt").exists());
    drop(stream);
    stop(&mut server).await;
    assert_eq!(
        fs::read_to_string(base.join("exports")).unwrap(),
        "0.onnx\n2.onnx\n4.onnx\n"
    );

    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    wait_for_model(&mut stream, &models.join("4.onnx")).await;
    assert!(!checkpoints.join("3.pt").exists());
    assert!(checkpoints.join("4.pt").exists());
    submit_checkpoint(&mut stream, &base, 5).await;
    wait_for_model(&mut stream, &models.join("6.onnx")).await;
    assert!(!checkpoints.join("3.pt").exists());
    assert!(!checkpoints.join("4.pt").exists());
    assert!(checkpoints.join("5.pt").exists());
    // An old lost-ack retry must not schedule another export.
    assert!(matches!(
        request(
            &mut stream,
            Request::SubmitCheckpoint {
                path: base.join("3.pt")
            }
        )
        .await,
        Response::CheckpointAccepted {}
    ));
    drop(stream);
    stop(&mut server).await;
    assert_eq!(
        fs::read_to_string(base.join("exports")).unwrap(),
        "0.onnx\n2.onnx\n4.onnx\n6.onnx\n"
    );
}

#[tokio::test]
async fn disabling_checkpoint_pruning_keeps_history_after_exports_and_restart() {
    let temp = tempfile::tempdir().unwrap();
    let base = fs::canonicalize(temp.path()).unwrap();
    let config = export_test_config(&base);
    let contents = fs::read_to_string(&config).unwrap();
    fs::write(
        &config,
        format!("{contents}\n[checkpoints]\nprune_old_checkpoints = false\n"),
    )
    .unwrap();
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    submit_checkpoint(&mut stream, &base, -1).await;
    wait_for_model(&mut stream, &base.join("server/models/0.onnx")).await;
    submit_checkpoint(&mut stream, &base, 1).await;
    wait_for_model(&mut stream, &base.join("server/models/2.onnx")).await;
    assert!(base.join("server/checkpoints/-1.pt").exists());
    drop(stream);
    stop(&mut server).await;

    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    wait_for_model(&mut stream, &base.join("server/models/2.onnx")).await;
    assert!(base.join("server/checkpoints/-1.pt").exists());
    assert!(base.join("server/checkpoints/1.pt").exists());
    drop(stream);
    stop(&mut server).await;
}

#[tokio::test]
async fn checkpoint_cleanup_failure_stops_the_server_and_restart_finishes_cleanup() {
    let temp = tempfile::tempdir().unwrap();
    let base = fs::canonicalize(temp.path()).unwrap();
    let config = export_test_config(&base);
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    submit_checkpoint(&mut stream, &base, -1).await;
    wait_for_model(&mut stream, &base.join("server/models/0.onnx")).await;
    // An invalid older entry makes pruning fail during checkpoint acceptance.
    let invalid = base.join("server/checkpoints/0.pt");
    fs::create_dir(&invalid).unwrap();
    let source = base.join("1.pt");
    fs::write(&source, b"checkpoint 1").unwrap();
    rgo_protocol::write(
        stream.get_mut(),
        &Request::SubmitCheckpoint { path: source },
    )
    .await
    .unwrap();
    assert!(
        !timeout(Duration::from_secs(5), server.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
    assert!(!base.join("server/models/2.onnx").exists());
    assert!(base.join("server/checkpoints/1.pt").exists());
    drop(stream);

    fs::remove_dir(invalid).unwrap();
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    wait_for_model(&mut stream, &base.join("server/models/2.onnx")).await;
    assert!(!base.join("server/checkpoints/-1.pt").exists());
    assert!(base.join("server/checkpoints/1.pt").exists());
    drop(stream);
    stop(&mut server).await;
    assert_eq!(
        fs::read_to_string(base.join("exports")).unwrap(),
        "0.onnx\n2.onnx\n"
    );
}

#[tokio::test]
async fn export_failure_stops_the_server_and_restart_recovers_publication() {
    let temp = tempfile::tempdir().unwrap();
    let base = fs::canonicalize(temp.path()).unwrap();
    let config = export_test_config(&base);
    fs::write(base.join("pause"), b"").unwrap();
    fs::write(base.join("fail"), b"").unwrap();
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    send_checkpoint(&mut stream, &base, -1).await;
    wait_for_path(&base.join("started")).await;
    let mut observer = connect(address).await;
    assert!(matches!(
        request(&mut observer, Request::GetLatestModel {}).await,
        Response::Model { path: None }
    ));
    // Model the helper exiting after final publication but before success.
    fs::write(base.join("release"), b"").unwrap();
    assert!(
        !timeout(Duration::from_secs(5), server.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
    drop(stream);
    drop(observer);
    fs::remove_file(base.join("fail")).unwrap();
    let models = base.join("server/models");
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    wait_for_model(&mut stream, &models.join("0.onnx")).await;
    drop(stream);
    stop(&mut server).await;
    assert_eq!(
        fs::read_to_string(base.join("exports")).unwrap(),
        "0.onnx\n"
    );

    // Model interruption before final publication. Reuse the accepted weights
    // and export again without another trainer submission.
    fs::remove_file(models.join("0.onnx")).unwrap();
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    wait_for_model(&mut stream, &models.join("0.onnx")).await;
    assert_eq!(fs::read(models.join("0.onnx")).unwrap(), b"checkpoint -1");
    drop(stream);
    stop(&mut server).await;
}

#[tokio::test]
async fn shutdown_keeps_the_storage_lock_until_export_finishes() {
    let temp = tempfile::tempdir().unwrap();
    let base = fs::canonicalize(temp.path()).unwrap();
    let config = export_test_config(&base);
    fs::write(base.join("pause"), b"").unwrap();
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    send_checkpoint(&mut stream, &base, -1).await;
    wait_for_path(&base.join("started")).await;
    assert!(
        std::process::Command::new("kill")
            .args(["-INT", &format!("-{}", server.id().unwrap())])
            .status()
            .unwrap()
            .success()
    );
    assert!(
        timeout(
            Duration::from_secs(5),
            rgo_protocol::read::<Response>(&mut stream)
        )
        .await
        .unwrap()
        .unwrap()
        .is_none()
    );
    let mut second = Command::new(env!("CARGO_BIN_EXE_rgo-server"))
        .arg(&config)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    assert!(
        !timeout(Duration::from_secs(5), second.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
    assert!(server.try_wait().unwrap().is_none());
    fs::write(base.join("release"), b"").unwrap();
    assert!(
        timeout(Duration::from_secs(5), server.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
}

#[tokio::test]
async fn model_polls_respond_while_export_holds_the_storage_queue() {
    let temp = tempfile::tempdir().unwrap();
    let base = fs::canonicalize(temp.path()).unwrap();
    let config = export_test_config(&base);
    fs::write(base.join("pause"), b"").unwrap();
    let (mut server, address) = start(&config).await;
    let mut submitter = connect(address).await;
    send_checkpoint(&mut submitter, &base, -1).await;
    wait_for_path(&base.join("started")).await;
    let mut observer = connect(address).await;
    assert!(matches!(
        request(&mut observer, Request::GetLatestModel {}).await,
        Response::Model { path: None }
    ));
    let mut stats_connection = connect(address).await;
    let stats =
        tokio::spawn(
            async move { request(&mut stats_connection, Request::GetReplayStats {}).await },
        );
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(!stats.is_finished());
    fs::write(base.join("release"), b"").unwrap();
    assert!(matches!(
        timeout(
            Duration::from_secs(5),
            rgo_protocol::read::<Response>(&mut submitter)
        )
        .await
        .unwrap()
        .unwrap(),
        Some(Response::CheckpointAccepted {})
    ));
    assert!(matches!(stats.await.unwrap(), Response::Stats { .. }));
    wait_for_model(&mut observer, &base.join("server/models/0.onnx")).await;
    drop(submitter);
    drop(observer);
    stop(&mut server).await;
}

#[tokio::test]
async fn checkpoint_handoff_retries_and_recovers_after_restart() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("server");
    let config = temp.path().join("server.toml");
    write_test_python(temp.path());
    fs::write(&config, format!("listen = '127.0.0.1:0'\ndata_dir = {root:?}\nprune_old_shards = false\n[checkpoints]\nprune_old_checkpoints = false\n[replay]\ncapacity_records = 100\nwindow_records = 100\nsample_group_records = 20\nsample_groups_per_shard = 4\nrecords_per_shard = 10\n[model_export]\nevery_shards = 10\n")).unwrap();
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    assert!(matches!(
        request(&mut stream, Request::GetLatestCheckpoint {}).await,
        Response::Checkpoint { path: None }
    ));
    let directory = fs::canonicalize(root.join("checkpoints")).unwrap();
    let shards = fs::canonicalize(root.join("shards")).unwrap();
    fs::write(shards.join("0.npz"), b"covered shard").unwrap();
    fs::write(shards.join("1.npz"), b"prefetched shard").unwrap();
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
            assert!(shards.join("0.npz").exists());
            assert!(shards.join("1.npz").exists());
        }
        assert!(matches!(
            request(&mut stream, Request::GetLatestCheckpoint {}).await,
            Response::Checkpoint { path: Some(path) } if path == directory.join(format!("{index}.pt"))
        ));
    }
    assert!(matches!(
        request(&mut stream, Request::GetTrainingShard { index: 0 }).await,
        Response::Error { .. }
    ));
    assert!(directory.join("-1.pt").exists());
    drop(stream);
    stop(&mut server).await;

    fs::write(
        &config,
        fs::read_to_string(&config)
            .unwrap()
            .replace("prune_old_shards = false", "prune_old_shards = true"),
    )
    .unwrap();
    let (mut server, address) = start(&config).await;
    let mut stream = connect(address).await;
    assert!(!shards.join("0.npz").exists());
    assert_eq!(fs::read(shards.join("1.npz")).unwrap(), b"prefetched shard");
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
    fs::write(&config, format!("listen = '127.0.0.1:0'\ndata_dir = {:?}\n[replay]\ncapacity_records = 100\nwindow_records = 100\nsample_group_records = 20\nsample_groups_per_shard = 4\nrecords_per_shard = 3\n[model_export]\nevery_shards = 10\n", root)).unwrap();
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
