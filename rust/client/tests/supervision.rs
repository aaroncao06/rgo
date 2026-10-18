#![cfg(unix)]

use rgo_artifacts::chunk_path;
use rgo_protocol::{
    PROTOCOL_VERSION,
    server::{Request, Response},
};
use std::{
    fs,
    net::SocketAddr,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    io::BufReader,
    net::TcpListener,
    process::{Child, Command},
    task::JoinHandle,
    time::{sleep, timeout},
};

struct Server {
    address: SocketAddr,
    accepted: Arc<Mutex<Vec<String>>>,
    requests: Arc<Mutex<Vec<PathBuf>>>,
    task: JoinHandle<()>,
}

impl Server {
    async fn start(root: &Path) -> Self {
        Self::start_with_delay(root, Duration::ZERO).await
    }

    async fn start_with_delay(root: &Path, delay: Duration) -> Self {
        Self::start_with_options(root, delay, false).await
    }

    async fn start_with_options(root: &Path, delay: Duration, lose_first_receipt: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let root = fs::canonicalize(root).unwrap();
        for name in ["models", "output", "shards"] {
            fs::create_dir(root.join(name)).unwrap();
        }
        let accepted = Arc::new(Mutex::new(Vec::new()));
        let log = accepted.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let request_log = requests.clone();
        let task = tokio::spawn(async move {
            let mut model_polls = 0;
            let mut lose_receipt = lose_first_receipt;
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let mut stream = BufReader::new(stream);
                while let Some(request) = rgo_protocol::read::<Request>(&mut stream).await.unwrap()
                {
                    let response = match request {
                        Request::Hello {
                            protocol_version: PROTOCOL_VERSION,
                            ..
                        } => Response::Welcome {
                            protocol_version: PROTOCOL_VERSION,
                        },
                        Request::LatestModel {} => {
                            model_polls += 1;
                            let version = if lose_first_receipt {
                                if lose_receipt { 42 } else { 43 }
                            } else if model_polls == 1 {
                                42
                            } else {
                                43
                            };
                            let directory = if lose_first_receipt && !lose_receipt {
                                "models-after-reconnect"
                            } else {
                                "models"
                            };
                            Response::Model {
                                path: Some(root.join(directory).join(format!("{version}.onnx"))),
                            }
                        }
                        Request::AcceptChunk { path } => {
                            request_log.lock().unwrap().push(path.clone());
                            sleep(delay).await;
                            let name = path.file_name().unwrap().to_str().unwrap().to_owned();
                            let already_accepted = log.lock().unwrap().contains(&name);
                            if !already_accepted {
                                fs::remove_file(&path).unwrap();
                                log.lock().unwrap().push(name);
                            }
                            // Simulate acceptance and producer-name removal before
                            // disconnecting without delivering the durable receipt.
                            if lose_receipt {
                                lose_receipt = false;
                                break;
                            }
                            Response::ChunkAccepted {
                                records: 1,
                                already_accepted,
                            }
                        }
                        other => panic!("unexpected request: {other:?}"),
                    };
                    if rgo_protocol::write(stream.get_mut(), &response)
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
        });
        Self {
            address,
            accepted,
            requests,
            task,
        }
    }
}

#[tokio::test]
async fn shutdown_interrupts_pending_recovery_without_starting_a_worker() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start_with_delay(temp.path(), Duration::from_secs(2)).await;
    for id in 1..=3 {
        fs::write(chunk_path(&temp.path().join("output"), id), b"pending").unwrap();
    }
    let mut client = start_client(
        temp.path(),
        server.address,
        "touch \"$RGO_TEST_OUTPUT_DIR/worker-started\"; exit 0",
    );
    wait_for(&mut client, || !server.requests.lock().unwrap().is_empty()).await;
    signal(&client);
    let exit = timeout(Duration::from_secs(1), client.wait())
        .await
        .expect("shutdown must interrupt recovery rather than await its backlog")
        .unwrap();
    assert!(exit.success());
    assert!(!temp.path().join("output/worker-started").exists());
    assert_eq!(server.requests.lock().unwrap().len(), 1);
    // The in-flight request may eventually succeed server-side. Until then,
    // all producer files remain available for the next client's recovery.
    for id in 1..=3 {
        assert!(chunk_path(&temp.path().join("output"), id).exists());
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn start_client(root: &Path, address: SocketAddr, script: &str) -> Child {
    start_client_with_budget(root, address, script, 1)
}

fn start_client_with_budget(root: &Path, address: SocketAddr, script: &str, budget: u64) -> Child {
    client_command(root, address, script, budget)
        .spawn()
        .unwrap()
}

fn client_command(root: &Path, address: SocketAddr, script: &str, budget: u64) -> Command {
    let worker = root.join("worker.sh");
    fs::write(&worker, format!("#!/bin/sh\n{script}\n")).unwrap();
    fs::set_permissions(&worker, fs::Permissions::from_mode(0o700)).unwrap();
    let worker_config = root.join("worker.toml");
    fs::write(
        &worker_config,
        format!("output_dir = {:?}\n", root.join("output")),
    )
    .unwrap();
    let config = root.join("client.toml");
    fs::write(&config, format!("server = '{address}'\nclient_id = 'test'\nselfplay_config = {worker_config:?}\nselfplay_binary = {:?}\nmodel_poll_interval_ms = 10\nrequest_timeout_secs = 2\nretry_timeout_secs = {budget}\n", worker)).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_rgo-client"));
    command
        .arg(config)
        .env("RGO_TEST_OUTPUT_DIR", root.join("output"))
        .stdout(Stdio::null())
        .stderr(Stdio::from(
            fs::File::create(root.join("client.log")).unwrap(),
        ))
        .kill_on_drop(true);
    command
}

#[tokio::test]
async fn bare_relative_worker_binary_uses_client_working_directory() {
    let temp = tempfile::tempdir().unwrap();
    let root = fs::canonicalize(temp.path()).unwrap();
    let server = Server::start(&root).await;
    let script = "out=\"$RGO_TEST_OUTPUT_DIR\"\necho '{\"type\":\"ready\",\"protocol_version\":1}'\necho started > \"$out/started\"\nwhile IFS= read -r command; do case \"$command\" in *finish*) echo '{\"type\":\"stopped\"}'; exit 0;; esac; done";
    let mut command = client_command(&root, server.address, script, 1);
    let config = root.join("client.toml");
    let original = fs::read_to_string(&config).unwrap();
    let absolute = format!("selfplay_binary = {:?}", root.join("worker.sh"));
    let relative = original.replacen(&absolute, "selfplay_binary = 'worker.sh'", 1);
    assert_ne!(relative, original);
    fs::write(config, relative).unwrap();
    command.current_dir(&root).env("PATH", "/usr/bin:/bin");
    let mut client = command.spawn().unwrap();
    wait_for(&mut client, || root.join("output/started").exists()).await;
    signal(&client);
    assert!(status(&mut client, &root).await.success());
}

async fn wait_for(client: &mut Child, condition: impl Fn() -> bool) {
    timeout(Duration::from_secs(10), async {
        while !condition() {
            assert!(client.try_wait().unwrap().is_none(), "client exited early");
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

fn signal(client: &Child) {
    assert!(
        std::process::Command::new("kill")
            .args(["-TERM", &client.id().unwrap().to_string()])
            .status()
            .unwrap()
            .success()
    );
}

async fn status(client: &mut Child, root: &Path) -> std::process::ExitStatus {
    timeout(Duration::from_secs(10), client.wait())
        .await
        .unwrap_or_else(|_| panic!("{}", fs::read_to_string(root.join("client.log")).unwrap()))
        .unwrap()
}

#[tokio::test]
async fn recovers_pending_chunks_forwards_models_and_accepts_final_chunks_before_exit() {
    let temp = tempfile::tempdir().unwrap();
    let root = fs::canonicalize(temp.path()).unwrap();
    let server = Server::start(&root).await;
    let pending = chunk_path(&root.join("output"), 1);
    fs::write(&pending, b"pending").unwrap();
    let live = chunk_path(&root.join("output"), 2);
    let final_chunk = chunk_path(&root.join("output"), 3);
    let live_event = serde_json::to_string(&rgo_protocol::selfplay::Event::ChunkReady {
        path: live.clone(),
        bytes: 4,
        records: 1,
    })
    .unwrap();
    let final_event = serde_json::to_string(&rgo_protocol::selfplay::Event::ChunkReady {
        path: final_chunk.clone(),
        bytes: 5,
        records: 1,
    })
    .unwrap();
    let script = format!(
        r#"
out="$RGO_TEST_OUTPUT_DIR"
echo '{{"type":"ready","protocol_version":1}}'
while IFS= read -r command; do
  echo "$command" >> "$out/commands.log"
  case "$command" in
    *model_ready*)
      if [ ! -e "$out/made" ]; then
        echo live > "$out/{}"
        touch "$out/made"
        echo '{live_event}'
      fi;;
    *finish*)
      echo final > "$out/{}"
      echo '{final_event}'
      echo '{{"type":"stopped"}}'
      exit 0;;
  esac
done
exit 2
"#,
        live.file_name().unwrap().to_str().unwrap(),
        final_chunk.file_name().unwrap().to_str().unwrap()
    );
    let mut client = start_client(&root, server.address, &script);
    wait_for(&mut client, || {
        fs::read_to_string(root.join("output/commands.log"))
            .unwrap_or_default()
            .contains("43")
            && server.accepted.lock().unwrap().len() == 2
    })
    .await;
    signal(&client);
    assert!(
        status(&mut client, &root).await.success(),
        "{}",
        fs::read_to_string(root.join("client.log")).unwrap()
    );
    let commands: Vec<rgo_protocol::selfplay::Command> =
        fs::read_to_string(root.join("output/commands.log"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
    for version in [42, 43] {
        assert!(commands.iter().any(|command| matches!(
            command,
            rgo_protocol::selfplay::Command::ModelReady { path }
                if path == &root.join(format!("models/{version}.onnx"))
        )));
    }
    assert!(matches!(
        commands.last(),
        Some(rgo_protocol::selfplay::Command::Finish {})
    ));
    assert_eq!(
        *server.accepted.lock().unwrap(),
        [pending, live, final_chunk].map(|path| path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned())
    );
}

#[tokio::test]
async fn retry_budget_expiry_reaps_the_worker() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path()).await;
    let script = "out=\"$RGO_TEST_OUTPUT_DIR\"\necho $$ > \"$out/pid\"\necho '{\"type\":\"ready\",\"protocol_version\":1}'\nwhile :; do :; done";
    let mut client = start_client(temp.path(), server.address, script);
    let pid_file = temp.path().join("output/pid");
    wait_for(&mut client, || pid_file.exists()).await;
    server.task.abort();
    assert!(!status(&mut client, temp.path()).await.success());
    assert!(
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap()
            .contains("retry budget expired")
    );
    let pid = fs::read_to_string(pid_file).unwrap();
    assert!(
        !std::process::Command::new("kill")
            .args(["-0", pid.trim()])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
    );
}

#[tokio::test]
async fn successful_child_exit_without_stopped_is_reported_instead_of_hanging() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path()).await;
    let mut client = start_client(
        temp.path(),
        server.address,
        "echo '{\"type\":\"ready\",\"protocol_version\":1}'\nexit 0",
    );
    assert!(!status(&mut client, temp.path()).await.success());
}

#[tokio::test]
async fn first_signal_after_stopped_remains_graceful() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path()).await;
    let script = "out=\"$RGO_TEST_OUTPUT_DIR\"\necho '{\"type\":\"ready\",\"protocol_version\":1}'\necho '{\"type\":\"stopped\"}'\necho started > \"$out/started\"\nwhile :; do :; done";
    let mut client = start_client(temp.path(), server.address, script);
    wait_for(&mut client, || temp.path().join("output/started").exists()).await;
    sleep(Duration::from_millis(100)).await;
    signal(&client);
    wait_for(&mut client, || {
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap_or_default()
            .contains("finishing active games")
    })
    .await;
    assert!(client.try_wait().unwrap().is_none());
    signal(&client);
    assert!(!status(&mut client, temp.path()).await.success());
}

#[tokio::test]
async fn second_shutdown_signal_kills_and_reaps_worker_before_releasing_output_lock() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path()).await;
    let script = "out=\"$RGO_TEST_OUTPUT_DIR\"\necho $$ > \"$out/pid\"\necho '{\"type\":\"ready\",\"protocol_version\":1}'\nwhile IFS= read -r command; do case \"$command\" in *finish*) echo received > \"$out/finish-received\"; while :; do :; done;; esac; done";
    let mut client = start_client(temp.path(), server.address, script);
    let pid_file = temp.path().join("output/pid");
    wait_for(&mut client, || pid_file.exists()).await;
    signal(&client);
    wait_for(&mut client, || {
        temp.path().join("output/finish-received").exists()
    })
    .await;
    sleep(Duration::from_millis(1200)).await;
    assert!(client.try_wait().unwrap().is_none());
    signal(&client);
    assert!(!status(&mut client, temp.path()).await.success());
    assert!(
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap()
            .contains("second shutdown signal")
    );
    let pid = fs::read_to_string(pid_file).unwrap();
    assert!(
        !std::process::Command::new("kill")
            .args(["-0", pid.trim()])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
    );
    let lock = fs::File::options()
        .read(true)
        .write(true)
        .open(temp.path().join("output/.client.lock"))
        .unwrap();
    lock.try_lock().unwrap();
}

#[tokio::test]
async fn refuses_recovery_while_previous_worker_owns_output() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path()).await;
    let lock = fs::File::create(temp.path().join("output/.selfplay.lock")).unwrap();
    lock.try_lock().unwrap();
    let pending: PathBuf = chunk_path(&temp.path().join("output"), 1);
    fs::write(&pending, b"pending").unwrap();
    let mut client = start_client(temp.path(), server.address, "exit 9");
    assert!(!status(&mut client, temp.path()).await.success());
    assert!(pending.exists());
    assert!(server.accepted.lock().unwrap().is_empty());
    assert!(
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap()
            .contains("previous worker")
    );
}

#[tokio::test]
async fn lost_chunk_receipt_reconnects_without_restarting_the_worker() {
    let temp = tempfile::tempdir().unwrap();
    let root = fs::canonicalize(temp.path()).unwrap();
    let server = Server::start_with_options(&root, Duration::ZERO, true).await;
    let chunk = chunk_path(&root.join("output"), 1);
    let event = serde_json::to_string(&rgo_protocol::selfplay::Event::ChunkReady {
        path: chunk.clone(),
        bytes: 4,
        records: 1,
    })
    .unwrap();
    let script = format!(
        r#"
out="$RGO_TEST_OUTPUT_DIR"
echo $$ >> "$out/pids"
echo '{{"type":"ready","protocol_version":1}}'
while IFS= read -r command; do
  echo "$command" >> "$out/commands.log"
  case "$command" in
    *model_ready*)
      if [ ! -e "$out/made" ]; then
        touch "$out/made"
        echo live > "$out/{}"
        echo '{event}'
      fi;;
    *finish*) echo '{{"type":"stopped"}}'; exit 0;;
  esac
done
exit 2
"#,
        chunk.file_name().unwrap().to_str().unwrap()
    );
    let mut client = start_client_with_budget(&root, server.address, &script, 10);
    wait_for(&mut client, || {
        fs::read_to_string(root.join("client.log"))
            .unwrap_or_default()
            .contains("(retry)")
    })
    .await;
    let expected_path = root.join("models-after-reconnect/43.onnx");
    wait_for(&mut client, || {
        fs::read_to_string(root.join("output/commands.log"))
        .unwrap_or_default().lines().any(|line| matches!(
            serde_json::from_str::<rgo_protocol::selfplay::Command>(line),
            Ok(rgo_protocol::selfplay::Command::ModelReady { path }) if path == expected_path
        ))
    })
    .await;
    assert_eq!(
        *server.requests.lock().unwrap(),
        [chunk.clone(), chunk.clone()]
    );
    assert_eq!(server.accepted.lock().unwrap().len(), 1);
    assert!(!chunk.exists());
    assert_eq!(
        fs::read_to_string(root.join("output/pids"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    assert!(client.try_wait().unwrap().is_none());
    signal(&client);
    assert!(status(&mut client, &root).await.success());
}

#[tokio::test]
async fn worker_failure_interrupts_connection_retries() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path()).await;
    let script = "out=\"$RGO_TEST_OUTPUT_DIR\"\necho started > \"$out/started\"\necho '{\"type\":\"ready\",\"protocol_version\":1}'\nwhile [ ! -e \"$out/fail\" ]; do :; done\nexit 9";
    let mut client = start_client_with_budget(temp.path(), server.address, script, 10);
    // First establish the session and launch the child before losing the server.
    wait_for(&mut client, || temp.path().join("output/started").exists()).await;
    server.task.abort();
    wait_for(&mut client, || {
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap_or_default()
            .contains("retrying")
    })
    .await;
    fs::write(temp.path().join("output/fail"), b"").unwrap();
    let exit = timeout(Duration::from_secs(1), client.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(!exit.success());
    assert!(
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap()
            .contains("self-play exited")
    );
}

#[tokio::test]
async fn second_shutdown_signal_interrupts_connection_retries_and_reaps_worker() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path()).await;
    let script = "out=\"$RGO_TEST_OUTPUT_DIR\"\necho $$ > \"$out/pid\"\necho '{\"type\":\"ready\",\"protocol_version\":1}'\nwhile IFS= read -r command; do case \"$command\" in *finish*) echo finished > \"$out/finished\"; while :; do :; done;; esac; done";
    let mut client = start_client_with_budget(temp.path(), server.address, script, 10);
    wait_for(&mut client, || temp.path().join("output/pid").exists()).await;
    server.task.abort();
    wait_for(&mut client, || {
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap_or_default()
            .contains("retrying")
    })
    .await;
    signal(&client);
    wait_for(&mut client, || temp.path().join("output/finished").exists()).await;
    sleep(Duration::from_millis(1200)).await;
    assert!(client.try_wait().unwrap().is_none());
    signal(&client);
    let exit = timeout(Duration::from_secs(3), client.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(!exit.success());
    assert!(temp.path().join("output/finished").exists());
    let pid = fs::read_to_string(temp.path().join("output/pid")).unwrap();
    assert!(
        !std::process::Command::new("kill")
            .args(["-0", pid.trim()])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
    );
    assert!(
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap()
            .contains("second shutdown signal")
    );
}

#[tokio::test]
async fn shutdown_interrupts_initial_connection_retries_without_starting_a_worker() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path()).await;
    server.task.abort();
    while !server.task.is_finished() {
        tokio::task::yield_now().await;
    }
    let mut client = start_client_with_budget(
        temp.path(),
        server.address,
        "touch \"$RGO_TEST_OUTPUT_DIR/worker-started\"; exit 0",
        10,
    );
    wait_for(&mut client, || {
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap_or_default()
            .contains("retrying")
    })
    .await;
    signal(&client);
    let exit = timeout(Duration::from_secs(1), client.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(exit.success());
    assert!(!temp.path().join("output/worker-started").exists());
}

async fn rejecting_server() -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
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
        assert!(
            rgo_protocol::read::<Request>(&mut stream)
                .await
                .unwrap()
                .is_some()
        );
        rgo_protocol::write(
            stream.get_mut(),
            &Response::Error {
                message: "test server rejection".into(),
            },
        )
        .await
        .unwrap();
        // The application caller decides to exit after receiving the response.
        assert!(
            rgo_protocol::read::<Request>(&mut stream)
                .await
                .unwrap()
                .is_none()
        );
    });
    (address, task)
}

#[tokio::test]
async fn recovery_reports_chunk_rejection_without_starting_a_worker() {
    let temp = tempfile::tempdir().unwrap();
    fs::create_dir(temp.path().join("output")).unwrap();
    let chunk = chunk_path(&temp.path().join("output"), 1);
    fs::write(&chunk, b"pending").unwrap();
    let (address, server) = rejecting_server().await;
    let mut client = start_client(
        temp.path(),
        address,
        "touch \"$RGO_TEST_OUTPUT_DIR/worker-started\"; exit 0",
    );
    assert!(!status(&mut client, temp.path()).await.success());
    assert!(
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap()
            .contains("test server rejection")
    );
    assert!(chunk.exists());
    assert!(!temp.path().join("output/worker-started").exists());
    server.await.unwrap();
}

#[tokio::test]
async fn control_command_reports_server_rejection_and_exits_unsuccessfully() {
    let temp = tempfile::tempdir().unwrap();
    let (address, server) = rejecting_server().await;
    let mut client = client_command(temp.path(), address, "exit 0", 1)
        .arg("stats")
        .spawn()
        .unwrap();
    assert!(!status(&mut client, temp.path()).await.success());
    assert!(
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap()
            .contains("test server rejection")
    );
    server.await.unwrap();
}
