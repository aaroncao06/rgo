#![cfg(unix)]

use rgo_artifacts::chunk_path;
use rgo_protocol::{
    PROTOCOL_VERSION,
    client_server::{Request, Response},
};
use std::{
    fs,
    net::SocketAddr,
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    io::BufReader,
    net::TcpListener,
    process::{Child, Command},
    sync::Notify,
    task::JoinHandle,
    time::{sleep, timeout},
};

struct Server {
    address: SocketAddr,
    accepted: Arc<Mutex<Vec<String>>>,
    requests: Arc<Mutex<Vec<PathBuf>>>,
    model_polled: Arc<Notify>,
    task: JoinHandle<()>,
}

impl Server {
    async fn start(root: &Path) -> Self {
        Self::start_with_delay(root, Duration::ZERO).await
    }

    async fn start_with_delay(root: &Path, delay: Duration) -> Self {
        Self::start_with_options(root, delay, Duration::ZERO, false).await
    }

    async fn start_with_options(
        root: &Path,
        chunk_delay: Duration,
        model_delay: Duration,
        lose_first_receipt: bool,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let root = fs::canonicalize(root).unwrap();
        fs::create_dir_all(root.join("test-run/selfplay")).unwrap();
        publish_model_fixtures(&root);
        let accepted = Arc::new(Mutex::new(Vec::new()));
        let log = accepted.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let request_log = requests.clone();
        let model_polled = Arc::new(Notify::new());
        let poll_notification = model_polled.clone();
        let task = tokio::spawn(async move {
            let mut model_polls = 0;
            let mut lose_receipt = lose_first_receipt;
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let mut stream = BufReader::new(stream);
                loop {
                    let request = match rgo_protocol::read::<Request>(&mut stream).await {
                        Ok(Some(request)) => request,
                        Ok(None) => break,
                        Err(error) if error.kind() == std::io::ErrorKind::ConnectionReset => break,
                        Err(error) => panic!("could not read client request: {error}"),
                    };
                    let response = match request {
                        Request::Hello {
                            protocol_version: PROTOCOL_VERSION,
                            ..
                        } => Response::Welcome {
                            protocol_version: PROTOCOL_VERSION,
                            run_name: "test-run".into(),
                        },
                        Request::GetLatestModel {} => {
                            model_polls += 1;
                            if !model_delay.is_zero() {
                                sleep(model_delay).await;
                            }
                            poll_notification.notify_one();
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
                                path: root.join(directory).join(format!("{version}.onnx")),
                            }
                        }
                        Request::SubmitChunk { path } => {
                            request_log.lock().unwrap().push(path.clone());
                            sleep(chunk_delay).await;
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
                            Response::ChunkAccepted {}
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
            model_polled,
            task,
        }
    }
}

#[tokio::test]
async fn shutdown_interrupts_pending_recovery_without_starting_a_worker() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start_with_delay(temp.path(), Duration::from_secs(2)).await;
    for id in 1..=3 {
        fs::write(
            chunk_path(&temp.path().join("test-run/selfplay"), id),
            b"pending",
        )
        .unwrap();
    }
    let mut client = start_client(
        temp.path(),
        server.address,
        "touch \"$output_dir/worker-started\"; exit 0",
    );
    wait_for(&mut client, || !server.requests.lock().unwrap().is_empty()).await;
    signal(&client);
    let exit = timeout(Duration::from_secs(1), client.wait())
        .await
        .expect("shutdown must interrupt recovery rather than await its backlog")
        .unwrap();
    assert!(exit.success());
    assert!(
        !temp
            .path()
            .join("test-run/selfplay/worker-started")
            .exists()
    );
    assert_eq!(server.requests.lock().unwrap().len(), 1);
    // The in-flight request may eventually succeed server-side. Until then,
    // all producer files remain available for the next client's recovery.
    for id in 1..=3 {
        assert!(chunk_path(&temp.path().join("test-run/selfplay"), id).exists());
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
    fs::write(
        &worker,
        format!(
            r#"#!/bin/sh
output_dir="$2"
model_loaded() {{
  filename=${{1##*/}}
  printf '{{"type":"model_loaded","version":%s}}\n' "${{filename%%.onnx*}}"
}}
{script}
"#
        ),
    )
    .unwrap();
    fs::set_permissions(&worker, fs::Permissions::from_mode(0o700)).unwrap();
    let worker_config = root.join("worker.toml");
    fs::write(&worker_config, "").unwrap();
    let config = root.join("client.toml");
    fs::write(&config, format!("server = '{address}'\nclient_id = 'test'\ndata_dir = {root:?}\nselfplay_config = {worker_config:?}\nselfplay_binary = {:?}\nmodel_poll_interval_ms = 10\nrequest_timeout_secs = 2\nretry_timeout_secs = {budget}\n", worker)).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_rgo-client"));
    command
        .arg(config)
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
    let script = "out=\"$output_dir\"\necho '{\"type\":\"ready\",\"protocol_version\":1}'\necho started > \"$out/started\"\nwhile IFS= read -r command; do case \"$command\" in *finish*) echo '{\"type\":\"stopped\"}'; exit 0;; esac; done";
    let mut command = client_command(&root, server.address, script, 1);
    let config = root.join("client.toml");
    let original = fs::read_to_string(&config).unwrap();
    let absolute = format!("selfplay_binary = {:?}", root.join("worker.sh"));
    let relative = original.replacen(&absolute, "selfplay_binary = 'worker.sh'", 1);
    assert_ne!(relative, original);
    fs::write(config, relative).unwrap();
    command.current_dir(&root).env("PATH", "/usr/bin:/bin");
    let mut client = command.spawn().unwrap();
    wait_for(&mut client, || {
        root.join("test-run/selfplay/started").exists()
    })
    .await;
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

fn worker_received_model(log_path: &Path, expected_path: &Path) -> bool {
    use rgo_protocol::selfplay::Command;

    fs::read_to_string(log_path)
        .unwrap_or_default()
        .lines()
        .any(|line| {
            matches!(
                serde_json::from_str::<Command>(line),
                Ok(Command::ModelReady { path }) if path == expected_path
            )
        })
}

fn publish_model_fixtures(root: &Path) {
    for name in [
        "models/42.onnx",
        "models/43.onnx",
        "models-after-reconnect/43.onnx",
    ] {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"model fixture").unwrap();
    }
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
    let pending = chunk_path(&root.join("test-run/selfplay"), 1);
    fs::write(&pending, b"pending").unwrap();
    let other_run = root.join("other-run/selfplay");
    fs::create_dir_all(&other_run).unwrap();
    let other_pending = chunk_path(&other_run, 1);
    fs::write(&other_pending, b"other-run pending").unwrap();
    let live = chunk_path(&root.join("test-run/selfplay"), 2);
    let final_chunk = chunk_path(&root.join("test-run/selfplay"), 3);
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
out="$output_dir"
echo '{{"type":"ready","protocol_version":1}}'
while IFS= read -r command; do
  echo "$command" >> "$out/commands.log"
  case "$command" in
    *model_ready*)
      model_loaded "$command"
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
    let command_log = root.join("test-run/selfplay/commands.log");
    let expected_path = root.join("test-run/selfplay/models/43.onnx");
    wait_for(&mut client, || {
        worker_received_model(&command_log, &expected_path)
            && server.accepted.lock().unwrap().len() == 2
    })
    .await;
    signal(&client);
    assert!(
        status(&mut client, &root).await.success(),
        "{}",
        fs::read_to_string(root.join("client.log")).unwrap()
    );
    let commands: Vec<rgo_protocol::selfplay::Command> = fs::read_to_string(&command_log)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for version in [42, 43] {
        assert!(commands.iter().any(|command| matches!(
            command,
            rgo_protocol::selfplay::Command::ModelReady { path }
                if path == &root.join(format!("test-run/selfplay/models/{version}.onnx"))
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
    assert_eq!(fs::read(other_pending).unwrap(), b"other-run pending");
}

#[tokio::test]
async fn model_links_survive_pruning_and_are_reused_on_restart() {
    let temp = tempfile::tempdir().unwrap();
    let root = fs::canonicalize(temp.path()).unwrap();
    let server = Server::start(&root).await;
    // The first poll advertises a version already pruned from server storage.
    fs::remove_file(root.join("models/42.onnx")).unwrap();
    let source = root.join("models/43.onnx");
    let selfplay_dir = root.join("test-run/selfplay");
    let models = selfplay_dir.join("models");
    fs::create_dir(&models).unwrap();
    let previous = models.join("41.onnx");
    fs::write(&previous, b"previous model").unwrap();
    let current = models.join("43.onnx");
    let command_log = selfplay_dir.join("model-command");
    let script = r#"
echo '{"type":"ready","protocol_version":1}'
while IFS= read -r command; do
  case "$command" in
    *model_ready*)
      printf '%s\n' "$command" > "$output_dir/model-command"
      attempts=0
      while [ ! -e "$output_dir/load-model" ]; do
        attempts=$((attempts + 1))
        [ "$attempts" -lt 500 ] || exit 7
        sleep 0.01
      done
      [ -f "$output_dir/models/43.onnx" ] || exit 8
      model_loaded "$command";;
    *finish*) echo '{"type":"stopped"}'; exit 0;;
  esac
done
exit 2
"#;
    let mut client = start_client_with_budget(&root, server.address, script, 5);
    wait_for(&mut client, || {
        worker_received_model(&command_log, &current)
    })
    .await;
    assert!(
        !previous.exists(),
        "prune before announcing the replacement"
    );
    assert_eq!(
        fs::metadata(&source).unwrap().ino(),
        fs::metadata(&current).unwrap().ino()
    );
    fs::remove_file(source).unwrap();
    // Consume the earlier poll notification while the worker is still waiting.
    server.model_polled.notified().await;
    fs::write(selfplay_dir.join("load-model"), b"").unwrap();
    timeout(Duration::from_secs(5), server.model_polled.notified())
        .await
        .unwrap();
    signal(&client);
    assert!(status(&mut client, &root).await.success());
    assert!(current.is_file());

    // Reuse the complete local file when the same version is advertised.
    fs::remove_file(&command_log).unwrap();
    let mut client = start_client(&root, server.address, script);
    wait_for(&mut client, || {
        worker_received_model(&command_log, &current)
    })
    .await;
    signal(&client);
    assert!(status(&mut client, &root).await.success());
    assert!(current.is_file());
}

#[tokio::test]
async fn shutdown_interrupts_model_acquisition_and_submits_final_chunk() {
    let temp = tempfile::tempdir().unwrap();
    let root = fs::canonicalize(temp.path()).unwrap();
    // Separate the acquisition deadline from the retry sleep's deadline.
    let server =
        Server::start_with_options(&root, Duration::ZERO, Duration::from_millis(150), false).await;
    // Advertise a model path whose server-side link has already been removed.
    fs::remove_file(root.join("models/42.onnx")).unwrap();
    let chunk = chunk_path(&root.join("test-run/selfplay"), 1);
    let event = serde_json::to_string(&rgo_protocol::selfplay::Event::ChunkReady {
        path: chunk.clone(),
        bytes: 5,
        records: 1,
    })
    .unwrap();
    let script = format!(
        r#"
echo '{{"type":"ready","protocol_version":1}}'
while IFS= read -r command; do
  case "$command" in
    *finish*)
      printf final > "$output_dir/{}"
      printf '%s\n' '{event}'
      echo '{{"type":"stopped"}}'
      exit 0;;
  esac
done
exit 2
"#,
        chunk.file_name().unwrap().to_str().unwrap(),
    );
    let mut client = start_client(&root, server.address, &script);
    timeout(Duration::from_secs(5), server.model_polled.notified())
        .await
        .unwrap();
    // Let the client receive the reply and reach its one-second retry sleep.
    sleep(Duration::from_millis(50)).await;
    signal(&client);
    assert!(
        status(&mut client, &root).await.success(),
        "{}",
        fs::read_to_string(root.join("client.log")).unwrap()
    );
    assert_eq!(*server.requests.lock().unwrap(), [chunk]);
}

#[tokio::test]
async fn retry_budget_expiry_reaps_the_worker() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path()).await;
    let script = r#"
echo $$ > "$output_dir/pid"
echo '{"type":"ready","protocol_version":1}'
while IFS= read -r command; do model_loaded "$command"; done
"#;
    let mut client = start_client(temp.path(), server.address, script);
    let pid_file = temp.path().join("test-run/selfplay/pid");
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
    let script = "out=\"$output_dir\"\necho '{\"type\":\"ready\",\"protocol_version\":1}'\necho '{\"type\":\"stopped\"}'\necho started > \"$out/started\"\nwhile :; do :; done";
    let mut client = start_client(temp.path(), server.address, script);
    wait_for(&mut client, || {
        temp.path().join("test-run/selfplay/started").exists()
    })
    .await;
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
    let script = "out=\"$output_dir\"\necho $$ > \"$out/pid\"\necho '{\"type\":\"ready\",\"protocol_version\":1}'\nwhile IFS= read -r command; do case \"$command\" in *finish*) echo received > \"$out/finish-received\"; while :; do :; done;; esac; done";
    let mut client = start_client(temp.path(), server.address, script);
    let pid_file = temp.path().join("test-run/selfplay/pid");
    wait_for(&mut client, || pid_file.exists()).await;
    signal(&client);
    wait_for(&mut client, || {
        temp.path()
            .join("test-run/selfplay/finish-received")
            .exists()
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
        .open(temp.path().join("test-run/selfplay/.client.lock"))
        .unwrap();
    lock.try_lock().unwrap();
}

#[tokio::test]
async fn refuses_recovery_while_previous_worker_owns_output() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path()).await;
    let lock = fs::File::create(temp.path().join("test-run/selfplay/.selfplay.lock")).unwrap();
    lock.try_lock().unwrap();
    let pending: PathBuf = chunk_path(&temp.path().join("test-run/selfplay"), 1);
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
    let server = Server::start_with_options(&root, Duration::ZERO, Duration::ZERO, true).await;
    let chunk = chunk_path(&root.join("test-run/selfplay"), 1);
    let event = serde_json::to_string(&rgo_protocol::selfplay::Event::ChunkReady {
        path: chunk.clone(),
        bytes: 4,
        records: 1,
    })
    .unwrap();
    let script = format!(
        r#"
out="$output_dir"
echo $$ >> "$out/pids"
echo '{{"type":"ready","protocol_version":1}}'
while IFS= read -r command; do
  echo "$command" >> "$out/commands.log"
  case "$command" in
    *model_ready*)
      model_loaded "$command"
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
            .contains("rgo-client: accepted")
    })
    .await;
    let command_log = root.join("test-run/selfplay/commands.log");
    let expected_path = root.join("test-run/selfplay/models/43.onnx");
    wait_for(&mut client, || {
        worker_received_model(&command_log, &expected_path)
    })
    .await;
    assert_eq!(
        fs::metadata(&expected_path).unwrap().ino(),
        fs::metadata(root.join("models-after-reconnect/43.onnx"))
            .unwrap()
            .ino()
    );
    assert_eq!(
        *server.requests.lock().unwrap(),
        [chunk.clone(), chunk.clone()]
    );
    assert_eq!(server.accepted.lock().unwrap().len(), 1);
    assert!(!chunk.exists());
    assert_eq!(
        fs::read_to_string(root.join("test-run/selfplay/pids"))
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
    let script = r#"
echo '{"type":"ready","protocol_version":1}'
while IFS= read -r command; do
  model_loaded "$command"
  case "$command" in *43.onnx*) break;; esac
done
touch "$output_dir/started"
while [ ! -e "$output_dir/fail" ]; do :; done
exit 9
"#;
    let mut client = start_client_with_budget(temp.path(), server.address, script, 10);
    // Acknowledge both fixture models so the next poll can detect the disconnect.
    wait_for(&mut client, || {
        temp.path().join("test-run/selfplay/started").exists()
    })
    .await;
    server.task.abort();
    wait_for(&mut client, || {
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap_or_default()
            .contains("retrying")
    })
    .await;
    fs::write(temp.path().join("test-run/selfplay/fail"), b"").unwrap();
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
    let script = r#"
echo $$ > "$output_dir/pid"
echo '{"type":"ready","protocol_version":1}'
while IFS= read -r command; do
  case "$command" in
    *model_ready*) model_loaded "$command";;
    *finish*) touch "$output_dir/finished"; while :; do :; done;;
  esac
done
"#;
    let mut client = start_client_with_budget(temp.path(), server.address, script, 10);
    wait_for(&mut client, || {
        temp.path().join("test-run/selfplay/pid").exists()
    })
    .await;
    server.task.abort();
    wait_for(&mut client, || {
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap_or_default()
            .contains("retrying")
    })
    .await;
    signal(&client);
    wait_for(&mut client, || {
        temp.path().join("test-run/selfplay/finished").exists()
    })
    .await;
    sleep(Duration::from_millis(1200)).await;
    assert!(client.try_wait().unwrap().is_none());
    signal(&client);
    let exit = timeout(Duration::from_secs(3), client.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(!exit.success());
    assert!(temp.path().join("test-run/selfplay/finished").exists());
    let pid = fs::read_to_string(temp.path().join("test-run/selfplay/pid")).unwrap();
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
        "touch \"$output_dir/worker-started\"; exit 0",
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
    assert!(
        !temp
            .path()
            .join("test-run/selfplay/worker-started")
            .exists()
    );
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
                run_name: "test-run".into(),
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
    fs::create_dir_all(temp.path().join("test-run/selfplay")).unwrap();
    let chunk = chunk_path(&temp.path().join("test-run/selfplay"), 1);
    fs::write(&chunk, b"pending").unwrap();
    let (address, server) = rejecting_server().await;
    let mut client = start_client(
        temp.path(),
        address,
        "touch \"$output_dir/worker-started\"; exit 0",
    );
    assert!(!status(&mut client, temp.path()).await.success());
    assert!(
        fs::read_to_string(temp.path().join("client.log"))
            .unwrap()
            .contains("test server rejection")
    );
    assert!(chunk.exists());
    assert!(
        !temp
            .path()
            .join("test-run/selfplay/worker-started")
            .exists()
    );
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
