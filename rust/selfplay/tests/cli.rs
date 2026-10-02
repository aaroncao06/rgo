use std::process::Command;

#[test]
fn help_and_usage_do_not_start_self_play() {
    let help = Command::new(env!("CARGO_BIN_EXE_rgo-selfplay"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("<config.toml>"));
    for args in [
        vec![],
        vec!["one.toml", "two.toml"],
        vec!["--managed", "one.toml"],
        vec!["--managed", "one.toml", "two.toml"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_rgo-selfplay"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("Usage:"));
    }
}

#[cfg(unix)]
mod unix {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        io::Write,
        path::PathBuf,
        process::{Child, ExitStatus, Stdio},
        sync::atomic::{AtomicU64, Ordering},
        thread,
        time::{Duration, Instant},
    };

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestProcess {
        dir: PathBuf,
        child: Child,
    }

    impl TestProcess {
        fn worker(model: Option<&str>) -> Self {
            Self::start(model, true)
        }

        fn start(model: Option<&str>, make_model_dir: bool) -> Self {
            Self::start_with_size(model, make_model_dir, 9)
        }

        fn start_with_size(model: Option<&str>, make_model_dir: bool, board_size: usize) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "rgo-cli-test-{}-{}",
                std::process::id(),
                NEXT_DIR.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&dir).unwrap();
            if make_model_dir {
                fs::create_dir(dir.join("models")).unwrap();
            }
            if let Some(model) = model {
                let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../engine/tests/data")
                    .join(model);
                fs::copy(fixture, dir.join("models/0.onnx")).unwrap();
            }
            let config = include_str!("../../../configs/self_play.toml")
                .replace("worker_threads = 2", "worker_threads = 1")
                .replace("workers_per_thread = 4", "workers_per_thread = 1")
                .replace("self_play_chunks", "output/chunks");
            fs::write(
                dir.join("config.toml"),
                format!("{config}\n[self_play]\nsearch_budget_policy = [{{ probability = 1.0, budget = {{ max_nodes = 0, max_playouts = 0 }} }}]\n[self_play.rules]\nboard_size = {board_size}\n"),
            )
            .unwrap();
            let log = fs::File::create(dir.join("stderr.log")).unwrap();
            let output = fs::File::create(dir.join("stdout.log")).unwrap();
            let child = Command::new(env!("CARGO_BIN_EXE_rgo-selfplay"))
                .arg("config.toml")
                .current_dir(&dir)
                .stdin(Stdio::piped())
                .stdout(output)
                .stderr(log)
                .spawn()
                .unwrap();
            Self { dir, child }
        }

        fn send(&mut self, command: &str) {
            writeln!(self.child.stdin.as_mut().unwrap(), "{command}").unwrap();
        }

        fn events(&self) -> Vec<serde_json::Value> {
            fs::read_to_string(self.dir.join("stdout.log"))
                .unwrap()
                .split_inclusive('\n')
                .filter(|line| line.ends_with('\n'))
                .map(|line| {
                    serde_json::from_str(line).expect("stdout must contain only JSON events")
                })
                .collect()
        }

        fn ready(&mut self) {
            self.until(|run| run.events().iter().any(|event| event["type"] == "ready"));
            assert_eq!(self.events()[0]["protocol_version"], 1);
        }

        fn log(&self) -> String {
            fs::read_to_string(self.dir.join("stderr.log")).unwrap()
        }

        fn until(&mut self, ready: impl Fn(&Self) -> bool) {
            let deadline = Instant::now() + Duration::from_secs(20);
            while !ready(self) {
                assert!(self.child.try_wait().unwrap().is_none(), "{}", self.log());
                assert!(Instant::now() < deadline, "timed out: {}", self.log());
                thread::sleep(Duration::from_millis(10));
            }
        }

        fn signal(&self, signal: &str) {
            assert!(
                Command::new("kill")
                    .args([signal, &self.child.id().to_string()])
                    .status()
                    .unwrap()
                    .success()
            );
        }

        fn finish(&mut self) -> ExitStatus {
            let deadline = Instant::now() + Duration::from_secs(20);
            loop {
                if let Some(status) = self.child.try_wait().unwrap() {
                    return status;
                }
                assert!(Instant::now() < deadline, "timed out: {}", self.log());
                thread::sleep(Duration::from_millis(10));
            }
        }

        fn chunks(&self) -> Vec<PathBuf> {
            fs::read_dir(self.dir.join("output/chunks"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| path.extension().is_some_and(|extension| extension == "rgo"))
                .collect()
        }
    }

    impl Drop for TestProcess {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn startup_errors_exit_with_a_diagnostic() {
        let mut missing = TestProcess::start(None, false);
        assert_eq!(missing.finish().code(), Some(1));
        assert!(missing.log().contains("could not open model directory"));
        assert!(!missing.dir.join("output").exists());
    }

    #[test]
    fn worker_waits_for_commands_even_with_an_existing_model() {
        let mut run = TestProcess::worker(Some("v0.onnx"));
        run.ready();
        run.send(r#"{"type":"finish"}"#);
        // The parent keeps stdin open: a blocked reader must not delay exit.
        assert!(run.finish().success(), "{}", run.log());
        assert!(run.chunks().is_empty());
        let events = run.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1]["type"], "stopped");
    }

    #[test]
    fn model_commands_publish_compact_chunks_and_drain() {
        let mut run = TestProcess::start_with_size(None, true, 9);
        run.ready();
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine/tests/data/v0.onnx");
        fs::copy(fixture, run.dir.join("models/42.onnx")).unwrap();
        run.send(r#"{"type":"model_ready","version":42}"#);
        run.until(|run| {
            run.events()
                .iter()
                .any(|event| event["type"] == "chunk_ready")
        });
        let first = run
            .events()
            .into_iter()
            .find(|event| event["type"] == "chunk_ready")
            .unwrap();
        assert!(run.dir.join(first["path"].as_str().unwrap()).is_file());
        // An installed model stays alive in memory. Duplicate/stale messages
        // must not reload it or require older model files to remain on disk.
        fs::remove_file(run.dir.join("models/42.onnx")).unwrap();
        run.send(r#"{"type":"model_ready","version":42}"#);
        run.send(r#"{"type":"model_ready","version":0}"#);
        run.send(r#"{"type":"finish"}"#);
        assert!(run.finish().success(), "{}", run.log());
        let events = run.events();
        assert_eq!(events.last().unwrap()["type"], "stopped");
        let chunks: Vec<_> = events
            .iter()
            .filter(|event| event["type"] == "chunk_ready")
            .collect();
        assert!(!chunks.is_empty());
        assert_eq!(chunks.len(), run.chunks().len());
        for event in chunks {
            let path = run.dir.join(event["path"].as_str().unwrap());
            let bytes = fs::read(path).unwrap();
            assert_eq!(&bytes[..8], b"RGOCHNK\0");
            assert_eq!(event["bytes"].as_u64().unwrap(), bytes.len() as u64);
            assert_eq!(
                event["records"].as_u64().unwrap(),
                u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as u64,
            );
            let checksum_offset = bytes.len() - 32;
            let records = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
            assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 3);
            assert_eq!(u32::from_le_bytes(bytes[20..24].try_into().unwrap()), 2);
            let mut offset = 24;
            for _ in 0..records {
                assert_eq!(
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()),
                    9
                );
                offset += 4 + (3 * 81 + 2 + 82 + 3) * 4 + 81;
            }
            assert_eq!(offset, checksum_offset);
            assert_eq!(
                Sha256::digest(&bytes[..checksum_offset]).as_slice(),
                &bytes[checksum_offset..]
            );
        }
        assert!(!run.dir.join("output/chunks/.pending-chunk.tmp").exists());
    }

    #[test]
    fn disconnected_client_finishes_active_games() {
        let mut run = TestProcess::worker(Some("v0.onnx"));
        run.ready();
        run.send(r#"{"type":"model_ready","version":0}"#);
        run.until(|run| {
            run.events()
                .iter()
                .any(|event| event["type"] == "chunk_ready")
        });
        drop(run.child.stdin.take());
        assert!(run.finish().success(), "{}", run.log());
        assert_eq!(run.events().last().unwrap()["type"], "stopped");
    }

    #[test]
    fn bad_commands_and_unavailable_models_report_errors() {
        for (command, diagnostic) in [
            ("not json".into(), "invalid control command"),
            (
                r#"{"type":"finish","unexpected":true}"#.into(),
                "invalid control command",
            ),
            (
                r#"{"type":"model_ready","version":0,"path":"wrong"}"#.into(),
                "invalid control command",
            ),
            (
                r#"{"type":"model_ready","version":0}"#.into(),
                "announced model models/0.onnx is unavailable",
            ),
            ("x".repeat(4097), "control command exceeds 4096 bytes"),
        ] {
            let mut run = TestProcess::worker(None);
            run.ready();
            run.send(&command);
            assert_eq!(run.finish().code(), Some(1), "{}", run.log());
            let events = run.events();
            let last = events.last().unwrap();
            assert_eq!(last["type"], "error");
            assert!(
                last["message"].as_str().unwrap().contains(diagnostic),
                "{last}"
            );
            assert!(run.chunks().is_empty());
        }

        let mut invalid = TestProcess::worker(Some("wrong_version.onnx"));
        invalid.ready();
        invalid.send(r#"{"type":"model_ready","version":0}"#);
        assert_eq!(invalid.finish().code(), Some(1));
        assert!(
            invalid.events().last().unwrap()["message"]
                .as_str()
                .unwrap()
                .contains("ModelLoad")
        );
    }

    #[test]
    fn signals_finish_with_stdin_still_open() {
        for (model, signal) in [(None, "-INT"), (None, "-TERM"), (Some("v0.onnx"), "-INT")] {
            let mut run = TestProcess::worker(model);
            run.ready();
            if model.is_some() {
                run.send(r#"{"type":"model_ready","version":0}"#);
                run.until(|run| {
                    run.events()
                        .iter()
                        .any(|event| event["type"] == "chunk_ready")
                });
            }
            run.signal(signal);
            assert!(run.finish().success(), "{}", run.log());
            assert_eq!(run.events().last().unwrap()["type"], "stopped");
            assert!(!run.dir.join("output/chunks/.pending-chunk.tmp").exists());
        }
    }
}
