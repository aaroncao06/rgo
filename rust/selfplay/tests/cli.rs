use std::process::Command;

#[test]
fn help_and_usage_do_not_start_self_play() {
    let help = Command::new(env!("CARGO_BIN_EXE_rgo-selfplay"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("<config.toml>"));
    for args in [vec![], vec!["one.toml", "two.toml"]] {
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
        fn start(model: Option<&str>, make_model_dir: bool) -> Self {
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
                format!("{config}\n[self_play]\nsearch_budget_policy = [{{ probability = 1.0, budget = {{ max_nodes = 0, max_playouts = 0 }} }}]\n"),
            )
            .unwrap();
            let log = fs::File::create(dir.join("stderr.log")).unwrap();
            let child = Command::new(env!("CARGO_BIN_EXE_rgo-selfplay"))
                .arg("config.toml")
                .current_dir(&dir)
                .stdout(Stdio::null())
                .stderr(log)
                .spawn()
                .unwrap();
            Self { dir, child }
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
    fn signals_exit_cleanly_while_waiting_for_the_first_model() {
        for signal in ["-INT", "-TERM"] {
            let mut run = TestProcess::start(None, true);
            run.until(|run| run.log().contains("Starting 1 self-play workers"));
            assert!(run.dir.join("output/chunks").is_dir());
            run.signal(signal);
            assert!(run.finish().success(), "{}", run.log());
            assert!(run.log().contains("training chunks drained"));
            assert!(run.chunks().is_empty());
        }
    }

    #[test]
    fn startup_errors_exit_with_a_diagnostic() {
        let mut missing = TestProcess::start(None, false);
        assert_eq!(missing.finish().code(), Some(1));
        assert!(missing.log().contains("could not open model directory"));
        assert!(!missing.dir.join("output").exists());

        let mut invalid = TestProcess::start(Some("wrong_version.onnx"), true);
        assert_eq!(invalid.finish().code(), Some(1));
        assert!(invalid.log().contains("ModelLoad"), "{}", invalid.log());
    }

    #[test]
    fn real_model_produces_chunks_and_drains_on_interrupt() {
        let mut run = TestProcess::start(Some("v0.onnx"), true);
        run.until(|run| run.log().contains("Starting 1 self-play workers"));
        run.until(|run| !run.chunks().is_empty());
        run.signal("-INT");
        assert!(run.finish().success(), "{}", run.log());
        assert!(run.log().contains("training chunks drained"));
        assert!(!run.dir.join("output/chunks/.pending-chunk.tmp").exists());
        for path in run.chunks() {
            let bytes = fs::read(path).unwrap();
            assert!(bytes.len() > 68);
            assert_eq!(&bytes[..8], b"RGOCHNK\0");
            let checksum_offset = bytes.len() - 32;
            assert_eq!(
                Sha256::digest(&bytes[..checksum_offset]).as_slice(),
                &bytes[checksum_offset..]
            );
        }
    }
}
