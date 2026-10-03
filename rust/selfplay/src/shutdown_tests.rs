use super::*;
use crate::inference::{
    backend::{InferenceBackend, InferenceError, InputBatch},
    outputs::NNOutput,
    runtime::start_test_runtime,
};
use std::{
    path::PathBuf,
    process::{Child, Command, ExitStatus},
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};

struct StalledBackend;

impl InferenceBackend for StalledBackend {
    fn evaluate_batch(
        &mut self,
        _: &dyn InputBatch,
        _: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        unreachable!("the test backend never completes startup")
    }
}

#[test]
#[ignore = "subprocess fixture invoked by blocked_startup_still_handles_shutdown_signals"]
fn blocked_startup_child() {
    let (_release_startup_tx, release_startup_rx) = mpsc::channel::<()>();
    build_runtime()
        .unwrap()
        .block_on(with_shutdown(|_finish_tx, mut finish_rx| async move {
            tokio::spawn(async move {
                finish_rx.changed().await.unwrap();
                assert!(*finish_rx.borrow());
                eprintln!("Finish request observed during blocked startup");
            });
            // This uses the engine's real synchronous executor-startup wait:
            // the calling thread blocks while the backend factory stalls.
            let _model = start_test_runtime(
                0,
                vec![(
                    move || {
                        eprintln!("Executor startup blocked");
                        release_startup_rx.recv().unwrap();
                        Err::<StalledBackend, _>("unexpected test backend release")
                    },
                    1,
                )],
                8,
                1,
            )
            .unwrap();
            Ok(())
        }))
        .unwrap();
}

struct ShutdownChild {
    child: Child,
    log: PathBuf,
}

impl ShutdownChild {
    fn output(&self) -> String {
        fs::read_to_string(&self.log).unwrap()
    }

    fn wait_for(&mut self, message: &str) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !self.output().contains(message) {
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "{}",
                self.output()
            );
            assert!(Instant::now() < deadline, "timed out: {}", self.output());
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

    fn wait_for_exit(&mut self) -> ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "timed out: {}", self.output());
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for ShutdownChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_file(&self.log);
    }
}

#[test]
fn blocked_startup_still_handles_shutdown_signals() {
    for signal in ["-INT", "-TERM"] {
        let log = env::temp_dir().join(format!(
            "rgo-blocked-startup-{}-{signal}.log",
            process::id()
        ));
        let child = Command::new(env::current_exe().unwrap())
            .args([
                "--exact",
                "shutdown_tests::blocked_startup_child",
                "--ignored",
                "--nocapture",
            ])
            .stderr(fs::File::create(&log).unwrap())
            .spawn()
            .unwrap();
        let mut child = ShutdownChild { child, log };
        child.wait_for("Executor startup blocked");
        child.signal(signal);
        child.wait_for("Finish request observed during blocked startup");
        // The graceful request must not cancel the in-progress startup wait.
        assert!(child.child.try_wait().unwrap().is_none());
        child.signal(signal);
        assert_eq!(child.wait_for_exit().code(), Some(1));
        assert!(child.output().contains("second shutdown signal received"));
    }
}
