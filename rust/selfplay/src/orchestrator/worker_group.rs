//! One OS thread and local Tokio task set for a group of search workers.

use std::thread;

use tokio::{
    sync::{mpsc, watch},
    task::{JoinError, JoinSet, LocalSet},
};

use crate::search::params::SearchParams;
use crate::{
    params::SelfPlayParams,
    training_data::CompletedGame,
    worker::{SelfPlayError, SelfPlayWorker, WorkerModelControl},
};

pub(super) struct WorkerSpec {
    pub(super) search_params: SearchParams,
    pub(super) self_play_params: SelfPlayParams,
    pub(super) completed_games_tx: mpsc::Sender<CompletedGame>,
    pub(super) model_control: WorkerModelControl,
}

pub(super) enum WorkerEvent {
    Finished(Result<Result<(), SelfPlayError>, JoinError>),
}

pub(super) struct WorkerGroup {
    member_specs: Vec<WorkerSpec>,
}

impl WorkerGroup {
    pub(super) fn new() -> Self {
        Self {
            member_specs: Vec::new(),
        }
    }

    pub(super) fn push(&mut self, worker: WorkerSpec) {
        self.member_specs.push(worker);
    }

    pub(super) fn spawn(
        self,
        max_games_per_worker: Option<usize>,
        finish_games_rx: watch::Receiver<bool>,
        events_tx: mpsc::UnboundedSender<WorkerEvent>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("worker-group Tokio runtime failed to start");
            let local = LocalSet::new();
            runtime.block_on(local.run_until(async {
                let mut workers = JoinSet::new();
                for spec in self.member_specs {
                    let finish_games_rx = finish_games_rx.clone();
                    let mut worker = SelfPlayWorker::new(
                        spec.search_params,
                        spec.self_play_params,
                        spec.completed_games_tx,
                        spec.model_control,
                    );
                    workers.spawn_local(async move {
                        let mut games_played = 0;
                        while !*finish_games_rx.borrow()
                            && max_games_per_worker.is_none_or(|limit| games_played < limit)
                        {
                            worker.play_game().await?;
                            games_played += 1;
                        }
                        Ok::<(), SelfPlayError>(())
                    });
                }
                drop(finish_games_rx);
                while let Some(result) = workers.join_next().await {
                    // sending events allows notifying the orchestrator about individual worker failures to kill promptly
                    if events_tx.send(WorkerEvent::Finished(result)).is_err() {
                        break;
                    }
                }
            }));
        })
    }
}
