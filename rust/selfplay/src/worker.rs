use rand::{SeedableRng, rngs::SmallRng};
use tokio::sync::{mpsc, watch};

use super::{
    RNG_SEED,
    params::SelfPlayParams,
    training_data::{CompletedGame, SelfPlayRecord},
};
use crate::{
    game::game_state::GameState,
    inference::runtime::{InferenceClient, ModelHandle},
    search::{
        node_store::FixedArenaNodeStore,
        params::SearchParams,
        worker::{SearchBudget, SearchError, SearchResult, SearchWorker},
    },
};

const GAMEPLAY_RNG_STREAM: u64 = 0x243f_6a88_85a3_08d3;
const SYMMETRY_RNG_STREAM: u64 = 0x1319_8a2e_0370_7344;

fn derive_rng_seed(global_seed: u64, worker_index: u64, stream: u64) -> u64 {
    // SplitMix64 gives each purpose and worker a deterministic, independent
    // stream while leaving one global seed as the only user-facing setting.
    let mut value = global_seed ^ worker_index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ stream;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Debug)]
pub(super) enum SelfPlayError {
    Search(SearchError),
    ChunkWriterClosed,
    ModelCoordinatorClosed,
}

impl From<SearchError> for SelfPlayError {
    fn from(error: SearchError) -> Self {
        Self::Search(error)
    }
}

// orchestrator says pause > workers say paused > orchestrator constructs new runtime and sends new handle
pub(super) struct WorkerModelControl {
    pub(super) worker_index: usize,
    pub(super) pause_rx: watch::Receiver<bool>,
    pub(super) paused_tx: mpsc::Sender<usize>,
    pub(super) resume_rx: mpsc::Receiver<ModelHandle>,
}

pub(super) struct SelfPlayWorker {
    game_state: GameState,
    completed_games_tx: mpsc::Sender<CompletedGame>,
    search_worker: SearchWorker<FixedArenaNodeStore>,
    inference_client: InferenceClient,
    model_control: WorkerModelControl,
    rng: SmallRng,
    params: SelfPlayParams,
}

impl SelfPlayWorker {
    pub(super) fn new(
        search_params: SearchParams,
        self_play_params: SelfPlayParams,
        completed_games_tx: mpsc::Sender<CompletedGame>,
        model_control: WorkerModelControl,
    ) -> Self {
        debug_assert!(
            *model_control.pause_rx.borrow(),
            "new workers must start paused until their first model arrives"
        );
        let worker_index = model_control.worker_index as u64;
        let gameplay_seed = derive_rng_seed(RNG_SEED, worker_index, GAMEPLAY_RNG_STREAM);
        let symmetry_seed = derive_rng_seed(RNG_SEED, worker_index, SYMMETRY_RNG_STREAM);
        let inference_client = InferenceClient::unbound(
            self_play_params
                .randomize_inference_symmetry
                .then_some(symmetry_seed),
        );

        let node_store = FixedArenaNodeStore::new(0);
        let game_state = GameState::new(self_play_params.rules);
        Self {
            game_state,
            completed_games_tx,
            search_worker: SearchWorker::new(node_store, search_params),
            inference_client,
            model_control,
            rng: SmallRng::seed_from_u64(gameplay_seed),
            params: self_play_params,
        }
    }

    async fn play_move(
        &mut self,
        search_budget: SearchBudget,
        records: &mut Vec<SelfPlayRecord>,
    ) -> Result<(), SelfPlayError> {
        let search_result = self
            .search_worker
            .search(
                &self.game_state,
                search_budget,
                &mut self.inference_client,
                &mut self.rng,
            )
            .await?;

        let SearchResult {
            selected_move,
            policy_target,
        } = search_result;
        records.push(SelfPlayRecord {
            player: self.game_state.next_player(),
            selected_move,
            policy_target,
        });

        let played = self.game_state.play(selected_move);
        debug_assert!(played, "search selected an illegal move");

        Ok(())
    }

    async fn sync_model(&mut self) -> Result<(), SelfPlayError> {
        if !*self.model_control.pause_rx.borrow() {
            return Ok(());
        }
        // This is called only between searches, so no inference is outstanding.
        if self.inference_client.has_model() {
            self.inference_client.release_model();
        }
        self.model_control
            .paused_tx
            .send(self.model_control.worker_index)
            .await
            .map_err(|_| SelfPlayError::ModelCoordinatorClosed)?;
        let model = self
            .model_control
            .resume_rx
            .recv()
            .await
            .ok_or(SelfPlayError::ModelCoordinatorClosed)?;
        self.inference_client.install_model(model);
        Ok(())
    }

    pub(super) async fn play_game(&mut self) -> Result<(), SelfPlayError> {
        self.game_state.reset(self.params.rules);
        // Provisional reservation; calibrate from observed game lengths later.
        let board_dim = self.game_state.board().dim();
        let mut records = Vec::with_capacity(board_dim * board_dim);
        while !self.game_state.is_finished() {
            self.sync_model().await?;
            let search_budget = self.params.search_budget_policy.sample(&mut self.rng);
            self.play_move(search_budget, &mut records).await?;
        }
        self.completed_games_tx
            .send(CompletedGame::new(&self.game_state, records))
            .await
            .map_err(|_| SelfPlayError::ChunkWriterClosed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_seed_derives_stable_distinct_worker_streams() {
        let global_seed = 42;

        let gameplay = derive_rng_seed(global_seed, 0, GAMEPLAY_RNG_STREAM);
        assert_eq!(
            gameplay,
            derive_rng_seed(global_seed, 0, GAMEPLAY_RNG_STREAM)
        );
        assert_ne!(
            gameplay,
            derive_rng_seed(global_seed, 0, SYMMETRY_RNG_STREAM)
        );
        assert_ne!(
            gameplay,
            derive_rng_seed(global_seed, 1, GAMEPLAY_RNG_STREAM)
        );
    }
}
