use rand::{SeedableRng, rngs::SmallRng};
use std::sync::atomic::{AtomicU64, Ordering};

use super::{RNG_SEED, params::SelfPlayParams};
use crate::{
    game::{
        board::{Loc, Player},
        game_state::GameState,
    },
    inference::{
        inputs::NNInput,
        policy::{BOARD_POLICY_SIZE, POLICY_SIZE},
        runtime::{InferenceClient, ModelVersion},
    },
    search::{
        node_store::FixedArenaNodeStore,
        params::SearchParams,
        worker::{SearchBudget, SearchError, SearchResult, SearchValueTarget, SearchWorker},
    },
};

const GAMEPLAY_RNG_STREAM: u64 = 0x243f_6a88_85a3_08d3;
const SYMMETRY_RNG_STREAM: u64 = 0x1319_8a2e_0370_7344;
static NEXT_WORKER_ID: AtomicU64 = AtomicU64::new(0);

fn derive_rng_seed(global_seed: u64, worker_id: u64, stream: u64) -> u64 {
    // SplitMix64 gives each purpose and worker a deterministic, independent
    // stream while leaving one global seed as the only user-facing setting.
    let mut value = global_seed ^ worker_id.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ stream;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct ValueTarget {
    win_probability: f32,
    score_mean: f32,
    score_stdev: f32,
    ownership: [u8; BOARD_POLICY_SIZE],
}

// eventually will make a packed version to minimize network transfer
// and then we will have the record -> sample conversion do the packing
// since we are allocating a new struct anyways
struct TrainingSample {
    input: NNInput,
    policy_target: [f32; POLICY_SIZE],
    value_target: ValueTarget,
}
struct SelfPlayRecord {
    model_version: ModelVersion,
    player: Player,
    selected_move: Loc, // to reconstruct nn input when we replay the game to finalize
    policy_target: [f32; POLICY_SIZE],
    search_value_target: SearchValueTarget,
}

struct SelfPlayWorker {
    game_state: GameState,
    records: Vec<SelfPlayRecord>,
    search_worker: SearchWorker<FixedArenaNodeStore>,
    inference_client: InferenceClient,
    rng: SmallRng,
    params: SelfPlayParams,
}

impl SelfPlayWorker {
    fn new(
        search_params: SearchParams,
        self_play_params: SelfPlayParams,
        mut inference_client: InferenceClient,
    ) -> Self {
        let worker_id = NEXT_WORKER_ID.fetch_add(1, Ordering::Relaxed);
        let gameplay_seed = derive_rng_seed(RNG_SEED, worker_id, GAMEPLAY_RNG_STREAM);
        let symmetry_seed = derive_rng_seed(RNG_SEED, worker_id, SYMMETRY_RNG_STREAM);
        inference_client.reseed_symmetry(symmetry_seed);

        let node_store = FixedArenaNodeStore::new(0);
        let game_state = GameState::new(self_play_params.rules);
        Self {
            game_state,
            records: Vec::new(),
            search_worker: SearchWorker::new(node_store, search_params),
            inference_client,
            rng: SmallRng::seed_from_u64(gameplay_seed),
            params: self_play_params,
        }
    }

    async fn play_move(&mut self, search_budget: SearchBudget) -> Result<(), SearchError> {
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
            value_target: search_value_target,
        } = search_result;
        self.records.push(SelfPlayRecord {
            model_version: self.inference_client.model_version(),
            player: self.game_state.next_player(),
            selected_move,
            policy_target,
            search_value_target,
        });

        let played = self.game_state.play(selected_move);
        assert!(played, "search selected an illegal move");

        Ok(())
    }
    async fn play_game(&mut self) -> Result<(), SearchError> {
        self.game_state.reset(self.params.rules);
        self.records.clear();
        let search_budget = self.params.search_budget_policy.sample(&mut self.rng);
        while !self.game_state.is_finished() {
            self.play_move(search_budget).await?;
        }
        Ok(())
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
