use rand::{SeedableRng, rngs::SmallRng};
use tokio::sync::{mpsc, oneshot, watch};

use super::{
    RNG_SEED,
    chunk_assembler::CompletedGame,
    params::SelfPlayParams,
    training_data::{TrainingSample, ValueTarget},
};
use crate::{
    game::{
        board::{Color, Loc, MAX_BOARD_AREA, Player},
        game_state::GameState,
    },
    inference::{
        inputs::NNInput,
        policy::{MAX_POLICY_SIZE, loc_to_spatial},
        runtime::{InferenceClient, ModelHandle},
    },
    search::{
        node_store::FixedArenaNodeStore,
        params::SearchParams,
        worker::{SearchBudget, SearchError, SearchResult, SearchValueTarget, SearchWorker},
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

struct SelfPlayRecord {
    player: Player,
    selected_move: Loc, // to reconstruct nn input when we replay the game to finalize
    policy_target: [f32; MAX_POLICY_SIZE],
    search_value_target: SearchValueTarget,
}

#[derive(Debug)]
pub(super) enum SelfPlayError {
    Search(SearchError),
    ChunkAssemblerClosed,
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

// training_samples and recycle_rx are option since they are sent to the chunk writer
pub(super) struct SelfPlayWorker {
    game_state: GameState,
    records: Vec<SelfPlayRecord>,
    training_samples: Option<Vec<TrainingSample>>,
    completed_games_tx: mpsc::Sender<CompletedGame>,
    recycle_rx: Option<oneshot::Receiver<Vec<TrainingSample>>>,
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
            records: Vec::new(),
            training_samples: Some(Vec::new()),
            completed_games_tx,
            recycle_rx: None,
            search_worker: SearchWorker::new(node_store, search_params),
            inference_client,
            model_control,
            rng: SmallRng::seed_from_u64(gameplay_seed),
            params: self_play_params,
        }
    }

    async fn play_move(&mut self, search_budget: SearchBudget) -> Result<(), SelfPlayError> {
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
            player: self.game_state.next_player(),
            selected_move,
            policy_target,
            search_value_target,
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
        self.records.clear();
        while !self.game_state.is_finished() {
            self.sync_model().await?;
            let search_budget = self.params.search_budget_policy.sample(&mut self.rng);
            self.play_move(search_budget).await?;
        }
        self.submit_finished_game().await
    }

    async fn submit_finished_game(&mut self) -> Result<(), SelfPlayError> {
        // its none when the chunk writer took it and now you take it back
        let mut samples = match self.training_samples.take() {
            Some(samples) => samples,
            None => self
                .recycle_rx
                .take()
                .expect("submitted buffer has a recycle receiver")
                .await
                .map_err(|_| SelfPlayError::ChunkAssemblerClosed)?,
        };
        build_training_samples(&mut self.records, &mut self.game_state, &mut samples);
        let (recycle_tx, recycle_rx) = oneshot::channel();
        let completed_game = CompletedGame {
            samples,
            recycle_tx,
        };
        if let Err(error) = self.completed_games_tx.send(completed_game).await {
            self.training_samples = Some(error.0.samples);
            return Err(SelfPlayError::ChunkAssemblerClosed);
        }
        self.recycle_rx = Some(recycle_rx);
        Ok(())
    }
}

fn build_training_samples(
    records: &mut Vec<SelfPlayRecord>,
    game_state: &mut GameState,
    samples: &mut Vec<TrainingSample>,
) {
    debug_assert!(
        game_state.is_finished(),
        "cannot finalize an unfinished game"
    );
    let final_ownership = game_state.final_ownership();
    let final_state_hash = game_state.current_state_hash();
    let final_turn_number = game_state.turn_number();
    let rules = *game_state.rules();
    let black_ownership = ownership_for_player(game_state.board(), &final_ownership, Player::Black);
    let white_ownership = ownership_for_player(game_state.board(), &final_ownership, Player::White);

    game_state.reset(rules);
    samples.clear();
    samples.reserve(records.len());
    for record in records.drain(..) {
        debug_assert_eq!(record.player, game_state.next_player());
        let ownership = match record.player {
            Player::Black => black_ownership,
            Player::White => white_ownership,
        };
        let SearchValueTarget {
            win_probability,
            score_mean,
            score_stdev,
        } = record.search_value_target;
        samples.push(TrainingSample {
            input: NNInput::encode(game_state),
            policy_target: record.policy_target,
            value_target: ValueTarget {
                win_probability,
                score_mean,
                score_stdev,
                ownership,
            },
        });
        let played = game_state.play(record.selected_move);
        debug_assert!(played, "recorded self-play move failed during replay");
    }
    debug_assert_eq!(
        game_state.current_state_hash(),
        final_state_hash,
        "replayed game differs from the completed game"
    );
    debug_assert_eq!(
        game_state.turn_number(),
        final_turn_number,
        "replayed game has the wrong length"
    );
}

fn ownership_for_player(
    board: &crate::game::board::Board,
    final_ownership: &[Color; crate::game::board::BOARD_STORAGE_LEN],
    player: Player,
) -> [u8; MAX_BOARD_AREA] {
    let mut ownership = [1; MAX_BOARD_AREA];
    for loc in board.locs() {
        ownership[loc_to_spatial(board, loc)] = match (final_ownership[loc.index()], player) {
            (Color::Empty, _) => 1,
            (Color::Black, Player::Black) | (Color::White, Player::White) => 2,
            (Color::Black, Player::White) | (Color::White, Player::Black) => 0,
            (Color::Wall, _) => unreachable!("board iterator yielded a wall"),
        };
    }
    ownership
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::rules::Rules;
    use crate::inference::policy::loc_to_policy;

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

    #[test]
    fn finished_game_replay_builds_player_relative_training_samples() {
        for board_dim in [5, 9] {
            let rules = Rules {
                board_dim,
                ..Rules::TROMP_TAYLORISH_9
            };
            let mut game_state = GameState::new(rules);
            let center = game_state
                .board()
                .loc(board_dim / 2, board_dim / 2)
                .unwrap();
            let moves = [center, Loc::PASS, Loc::PASS];
            let mut records = Vec::new();

            for (turn, move_loc) in moves.into_iter().enumerate() {
                let mut policy_target = [0.0; MAX_POLICY_SIZE];
                policy_target[loc_to_policy(game_state.board(), move_loc)] = 1.0;
                records.push(SelfPlayRecord {
                    player: game_state.next_player(),
                    selected_move: move_loc,
                    policy_target,
                    search_value_target: SearchValueTarget {
                        win_probability: 0.1 + turn as f32 * 0.1,
                        score_mean: turn as f32 + 0.5,
                        score_stdev: turn as f32 + 1.5,
                    },
                });
                assert!(game_state.play(move_loc));
            }
            assert!(game_state.is_finished());

            let mut samples = Vec::new();
            build_training_samples(&mut records, &mut game_state, &mut samples);

            assert!(records.is_empty());
            assert_eq!(samples.len(), 3);
            assert_eq!(samples[0].input.global, [-7.5, 0.0]);
            assert_eq!(samples[1].input.global, [7.5, 0.0]);
            assert_eq!(samples[2].input.global, [-7.5, 1.0]);

            let center_policy = loc_to_policy(game_state.board(), center);
            let center_spatial = loc_to_spatial(game_state.board(), center);
            assert_eq!(samples[0].input.spatial[center_spatial], 0);
            assert_eq!(samples[1].input.spatial[MAX_BOARD_AREA + center_spatial], 1);
            assert_eq!(samples[2].input.spatial[center_spatial], 1);
            assert_eq!(samples[0].policy_target[center_policy], 1.0);

            assert_eq!(samples[0].value_target.win_probability, 0.1);
            assert_eq!(samples[0].value_target.score_mean, 0.5);
            assert_eq!(samples[0].value_target.score_stdev, 1.5);
            assert_eq!(samples[0].value_target.ownership[center_spatial], 2);
            assert_eq!(samples[1].value_target.ownership[center_spatial], 0);
            for sample in &samples {
                assert_eq!(sample.input.board_dim, board_dim);
                assert!(
                    sample.policy_target[board_dim * board_dim + 1..]
                        .iter()
                        .all(|&p| p == 0.0)
                );
                for i in 0..MAX_BOARD_AREA {
                    let x = i % crate::game::board::MAX_BOARD_DIM;
                    let y = i / crate::game::board::MAX_BOARD_DIM;
                    if x >= board_dim || y >= board_dim {
                        assert_eq!(sample.value_target.ownership[i], 1);
                    }
                }
            }
        }
    }
}
