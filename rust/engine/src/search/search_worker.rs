use std::{ptr::NonNull, sync::Arc};

use crate::{
    game::{
        board::{Loc, Player},
        game_state::GameState,
    },
    inference::{
        backend::InferenceError,
        outputs::NNOutput,
        policy::{POLICY_SIZE, policy_to_loc},
        runtime::InferenceClient,
    },
    search::{
        graph_key::GraphKey,
        node::{EdgeIndex, SearchNode},
        node_store::{NodeStore, StoreFull},
        search_params::SearchParams,
        utility::{recent_score_center, white_utility},
    },
};

struct SearchRoot {
    game_state: GameState, // playouts will start from this
    key: GraphKey,
    node: Box<SearchNode>, // owned separately from the rest of the nodestore
}

//search state tracks traversal through mcts tree. reset in between moves, not playouts. eventually will support pruning
//owned by search worker
pub(crate) struct SearchGraph<N: NodeStore> {
    root: Option<SearchRoot>,
    node_store: N,
}

impl<N: NodeStore> SearchGraph<N> {
    pub(crate) fn new(node_store: N) -> Self {
        Self {
            root: None,
            node_store,
        }
    }
    pub(crate) fn reset(
        &mut self,
        root_game_state: GameState,
        root_output: Arc<NNOutput>,
        root_utility: f64,
    ) {
        self.node_store.clear();
        let key = GraphKey::new(&root_game_state);
        let node = Self::initialized_root(root_output, root_utility);
        self.root = Some(SearchRoot {
            game_state: root_game_state,
            key,
            node,
        });
    }
    fn initialized_root(output: Arc<NNOutput>, utility: f64) -> Box<SearchNode> {
        let mut root = Box::new(SearchNode::new());
        root.attach_nn_output(output.clone());
        root.record_visit(
            f64::from(output.white_win_prob()),
            f64::from(output.white_score_mean()),
            f64::from(output.white_score_mean_sq()),
            utility,
        );
        root
    }
    pub(crate) fn advance_root(&mut self, move_loc: Loc) {
        // later for normal play
    }
}

struct PlayoutStep {
    parent: NonNull<SearchNode>,
    edge_index: EdgeIndex,
}

struct PlayoutPath {
    steps: Vec<PlayoutStep>,
}
impl PlayoutPath {
    fn new() -> Self {
        Self { steps: Vec::new() }
    }
    fn clear(&mut self) {
        self.steps.clear();
    }
    fn push(&mut self, parent: NonNull<SearchNode>, edge_index: EdgeIndex) {
        self.steps.push(PlayoutStep { parent, edge_index });
    }
}

struct SearchWorker<N: NodeStore> {
    // keeps mutating its search state, one worker per game, eventually supports subgraph reuse
    search_graph: SearchGraph<N>,
    playout_path: PlayoutPath, // scratch work to avoid reallocating
    scratch_game_state: Option<GameState>, // mutates through each playout
    params: SearchParams,
    recent_score_center: f64,
}

#[derive(Debug)]
pub(crate) enum SearchError {
    StoreFull,
    InferenceError(InferenceError),
}

impl From<InferenceError> for SearchError {
    fn from(error: InferenceError) -> Self {
        Self::InferenceError(error)
    }
}

impl From<StoreFull> for SearchError {
    fn from(_: StoreFull) -> Self {
        Self::StoreFull
    }
}

impl<N: NodeStore> SearchWorker<N> {
    fn new(node_store: N, params: SearchParams) -> Self {
        Self {
            search_graph: SearchGraph::new(node_store),
            playout_path: PlayoutPath::new(),
            scratch_game_state: None,
            params,
            recent_score_center: 0.0,
        }
    }
    async fn start_game(
        &mut self,
        root_game_state: GameState,
        inference_client: &mut InferenceClient,
    ) -> Result<(), SearchError> {
        // just so that cross game graph reuse is not allowed
        self.reset_graph(root_game_state, inference_client).await
    }
    async fn reset_graph(
        &mut self,
        root_game_state: GameState,
        inference_client: &mut InferenceClient,
    ) -> Result<(), SearchError> {
        // full reset, in the future can have a version where you retain subgraph between moves
        debug_assert!(!root_game_state.is_finished());
        let root_output = inference_client.evaluate(&root_game_state).await?;
        let score_center =
            recent_score_center(f64::from(root_output.white_score_mean()), self.params);
        let root_utility = white_utility(
            f64::from(root_output.white_win_prob()),
            f64::from(root_output.white_score_mean()),
            f64::from(root_output.white_score_mean_sq()),
            score_center,
            self.params,
        );

        match self.scratch_game_state.as_mut() {
            Some(scratch) => scratch.reset_from(&root_game_state),
            None => self.scratch_game_state = Some(root_game_state.clone()),
        }
        self.search_graph
            .reset(root_game_state, root_output, root_utility);
        self.recent_score_center = score_center;
        self.playout_path.clear();
        Ok(())
    }
    async fn playout(&mut self, inference_client: &mut InferenceClient) -> Result<(), SearchError> {
        self.playout_path.clear();
        let root = self
            .search_graph
            .root
            .as_mut()
            .expect("playout requires an active game");
        let scratch_game_state = self
            .scratch_game_state
            .as_mut()
            .expect("active game requires scratch state");
        scratch_game_state.reset_from(&root.game_state);
        let mut current_node = NonNull::from(root.node.as_mut());

        loop {
            if self
                .scratch_game_state
                .as_ref()
                .expect("active game requires scratch state")
                .is_finished()
            {
                break;
            }
            let node = unsafe { current_node.as_ref() };

            // choose move, check if move is in children. if so go down, if not break
        }

        Ok(())
    }
    fn choose_move(&self, node: &SearchNode, game_state: &GameState, is_root: bool) -> Loc {
        // KataGo increases exploration slowly as the node accumulates visits.
        let total_child_visits = f64::from(node.edge_visit_sum());
        let cpuct = self.params.cpuct_exploration
            + self.params.cpuct_exploration_log
                * ((total_child_visits + self.params.cpuct_exploration_base)
                    / self.params.cpuct_exploration_base)
                    .ln();
        let exploration_scaling = cpuct * (total_child_visits + 0.01).sqrt();

        // Estimate unvisited children using first-play urgency (FPU). As more
        // policy mass is visited, trust backed-up utility more than the direct
        // neural-network evaluation of this node.
        let visited_policy_mass = node.visited_policy_mass().min(1.0);
        let direct_output = node.nn_output();
        let direct_utility = white_utility(
            f64::from(direct_output.white_win_prob()),
            f64::from(direct_output.white_score_mean()),
            f64::from(direct_output.white_score_mean_sq()),
            self.recent_score_center,
            self.params,
        );
        let backed_up_weight = visited_policy_mass
            .powf(self.params.fpu_parent_weight_by_visited_policy_pow)
            .min(1.0);
        let parent_utility_for_fpu =
            backed_up_weight * node.white_utility() + (1.0 - backed_up_weight) * direct_utility;
        let fpu_reduction_max = if is_root {
            self.params.root_fpu_reduction_max
        } else {
            self.params.fpu_reduction_max
        };
        let fpu_reduction = fpu_reduction_max * visited_policy_mass.sqrt();
        let fpu = match game_state.next_player() {
            Player::White => parent_utility_for_fpu - fpu_reduction,
            Player::Black => parent_utility_for_fpu + fpu_reduction,
        };

        // Score every legal move. Existing edges use their child's backed-up
        // utility and edge visits; unexpanded moves use FPU and zero visits.
        let mut best_move = Loc::NULL;
        let mut best_selection_value = f64::NEG_INFINITY;
        for policy_index in 0..POLICY_SIZE {
            let move_loc = policy_to_loc(policy_index);
            if !game_state.is_legal(move_loc) {
                continue;
            }

            let policy_probability = f64::from(node.policy_probs()[policy_index]);
            let (child_utility, edge_visits) = match node.edge_for_move(move_loc) {
                Some((_, edge)) => {
                    let child = unsafe { edge.child().as_ref() };
                    (child.white_utility(), f64::from(edge.visits()))
                }
                None => (fpu, 0.0),
            };
            let selection_value = selection_value(
                child_utility,
                game_state.next_player(),
                policy_probability,
                edge_visits,
                exploration_scaling,
            );

            if selection_value > best_selection_value {
                best_selection_value = selection_value;
                best_move = move_loc;
            }
        }

        debug_assert!(best_move != Loc::NULL, "pass must always be legal");
        best_move
    }
}

fn selection_value(
    white_utility: f64,
    player: Player,
    policy_probability: f64,
    edge_visits: f64,
    exploration_scaling: f64,
) -> f64 {
    let value = match player {
        Player::White => white_utility,
        Player::Black => -white_utility,
    };
    let exploration = exploration_scaling * policy_probability / (1.0 + edge_visits);
    value + exploration
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        game::rules::Rules,
        inference::{
            backend::InferenceBackend, inputs::NNInput, policy::POLICY_SIZE, runtime::ModelRuntime,
        },
        search::node_store::FixedArenaNodeStore,
    };

    struct TestBackend {
        fail: bool,
    }

    impl InferenceBackend for TestBackend {
        fn evaluate_batch(
            &mut self,
            inputs: &[NNInput],
            outputs: &mut Vec<Arc<NNOutput>>,
        ) -> Result<(), InferenceError> {
            if self.fail {
                return Err(InferenceError::ExecutionFailed);
            }
            for _ in inputs {
                outputs.push(Arc::new(NNOutput::from_raw(
                    [0.0; POLICY_SIZE],
                    0.0,
                    0.0,
                    0.0,
                )));
            }
            Ok(())
        }
    }

    fn inference_client(fail: bool) -> InferenceClient {
        let model_handle = ModelRuntime::start(vec![TestBackend { fail }], 1, 1, 16, 1);
        InferenceClient::new(model_handle)
    }

    fn worker() -> SearchWorker<FixedArenaNodeStore> {
        SearchWorker::new(
            FixedArenaNodeStore::new(16),
            SearchParams::KATAGO_SELFPLAY8_MAIN_B18,
        )
    }

    fn processed_output(policy_logits: [f32; POLICY_SIZE], white_win_logit: f32) -> Arc<NNOutput> {
        let mut output = Arc::new(NNOutput::from_raw(
            policy_logits,
            white_win_logit,
            0.0,
            -20.0,
        ));
        Arc::get_mut(&mut output)
            .unwrap()
            .process_in_place(Player::White, &[true; POLICY_SIZE]);
        output
    }

    fn initialized_node(output: Arc<NNOutput>, utility: f64) -> Box<SearchNode> {
        let mut node = Box::new(SearchNode::new());
        node.attach_nn_output(output.clone());
        node.record_visit(
            f64::from(output.white_win_prob()),
            f64::from(output.white_score_mean()),
            f64::from(output.white_score_mean_sq()),
            utility,
        );
        node
    }

    #[test]
    fn new_worker_has_no_active_graph_or_scratch_state() {
        let worker = worker();

        assert!(worker.search_graph.root.is_none());
        assert_eq!(worker.search_graph.node_store.len(), 0);
        assert!(worker.scratch_game_state.is_none());
        assert!(worker.playout_path.steps.is_empty());
    }

    #[test]
    fn selection_chooses_the_highest_policy_unexpanded_move() {
        let worker = worker();
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let expected_move = Loc::new(4, 4).unwrap();
        let mut logits = [0.0; POLICY_SIZE];
        logits[crate::inference::policy::loc_to_policy(expected_move)] = 5.0;
        let output = processed_output(logits, 0.0);
        let node = initialized_node(output, 0.0);

        assert_eq!(worker.choose_move(&node, &game_state, true), expected_move);
    }

    #[test]
    fn selection_orients_child_utility_for_the_player_to_move() {
        let worker = worker();
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        assert_eq!(game_state.next_player(), Player::Black);
        let parent_output = processed_output([0.0; POLICY_SIZE], 0.0);
        let mut parent = initialized_node(parent_output, 0.0);

        let white_favored_move = Loc::new(3, 3).unwrap();
        let black_favored_move = Loc::new(4, 4).unwrap();
        let mut white_favored_child =
            initialized_node(processed_output([0.0; POLICY_SIZE], 9.0_f32.ln()), 0.8);
        let mut black_favored_child = initialized_node(
            processed_output([0.0; POLICY_SIZE], (1.0_f32 / 9.0).ln()),
            -0.8,
        );
        parent.add_child(
            white_favored_move,
            parent.policy_probs()[crate::inference::policy::loc_to_policy(white_favored_move)],
            NonNull::from(white_favored_child.as_mut()),
        );
        parent.add_child(
            black_favored_move,
            parent.policy_probs()[crate::inference::policy::loc_to_policy(black_favored_move)],
            NonNull::from(black_favored_child.as_mut()),
        );

        assert_eq!(
            worker.choose_move(&parent, &game_state, true),
            black_favored_move
        );
    }

    #[tokio::test]
    async fn start_game_installs_an_evaluated_root_and_scratch_state() {
        let mut worker = worker();
        let mut client = inference_client(false);
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let expected_key = GraphKey::new(&game_state);

        worker.start_game(game_state, &mut client).await.unwrap();

        let root = worker.search_graph.root.as_ref().unwrap();
        assert_eq!(root.key, expected_key);
        assert_eq!(root.node.visits(), 1);
        assert_eq!(
            GraphKey::new(worker.scratch_game_state.as_ref().unwrap()),
            expected_key
        );
    }

    #[tokio::test]
    async fn resetting_graph_clears_stored_nodes_and_replaces_the_root() {
        let mut worker = worker();
        let mut client = inference_client(false);
        worker
            .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut client)
            .await
            .unwrap();
        worker
            .search_graph
            .node_store
            .find_or_insert(GraphKey::from_raw(1))
            .unwrap();

        let mut next_state = GameState::new(Rules::TROMP_TAYLORISH);
        assert!(next_state.play(Loc::new(4, 4).unwrap()));
        let expected_key = GraphKey::new(&next_state);
        worker.reset_graph(next_state, &mut client).await.unwrap();

        assert_eq!(worker.search_graph.node_store.len(), 0);
        assert_eq!(worker.search_graph.root.as_ref().unwrap().key, expected_key);
    }

    #[tokio::test]
    async fn inference_failure_preserves_the_existing_graph() {
        let mut worker = worker();
        let mut working_client = inference_client(false);
        worker
            .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut working_client)
            .await
            .unwrap();
        worker
            .search_graph
            .node_store
            .find_or_insert(GraphKey::from_raw(1))
            .unwrap();
        let original_key = worker.search_graph.root.as_ref().unwrap().key;

        let mut failing_client = inference_client(true);
        let mut replacement = GameState::new(Rules::TROMP_TAYLORISH);
        assert!(replacement.play(Loc::new(4, 4).unwrap()));
        let result = worker.reset_graph(replacement, &mut failing_client).await;

        assert!(matches!(result, Err(SearchError::InferenceError(_))));
        assert_eq!(worker.search_graph.root.as_ref().unwrap().key, original_key);
        assert_eq!(worker.search_graph.node_store.len(), 1);
    }
}
