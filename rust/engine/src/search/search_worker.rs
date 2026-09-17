use std::{ptr::NonNull, sync::Arc};

use crate::{
    game::{
        board::{Loc, Player},
        game_state::GameState,
    },
    inference::{
        backend::InferenceError,
        outputs::NNOutput,
        policy::{POLICY_SIZE, loc_to_policy, policy_to_loc},
        runtime::InferenceClient,
    },
    search::{
        graph_key::GraphKey,
        node::{EdgeIndex, SearchNode},
        node_store::{InsertError, NodeStore},
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

struct SearchWorker<N: NodeStore> {
    // keeps mutating its search state, one worker per game, eventually supports subgraph reuse
    search_graph: SearchGraph<N>,
    playout_path: Vec<PlayoutStep>, // scratch work to avoid reallocating
    scratch_game_state: Option<GameState>, // mutates through each playout
    params: SearchParams,
    recent_score_center: f64,
}

#[derive(Debug)]
pub(crate) enum SearchError {
    NodeStore(InsertError),
    InferenceError(InferenceError),
}

impl From<InferenceError> for SearchError {
    fn from(error: InferenceError) -> Self {
        Self::InferenceError(error)
    }
}

impl From<InsertError> for SearchError {
    fn from(error: InsertError) -> Self {
        Self::NodeStore(error)
    }
}

impl<N: NodeStore> SearchWorker<N> {
    fn new(node_store: N, params: SearchParams) -> Self {
        Self {
            search_graph: SearchGraph::new(node_store),
            playout_path: Vec::new(),
            scratch_game_state: None,
            params,
            recent_score_center: 0.0,
        }
    }
    fn scratch_game_state(&self) -> &GameState {
        self.scratch_game_state
            .as_ref()
            .expect("active game requires scratch state")
    }
    fn scratch_game_state_mut(&mut self) -> &mut GameState {
        self.scratch_game_state
            .as_mut()
            .expect("active game requires scratch state")
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
        // mut borrow search root
        let (mut current_node, mut current_key) = {
            let root = self
                .search_graph
                .root
                .as_mut()
                .expect("playout requires an active game");
            self.scratch_game_state
                .as_mut()
                .expect("active game requires scratch state")
                .reset_from(&root.game_state);
            (NonNull::from(root.node.as_mut()), root.key)
        };
        let mut is_root = true;

        let leaf_node = loop {
            // scope for borrowing node: choose move and get relevant info regarding it (policy for it, whether edge already exists)
            let (move_loc, policy_prior, existing_edge) = {
                let node = unsafe { current_node.as_ref() };
                let move_loc = self.choose_move(node, self.scratch_game_state(), is_root);
                let policy_prior = node.policy_probs()[loc_to_policy(move_loc)];
                let existing_edge = node
                    .edge_for_move(move_loc)
                    .map(|(edge_index, edge)| (edge_index, edge.child()));
                (move_loc, policy_prior, existing_edge)
            };

            // update the game state
            let played = self.scratch_game_state_mut().play(move_loc);
            debug_assert!(played);

            // scope for borrowing game state (now the child game state, move already played): update key and check if it is finished
            let is_finished = {
                let scratch_game_state = self.scratch_game_state();
                current_key.advance(scratch_game_state, move_loc);
                scratch_game_state.is_finished()
            };

            match existing_edge {
                Some((edge_index, child)) => {
                    // descend
                    self.playout_path.push(PlayoutStep {
                        parent: current_node,
                        edge_index,
                    });
                    if is_finished {
                        break child;
                    }
                    current_node = child;
                    is_root = false;
                }
                None => {
                    // child already exists in the node store, can just connect a new edge to it for the current parent
                    // can keep descending to find a new node to create if current isnt finished
                    if let Some(child) = self.search_graph.node_store.find(current_key) {
                        let edge_index = unsafe { current_node.as_mut() }.add_child(
                            move_loc,
                            policy_prior,
                            child,
                        );
                        self.playout_path.push(PlayoutStep {
                            parent: current_node,
                            edge_index,
                        });
                        if is_finished {
                            break child;
                        }
                        current_node = child;
                        is_root = false;
                        continue;
                    }
                    //create a new node. inference before creating node so that if inference fails we dont leave an invalid node
                    let output = if is_finished {
                        None
                    } else {
                        Some(inference_client.evaluate(self.scratch_game_state()).await?)
                    };
                    let mut child = self.search_graph.node_store.insert(current_key)?;

                    // populate node
                    if let Some(output) = output {
                        let utility = white_utility(
                            f64::from(output.white_win_prob()),
                            f64::from(output.white_score_mean()),
                            f64::from(output.white_score_mean_sq()),
                            self.recent_score_center,
                            self.params,
                        );
                        let child = unsafe { child.as_mut() };
                        child.attach_nn_output(output.clone());
                        child.record_visit(
                            f64::from(output.white_win_prob()),
                            f64::from(output.white_score_mean()),
                            f64::from(output.white_score_mean_sq()),
                            utility,
                        );
                    } else {
                        // use terminal state stats not model output
                        let white_score =
                            f64::from(self.scratch_game_state().final_score_white_minus_black());
                        let white_win = if white_score > 0.0 {
                            1.0
                        } else if white_score < 0.0 {
                            0.0
                        } else {
                            0.5
                        };
                        let white_utility = white_utility(
                            white_win,
                            white_score,
                            white_score * white_score,
                            self.recent_score_center,
                            self.params,
                        );
                        unsafe { child.as_mut() }.record_visit(
                            white_win,
                            white_score,
                            white_score * white_score,
                            white_utility,
                        );
                    }
                    // connect the child to the parent
                    let edge_index =
                        unsafe { current_node.as_mut() }.add_child(move_loc, policy_prior, child);
                    self.playout_path.push(PlayoutStep {
                        parent: current_node,
                        edge_index,
                    });
                    break child;
                }
            }
        };
        self.backup(leaf_node);

        Ok(())
    }
    fn backup(&mut self, leaf_node: NonNull<SearchNode>) {
        let leaf = unsafe { leaf_node.as_ref() };
        let white_win = leaf.white_win_rate();
        let white_score = leaf.white_score_mean();
        let white_score_mean_sq = leaf.white_score_mean_sq();
        let white_utility = leaf.white_utility();

        while let Some(step) = self.playout_path.pop() {
            let mut parent = step.parent;
            let parent = unsafe { parent.as_mut() };
            parent.edge_mut(step.edge_index).record_visit();
            parent.record_visit(white_win, white_score, white_score_mean_sq, white_utility);
        }
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
            let (child_utility, edge_visits, force_root_visit) = match node.edge_for_move(move_loc)
            {
                Some((_, edge)) => {
                    let child = unsafe { edge.child().as_ref() };
                    let edge_visits = f64::from(edge.visits());
                    let desired_visits = (policy_probability
                        * total_child_visits
                        * self.params.root_desired_per_child_visits_coeff)
                        .sqrt();
                    (
                        child.white_utility(),
                        edge_visits,
                        is_root && policy_probability > 0.0 && edge_visits < desired_visits,
                    )
                }
                None => (fpu, 0.0, false),
            };
            let selection_value = if force_root_visit {
                f64::INFINITY
            } else {
                selection_value(
                    child_utility,
                    game_state.next_player(),
                    policy_probability,
                    edge_visits,
                    exploration_scaling,
                )
            };

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
        policy_logits: [f32; POLICY_SIZE],
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
                    self.policy_logits,
                    0.0,
                    0.0,
                    0.0,
                )));
            }
            Ok(())
        }
    }

    fn inference_client(fail: bool) -> InferenceClient {
        inference_client_with_policy(fail, [0.0; POLICY_SIZE])
    }

    fn inference_client_with_policy(
        fail: bool,
        policy_logits: [f32; POLICY_SIZE],
    ) -> InferenceClient {
        let model_handle = ModelRuntime::start(
            vec![TestBackend {
                fail,
                policy_logits,
            }],
            1,
            1,
            16,
            1,
        );
        InferenceClient::new(model_handle)
    }

    struct OneShotBackend {
        policy_logits: [f32; POLICY_SIZE],
        evaluated: bool,
    }

    impl InferenceBackend for OneShotBackend {
        fn evaluate_batch(
            &mut self,
            inputs: &[NNInput],
            outputs: &mut Vec<Arc<NNOutput>>,
        ) -> Result<(), InferenceError> {
            if self.evaluated {
                return Err(InferenceError::ExecutionFailed);
            }
            self.evaluated = true;
            for _ in inputs {
                outputs.push(Arc::new(NNOutput::from_raw(
                    self.policy_logits,
                    0.0,
                    0.0,
                    0.0,
                )));
            }
            Ok(())
        }
    }

    fn one_shot_inference_client(policy_logits: [f32; POLICY_SIZE]) -> InferenceClient {
        let model_handle = ModelRuntime::start(
            vec![OneShotBackend {
                policy_logits,
                evaluated: false,
            }],
            1,
            1,
            16,
            1,
        );
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
        assert!(worker.playout_path.is_empty());
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

    #[test]
    fn root_selection_forces_an_existing_child_below_its_desired_visits() {
        let worker = worker();
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let forced_move = Loc::new(3, 3).unwrap();
        let otherwise_best_move = Loc::new(4, 4).unwrap();
        let mut logits = [-20.0; POLICY_SIZE];
        logits[loc_to_policy(forced_move)] = 4.0;
        logits[loc_to_policy(otherwise_best_move)] = 0.0;
        let parent_output = processed_output(logits, 0.0);
        let mut parent = initialized_node(parent_output, 0.0);
        let mut forced_child =
            initialized_node(processed_output([0.0; POLICY_SIZE], 9.0_f32.ln()), 1.0);
        let mut otherwise_best_child = initialized_node(
            processed_output([0.0; POLICY_SIZE], (1.0_f32 / 9.0).ln()),
            -1.0,
        );
        let forced_edge = parent.add_child(
            forced_move,
            parent.policy_probs()[loc_to_policy(forced_move)],
            NonNull::from(forced_child.as_mut()),
        );
        let otherwise_best_edge = parent.add_child(
            otherwise_best_move,
            parent.policy_probs()[loc_to_policy(otherwise_best_move)],
            NonNull::from(otherwise_best_child.as_mut()),
        );
        for _ in 0..10 {
            parent.edge_mut(forced_edge).record_visit();
        }
        for _ in 0..100 {
            parent.edge_mut(otherwise_best_edge).record_visit();
        }

        assert_eq!(
            worker.choose_move(&parent, &game_state, false),
            otherwise_best_move
        );
        assert_eq!(worker.choose_move(&parent, &game_state, true), forced_move);
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
    async fn playout_expands_and_evaluates_a_missing_child() {
        let mut worker = worker();
        let mut client = inference_client(false);
        worker
            .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut client)
            .await
            .unwrap();

        worker.playout(&mut client).await.unwrap();

        assert_eq!(worker.search_graph.node_store.len(), 1);
        assert!(worker.playout_path.is_empty());
        let root = &worker.search_graph.root.as_ref().unwrap().node;
        assert_eq!(root.visits(), 2);
        assert_eq!(root.edge_visit_sum(), 1);
    }

    #[tokio::test]
    async fn consecutive_playouts_descend_and_back_up_through_existing_edges() {
        let preferred_move = Loc::new(4, 4).unwrap();
        let mut logits = [0.0; POLICY_SIZE];
        logits[loc_to_policy(preferred_move)] = 20.0;
        let mut client = inference_client_with_policy(false, logits);
        let mut worker = worker();
        worker
            .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut client)
            .await
            .unwrap();

        worker.playout(&mut client).await.unwrap();
        worker.playout(&mut client).await.unwrap();

        assert_eq!(worker.search_graph.node_store.len(), 2);
        assert!(worker.playout_path.is_empty());
        let root = &worker.search_graph.root.as_ref().unwrap().node;
        assert_eq!(root.visits(), 3);
        assert_eq!(root.edge_visit_sum(), 2);
        let root_edge = root.edge_for_move(preferred_move).unwrap().1;
        assert_eq!(root_edge.visits(), 2);
        let child = unsafe { root_edge.child().as_ref() };
        assert_eq!(child.visits(), 2);
        assert_eq!(child.edge_visit_sum(), 1);
    }

    #[tokio::test]
    async fn terminal_leaf_uses_exact_score_without_inference() {
        let mut root_state = GameState::new(Rules::TROMP_TAYLORISH);
        assert!(root_state.play(Loc::PASS));
        let mut logits = [0.0; POLICY_SIZE];
        logits[loc_to_policy(Loc::PASS)] = 20.0;
        let mut client = one_shot_inference_client(logits);
        let mut worker = worker();
        worker.start_game(root_state, &mut client).await.unwrap();

        worker.playout(&mut client).await.unwrap();

        assert_eq!(worker.search_graph.node_store.len(), 1);
        assert!(worker.scratch_game_state().is_finished());
        let root = &worker.search_graph.root.as_ref().unwrap().node;
        let pass_edge = root.edge_for_move(Loc::PASS).unwrap().1;
        assert_eq!(pass_edge.visits(), 1);
        let terminal = unsafe { pass_edge.child().as_ref() };
        assert_eq!(terminal.visits(), 1);
        assert_eq!(terminal.white_win_rate(), 1.0);
        assert_eq!(terminal.white_score_mean(), 7.5);
        assert_eq!(terminal.white_score_mean_sq(), 7.5 * 7.5);
    }

    #[tokio::test]
    async fn playout_inference_failure_does_not_attach_an_unevaluated_child() {
        let mut worker = worker();
        let mut working_client = inference_client(false);
        worker
            .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut working_client)
            .await
            .unwrap();

        let mut failing_client = inference_client(true);
        let result = worker.playout(&mut failing_client).await;

        assert!(matches!(result, Err(SearchError::InferenceError(_))));
        assert_eq!(worker.search_graph.node_store.len(), 0);
        assert!(worker.playout_path.is_empty());
        assert_eq!(
            worker
                .search_graph
                .root
                .as_ref()
                .unwrap()
                .node
                .visited_policy_mass(),
            0.0
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
            .insert(GraphKey::from_raw(1))
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
            .insert(GraphKey::from_raw(1))
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
