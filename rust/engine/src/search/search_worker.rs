use std::{
    collections::HashSet,
    ptr::NonNull,
    sync::{Arc, OnceLock},
};

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
        node::{EdgeIndex, SearchNode, SearchStats},
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
    player: Player,
}

struct ChildContribution {
    white_win: f64,
    white_score: f64,
    white_score_mean_sq: f64,
    white_utility: f64,
    white_utility_mean_sq: f64,
    raw_weight: f64,
    adjusted_weight: f64,
    weight_sq_sum: f64,
}

struct SearchWorker<N: NodeStore> {
    // keeps mutating its search state, one worker per game, eventually supports subgraph reuse
    search_graph: SearchGraph<N>,
    playout_path: Vec<PlayoutStep>, // scratch work to avoid reallocating
    visited_nodes: HashSet<NonNull<SearchNode>>, // cycle detection within one playout
    scratch_game_state: Option<GameState>, // mutates through each playout
    child_contributions: Vec<ChildContribution>, // scratch work for weighted backup
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
            visited_nodes: HashSet::new(),
            scratch_game_state: None,
            child_contributions: Vec::new(),
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
        self.visited_nodes.clear();
        Ok(())
    }
    async fn playout(&mut self, inference_client: &mut InferenceClient) -> Result<(), SearchError> {
        self.playout_path.clear();
        self.visited_nodes.clear();
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
        self.visited_nodes.insert(current_node);

        loop {
            // scope for borrowing node: choose move and get relevant info regarding it (policy for it, whether edge already exists)
            let (move_loc, policy_prior, existing_edge, player) = {
                let node = unsafe { current_node.as_ref() };
                let game_state = self.scratch_game_state();
                let (move_loc, edge_index) = self.choose_move(node, game_state, is_root);
                let policy_prior = node.policy_probs()[loc_to_policy(move_loc)];
                let existing_edge = edge_index.map(|edge_index| {
                    let edge = node.edge(edge_index);
                    (edge_index, edge.child(), edge.visits())
                });
                (
                    move_loc,
                    policy_prior,
                    existing_edge,
                    game_state.next_player(),
                )
            };

            // Catch up existing edges before playing a move or rebuilding masks.
            // SAFETY: the graph owns the child, and no mutable child borrow is live.
            if let Some((edge_index, child, edge_visits)) = existing_edge {
                if edge_visits < unsafe { child.as_ref() }.visits() {
                    self.playout_path.push(PlayoutStep {
                        parent: current_node,
                        edge_index,
                        player,
                    });
                    // Backup increments this edge exactly once.
                    break;
                }
            }

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
                Some((edge_index, mut child, _)) => {
                    // descend
                    self.playout_path.push(PlayoutStep {
                        parent: current_node,
                        edge_index,
                        player,
                    });
                    // Match KataGo: a repeated node ends this playout. The
                    // closing edge and its ancestors still receive backup.
                    if !self.visited_nodes.insert(child) {
                        break;
                    }
                    if is_finished {
                        // Terminal nodes have no NN output to recompute, so a
                        // repeated playout contributes the same exact result.
                        unsafe { child.as_mut() }.repeat_visit();
                        break;
                    }
                    current_node = child;
                    is_root = false;
                }
                None => {
                    // A new edge to an existing, evaluated node starts at zero
                    // visits, so this playout catches up using its existing stats.
                    if let Some(child) = self.search_graph.node_store.find(current_key) {
                        let edge_index = unsafe { current_node.as_mut() }.add_child(
                            move_loc,
                            policy_prior,
                            child,
                        );
                        self.playout_path.push(PlayoutStep {
                            parent: current_node,
                            edge_index,
                            player,
                        });
                        debug_assert!(unsafe { child.as_ref() }.visits() > 0);
                        break;
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
                        // KataGo represents integer scores as an equal mixture
                        // of adjacent half-integers with default draw weighting.
                        let white_score_mean_sq = white_score * white_score
                            + if white_score.fract() == 0.0 {
                                0.25
                            } else {
                                0.0
                            };
                        let white_utility = white_utility(
                            white_win,
                            white_score,
                            white_score_mean_sq,
                            self.recent_score_center,
                            self.params,
                        );
                        unsafe { child.as_mut() }.record_visit(
                            white_win,
                            white_score,
                            white_score_mean_sq,
                            white_utility,
                        );
                    }
                    // connect the child to the parent
                    let edge_index =
                        unsafe { current_node.as_mut() }.add_child(move_loc, policy_prior, child);
                    self.playout_path.push(PlayoutStep {
                        parent: current_node,
                        edge_index,
                        player,
                    });
                    break;
                }
            }
        }
        self.backup();
        self.visited_nodes.clear();

        Ok(())
    }
    fn backup(&mut self) {
        while let Some(step) = self.playout_path.pop() {
            let mut parent = step.parent;
            let parent = unsafe { parent.as_mut() };
            parent.edge_mut(step.edge_index).record_visit();
            self.recompute_node_stats(parent, step.player);
        }
    }
    fn choose_move(
        &self,
        node: &SearchNode,
        game_state: &GameState,
        is_root: bool,
    ) -> (Loc, Option<EdgeIndex>) {
        // KataGo increases exploration slowly as the node accumulates visits.
        // SAFETY: selection runs while all graph nodes are live and none of the
        // children are mutably borrowed. The store cannot be cleared during it.
        let total_child_weight = unsafe { node.child_weight_sum() };
        let cpuct = self.params.cpuct_exploration
            + self.params.cpuct_exploration_log
                * ((total_child_weight + self.params.cpuct_exploration_base)
                    / self.params.cpuct_exploration_base)
                    .ln();
        let exploration_scaling = cpuct * (total_child_weight + 0.01).sqrt();

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

        // Score existing children in insertion order. Strict comparisons retain
        // the first child on ties, including ties between forced root visits.
        let mut best_move = Loc::NULL;
        let mut best_edge = None;
        let mut best_selection_value = f64::NEG_INFINITY;
        let policy_probs = node.policy_probs();
        let mut expanded = [false; POLICY_SIZE];
        for (edge_index, edge) in node.indexed_edges() {
            let move_loc = edge.move_loc();
            let policy_index = loc_to_policy(move_loc);
            expanded[policy_index] = true;
            if !game_state.is_legal(move_loc) {
                continue;
            }

            let policy_probability = f64::from(policy_probs[policy_index]);
            let child = unsafe { edge.child().as_ref() };
            // SAFETY: the same graph-lifetime guarantee as above applies here.
            let child_weight = unsafe { edge.child_weight() };
            let force_root_visit = is_root
                && self.params.root_desired_per_child_visits_coeff > 0.0
                && policy_probability > 0.0
                && child_weight
                    < (policy_probability
                        * total_child_weight
                        * self.params.root_desired_per_child_visits_coeff)
                        .sqrt();
            let selection_value = if force_root_visit {
                1e20 // KataGo's forced-visit selection value.
            } else {
                selection_value(
                    child.white_utility(),
                    game_state.next_player(),
                    policy_probability,
                    child_weight,
                    exploration_scaling,
                )
            };

            if selection_value > best_selection_value {
                best_selection_value = selection_value;
                best_move = move_loc;
                best_edge = Some(edge_index);
            }
        }

        // All unexpanded moves share FPU and zero child weight, so only the
        // highest-policy legal candidate needs scoring. The mask avoids lookups.
        let mut best_new_move = Loc::NULL;
        let mut best_new_policy = -1.0_f32;
        for (policy_index, &probability) in policy_probs.iter().enumerate() {
            if expanded[policy_index] {
                continue;
            }
            let move_loc = policy_to_loc(policy_index);
            if game_state.is_legal(move_loc) && probability > best_new_policy {
                best_new_move = move_loc;
                best_new_policy = probability;
            }
        }
        if best_new_move != Loc::NULL {
            let new_selection_value = selection_value(
                fpu,
                game_state.next_player(),
                f64::from(best_new_policy),
                0.0,
                exploration_scaling,
            );
            // Existing children win exact ties against the unexpanded candidate.
            if new_selection_value > best_selection_value {
                best_move = best_new_move;
                best_edge = None;
            }
        }

        debug_assert!(best_move != Loc::NULL, "pass must always be legal");
        (best_move, best_edge)
    }
    fn recompute_node_stats(&mut self, node: &mut SearchNode, player: Player) {
        let contributions = &mut self.child_contributions;
        contributions.clear();
        let mut total_child_weight = 0.0;
        // Like KataGo, count the completed playout independently of child
        // statistics. Cycles can make edge visits exceed a child's visits.
        let visits = node.visits() + 1;

        for edge in node.edges() {
            if edge.visits() <= 0 {
                continue;
            }
            // A self-loop must read through the current borrow, not create an
            // alias from its raw pointer while `node` is mutably borrowed.
            let child = if std::ptr::eq(edge.child().as_ptr(), node) {
                &*node
            } else {
                // SAFETY: other graph nodes remain live and are not mutably
                // borrowed during this parent's recomputation.
                unsafe { edge.child().as_ref() }
            };
            if child.visits() <= 0 || child.weight_sum() <= 0.0 {
                continue;
            }

            // A transposed child may have visits from other parent edges. Only the
            // fraction attributable to this edge contributes to this parent.
            let edge_fraction = f64::from(edge.visits()) / f64::from(child.visits().max(1));
            let raw_weight = child.weight_sum() * edge_fraction;
            total_child_weight += raw_weight;
            contributions.push(ChildContribution {
                white_win: child.white_win_rate(),
                white_score: child.white_score_mean(),
                white_score_mean_sq: child.white_score_mean_sq(),
                white_utility: child.white_utility(),
                white_utility_mean_sq: child.white_utility_mean_sq(),
                raw_weight,
                adjusted_weight: raw_weight,
                // Scaling every sample's weight scales its square quadratically.
                weight_sq_sum: child.weight_sq_sum() * edge_fraction * edge_fraction,
            });
        }

        if total_child_weight > 0.0 && self.params.value_weight_exponent != 0.0 {
            let simple_utility = contributions
                .iter()
                .map(|child| {
                    let utility = match player {
                        Player::White => child.white_utility,
                        Player::Black => -child.white_utility,
                    };
                    utility * child.raw_weight
                })
                .sum::<f64>()
                / total_child_weight;

            let mut adjusted_weight_sum = 0.0;
            for child in contributions.iter_mut() {
                let utility = match player {
                    Player::White => child.white_utility,
                    Player::Black => -child.white_utility,
                };
                let precision = 1.5 * child.raw_weight.sqrt();
                let stdev = (1e-8 + 1.0 / precision).sqrt();
                let z = (utility - simple_utility) / stdev;
                let value_weight =
                    (student_t_cdf_degrees_3(z) + 0.0001).powf(self.params.value_weight_exponent);
                child.adjusted_weight *= value_weight;
                adjusted_weight_sum += child.adjusted_weight;
            }

            let normalization = total_child_weight / adjusted_weight_sum;
            for child in contributions.iter_mut() {
                child.adjusted_weight *= normalization;
            }
        }

        let output = node.nn_output();
        let direct_white_win = f64::from(output.white_win_prob());
        let direct_white_score = f64::from(output.white_score_mean());
        let direct_white_score_mean_sq = f64::from(output.white_score_mean_sq());
        let direct_white_utility = white_utility(
            direct_white_win,
            direct_white_score,
            direct_white_score_mean_sq,
            self.recent_score_center,
            self.params,
        );

        let mut white_win_sum = direct_white_win;
        let mut white_score_sum = direct_white_score;
        let mut white_score_mean_sq_sum = direct_white_score_mean_sq;
        let mut white_utility_sum = direct_white_utility;
        let mut white_utility_sq_sum = direct_white_utility * direct_white_utility;
        let mut weight_sq_sum = 1.0;
        for child in contributions.iter() {
            let weight = child.adjusted_weight;
            let weight_scaling = weight / child.raw_weight;
            white_win_sum += weight * child.white_win;
            white_score_sum += weight * child.white_score;
            white_score_mean_sq_sum += weight * child.white_score_mean_sq;
            white_utility_sum += weight * child.white_utility;
            white_utility_sq_sum += weight * child.white_utility_mean_sq;
            weight_sq_sum += weight_scaling * weight_scaling * child.weight_sq_sum;
        }

        node.replace_stats(SearchStats {
            visits,
            white_win_sum,
            white_score_sum,
            white_score_mean_sq_sum,
            white_utility_sum,
            white_utility_sq_sum,
            weight_sum: 1.0 + total_child_weight,
            weight_sq_sum,
        });
    }
}

// Match KataGo's DistributionTable: 2,000 samples across [-50, 50], forced
// endpoint probabilities of 0 and 1, and linear interpolation between samples.
fn student_t_cdf_degrees_3(value: f64) -> f64 {
    const SIZE: usize = 2000;
    static TABLE: OnceLock<[f64; SIZE]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        std::array::from_fn(|i| {
            if i == 0 {
                0.0
            } else if i == SIZE - 1 {
                1.0
            } else {
                let z = -50.0 + i as f64 * 100.0 / (SIZE - 1) as f64;
                // Closed-form Student-t(3) CDF used only to initialize the table.
                let sqrt_three = 3.0_f64.sqrt();
                0.5 + (z.atan2(sqrt_three) + z * sqrt_three / (z * z + 3.0)) / std::f64::consts::PI
            }
        })
    });
    let position = (SIZE - 1) as f64 * (value + 50.0) / 100.0;
    if position <= 0.0 {
        return 0.0;
    }
    let index = position as usize;
    if index >= SIZE - 1 {
        return 1.0;
    }
    let fraction = position - index as f64;
    table[index] + fraction * (table[index + 1] - table[index])
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

    #[derive(Debug, PartialEq)]
    struct NodeSnapshot {
        visits: i32,
        values: [f64; 7],
        edges: Vec<(Loc, i32, f32, NonNull<SearchNode>)>,
    }

    fn node_snapshot(node: &SearchNode) -> NodeSnapshot {
        NodeSnapshot {
            visits: node.visits(),
            values: [
                node.white_win_rate(),
                node.white_score_mean(),
                node.white_score_mean_sq(),
                node.white_utility(),
                node.white_utility_mean_sq(),
                node.weight_sum(),
                node.weight_sq_sum(),
            ],
            edges: node
                .edges()
                .map(|edge| {
                    (
                        edge.move_loc(),
                        edge.visits(),
                        edge.policy_prior(),
                        edge.child(),
                    )
                })
                .collect(),
        }
    }

    // Snapshot every reachable node, including edge identity and all aggregate
    // values, so failure tests detect partial backup or partial attachment.
    fn graph_snapshot(worker: &SearchWorker<FixedArenaNodeStore>) -> Vec<NodeSnapshot> {
        let root = &worker.search_graph.root.as_ref().unwrap().node;
        let mut snapshots = vec![node_snapshot(root)];
        let mut pending: Vec<_> = root.edges().map(|edge| edge.child()).collect();
        let mut seen = HashSet::new();
        while let Some(pointer) = pending.pop() {
            if !seen.insert(pointer) {
                continue;
            }
            // SAFETY: the worker owns these nodes and is only shared-borrowed.
            let node = unsafe { pointer.as_ref() };
            snapshots.push(node_snapshot(node));
            pending.extend(node.edges().map(|edge| edge.child()));
        }
        snapshots
    }

    fn focused_policy() -> [f32; POLICY_SIZE] {
        let mut logits = [-20.0; POLICY_SIZE];
        logits[loc_to_policy(Loc::new(4, 4).unwrap())] = 20.0;
        logits[loc_to_policy(Loc::new(3, 3).unwrap())] = 10.0;
        logits
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

        assert_eq!(
            worker.choose_move(&node, &game_state, true),
            (expected_move, None)
        );
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
            worker.choose_move(&parent, &game_state, true).0,
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
            worker.choose_move(&parent, &game_state, false).0,
            otherwise_best_move
        );
        assert_eq!(
            worker.choose_move(&parent, &game_state, true).0,
            forced_move
        );
    }

    #[test]
    fn forced_root_visit_ties_follow_child_insertion_order() {
        let worker = worker();
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let first_move = Loc::new(4, 4).unwrap();
        let second_move = Loc::new(3, 3).unwrap();
        assert!(loc_to_policy(first_move) > loc_to_policy(second_move));
        let mut logits = [-20.0; POLICY_SIZE];
        logits[loc_to_policy(first_move)] = 0.0;
        logits[loc_to_policy(second_move)] = 0.0;
        let mut parent = initialized_node(processed_output(logits, 0.0), 0.0);
        let mut first_child = initialized_node(processed_output(logits, 0.0), 0.0);
        let mut second_child = initialized_node(processed_output(logits, 0.0), 0.0);
        let first_edge = parent.add_child(first_move, 0.5, NonNull::from(first_child.as_mut()));
        let second_edge = parent.add_child(second_move, 0.5, NonNull::from(second_child.as_mut()));
        parent.edge_mut(first_edge).record_visit();
        parent.edge_mut(second_edge).record_visit();
        // Each child has weight 1, below sqrt(0.5 * 2 * 2).
        assert_eq!(
            worker.choose_move(&parent, &game_state, true),
            (first_move, Some(first_edge))
        );
    }

    #[test]
    fn existing_child_wins_exact_tie_against_unexpanded_move() {
        let mut worker = worker();
        worker.params.cpuct_exploration = 0.0;
        worker.params.cpuct_exploration_log = 0.0;
        worker.params.fpu_reduction_max = 0.0;
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let existing_move = Loc::new(4, 4).unwrap();
        let output = processed_output([0.0; POLICY_SIZE], 0.0);
        let mut parent = initialized_node(output.clone(), 0.0);
        let mut child = initialized_node(output, 0.0);
        let edge = parent.add_child(existing_move, 0.5, NonNull::from(child.as_mut()));
        parent.edge_mut(edge).record_visit();
        assert_eq!(
            worker.choose_move(&parent, &game_state, false).0,
            existing_move
        );
    }

    #[test]
    fn value_weighting_favors_the_better_child_and_preserves_total_weight() {
        let mut worker = worker();
        let mut parent = initialized_node(processed_output([0.0; POLICY_SIZE], 0.0), 0.0);
        let mut good_child =
            initialized_node(processed_output([0.0; POLICY_SIZE], 9.0_f32.ln()), 1.0);
        let mut bad_child = initialized_node(
            processed_output([0.0; POLICY_SIZE], (1.0_f32 / 9.0).ln()),
            -1.0,
        );
        for _ in 0..9 {
            good_child.record_visit(1.0, 0.0, 0.0, 1.0);
            bad_child.record_visit(0.0, 0.0, 0.0, -1.0);
        }

        let good_edge = parent.add_child(
            Loc::new(3, 3).unwrap(),
            0.5,
            NonNull::from(good_child.as_mut()),
        );
        let bad_edge = parent.add_child(
            Loc::new(4, 4).unwrap(),
            0.5,
            NonNull::from(bad_child.as_mut()),
        );
        for _ in 0..10 {
            parent.edge_mut(good_edge).record_visit();
            parent.edge_mut(bad_edge).record_visit();
        }

        worker.recompute_node_stats(&mut parent, Player::White);

        assert_eq!(parent.visits(), 2); // One completed playout, not a sum of child visits.
        assert!((parent.weight_sum() - 21.0).abs() < 1e-12);
        assert!(parent.white_utility() > 0.0);
    }

    #[test]
    fn transposed_child_squared_weights_scale_by_squared_edge_fraction() {
        let mut worker = worker();
        let output = processed_output([0.0; POLICY_SIZE], 0.0);
        let mut child = initialized_node(output.clone(), 0.0);
        for _ in 0..9 {
            child.record_visit(0.5, 0.0, 0.0, 0.0);
        }
        let mut first_parent = initialized_node(output.clone(), 0.0);
        let mut second_parent = initialized_node(output, 0.0);
        let child_ptr = NonNull::from(child.as_mut());
        for parent in [&mut first_parent, &mut second_parent] {
            let edge = parent.add_child(Loc::PASS, 1.0, child_ptr);
            for _ in 0..5 {
                parent.edge_mut(edge).record_visit();
            }
            worker.recompute_node_stats(parent, Player::White);

            assert_eq!(parent.visits(), 2);
            assert!((parent.weight_sum() - 6.0).abs() < 1e-12);
            // Ten child samples scaled by 1/2, plus the parent's unit-weight eval.
            assert!((parent.weight_sq_sum() - 3.5).abs() < 1e-12);
        }
        assert_eq!(child.weight_sq_sum(), 10.0);
    }

    #[test]
    fn student_t_cdf_degrees_3_is_centered_and_symmetric() {
        assert!((student_t_cdf_degrees_3(0.0) - 0.5).abs() < 1e-12);
        let positive = student_t_cdf_degrees_3(2.0);
        let negative = student_t_cdf_degrees_3(-2.0);
        assert!((positive + negative - 1.0).abs() < 1e-12);
        assert!(positive > 0.5);
    }

    #[tokio::test]
    async fn graph_cycles_end_the_playout_and_back_up_without_inference() {
        for self_loop in [false, true] {
            let root_move = Loc::new(0, 0).unwrap();
            let first_move = Loc::new(1, 0).unwrap();
            let second_move = Loc::new(2, 0).unwrap();
            let policy = |loc| {
                let mut logits = [-20.0; POLICY_SIZE];
                logits[loc_to_policy(loc)] = 20.0;
                logits
            };
            let mut client = one_shot_inference_client(policy(root_move));
            let mut worker = worker();
            worker
                .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut client)
                .await
                .unwrap();
            let mut first = worker
                .search_graph
                .node_store
                .insert(GraphKey::from_raw(1))
                .unwrap();
            let mut second = if self_loop {
                first
            } else {
                worker
                    .search_graph
                    .node_store
                    .insert(GraphKey::from_raw(2))
                    .unwrap()
            };
            // Synthetic graph: deliberately seed equal edge/node counts to
            // exercise the cycle guard rather than the earlier catch-up check.
            unsafe {
                first
                    .as_mut()
                    .attach_nn_output(processed_output(policy(first_move), 0.0));
                first.as_mut().record_visit(0.5, 0.0, 0.0, 0.0);
                if !self_loop {
                    second
                        .as_mut()
                        .attach_nn_output(processed_output(policy(second_move), 0.0));
                    second.as_mut().record_visit(0.5, 0.0, 0.0, 0.0);
                }
            }
            let root = &mut worker.search_graph.root.as_mut().unwrap().node;
            let root_edge = root.add_child(root_move, 1.0, first);
            root.edge_mut(root_edge).record_visit();
            let first_edge = unsafe { first.as_mut() }.add_child(first_move, 1.0, second);
            unsafe { first.as_mut() }
                .edge_mut(first_edge)
                .record_visit();
            let second_edge = if self_loop {
                None
            } else {
                let edge = unsafe { second.as_mut() }.add_child(second_move, 1.0, first);
                unsafe { second.as_mut() }.edge_mut(edge).record_visit();
                Some(edge)
            };

            // Repeat to check that cycle tracking resets between playouts and
            // every closing edge/ancestor receives exactly one update each time.
            for expected_visits in 2..=6 {
                worker.playout(&mut client).await.unwrap();
                let root = &worker.search_graph.root.as_ref().unwrap().node;
                assert_eq!(root.visits(), expected_visits);
                assert_eq!(root.edge(root_edge).visits(), expected_visits);
                let first_node = unsafe { first.as_ref() };
                assert_eq!(first_node.visits(), expected_visits);
                assert_eq!(first_node.edge(first_edge).visits(), expected_visits);
                if let Some(edge) = second_edge {
                    let second_node = unsafe { second.as_ref() };
                    assert_eq!(second_node.visits(), expected_visits);
                    assert_eq!(second_node.edge(edge).visits(), expected_visits);
                }
                assert_eq!(
                    worker.search_graph.node_store.len(),
                    if self_loop { 1 } else { 2 }
                );
                assert!(
                    graph_snapshot(&worker)
                        .iter()
                        .all(|node| node.values.iter().all(|value| value.is_finite()))
                );
                assert!(worker.playout_path.is_empty());
                assert!(worker.visited_nodes.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn transposed_child_catches_up_before_requesting_more_inference() {
        let move_loc = Loc::new(4, 4).unwrap();
        let mut logits = [-20.0; POLICY_SIZE];
        logits[loc_to_policy(move_loc)] = 20.0;
        let mut client = one_shot_inference_client(logits);
        let mut worker = worker();
        let root_state = GameState::new(Rules::TROMP_TAYLORISH);
        let mut child_state = root_state.clone();
        let mut child_key = GraphKey::new(&root_state);
        assert!(child_state.play(move_loc));
        child_key.advance(&child_state, move_loc);
        worker.start_game(root_state, &mut client).await.unwrap();

        // Model a node already searched three times through another parent.
        let mut child_ptr = worker.search_graph.node_store.insert(child_key).unwrap();
        {
            let child = unsafe { child_ptr.as_mut() };
            child.attach_nn_output(processed_output([0.0; POLICY_SIZE], 0.0));
            for _ in 0..3 {
                child.record_visit(0.5, 0.0, 0.0, 0.0);
            }
        }
        // First playout links a new edge; subsequent ones catch up that edge.
        // The backend rejects any inference after the root's evaluation.
        for expected_visits in 1..=3 {
            worker.playout(&mut client).await.unwrap();
            let root = &worker.search_graph.root.as_ref().unwrap().node;
            assert_eq!(
                root.edge_for_move(move_loc).unwrap().1.visits(),
                expected_visits
            );
            assert_eq!(root.visits(), expected_visits + 1);
            assert_eq!(unsafe { child_ptr.as_ref() }.visits(), 3);
            assert_eq!(worker.search_graph.node_store.len(), 1);
            assert!(worker.playout_path.is_empty());
            if expected_visits > 1 {
                // Existing-edge catch-up stops before changing the board,
                // player, or superko state. New-edge lookup still plays the move.
                let root_state = &worker.search_graph.root.as_ref().unwrap().game_state;
                assert_eq!(
                    GraphKey::new(worker.scratch_game_state()),
                    GraphKey::new(root_state)
                );
                assert_eq!(
                    worker.scratch_game_state().next_player(),
                    root_state.next_player()
                );
            }
        }
        // Once caught up, search must descend and request a fresh evaluation.
        assert!(matches!(
            worker.playout(&mut client).await,
            Err(SearchError::InferenceError(_))
        ));
    }

    #[test]
    fn student_t_table_clamps_and_linearly_interpolates() {
        assert_eq!(student_t_cdf_degrees_3(-100.0), 0.0);
        assert_eq!(student_t_cdf_degrees_3(-50.0), 0.0);
        assert_eq!(student_t_cdf_degrees_3(50.0), 1.0);
        assert_eq!(student_t_cdf_degrees_3(100.0), 1.0);
        let left = -50.0 + 1030.0 * 100.0 / 1999.0;
        let right = -50.0 + 1031.0 * 100.0 / 1999.0;
        let midpoint = (left + right) * 0.5;
        let expected = (student_t_cdf_degrees_3(left) + student_t_cdf_degrees_3(right)) * 0.5;
        assert!((student_t_cdf_degrees_3(midpoint) - expected).abs() < 1e-12);
        // Reference Student-t(3) probability at x=2 (allow table interpolation error).
        assert!((student_t_cdf_degrees_3(2.0) - 0.9303370157205785).abs() < 1e-4);
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
    async fn integer_terminal_scores_use_katago_gridded_second_moment() {
        for komi in [-7.0, 0.0, 7.0] {
            let mut root_state = GameState::new(Rules {
                komi,
                ..Rules::TROMP_TAYLORISH
            });
            assert!(root_state.play(Loc::PASS));
            let mut logits = [-20.0; POLICY_SIZE];
            logits[loc_to_policy(Loc::PASS)] = 20.0;
            let mut client = one_shot_inference_client(logits);
            let mut worker = worker();
            worker.start_game(root_state, &mut client).await.unwrap();

            for _ in 0..2 {
                worker.playout(&mut client).await.unwrap();
                let root = &worker.search_graph.root.as_ref().unwrap().node;
                let terminal = unsafe { root.edge_for_move(Loc::PASS).unwrap().1.child().as_ref() };
                let score = f64::from(komi);
                let win = if score > 0.0 {
                    1.0
                } else if score < 0.0 {
                    0.0
                } else {
                    0.5
                };
                assert_eq!(terminal.white_win_rate(), win);
                assert_eq!(terminal.white_score_mean(), score);
                assert_eq!(terminal.white_score_mean_sq(), score * score + 0.25);
                let expected_utility = white_utility(
                    win,
                    score,
                    score * score + 0.25,
                    worker.recent_score_center,
                    worker.params,
                );
                assert!((terminal.white_utility() - expected_utility).abs() < 1e-12);
            }
        }
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
        worker.playout(&mut client).await.unwrap();

        assert_eq!(worker.search_graph.node_store.len(), 1);
        assert!(worker.scratch_game_state().is_finished());
        let root = &worker.search_graph.root.as_ref().unwrap().node;
        let pass_edge = root.edge_for_move(Loc::PASS).unwrap().1;
        assert_eq!(pass_edge.visits(), 2);
        let terminal = unsafe { pass_edge.child().as_ref() };
        assert_eq!(terminal.visits(), 2);
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
    async fn deep_inference_failure_preserves_graph_and_retry_backs_up_once() {
        let mut worker = worker();
        let mut client = inference_client_with_policy(false, focused_policy());
        worker
            .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut client)
            .await
            .unwrap();
        for _ in 0..2 {
            worker.playout(&mut client).await.unwrap();
        }
        let before = graph_snapshot(&worker);
        assert_eq!(before.len(), 3);
        let mut failing = inference_client(true);
        assert!(matches!(
            worker.playout(&mut failing).await,
            Err(SearchError::InferenceError(_))
        ));
        assert_eq!(
            worker.playout_path.len(),
            2,
            "failure must occur below two existing edges"
        );
        assert_eq!(worker.search_graph.node_store.len(), 2);
        assert_eq!(graph_snapshot(&worker), before);

        worker.playout(&mut client).await.unwrap();
        let after = graph_snapshot(&worker);
        assert_eq!(after.len(), 4);
        assert_eq!(worker.search_graph.node_store.len(), 3);
        for (old, new) in before.iter().zip(&after) {
            assert_eq!(new.visits, old.visits + 1);
            for (old_edge, new_edge) in old.edges.iter().zip(&new.edges) {
                assert_eq!(new_edge.1, old_edge.1 + 1);
                assert_eq!(new_edge.3, old_edge.3);
            }
        }
        assert_eq!(after[2].edges.len(), 1);
        assert_eq!(after[2].edges[0].1, 1);
        assert_eq!(after[3].visits, 1);
        assert!(after[3].edges.is_empty());
        assert!(worker.playout_path.is_empty());
        assert!(worker.visited_nodes.is_empty());
    }

    #[tokio::test]
    async fn full_store_preserves_graph_and_reset_reuses_capacity() {
        let mut worker = SearchWorker::new(
            FixedArenaNodeStore::new(1),
            SearchParams::KATAGO_SELFPLAY8_MAIN_B18,
        );
        let mut client = inference_client_with_policy(false, focused_policy());
        worker
            .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut client)
            .await
            .unwrap();
        worker.playout(&mut client).await.unwrap();
        let before = graph_snapshot(&worker);
        // A fresh one-shot client proves the inference before insertion succeeds.
        let mut one_shot = one_shot_inference_client(focused_policy());
        assert!(matches!(
            worker.playout(&mut one_shot).await,
            Err(SearchError::NodeStore(InsertError::StoreFull))
        ));
        assert_eq!(worker.playout_path.len(), 1);
        assert_eq!(worker.search_graph.node_store.len(), 1);
        assert_eq!(graph_snapshot(&worker), before);

        worker
            .reset_graph(GameState::new(Rules::TROMP_TAYLORISH), &mut client)
            .await
            .unwrap();
        assert_eq!(worker.search_graph.node_store.len(), 0);
        assert_eq!(worker.search_graph.node_store.capacity(), 1);
        assert!(worker.playout_path.is_empty());
        assert!(worker.visited_nodes.is_empty());
        worker.playout(&mut client).await.unwrap();
        assert_eq!(worker.search_graph.node_store.len(), 1);
        let root = &worker.search_graph.root.as_ref().unwrap().node;
        assert_eq!(root.visits(), 2);
        assert_eq!(root.edge_visit_sum(), 1);
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
        let mut working_client = inference_client_with_policy(false, focused_policy());
        worker
            .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut working_client)
            .await
            .unwrap();
        // Use a nonzero center so an accidental reset to zero is observable.
        // Subsequent playouts back up the graph using this center.
        worker.recent_score_center = 3.0;
        for _ in 0..2 {
            worker.playout(&mut working_client).await.unwrap();
        }
        let before = graph_snapshot(&worker);
        let original_center = worker.recent_score_center;
        let original_scratch_key = GraphKey::new(worker.scratch_game_state());
        let original_key = worker.search_graph.root.as_ref().unwrap().key;

        let mut failing_client = inference_client(true);
        let mut replacement = GameState::new(Rules::TROMP_TAYLORISH);
        assert!(replacement.play(Loc::new(4, 4).unwrap()));
        let result = worker.reset_graph(replacement, &mut failing_client).await;

        assert!(matches!(result, Err(SearchError::InferenceError(_))));
        assert_eq!(worker.search_graph.root.as_ref().unwrap().key, original_key);
        assert_eq!(worker.search_graph.node_store.len(), 2);
        assert_eq!(graph_snapshot(&worker), before);
        assert_eq!(worker.recent_score_center, original_center);
        assert_eq!(
            GraphKey::new(worker.scratch_game_state()),
            original_scratch_key
        );
        worker.playout(&mut working_client).await.unwrap();
        assert_eq!(worker.search_graph.node_store.len(), 3);
        let after = graph_snapshot(&worker);
        for (old, new) in before.iter().zip(&after) {
            assert_eq!(new.visits, old.visits + 1);
        }
        assert!(worker.playout_path.is_empty());
        assert!(worker.visited_nodes.is_empty());
    }
}
