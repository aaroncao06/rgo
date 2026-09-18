use rand::Rng;
use std::{
    collections::HashSet,
    ptr::NonNull,
    sync::{Arc, OnceLock},
};

use crate::{
    game::{
        board::{BOARD_SIZE, Loc, Player},
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
        move_selection,
        node::{EdgeIndex, SearchNode, SearchStats},
        node_store::{InsertError, NodeStore},
        root_policy,
        search_params::SearchParams,
        utility::{recent_score_center, white_utility},
    },
};

struct SearchRoot {
    game_state: GameState, // playouts will start from this
    key: GraphKey,
    node: Box<SearchNode>, // owned separately from the rest of the nodestore
}

/// Owns the root position and stored graph nodes. Self-play resets the graph
/// between moves; traversal and reusable scratch buffers belong to the worker.
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
        root_game_state: &GameState,
        root_output: Arc<NNOutput>,
        root_utility: f64,
    ) {
        self.node_store.clear();
        let key = GraphKey::new(root_game_state);
        let node = Self::initialized_root(root_output, root_utility);
        match self.root.as_mut() {
            Some(root) => {
                // Retain the root state's history allocation between moves.
                root.game_state.reset_from(root_game_state);
                root.key = key;
                root.node = node;
            }
            None => {
                self.root = Some(SearchRoot {
                    game_state: root_game_state.clone(),
                    key,
                    node,
                });
            }
        }
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
        todo!();
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
    // The worker reuses its graph storage and scratch buffers across moves.
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
        root_game_state: &GameState,
        inference_client: &mut InferenceClient,
    ) -> Result<(), SearchError> {
        // just so that cross game graph reuse is not allowed
        self.reset_graph(root_game_state, inference_client).await
    }
    async fn reset_graph(
        &mut self,
        root_game_state: &GameState,
        inference_client: &mut InferenceClient,
    ) -> Result<(), SearchError> {
        // full reset, in the future can have a version where you retain subgraph between moves
        debug_assert!(!root_game_state.is_finished());
        let root_output = inference_client.evaluate(root_game_state).await?;
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
            Some(scratch) => scratch.reset_from(root_game_state),
            None => self.scratch_game_state = Some(root_game_state.clone()),
        }
        self.search_graph
            .reset(root_game_state, root_output, root_utility);
        self.recent_score_center = score_center;
        self.playout_path.clear();
        self.visited_nodes.clear();
        Ok(())
    }
    async fn search(
        &mut self,
        budget: usize,
        inference_client: &mut InferenceClient,
    ) -> Result<(), SearchError> {
        for _ in 0..budget {
            // Backends handle retryable failures; search propagates remaining errors.
            self.playout(inference_client).await?;
        }
        Ok(())
    }
    async fn choose_move<R: Rng + ?Sized>(
        &mut self,
        game_state: &GameState,
        budget: usize,
        inference_client: &mut InferenceClient,
        rng: &mut R,
    ) -> Result<Loc, SearchError> {
        // choose move for self play, without retaining subgraph between moves
        self.reset_graph(game_state, inference_client).await?;
        self.apply_root_policy_temperature_and_noise(game_state, rng);
        self.search(budget, inference_client).await?;

        let (moves, weights) = self.root_selection_weights();
        let temperature = move_selection::temperature(
            game_state.turn_number(),
            BOARD_SIZE,
            self.params.chosen_move_temperature_early,
            self.params.chosen_move_temperature,
            self.params.chosen_move_temperature_halflife,
        );
        Ok(moves[move_selection::sample_index(&weights, temperature, rng)])
    }

    fn apply_root_policy_temperature_and_noise<R: Rng + ?Sized>(
        &mut self,
        game_state: &GameState,
        rng: &mut R,
    ) {
        let params = self.params;
        let temperature = move_selection::temperature(
            game_state.turn_number(),
            BOARD_SIZE,
            params.root_policy_temperature_early,
            params.root_policy_temperature,
            params.chosen_move_temperature_halflife,
        );
        let legal = crate::inference::policy::legal_mask(game_state);
        let root = self
            .search_graph
            .root
            .as_mut()
            .expect("root policy requires an active game");
        let policy = root.node.policy_probs_mut();
        // KataGo shapes the policy before constructing the noise distribution.
        root_policy::apply_temperature(policy, &legal, temperature);
        if params.root_noise_enabled {
            root_policy::add_dirichlet_noise(
                policy,
                &legal,
                params.root_dirichlet_noise_total_concentration,
                params.root_dirichlet_noise_weight,
                rng,
            );
        }
    }

    // Final play-selection weights, not raw visits or a training-policy target.
    // This reads the graph without changing its statistics or stored policy.
    fn root_selection_weights(&self) -> (Vec<Loc>, Vec<f64>) {
        let root = self
            .search_graph
            .root
            .as_ref()
            .expect("selection requires an active game");
        let policy = root.node.policy_probs();
        let player = root.game_state.next_player();
        let mut moves = Vec::new();
        let mut weights = Vec::new();
        let mut total_child_weight = 0.0;
        let mut best_index = 0;
        let mut best_goodness = f64::NEG_INFINITY;
        for (i, edge) in root.node.edges().enumerate() {
            // SAFETY: the worker owns the entire graph; selection only borrows it.
            let weight = unsafe { edge.child_weight() };
            total_child_weight += weight;
            let loc = edge.move_loc();
            let weight = if root.game_state.is_legal(loc) {
                weight
            } else {
                0.0
            };
            moves.push(loc);
            weights.push(weight);
            let visits = f64::from(edge.visits());
            // KataGo's stable reference child: discount its newest visit and
            // add a small policy contribution to stabilize low-budget searches.
            let goodness = weight * (visits - 1.0).max(0.0) / visits.max(1.0)
                + 2.0 * f64::from(policy[loc_to_policy(loc)]);
            if goodness > best_goodness {
                best_goodness = goodness;
                best_index = i;
            }
        }

        if !moves.is_empty() {
            let reference_weight = weights[best_index];
            let explore_scaling = exploration_scaling(total_child_weight, self.params);
            let best_edge = root.node.edges().nth(best_index).unwrap();
            // SAFETY: all node pointers remain live and there are no mutable borrows.
            let best_child = unsafe { best_edge.child().as_ref() };
            let best_selection = selection_value(
                best_child.white_utility(),
                player,
                f64::from(policy[loc_to_policy(best_edge.move_loc())]),
                reference_weight,
                explore_scaling,
            );
            for (i, edge) in root.node.edges().enumerate() {
                // SAFETY: same graph-lifetime and shared-borrow guarantee as above.
                let child = unsafe { edge.child().as_ref() };
                let self_utility = match player {
                    Player::White => child.white_utility(),
                    Player::Black => -child.white_utility(),
                };
                if i != best_index && root.game_state.is_legal(edge.move_loc()) {
                    weights[i] = move_selection::reduced_weight(
                        weights[i],
                        self_utility,
                        f64::from(policy[loc_to_policy(edge.move_loc())]),
                        explore_scaling,
                        best_selection,
                    );
                }
            }
            if self.params.use_lcb_for_selection {
                self.adjust_root_weights_by_lcb(&mut weights, reference_weight);
            }
        } else {
            // Zero-budget search still has a root evaluation: use its legal policy.
            for (i, &probability) in policy.iter().enumerate() {
                let loc = policy_to_loc(i);
                if root.game_state.is_legal(loc) {
                    moves.push(loc);
                    weights.push(f64::from(probability));
                }
            }
        }
        if weights.iter().copied().fold(0.0, f64::max) <= 1e-50 {
            for (loc, weight) in moves.iter().zip(&mut weights) {
                *weight = if root.game_state.is_legal(*loc) {
                    f64::from(policy[loc_to_policy(*loc)])
                } else {
                    0.0
                };
            }
        }
        move_selection::prune_weights(
            &mut weights,
            self.params.chosen_move_subtract,
            self.params.chosen_move_prune,
        );
        (moves, weights)
    }
    /// Apply LCB after exploration-weight reduction, using the reference weight
    /// saved before reduction. Weights follow root edge iteration order.
    fn adjust_root_weights_by_lcb(&self, weights: &mut [f64], reference_weight: f64) {
        let root = self
            .search_graph
            .root
            .as_ref()
            .expect("selection requires an active game");
        let player = root.game_state.next_player();
        let utility_radius = self.params.win_loss_utility_factor
            + self.params.static_score_utility_factor
            + self.params.dynamic_score_utility_factor;
        let mut lcbs = Vec::with_capacity(weights.len());
        for edge in root.node.edges() {
            // SAFETY: the worker owns the graph and only shared borrows are live.
            let child = unsafe { edge.child().as_ref() };
            let self_utility = match player {
                Player::White => child.white_utility(),
                Player::Black => -child.white_utility(),
            };
            let fraction = f64::from(edge.visits()) / f64::from(child.visits().max(1));
            // LCB matches KataGo's getChildWeightSq: LINEAR edge fraction.
            // This estimates edge sample count, unlike the squared scaling
            // used when combining weighted child samples during backup.
            lcbs.push(move_selection::lcb_and_radius(
                self_utility,
                child.white_utility_mean_sq(),
                child.weight_sum() * fraction,
                child.weight_sq_sum() * fraction,
                utility_radius,
                self.params.lcb_stdevs,
            ));
        }
        // The selected config enables useNonBuggyLcb, so index zero is
        // eligible for the same bonus as every other child.
        move_selection::adjust_lcb(
            weights,
            &lcbs,
            reference_weight,
            self.params.min_visit_prop_for_lcb,
        );
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
                let (move_loc, edge_index) = self.select_child(node, game_state, is_root);
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
    fn select_child(
        &self,
        node: &SearchNode,
        game_state: &GameState,
        is_root: bool,
    ) -> (Loc, Option<EdgeIndex>) {
        // KataGo increases exploration slowly as the node accumulates visits.
        // SAFETY: selection runs while all graph nodes are live and none of the
        // children are mutably borrowed. The store cannot be cleared during it.
        let total_child_weight = unsafe { node.child_weight_sum() };
        let exploration_scaling = exploration_scaling(total_child_weight, self.params);

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

fn exploration_scaling(total_child_weight: f64, params: SearchParams) -> f64 {
    let cpuct = params.cpuct_exploration
        + params.cpuct_exploration_log
            * ((total_child_weight + params.cpuct_exploration_base)
                / params.cpuct_exploration_base)
                .ln();
    cpuct * (total_child_weight + 0.01).sqrt()
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
mod tests;
