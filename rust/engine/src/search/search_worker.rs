//! Graph lifecycle, playout execution, and the reverse backup walk.
//! Private policy modules decide descent, parent estimates, and final moves.

use rand::Rng;
use std::{collections::HashSet, ptr::NonNull, sync::Arc};

use crate::{
    game::{
        board::{ARRAY_LEN, Color, Loc, Player},
        game_state::GameState,
    },
    inference::{
        backend::InferenceError, outputs::NNOutput, policy::loc_to_policy, runtime::InferenceClient,
    },
    search::{
        graph_key::GraphKey,
        node::{EdgeIndex, SearchNode},
        node_store::{InsertError, NodeStore},
        search_params::SearchParams,
        utility::{recent_score_center, white_utility},
    },
};

mod backup_policy;
mod root_endgame;
mod root_policy;
mod selection_policy;

use backup_policy::ChildContribution;

struct SearchRoot {
    game_state: GameState, // playouts will start from this
    key: GraphKey,
    node: Box<SearchNode>, // owned separately from the rest of the nodestore
    safe_area: [Color; ARRAY_LEN],
}

/// Owns the root position and stored graph nodes. Self-play resets the graph
/// between moves; traversal and reusable scratch buffers belong to the worker.
struct SearchGraph<N: NodeStore> {
    root: Option<SearchRoot>,
    node_store: N,
}

impl<N: NodeStore> SearchGraph<N> {
    fn new(node_store: N) -> Self {
        Self {
            root: None,
            node_store,
        }
    }
    fn reset(
        &mut self,
        root_game_state: &GameState,
        root_output: Arc<NNOutput>,
        root_utility: f64,
    ) {
        self.node_store.clear();
        let key = GraphKey::new(root_game_state);
        let node = Self::initialized_root(root_output, root_utility);
        let safe_area = root_game_state
            .board()
            .calculate_pass_alive_area(root_game_state.rules().multi_stone_suicide_legal);
        match self.root.as_mut() {
            Some(root) => {
                // Retain the root state's history allocation between moves.
                root.game_state.reset_from(root_game_state);
                root.key = key;
                root.node = node;
                root.safe_area = safe_area;
            }
            None => {
                self.root = Some(SearchRoot {
                    game_state: root_game_state.clone(),
                    key,
                    node,
                    safe_area,
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
    fn advance_root(&mut self, move_loc: Loc) {
        // later for normal play
        todo!();
    }
}

struct PlayoutStep {
    parent: NonNull<SearchNode>,
    edge_index: EdgeIndex,
    player: Player,
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
enum SearchError {
    NodeStore(InsertError),
    InferenceError(InferenceError),
    NoSelectableMove,
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
        let root_output = inference_client
            .evaluate(root_game_state, self.params.root_ending_bonus_points != 0.0)
            .await?;
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

        self.select_root_move(game_state, rng)
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
            if let Some((edge_index, child, edge_visits)) = existing_edge
                && edge_visits < unsafe { child.as_ref() }.visits()
            {
                self.playout_path.push(PlayoutStep {
                    parent: current_node,
                    edge_index,
                    player,
                });
                // Backup increments this edge exactly once.
                break;
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
                        Some(
                            inference_client
                                .evaluate(self.scratch_game_state(), false)
                                .await?,
                        )
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
}

#[cfg(test)]
mod tests;
