//! Root preprocessing and complete final move selection: weights, fallback,
//! reduced-weight/LCB adjustments, and temperature sampling.

use super::{
    SearchError, SearchResult, SearchValueTarget, SearchWorker,
    selection_policy::{exploration_scaling, selection_value},
};
use crate::{
    game::{
        board::{Board, Loc, Player},
        game_state::GameState,
    },
    inference::policy::loc_to_policy,
    search::{move_selection, node::SearchNode, node_store::NodeStore, root_policy},
};
use rand::Rng;

impl<N: NodeStore> SearchWorker<N> {
    pub(super) fn build_search_result<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Result<SearchResult, SearchError> {
        // KataGo's self-play path includes LCB in the recorded policy target,
        // but disables it while sampling the move that will actually be played.
        let (moves, mut policy_weights, reference_weight) = self.root_base_weights();
        // Split before LCB, fallback, and pruning: both paths retain their
        // original operation order. With no children, LCB has no effect.
        let mut move_weights = (self.params.use_lcb_for_selection && reference_weight.is_some())
            .then(|| policy_weights.clone());
        self.finish_root_weights(
            &moves,
            &mut policy_weights,
            reference_weight,
            self.params.use_lcb_for_selection,
        )?;
        if let Some(weights) = move_weights.as_mut() {
            self.finish_root_weights(&moves, weights, reference_weight, false)?;
        }
        let game_state = &self
            .search_graph
            .root
            .as_ref()
            .expect("search result requires an active game")
            .game_state;
        let temperature = move_selection::temperature(
            game_state.turn_number(),
            game_state.board().size(),
            self.params.chosen_move_temperature_early,
            self.params.chosen_move_temperature,
            self.params.chosen_move_temperature_halflife,
        );
        let selected_move = moves[move_selection::sample_index(
            move_weights.as_deref().unwrap_or(&policy_weights),
            temperature,
            rng,
        )];
        let policy_target = normalized_policy_target(game_state.board(), &moves, &policy_weights);
        let root = self
            .search_graph
            .root
            .as_ref()
            .expect("search result requires an active game");
        let value_target = search_value_target(&root.node, root.game_state.next_player());
        Ok(SearchResult {
            selected_move,
            policy_target,
            value_target,
        })
    }

    fn root_base_weights(&self) -> (Vec<Loc>, Vec<f64>, Option<f64>) {
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
            let weight = if root.game_state.is_legal(loc) && root.is_allowed_move(loc, self.params)
            {
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
                + 2.0 * f64::from(policy[loc_to_policy(root.game_state.board(), loc)]);
            if goodness > best_goodness {
                best_goodness = goodness;
                best_index = i;
            }
        }

        let reference_weight = (!moves.is_empty()).then(|| weights[best_index]);
        if let Some(reference_weight) = reference_weight {
            let explore_scaling = exploration_scaling(total_child_weight, self.params);
            let best_edge = root.node.edges().nth(best_index).unwrap();
            // SAFETY: all node pointers remain live and there are no mutable borrows.
            let best_child = unsafe { best_edge.child().as_ref() };
            let best_selection = selection_value(
                best_child.white_utility()
                    + self.root_ending_utility_bonus(best_child, best_edge.move_loc()),
                player,
                f64::from(policy[loc_to_policy(root.game_state.board(), best_edge.move_loc())]),
                reference_weight,
                explore_scaling,
            );
            for (i, edge) in root.node.edges().enumerate() {
                // SAFETY: same graph-lifetime and shared-borrow guarantee as above.
                let child = unsafe { edge.child().as_ref() };
                let utility =
                    child.white_utility() + self.root_ending_utility_bonus(child, edge.move_loc());
                let self_utility = match player {
                    Player::White => utility,
                    Player::Black => -utility,
                };
                if i != best_index
                    && root.game_state.is_legal(edge.move_loc())
                    && root.is_allowed_move(edge.move_loc(), self.params)
                {
                    weights[i] = move_selection::reduced_weight(
                        weights[i],
                        self_utility,
                        f64::from(policy[loc_to_policy(root.game_state.board(), edge.move_loc())]),
                        explore_scaling,
                        best_selection,
                    );
                }
            }
        } else {
            // Zero-budget search still has a root evaluation: use its legal policy.
            let board = root.game_state.board();
            for loc in board.locs().chain(std::iter::once(Loc::PASS)) {
                let probability = policy[loc_to_policy(board, loc)];
                if root.game_state.is_legal(loc) && root.is_allowed_move(loc, self.params) {
                    moves.push(loc);
                    weights.push(f64::from(probability));
                }
            }
        }
        (moves, weights, reference_weight)
    }

    fn finish_root_weights(
        &self,
        moves: &[Loc],
        weights: &mut [f64],
        reference_weight: Option<f64>,
        use_lcb: bool,
    ) -> Result<(), SearchError> {
        if use_lcb && let Some(reference_weight) = reference_weight {
            self.adjust_root_weights_by_lcb(weights, reference_weight);
        }
        let root = self
            .search_graph
            .root
            .as_ref()
            .expect("selection requires an active game");
        let policy = root.node.policy_probs();
        if weights.iter().copied().fold(0.0, f64::max) <= 1e-50 {
            for (loc, weight) in moves.iter().zip(weights.iter_mut()) {
                *weight =
                    if root.game_state.is_legal(*loc) && root.is_allowed_move(*loc, self.params) {
                        f64::from(policy[loc_to_policy(root.game_state.board(), *loc)])
                    } else {
                        0.0
                    };
            }
            // Root move filtering can remove all positive policy mass. Like
            // KataGo, report failure if even the policy fallback is weightless.
            if weights.iter().copied().fold(0.0, f64::max) <= 1e-50 {
                return Err(SearchError::NoSelectableMove);
            }
        }
        move_selection::prune_weights(
            weights,
            self.params.chosen_move_subtract,
            self.params.chosen_move_prune,
        );
        Ok(())
    }

    pub(super) fn apply_root_policy_temperature_and_noise<R: Rng + ?Sized>(
        &mut self,
        game_state: &GameState,
        rng: &mut R,
    ) {
        let params = self.params;
        let temperature = move_selection::temperature(
            game_state.turn_number(),
            game_state.board().size(),
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
        let legal = &legal[..policy.len()];
        // KataGo shapes the policy before constructing the noise distribution.
        root_policy::apply_temperature(policy, legal, temperature);
        if params.root_noise_enabled {
            root_policy::add_dirichlet_noise(
                policy,
                legal,
                params.root_dirichlet_noise_total_concentration,
                params.root_dirichlet_noise_weight,
                rng,
            );
        }
    }

    /// Apply LCB after exploration-weight reduction, using the reference weight
    /// saved before reduction. Weights follow root edge iteration order.
    pub(super) fn adjust_root_weights_by_lcb(&self, weights: &mut [f64], reference_weight: f64) {
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
            let (mut lcb, radius) = move_selection::lcb_and_radius(
                self_utility,
                child.white_utility_mean_sq(),
                child.weight_sum() * fraction,
                child.weight_sq_sum() * fraction,
                utility_radius,
                self.params.lcb_stdevs,
            );
            // Shift the LCB mean, not its variance: the bonus is deterministic.
            if child.weight_sum() * fraction > 0.0 && child.weight_sq_sum() * fraction > 0.0 {
                let bonus = self.root_ending_utility_bonus(child, edge.move_loc());
                lcb += match player {
                    Player::White => bonus,
                    Player::Black => -bonus,
                };
            }
            lcbs.push((lcb, radius));
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
}

fn normalized_policy_target(
    board: &Board,
    moves: &[Loc],
    weights: &[f64],
) -> [f32; crate::inference::policy::MAX_POLICY_SIZE] {
    debug_assert_eq!(moves.len(), weights.len());
    let weight_sum: f64 = weights.iter().sum();
    debug_assert!(weight_sum > 0.0);
    let mut target = [0.0; crate::inference::policy::MAX_POLICY_SIZE];
    for (&move_loc, &weight) in moves.iter().zip(weights) {
        target[loc_to_policy(board, move_loc)] = (weight / weight_sum) as f32;
    }
    target
}

fn search_value_target(node: &SearchNode, player: Player) -> SearchValueTarget {
    let white_win_probability = node.white_win_rate();
    let white_score_mean = node.white_score_mean();
    let score_variance =
        (node.white_score_mean_sq() - white_score_mean * white_score_mean).max(0.0);
    let (win_probability, score_mean) = match player {
        Player::White => (white_win_probability, white_score_mean),
        Player::Black => (1.0 - white_win_probability, -white_score_mean),
    };
    SearchValueTarget {
        win_probability: win_probability as f32,
        score_mean: score_mean as f32,
        score_stdev: score_variance.sqrt() as f32,
    }
}

#[cfg(test)]
impl<N: NodeStore> SearchWorker<N> {
    // Final weights exposed to the policy tests. Production computes the base
    // once and finishes the training and sampling weights separately.
    pub(super) fn root_selection_weights(&self) -> Result<(Vec<Loc>, Vec<f64>), SearchError> {
        let (moves, mut weights, reference_weight) = self.root_base_weights();
        self.finish_root_weights(
            &moves,
            &mut weights,
            reference_weight,
            self.params.use_lcb_for_selection,
        )?;
        Ok((moves, weights))
    }
}
