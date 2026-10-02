//! Complete tree-descent policy: child scanning, PUCT, FPU, and forced root visits.

use super::SearchWorker;
use crate::{
    game::{
        board::{Loc, Player},
        game_state::GameState,
    },
    inference::policy::{MAX_POLICY_SIZE, loc_to_policy},
    search::{
        node::{EdgeIndex, SearchNode},
        node_store::NodeStore,
        params::SearchParams,
        utility::white_utility,
    },
};

impl<N: NodeStore> SearchWorker<N> {
    pub(super) fn select_child(
        &self,
        node: &SearchNode,
        game_state: &GameState,
        is_root: bool,
    ) -> (Loc, Option<EdgeIndex>) {
        let root = is_root.then(|| {
            self.search_graph
                .root
                .as_ref()
                .expect("root selection requires a graph")
        });
        // KataGo increases exploration slowly as the node accumulates visits.
        // SAFETY: selection runs while all graph nodes are live and none of the
        // children are mutably borrowed. The store cannot be cleared during it.
        let total_child_weight = unsafe { node.child_weight_sum() };
        let exploration_scaling = exploration_scaling(total_child_weight, self.params);

        let fpu = self.first_play_urgency(node, game_state, is_root);

        // Score existing children in insertion order. Strict comparisons retain
        // the first child on ties, including ties between forced root visits.
        let mut best_move = Loc::NULL;
        let mut best_edge = None;
        let mut best_selection_value = f64::NEG_INFINITY;
        let policy_probs = node.policy_probs();
        let mut expanded = [false; MAX_POLICY_SIZE];
        for (edge_index, edge) in node.indexed_edges() {
            let move_loc = edge.move_loc();
            let policy_index = loc_to_policy(game_state.board(), move_loc);
            expanded[policy_index] = true;
            if !game_state.is_legal(move_loc)
                || root.is_some_and(|root| !root.is_allowed_move(move_loc, self.params))
            {
                continue;
            }

            let policy_probability = f64::from(policy_probs[policy_index]);
            let child = unsafe { edge.child().as_ref() };
            // SAFETY: the same graph-lifetime guarantee as above applies here.
            let child_weight = unsafe { edge.child_weight() };
            let force_root_visit = is_root
                && force_root_visit(
                    child_weight,
                    policy_probability,
                    total_child_weight,
                    self.params,
                );
            let selection_value = if force_root_visit {
                1e20 // KataGo's forced-visit selection value.
            } else {
                selection_value(
                    child.white_utility()
                        + if is_root {
                            self.root_ending_utility_bonus(child, move_loc)
                        } else {
                            0.0
                        },
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
        let board = game_state.board();
        for move_loc in board.locs().chain(std::iter::once(Loc::PASS)) {
            let policy_index = loc_to_policy(board, move_loc);
            if expanded[policy_index] {
                continue;
            }
            let probability = policy_probs[policy_index];
            if game_state.is_legal(move_loc)
                && probability > best_new_policy
                && root.is_none_or(|root| root.is_allowed_move(move_loc, self.params))
            {
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

    fn first_play_urgency(&self, node: &SearchNode, game_state: &GameState, is_root: bool) -> f64 {
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
            game_state.board().size(),
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
        match game_state.next_player() {
            Player::White => parent_utility_for_fpu - fpu_reduction,
            Player::Black => parent_utility_for_fpu + fpu_reduction,
        }
    }
}

pub(super) fn exploration_scaling(total_child_weight: f64, params: SearchParams) -> f64 {
    let cpuct = params.cpuct_exploration
        + params.cpuct_exploration_log
            * ((total_child_weight + params.cpuct_exploration_base)
                / params.cpuct_exploration_base)
                .ln();
    cpuct * (total_child_weight + 0.01).sqrt()
}

fn force_root_visit(
    child_weight: f64,
    policy_probability: f64,
    total_child_weight: f64,
    params: SearchParams,
) -> bool {
    params.root_desired_per_child_visits_coeff > 0.0
        && policy_probability > 0.0
        && child_weight
            < (policy_probability * total_child_weight * params.root_desired_per_child_visits_coeff)
                .sqrt()
}

pub(super) fn selection_value(
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
