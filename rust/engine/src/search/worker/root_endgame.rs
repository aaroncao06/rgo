//! Root-only area-scoring heuristics from KataGo's searchhelpers.cpp.
//! Neither heuristic changes game legality, neural policy, or backed-up values.

use super::*;
use crate::search::utility::score_utility_diff;

impl SearchRoot {
    pub(super) fn is_allowed_move(&self, loc: Loc, params: SearchParams) -> bool {
        !params.root_prune_useless_moves
            || loc == Loc::PASS
            || !self.game_state.opponent_passed_last_four_turns()
            || self.safe_area[loc.index()] == Color::Empty
    }

    pub(super) fn ending_white_score_bonus(&self, loc: Loc, params: SearchParams) -> f64 {
        let board = self.game_state.board();
        if params.root_ending_bonus_points == 0.0
            || loc == Loc::PASS
            || loc == Loc::NULL
            || board.simple_ko().is_some()
        {
            return 0.0;
        }
        let Some(ownership) = self.node.nn_output().white_ownership() else {
            return 0.0;
        };
        let player = self.game_state.next_player();
        let white_ownership = f64::from(ownership[loc_to_policy(board, loc)]);
        let player_ownership = match player {
            Player::White => white_ownership,
            Player::Black => -white_ownership,
        };
        // Leave captures, cleanup next to opponents, and unsettled connections
        // alone. Only discourage confidently pointless territory filling.
        let excess = if player_ownership <= -0.95 && !board.would_capture(loc, player) {
            -0.95 - player_ownership
        } else if player_ownership >= 0.95
            && !board.is_adjacent_to_player(loc, player.opponent())
            && !board.is_non_pass_alive_self_connection(loc, player, &self.safe_area)
        {
            player_ownership - 0.95
        } else {
            0.0
        };
        let player_bonus = -params.root_ending_bonus_points * excess / 0.05;
        match player {
            Player::White => player_bonus,
            Player::Black => -player_bonus,
        }
    }
}

impl<N: NodeStore> SearchWorker<N> {
    pub(super) fn root_ending_utility_bonus(&self, child: &SearchNode, loc: Loc) -> f64 {
        let root = self
            .search_graph
            .root
            .as_ref()
            .expect("root selection requires a graph");
        score_utility_diff(
            child.white_score_mean(),
            child.white_score_mean_sq(),
            root.ending_white_score_bonus(loc, self.params),
            self.recent_score_center,
            self.params,
            root.game_state.board().dim(),
        )
    }
}
