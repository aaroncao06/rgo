use crate::game::board::BOARD_STORAGE_LEN;

use super::board::{Board, Color, Loc, Player};
use super::hash::{Hash128, pass_hash, player_hash, superko_hash};
use super::rules::Rules;

use std::collections::HashSet;

#[derive(Clone)]
pub struct GameState {
    board: Board,
    rules: Rules,
    next_player: Player,
    turn_number: usize,
    consecutive_ending_passes: u8,
    // Each player's consecutive passes on their own turns, capped at four.
    // Root pruning needs this, not just consecutive passes by either player.
    passes_by_player: [u8; 2],
    superko_banned: [bool; BOARD_STORAGE_LEN],
    seen_position_hashes: HashSet<Hash128>,
}

impl GameState {
    pub fn new(rules: Rules) -> Self {
        let board = Board::new(rules.board_dim);
        let mut seen_position_hashes = HashSet::new();
        seen_position_hashes.insert(board.position_hash());
        let superko_banned = [false; BOARD_STORAGE_LEN];
        Self {
            board,
            rules,
            next_player: Player::Black,
            turn_number: 0,
            consecutive_ending_passes: 0,
            passes_by_player: [0; 2],
            superko_banned,
            seen_position_hashes,
        }
    }
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Starts a fresh game while retaining reusable history allocation.
    pub fn reset(&mut self, rules: Rules) {
        self.board = Board::new(rules.board_dim);
        self.rules = rules;
        self.next_player = Player::Black;
        self.turn_number = 0;
        self.consecutive_ending_passes = 0;
        self.passes_by_player = [0; 2];
        self.superko_banned.fill(false);

        self.seen_position_hashes.clear();
        self.seen_position_hashes.insert(self.board.position_hash());
    }

    /// Restores this state from `source` while retaining the hash-set's
    /// allocated capacity for reuse across playouts.
    pub(crate) fn reset_from(&mut self, source: &GameState) {
        self.board.clone_from(&source.board);
        self.rules = source.rules;
        self.next_player = source.next_player;
        self.turn_number = source.turn_number;
        self.consecutive_ending_passes = source.consecutive_ending_passes;
        self.passes_by_player = source.passes_by_player;
        self.superko_banned = source.superko_banned;

        self.seen_position_hashes.clear();
        self.seen_position_hashes
            .extend(source.seen_position_hashes.iter().copied());
    }
    pub fn rules(&self) -> &Rules {
        &self.rules
    }
    pub fn next_player(&self) -> Player {
        self.next_player
    }
    /// Board size, stones, player to move, superko bans, and consecutive passes.
    /// This common key component excludes komi, the suicide rule, and repetition history
    /// beyond the current ban mask. It is not a complete graph identity.
    pub fn current_state_hash(&self) -> Hash128 {
        let mut key = self.board.position_hash();
        key ^= player_hash(self.next_player);
        for loc in self.board.locs() {
            if self.is_superko_banned(loc) {
                key ^= superko_hash(loc);
            }
        }
        key ^= pass_hash(self.consecutive_ending_passes);
        key
    }
    /// Number of successfully played moves, including passes. Starts at zero.
    pub fn turn_number(&self) -> usize {
        self.turn_number
    }
    pub fn consecutive_ending_passes(&self) -> u8 {
        self.consecutive_ending_passes
    }
    pub(crate) fn opponent_passed_last_four_turns(&self) -> bool {
        let opponent_index = match self.next_player {
            Player::Black => 1,
            Player::White => 0,
        };
        self.passes_by_player[opponent_index] >= 4
    }
    pub(crate) fn is_superko_banned(&self, loc: Loc) -> bool {
        debug_assert!(self.board.is_on_board(loc));
        self.superko_banned[loc.index()]
    }
    fn rebuild_superko_banned(&mut self) {
        // Maintain separately from the legal mask.
        for loc in self.board.locs() {
            self.superko_banned[loc.index()] = self.board.is_legal_ignoring_ko(
                loc,
                self.next_player,
                self.rules.multi_stone_suicide_legal,
            ) && self.seen_position_hashes.contains(
                &self
                    .board
                    .get_position_hash_after_move(loc, self.next_player),
            );
        }
    }
    pub fn is_legal(&self, loc: Loc) -> bool {
        if !self.board.is_legal_ignoring_ko(
            loc,
            self.next_player,
            self.rules.multi_stone_suicide_legal,
        ) {
            return false;
        }
        loc == Loc::PASS || !self.superko_banned[loc.index()]
    }
    pub fn is_finished(&self) -> bool {
        self.consecutive_ending_passes >= 2
    }
    pub fn play(&mut self, loc: Loc) -> bool {
        if !self.is_legal(loc) {
            return false;
        }
        self.board.play_move_assume_legal(loc, self.next_player);
        let player_index = match self.next_player {
            Player::Black => 0,
            Player::White => 1,
        };
        self.passes_by_player[player_index] = if loc == Loc::PASS {
            (self.passes_by_player[player_index] + 1).min(4)
        } else {
            0
        };
        if loc == Loc::PASS {
            self.consecutive_ending_passes += 1;
        } else {
            self.consecutive_ending_passes = 0;
            self.seen_position_hashes.insert(self.board.position_hash());
        }
        self.next_player = self.next_player.opponent();
        self.turn_number += 1;
        // recompute mask
        self.rebuild_superko_banned();
        true
    }
    fn count_area_score_white_minus_black(&self) -> i16 {
        let area = self
            .board
            .calculate_area(self.rules.multi_stone_suicide_legal);
        let mut score = 0_i16;
        for loc in self.board.locs() {
            match area[loc.index()] {
                Color::White => score += 1,
                Color::Black => score -= 1,
                Color::Empty | Color::Wall => {}
            }
        }
        score
    }
    pub fn final_score_white_minus_black(&self) -> f32 {
        self.count_area_score_white_minus_black() as f32 + self.rules.komi
    }

    /// Final area ownership on the board's padded coordinate array.
    pub fn final_ownership(&self) -> [Color; BOARD_STORAGE_LEN] {
        debug_assert!(self.is_finished(), "cannot score an unfinished game");
        self.board
            .calculate_area(self.rules.multi_stone_suicide_legal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> Rules {
        Rules {
            board_dim: 9,
            komi: 7.5,
            multi_stone_suicide_legal: true,
        }
    }

    fn loc(x: usize, y: usize) -> Loc {
        crate::game::board::Board::new(9)
            .loc(x, y)
            .expect("test coordinates must be on the board")
    }

    #[test]
    fn a_location_from_a_larger_board_is_rejected_without_changing_state() {
        let mut state = GameState::new(Rules {
            board_dim: 5,
            ..rules()
        });
        let outside = Board::new(9).loc(5, 0).unwrap();
        let initial_hash = state.current_state_hash();
        assert!(!state.board.is_on_board(outside));
        assert_eq!(state.board.coords(outside), None);
        assert_eq!(state.board.color_at(outside), Color::Wall);
        assert!(!state.is_legal(outside));
        assert!(!state.play(outside));
        assert_eq!(state.current_state_hash(), initial_hash);
        assert_eq!(state.turn_number(), 0);
    }

    #[test]
    fn reset_and_reset_from_preserve_runtime_geometry_and_history() {
        let mut state = GameState::new(rules());
        let outside = state.board.loc(8, 8).unwrap();
        assert!(state.play(outside));
        let smaller_rules = Rules {
            board_dim: 3,
            ..rules()
        };
        state.reset(smaller_rules);
        assert_eq!(state.board.dim(), 3);
        assert_eq!(state.rules().board_dim, 3);
        assert!(!state.is_legal(outside));
        assert_eq!(state.turn_number(), 0);
        assert_eq!(state.seen_position_hashes.len(), 1);
        let corner = state.board.loc(0, 0).unwrap();
        assert!(state.play(corner));
        assert!(state.play(Loc::PASS));

        let mut scratch = GameState::new(rules());
        scratch.reset_from(&state);
        assert_eq!(scratch.board.dim(), 3);
        assert_eq!(scratch.rules().board_dim, 3);
        assert_eq!(scratch.current_state_hash(), state.current_state_hash());
        assert_eq!(scratch.seen_position_hashes, state.seen_position_hashes);
        assert!(!scratch.is_legal(outside));
        assert!(scratch.play(Loc::PASS));
        assert!(scratch.is_finished());
        assert_eq!(
            scratch.final_score_white_minus_black(),
            -9.0 + smaller_rules.komi
        );
    }

    #[test]
    fn positional_superko_applies_on_a_smaller_board() {
        let mut state = GameState::new(Rules {
            board_dim: 4,
            ..rules()
        });
        for (x, y) in [
            (1, 0),
            (1, 1),
            (0, 1),
            (1, 3),
            (2, 1),
            (0, 2),
            (3, 0),
            (2, 2),
            (1, 2),
        ] {
            let point = state.board.loc(x, y).unwrap();
            assert!(state.play(point));
        }
        let recapture = state.board.loc(1, 1).unwrap();
        assert!(
            state
                .board
                .is_legal_ignoring_ko(recapture, Player::White, true)
        );
        assert!(!state.is_legal(recapture));
    }

    #[test]
    fn black_moves_first_and_a_successful_move_flips_turn() {
        let mut history = GameState::new(rules());

        assert_eq!(history.next_player, Player::Black);
        assert!(history.play(loc(4, 4)));
        assert_eq!(history.next_player, Player::White);
        assert!(!history.is_legal(loc(4, 4)));
    }

    #[test]
    fn reset_from_restores_the_complete_game_state() {
        let mut source = GameState::new(rules());
        assert!(source.play(loc(4, 4)));
        assert!(source.play(Loc::PASS));

        let mut scratch = GameState::new(Rules::OGS_CHINESE_9);
        assert!(scratch.play(loc(0, 0)));
        scratch.reset_from(&source);

        assert_eq!(scratch.board.position_hash(), source.board.position_hash());
        assert_eq!(scratch.rules.komi, source.rules.komi);
        assert_eq!(
            scratch.rules.multi_stone_suicide_legal,
            source.rules.multi_stone_suicide_legal
        );
        assert_eq!(scratch.next_player, source.next_player);
        assert_eq!(scratch.turn_number(), source.turn_number());
        assert_eq!(
            scratch.consecutive_ending_passes,
            source.consecutive_ending_passes
        );
        assert_eq!(scratch.superko_banned, source.superko_banned);
        assert_eq!(scratch.seen_position_hashes, source.seen_position_hashes);
    }

    #[test]
    fn reset_starts_a_fresh_game_and_reuses_history_allocation() {
        let mut state = GameState::new(Rules::OGS_CHINESE_9);
        assert!(state.play(loc(4, 4)));
        assert!(state.play(loc(3, 3)));
        assert!(state.play(Loc::PASS));
        let history_capacity = state.seen_position_hashes.capacity();

        state.reset(rules());

        assert_eq!(
            state.board.position_hash(),
            Board::new(rules().board_dim).position_hash()
        );
        assert_eq!(state.rules.komi, rules().komi);
        assert_eq!(
            state.rules.multi_stone_suicide_legal,
            rules().multi_stone_suicide_legal
        );
        assert_eq!(state.next_player, Player::Black);
        assert_eq!(state.turn_number, 0);
        assert_eq!(state.consecutive_ending_passes, 0);
        assert_eq!(state.passes_by_player, [0; 2]);
        assert!(state.superko_banned.iter().all(|&banned| !banned));
        assert_eq!(state.seen_position_hashes.len(), 1);
        assert!(
            state
                .seen_position_hashes
                .contains(&state.board.position_hash())
        );
        assert_eq!(state.seen_position_hashes.capacity(), history_capacity);
    }

    #[test]
    fn turn_number_counts_successful_moves_and_passes_only() {
        let mut state = GameState::new(rules());
        assert_eq!(state.turn_number(), 0);
        assert!(state.play(loc(4, 4)));
        assert_eq!(state.turn_number(), 1);
        assert!(!state.play(loc(4, 4)));
        assert_eq!(state.turn_number(), 1);
        assert!(state.play(Loc::PASS));
        assert_eq!(state.turn_number(), 2);

        let mut cloned = state.clone();
        assert_eq!(cloned.turn_number(), 2);
        assert!(cloned.play(Loc::PASS));
        assert_eq!(cloned.turn_number(), 3);
        assert_eq!(state.turn_number(), 2);
        cloned.reset_from(&GameState::new(rules()));
        assert_eq!(cloned.turn_number(), 0);
    }

    #[test]
    fn opponent_pass_streak_tracks_own_turns_and_survives_reset() {
        let mut state = GameState::new(rules());
        for x in 0..4 {
            assert!(state.play(loc(x, 0)));
            assert!(state.play(Loc::PASS));
            assert_eq!(state.opponent_passed_last_four_turns(), x == 3);
            assert_eq!(state.consecutive_ending_passes(), 1);
        }
        let mut scratch = GameState::new(rules());
        scratch.reset_from(&state);
        assert!(scratch.opponent_passed_last_four_turns());
        assert!(state.clone().opponent_passed_last_four_turns());
        assert!(!scratch.play(loc(0, 0))); // Illegal moves do not reset the streak.
        assert!(scratch.opponent_passed_last_four_turns());
        assert!(scratch.play(loc(4, 0)));
        assert!(!scratch.opponent_passed_last_four_turns()); // White sees Black's streak.
        assert!(scratch.play(loc(8, 8))); // White stops passing.
        assert!(!scratch.opponent_passed_last_four_turns());
        scratch.reset_from(&GameState::new(rules()));
        assert!(!scratch.opponent_passed_last_four_turns());
    }

    #[test]
    fn two_passes_finish_the_game() {
        let mut history = GameState::new(rules());

        assert!(history.play(Loc::PASS));
        assert!(!history.is_finished());
        assert!(history.play(Loc::PASS));
        assert!(history.is_finished());
        assert_eq!(history.count_area_score_white_minus_black(), 0);
        assert_eq!(history.final_score_white_minus_black(), 7.5);
    }

    #[test]
    fn area_score_counts_a_two_eye_group_and_neutral_exterior() {
        let mut history = GameState::new(rules());

        // Construct the position directly: a pass-alive Black group with two
        // eyes, and one distant White stone in the exterior region.
        for (x, y) in [
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 3),
            (3, 4),
            (5, 4),
            (7, 4),
            (3, 5),
            (4, 5),
            (5, 5),
            (6, 5),
            (7, 5),
        ] {
            history
                .board
                .play_move_assume_legal(loc(x, y), Player::Black);
        }
        history
            .board
            .play_move_assume_legal(loc(0, 0), Player::White);

        // Black has 13 stones and two eyes; White has one stone.
        assert_eq!(history.count_area_score_white_minus_black(), -14);
        assert_eq!(history.final_score_white_minus_black(), -6.5);
    }

    #[test]
    fn stone_move_resets_consecutive_passes() {
        let mut history = GameState::new(rules());

        assert!(history.play(Loc::PASS));
        assert_eq!(history.consecutive_ending_passes, 1);
        assert!(history.play(loc(4, 4)));
        assert_eq!(history.consecutive_ending_passes, 0);
        assert!(!history.is_finished());
    }

    #[test]
    fn illegal_move_does_not_change_history_state() {
        let mut history = GameState::new(rules());
        let point = loc(4, 4);

        assert!(history.play(point));
        assert_eq!(history.next_player, Player::White);
        assert!(!history.play(point));
        assert_eq!(history.next_player, Player::White);
        assert_eq!(history.consecutive_ending_passes, 0);
    }

    #[test]
    fn positional_superko_rejects_immediate_ko_recapture() {
        let mut history = GameState::new(rules());
        let capture = loc(4, 5);
        let recapture = loc(4, 4);

        for move_loc in [
            loc(4, 3),
            loc(4, 4),
            loc(3, 4),
            loc(4, 6),
            loc(5, 4),
            loc(3, 5),
            loc(0, 0),
            loc(5, 5),
            capture,
        ] {
            assert!(history.play(move_loc));
        }

        assert_eq!(history.next_player, Player::White);
        assert!(history.board.is_legal_ignoring_ko(
            recapture,
            Player::White,
            history.rules.multi_stone_suicide_legal,
        ));
        assert!(!history.is_legal(recapture));
    }

    #[test]
    fn positional_superko_rejects_recapture_after_intervening_passes() {
        let mut history = GameState::new(rules());
        let capture = loc(4, 5);
        let recapture = loc(4, 4);

        for move_loc in [
            loc(4, 3),
            recapture,
            loc(3, 4),
            loc(4, 6),
            loc(5, 4),
            loc(3, 5),
            loc(0, 0),
            loc(5, 5),
            capture,
        ] {
            assert!(history.play(move_loc));
        }

        assert!(history.play(Loc::PASS));
        assert!(history.play(Loc::PASS));
        assert_eq!(history.next_player, Player::White);

        assert!(history.board.is_legal_ignoring_ko(
            recapture,
            history.next_player,
            history.rules.multi_stone_suicide_legal,
        ));
        assert!(history.superko_banned[recapture.index()]);
        assert!(!history.is_legal(recapture));
    }

    #[test]
    fn cached_superko_mask_matches_direct_legality_after_a_ko_capture() {
        let mut history = GameState::new(rules());

        for move_loc in [
            loc(4, 3),
            loc(4, 4),
            loc(3, 4),
            loc(4, 6),
            loc(5, 4),
            loc(3, 5),
            loc(0, 0),
            loc(5, 5),
            loc(4, 5),
        ] {
            assert!(history.play(move_loc));
        }

        for loc in history.board.locs() {
            let locally_legal = history.board.is_legal_ignoring_ko(
                loc,
                history.next_player,
                history.rules.multi_stone_suicide_legal,
            );
            let directly_superko_banned = locally_legal
                && history.seen_position_hashes.contains(
                    &history
                        .board
                        .get_position_hash_after_move(loc, history.next_player),
                );

            assert_eq!(history.superko_banned[loc.index()], directly_superko_banned);
            assert_eq!(
                history.is_legal(loc),
                locally_legal && !directly_superko_banned
            );
        }
    }
}
