use crate::game::board::ARRAY_LEN;

use super::board::{Board, Color, Loc, Player};
use super::hash::PositionHash;
use super::rules::Rules;

use std::collections::HashSet;

pub struct GameState {
    pub board: Board,
    pub rules: Rules,
    pub next_player: Player,
    pub consecutive_ending_passes: u8,
    pub superko_banned: [bool; ARRAY_LEN],
    seen_position_hashes: HashSet<PositionHash>,
}

impl GameState {
    pub fn new(rules: Rules) -> Self {
        let board = Board::new();
        let mut seen_position_hashes = HashSet::new();
        seen_position_hashes.insert(board.position_hash());
        let superko_banned = [false; ARRAY_LEN];
        Self {
            board,
            rules,
            next_player: Player::Black,
            consecutive_ending_passes: 0,
            superko_banned,
            seen_position_hashes,
        }
    }
    fn rebuild_superko_banned(&mut self) {
        // Maintain separately from the legal mask.
        for loc in Loc::board_iter() {
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
    fn is_finished(&self) -> bool {
        self.consecutive_ending_passes >= 2
    }
    pub fn play(&mut self, loc: Loc) -> bool {
        if !self.is_legal(loc) {
            return false;
        }
        self.board.play_move_assume_legal(loc, self.next_player);
        if loc == Loc::PASS {
            self.consecutive_ending_passes += 1;
        } else {
            self.consecutive_ending_passes = 0;
            self.seen_position_hashes.insert(self.board.position_hash());
        }
        self.next_player = self.next_player.opponent();
        // recompute mask
        self.rebuild_superko_banned();
        true
    }
    fn count_area_score_white_minus_black(&self) -> i16 {
        let area = self
            .board
            .calculate_area(self.rules.multi_stone_suicide_legal);
        let mut score = 0_i16;
        for loc in Loc::board_iter() {
            match area[loc.index()] {
                Color::White => score += 1,
                Color::Black => score -= 1,
                Color::Empty | Color::Wall => {}
            }
        }
        score
    }
    fn final_score_white_minus_black(&self) -> f32 {
        self.count_area_score_white_minus_black() as f32 + self.rules.komi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> Rules {
        Rules {
            komi: 7.5,
            multi_stone_suicide_legal: true,
        }
    }

    fn loc(x: usize, y: usize) -> Loc {
        Loc::new(x, y).expect("test coordinates must be on the board")
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

        for loc in Loc::board_iter() {
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
