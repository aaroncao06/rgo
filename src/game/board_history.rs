use super::board::{Board, Loc, Player};
use super::hash::PositionHash;
use super::rules::Rules;

use std::collections::HashSet;

struct BoardHistory {
    board: Board,
    rules: Rules,
    next_player: Player,
    consecutive_ending_passes: u8,
    seen_position_hashes: HashSet<PositionHash>,
}

impl BoardHistory {
    fn new(rules: Rules) -> Self {
        let board = Board::new();
        let mut seen_position_hashes = HashSet::new();
        seen_position_hashes.insert(board.position_hash());
        Self {
            board,
            rules,
            next_player: Player::Black,
            consecutive_ending_passes: 0,
            seen_position_hashes,
        }
    }
    fn is_legal(&self, loc: Loc) -> bool {
        if !self.board.is_legal_ignoring_ko(
            loc,
            self.next_player,
            self.rules.multi_stone_suicide_legal,
        ) {
            return false;
        }
        if loc == Loc::PASS {
            return true;
        }
        !self.seen_position_hashes.contains(
            &self
                .board
                .get_position_hash_after_move(loc, self.next_player),
        )
    }

    fn is_finished(&self) -> bool {
        self.consecutive_ending_passes >= 2
    }
    fn play(&mut self, loc: Loc) -> bool {
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
        true
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
        let mut history = BoardHistory::new(rules());

        assert_eq!(history.next_player, Player::Black);
        assert!(history.play(loc(4, 4)));
        assert_eq!(history.next_player, Player::White);
        assert!(!history.is_legal(loc(4, 4)));
    }

    #[test]
    fn two_passes_finish_the_game() {
        let mut history = BoardHistory::new(rules());

        assert!(history.play(Loc::PASS));
        assert!(!history.is_finished());
        assert!(history.play(Loc::PASS));
        assert!(history.is_finished());
    }

    #[test]
    fn stone_move_resets_consecutive_passes() {
        let mut history = BoardHistory::new(rules());

        assert!(history.play(Loc::PASS));
        assert_eq!(history.consecutive_ending_passes, 1);
        assert!(history.play(loc(4, 4)));
        assert_eq!(history.consecutive_ending_passes, 0);
        assert!(!history.is_finished());
    }

    #[test]
    fn illegal_move_does_not_change_history_state() {
        let mut history = BoardHistory::new(rules());
        let point = loc(4, 4);

        assert!(history.play(point));
        assert_eq!(history.next_player, Player::White);
        assert!(!history.play(point));
        assert_eq!(history.next_player, Player::White);
        assert_eq!(history.consecutive_ending_passes, 0);
    }

    #[test]
    fn positional_superko_rejects_immediate_ko_recapture() {
        let mut history = BoardHistory::new(rules());
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
}
