use super::board::{Board, Loc, Player};
use super::rules::Rules;

struct BoardHistory {
    board: Board,
    rules: Rules,
    next_player: Player,
    consecutive_ending_passes: u8,
}

impl BoardHistory {
    fn new(rules: Rules) -> Self {
        Self {
            board: Board::new(),
            rules,
            next_player: Player::Black,
            consecutive_ending_passes: 0,
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
        //check against history hash

        true
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
}
