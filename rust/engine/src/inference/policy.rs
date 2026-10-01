use crate::game::board::{Board, Loc};
use crate::game::game_state::GameState;

/// Current model geometry, independent of the engine's storage capacity.
pub const MODEL_BOARD_SIZE: usize = 9;
pub const BOARD_POLICY_SIZE: usize = MODEL_BOARD_SIZE * MODEL_BOARD_SIZE;
pub const PASS_POLICY_INDEX: usize = BOARD_POLICY_SIZE;
pub const POLICY_SIZE: usize = BOARD_POLICY_SIZE + 1;

/// Convert an on-board location or pass after validating model geometry.
pub fn loc_to_policy(board: &Board, loc: Loc) -> usize {
    // Map a Loc action to its index in the dense NN policy.
    debug_assert_eq!(
        board.size(),
        MODEL_BOARD_SIZE,
        "policy geometry must match the board"
    );
    debug_assert!(loc != Loc::NULL);
    if loc == Loc::PASS {
        return PASS_POLICY_INDEX;
    }
    let (x, y) = board.coords_assume_on_board(loc);
    x + y * MODEL_BOARD_SIZE
}

/// Convert a valid policy index after validating model geometry.
pub fn policy_to_loc(board: &Board, i: usize) -> Loc {
    debug_assert_eq!(
        board.size(),
        MODEL_BOARD_SIZE,
        "policy geometry must match the board"
    );
    debug_assert!(i < POLICY_SIZE);
    if i == PASS_POLICY_INDEX {
        return Loc::PASS;
    }
    board.loc_assume_on_board(i % MODEL_BOARD_SIZE, i / MODEL_BOARD_SIZE)
}

pub fn legal_mask(game_state: &GameState) -> [bool; POLICY_SIZE] {
    let mut mask = [false; POLICY_SIZE];
    for (i, allowed) in mask.iter_mut().enumerate() {
        *allowed = game_state.is_legal(policy_to_loc(game_state.board(), i));
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::rules::Rules;

    #[test]
    fn corner_policy_indices_are_row_major() {
        let board = Board::new(MODEL_BOARD_SIZE);
        assert_eq!(loc_to_policy(&board, board.loc(0, 0).unwrap()), 0);
        assert_eq!(loc_to_policy(&board, board.loc(8, 0).unwrap()), 8);
        assert_eq!(loc_to_policy(&board, board.loc(0, 8).unwrap()), 72);
        assert_eq!(loc_to_policy(&board, board.loc(8, 8).unwrap()), 80);
    }

    #[test]
    fn every_policy_slot_round_trips_to_its_location() {
        let board = Board::new(MODEL_BOARD_SIZE);
        for i in 0..POLICY_SIZE {
            assert_eq!(loc_to_policy(&board, policy_to_loc(&board, i)), i);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "policy geometry must match the board")]
    fn incompatible_policy_geometry_is_a_contract_violation() {
        policy_to_loc(&Board::new(5), 0);
    }

    #[test]
    fn empty_board_legal_mask_allows_every_policy_slot() {
        let game_state = GameState::new(Rules {
            board_size: 9,
            komi: 7.5,
            multi_stone_suicide_legal: true,
        });

        assert!(legal_mask(&game_state).into_iter().all(|legal| legal));
    }
}
