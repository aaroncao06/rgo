use crate::game::board::{Board, Loc, MAX_BOARD_AREA, MAX_BOARD_DIM};
use crate::game::game_state::GameState;

/// Capacity for fixed policy scratch arrays; actual policies have board_dim² + 1 entries.
pub const MAX_POLICY_SIZE: usize = MAX_BOARD_AREA + 1;

/// Convert an active-board location or pass to the compact policy.
pub fn loc_to_policy(board: &Board, loc: Loc) -> usize {
    // Map a Loc action to its index in the dense NN policy.
    debug_assert!(loc != Loc::NULL);
    if loc == Loc::PASS {
        return board.dim() * board.dim();
    }
    let (x, y) = board.coords_assume_on_board(loc);
    x + y * board.dim()
}

/// Convert an active-board policy index or pass to a location.
pub fn policy_to_loc(board: &Board, i: usize) -> Loc {
    let dim = board.dim();
    debug_assert!(i <= dim * dim);
    if i == dim * dim {
        return Loc::PASS;
    }
    board.loc_assume_on_board(i % dim, i / dim)
}

/// Index an active point in a fixed-capacity spatial plane.
pub fn loc_to_spatial(board: &Board, loc: Loc) -> usize {
    let (x, y) = board.coords_assume_on_board(loc);
    x + y * MAX_BOARD_DIM
}

/// Active rows of a fixed-capacity plane, without the trailing storage columns.
pub fn active_rows<T>(plane: &[T; MAX_BOARD_AREA], board_dim: usize) -> impl Iterator<Item = &[T]> {
    debug_assert!((1..=MAX_BOARD_DIM).contains(&board_dim));
    plane
        .as_chunks::<MAX_BOARD_DIM>()
        .0
        .iter()
        .take(board_dim)
        .map(move |row| &row[..board_dim])
}

/// Compact policy mask in a fixed-capacity scratch array; the unused tail is false.
pub fn legal_mask(game_state: &GameState) -> [bool; MAX_POLICY_SIZE] {
    let mut mask = [false; MAX_POLICY_SIZE];
    let board = game_state.board();
    for loc in board.locs().chain(std::iter::once(Loc::PASS)) {
        mask[loc_to_policy(board, loc)] = game_state.is_legal(loc);
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::rules::Rules;

    #[test]
    fn corner_policy_indices_are_row_major() {
        let board = Board::new(MAX_BOARD_DIM);
        assert_eq!(loc_to_policy(&board, board.loc(0, 0).unwrap()), 0);
        assert_eq!(loc_to_policy(&board, board.loc(8, 0).unwrap()), 8);
        assert_eq!(loc_to_policy(&board, board.loc(0, 8).unwrap()), 72);
        assert_eq!(loc_to_policy(&board, board.loc(8, 8).unwrap()), 80);
    }

    #[test]
    fn every_policy_slot_round_trips_to_its_location() {
        for dim in 1..=MAX_BOARD_DIM {
            let board = Board::new(dim);
            for i in 0..=dim * dim {
                assert_eq!(loc_to_policy(&board, policy_to_loc(&board, i)), i);
            }
        }
    }

    #[test]
    fn compact_policy_uses_the_active_stride_and_masks_the_unused_scratch_tail() {
        let state = GameState::new(Rules {
            board_dim: 5,
            ..Rules::default()
        });
        let board = state.board();
        let mask = legal_mask(&state);
        for (i, &legal) in mask[..25].iter().enumerate() {
            assert!(legal);
            let point = policy_to_loc(board, i);
            assert!(board.is_on_board(point));
            assert_eq!(loc_to_policy(board, point), i);
            assert_eq!(board.coords(point), Some((i % 5, i / 5)));
        }
        assert!(mask[25]);
        assert!(mask[26..].iter().all(|&legal| !legal));
        assert_eq!(policy_to_loc(board, 25), Loc::PASS);
        let corner = board.loc(4, 4).unwrap();
        assert_eq!(loc_to_policy(board, corner), 24);
        assert_eq!(loc_to_spatial(board, corner), 4 + 4 * MAX_BOARD_DIM);
    }

    #[test]
    fn empty_board_legal_mask_allows_every_policy_slot() {
        let game_state = GameState::new(Rules {
            board_dim: 9,
            komi: 7.5,
            multi_stone_suicide_legal: true,
        });

        assert!(legal_mask(&game_state).into_iter().all(|legal| legal));
    }
}
