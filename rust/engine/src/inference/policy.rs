use crate::game::board::{BOARD_SIZE, Loc, STRIDE};
use crate::game::game_state::GameState;

pub const BOARD_POLICY_SIZE: usize = BOARD_SIZE * BOARD_SIZE;
pub const PASS_POLICY_INDEX: usize = BOARD_POLICY_SIZE;
pub const POLICY_SIZE: usize = BOARD_POLICY_SIZE + 1;

pub fn loc_to_policy(loc: Loc) -> usize {
    // Map a Loc action to its index in the dense NN policy.
    debug_assert!(loc != Loc::NULL);
    if loc == Loc::PASS {
        return PASS_POLICY_INDEX;
    }
    debug_assert!(loc.is_on_board());
    loc.x() + loc.y() * BOARD_SIZE
}

pub fn policy_to_loc(i: usize) -> Loc {
    debug_assert!(i < POLICY_SIZE);
    if i == PASS_POLICY_INDEX {
        return Loc::PASS;
    }
    let x = i % BOARD_SIZE;
    let y = i / BOARD_SIZE;
    Loc::from_index((x + 1) + (y + 1) * STRIDE)
}

pub fn legal_mask(game_state: &GameState) -> [bool; POLICY_SIZE] {
    let mut mask = [false; POLICY_SIZE];
    for i in 0..POLICY_SIZE {
        mask[i] = game_state.is_legal(policy_to_loc(i));
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::rules::Rules;

    #[test]
    fn corner_policy_indices_are_row_major() {
        assert_eq!(loc_to_policy(Loc::new(0, 0).unwrap()), 0);
        assert_eq!(loc_to_policy(Loc::new(BOARD_SIZE - 1, 0).unwrap()), 8);
        assert_eq!(loc_to_policy(Loc::new(0, BOARD_SIZE - 1).unwrap()), 72);
        assert_eq!(
            loc_to_policy(Loc::new(BOARD_SIZE - 1, BOARD_SIZE - 1).unwrap()),
            80
        );
    }

    #[test]
    fn every_policy_slot_round_trips_to_its_location() {
        for i in 0..POLICY_SIZE {
            assert_eq!(loc_to_policy(policy_to_loc(i)), i);
        }
    }

    #[test]
    fn empty_board_legal_mask_allows_every_policy_slot() {
        let game_state = GameState::new(Rules {
            komi: 7.5,
            multi_stone_suicide_legal: true,
        });

        assert!(legal_mask(&game_state).into_iter().all(|legal| legal));
    }
}
