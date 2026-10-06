//! Move records and completed-game data passed from workers to the chunk writer.

use crate::{
    game::{
        board::{BOARD_STORAGE_LEN, Color, Loc, Player},
        game_state::GameState,
        rules::Rules,
    },
    search::worker::PolicyTarget,
};

pub(super) struct SelfPlayRecord {
    pub(super) player: Player,
    pub(super) selected_move: Loc,
    pub(super) policy_target: PolicyTarget,
}

/// Games currently start empty; rules and moves suffice to reconstruct inputs.
/// Final ownership is calculated once; the writer prepares player-relative labels.
pub(super) struct CompletedGame {
    pub(super) rules: Rules,
    pub(super) records: Vec<SelfPlayRecord>,
    pub(super) final_ownership: [Color; BOARD_STORAGE_LEN],
}

impl CompletedGame {
    pub(super) fn new(game_state: &GameState, records: Vec<SelfPlayRecord>) -> Self {
        debug_assert!(game_state.is_finished(), "cannot submit an unfinished game");
        debug_assert_eq!(records.len(), game_state.turn_number());
        Self {
            rules: *game_state.rules(),
            records,
            final_ownership: game_state.final_ownership(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::policy::MAX_POLICY_SIZE;
    use half::f16;

    #[test]
    fn completed_game_preserves_final_ownership_and_moves_record_storage() {
        for board_dim in [9, 13, 19] {
            let mut state = GameState::new(Rules {
                board_dim,
                ..Rules::TROMP_TAYLORISH_9
            });
            let center = state.board().loc(board_dim / 2, board_dim / 2).unwrap();
            let mut records = Vec::new();
            for selected_move in [center, Loc::PASS, Loc::PASS] {
                records.push(SelfPlayRecord {
                    player: state.next_player(),
                    selected_move,
                    policy_target: [f16::ZERO; MAX_POLICY_SIZE],
                });
                assert!(state.play(selected_move));
            }
            let allocation = records.as_ptr();
            let game = CompletedGame::new(&state, records);
            assert_eq!(game.records.as_ptr(), allocation);
            assert_eq!(game.final_ownership, state.final_ownership());
            assert_eq!(game.final_ownership[center.index()], Color::Black);
        }
    }
}
