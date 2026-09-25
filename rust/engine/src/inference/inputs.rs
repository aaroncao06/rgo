use super::policy::{BOARD_POLICY_SIZE, loc_to_policy};
use super::symmetry::Symmetry;
use crate::game::board::{Color, Loc};
use crate::game::game_state::GameState;

pub(crate) const NUM_SPATIAL_FEATURES: usize = 3; // player masks, superko banned
pub(crate) const NUM_GLOBAL_FEATURES: usize = 2; // komi, passes

// takes in current board,
pub(crate) struct NNInput {
    /// Request metadata, NOT a model feature. Only roots need ownership output.
    pub(crate) include_ownership: bool,
    pub(crate) spatial: [f32; NUM_SPATIAL_FEATURES * BOARD_POLICY_SIZE],
    pub(crate) global: [f32; NUM_GLOBAL_FEATURES],
}

impl NNInput {
    pub(crate) fn encode(game_state: &GameState) -> Self {
        let current_player = game_state.next_player();
        let current_color = Color::from(current_player);
        let opponent_color = Color::from(current_player.opponent());

        let mut global = [0_f32; NUM_GLOBAL_FEATURES];
        match current_color {
            Color::White => global[0] = game_state.rules().komi,
            Color::Black => global[0] = -game_state.rules().komi,
            _ => {}
        }
        global[1] = game_state.consecutive_ending_passes() as f32;

        // encode spatial maps
        let mut spatial = [0_f32; NUM_SPATIAL_FEATURES * BOARD_POLICY_SIZE];
        for loc in Loc::board_iter() {
            let color = game_state.board().color_at(loc);
            let policy_idx = loc_to_policy(loc);

            if color == current_color {
                spatial[policy_idx] = 1_f32;
            } else if color == opponent_color {
                spatial[BOARD_POLICY_SIZE + policy_idx] = 1_f32;
            }
            if game_state.is_superko_banned(loc) {
                spatial[2 * BOARD_POLICY_SIZE + policy_idx] = 1_f32;
            }
        }
        Self {
            spatial,
            global,
            include_ownership: false,
        }
    }

    pub(super) fn apply_symmetry_in_place(&mut self, symmetry: Symmetry) {
        symmetry.transform_planes(&mut self.spatial);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::Player;
    use crate::game::rules::Rules;

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
    fn empty_position_has_empty_planes_and_black_relative_komi() {
        let inputs = NNInput::encode(&GameState::new(rules()));

        assert_eq!(
            inputs.spatial,
            [0.0; NUM_SPATIAL_FEATURES * BOARD_POLICY_SIZE]
        );
        assert_eq!(inputs.global, [-7.5, 0.0]);
    }

    #[test]
    fn spatial_planes_are_relative_to_the_player_to_move() {
        let mut game_state = GameState::new(rules());
        let black_stone = loc(4, 4);
        let white_stone = loc(3, 3);

        assert!(game_state.play(black_stone));
        let white_turn_inputs = NNInput::encode(&game_state);
        let black_pos = loc_to_policy(black_stone);
        assert_eq!(
            white_turn_inputs.spatial[BOARD_POLICY_SIZE + black_pos],
            1.0
        );
        assert_eq!(white_turn_inputs.global, [7.5, 0.0]);

        assert!(game_state.play(white_stone));
        assert_eq!(game_state.next_player(), Player::Black);

        let inputs = NNInput::encode(&game_state);
        let white_pos = loc_to_policy(white_stone);

        assert_eq!(inputs.spatial[black_pos], 1.0);
        assert_eq!(inputs.spatial[BOARD_POLICY_SIZE + white_pos], 1.0);
        assert_eq!(inputs.spatial[BOARD_POLICY_SIZE + black_pos], 0.0);
        assert_eq!(inputs.spatial[white_pos], 0.0);
        assert_eq!(inputs.global, [-7.5, 0.0]);
    }

    #[test]
    fn ko_plane_contains_the_current_repetition_ban() {
        let mut game_state = GameState::new(rules());
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
            loc(4, 5),
        ] {
            assert!(game_state.play(move_loc));
        }

        let inputs = NNInput::encode(&game_state);
        let recapture_pos = loc_to_policy(recapture);

        assert!(game_state.is_superko_banned(recapture));
        assert_eq!(inputs.spatial[2 * BOARD_POLICY_SIZE + recapture_pos], 1.0);
    }
}
