use super::policy::{active_rows, loc_to_spatial};
use super::symmetry::Symmetry;
use crate::game::board::{Color, MAX_BOARD_POINTS};
use crate::game::game_state::GameState;

pub const NUM_SPATIAL_FEATURES: usize = 3; // player stones, opponent stones, superko
pub const NUM_GLOBAL_FEATURES: usize = 2; // komi, passes

// takes in current board,
pub struct NNInput {
    /// Active tensor dimensions; storage retains the maximum board stride.
    pub board_size: usize,
    /// Request metadata, NOT a model feature. Only roots need ownership output.
    pub include_ownership: bool,
    pub spatial: [f32; NUM_SPATIAL_FEATURES * MAX_BOARD_POINTS],
    pub global: [f32; NUM_GLOBAL_FEATURES],
}

impl NNInput {
    pub fn encode(game_state: &GameState) -> Self {
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
        let mut spatial = [0_f32; NUM_SPATIAL_FEATURES * MAX_BOARD_POINTS];
        for loc in game_state.board().locs() {
            let color = game_state.board().color_at(loc);
            let policy_idx: usize = loc_to_spatial(game_state.board(), loc);

            if color == current_color {
                spatial[policy_idx] = 1_f32;
            } else if color == opponent_color {
                spatial[MAX_BOARD_POINTS + policy_idx] = 1_f32;
            }
            if game_state.is_superko_banned(loc) {
                spatial[2 * MAX_BOARD_POINTS + policy_idx] = 1_f32;
            }
        }
        Self {
            board_size: game_state.board().size(),
            spatial,
            global,
            include_ownership: false,
        }
    }

    /// Active rows in channel-major order, without storage padding.
    pub fn spatial_rows(&self) -> impl Iterator<Item = &[f32]> {
        self.spatial
            .as_chunks::<MAX_BOARD_POINTS>()
            .0
            .iter()
            .flat_map(|plane| active_rows(plane, self.board_size))
    }

    pub(super) fn apply_symmetry_in_place(&mut self, symmetry: Symmetry) {
        symmetry.transform_planes(&mut self.spatial, self.board_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::{Loc, Player};
    use crate::game::rules::Rules;

    fn rules() -> Rules {
        Rules {
            board_size: 9,
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
    fn empty_position_has_empty_planes_and_black_relative_komi() {
        let inputs = NNInput::encode(&GameState::new(rules()));
        assert_eq!(inputs.board_size, 9);
        assert_eq!(
            inputs.spatial,
            [0.0; NUM_SPATIAL_FEATURES * MAX_BOARD_POINTS]
        );
        assert_eq!(inputs.global, [-7.5, 0.0]);
    }

    #[test]
    fn active_rows_exclude_padding_for_every_supported_size() {
        for size in 1..=crate::game::board::MAX_BOARD_SIZE {
            let state = GameState::new(Rules {
                board_size: size,
                ..rules()
            });
            let input = NNInput::encode(&state);
            assert_eq!(input.board_size, size);
            assert_eq!(input.spatial_rows().count(), NUM_SPATIAL_FEATURES * size);
            assert!(input.spatial_rows().all(|row| row.len() == size));
        }
    }

    #[test]
    fn symmetries_keep_stones_inside_the_active_board_and_restore_them() {
        let mut state = GameState::new(Rules {
            board_size: 5,
            ..rules()
        });
        assert!(state.play(state.board().loc(0, 0).unwrap()));
        assert!(state.play(state.board().loc(4, 4).unwrap()));
        let canonical = NNInput::encode(&state);
        for symmetry in Symmetry::ALL {
            let mut input = NNInput::encode(&state);
            input.apply_symmetry_in_place(symmetry);
            for plane in input.spatial.as_chunks::<MAX_BOARD_POINTS>().0 {
                for (i, &value) in plane.iter().enumerate() {
                    if value != 0.0 {
                        let point = state
                            .board()
                            .loc(
                                i % crate::game::board::MAX_BOARD_SIZE,
                                i / crate::game::board::MAX_BOARD_SIZE,
                            )
                            .unwrap();
                        assert!(state.board().is_on_board(point));
                    }
                }
            }
            for plane in input.spatial.as_chunks_mut::<MAX_BOARD_POINTS>().0 {
                symmetry.restore_output(
                    plane,
                    input.board_size,
                    crate::game::board::MAX_BOARD_SIZE,
                );
            }
            assert_eq!(input.spatial, canonical.spatial, "{symmetry:?}");
        }
    }

    #[test]
    fn spatial_planes_are_relative_to_the_player_to_move() {
        let mut game_state = GameState::new(rules());
        let black_stone = loc(4, 4);
        let white_stone = loc(3, 3);

        assert!(game_state.play(black_stone));
        let white_turn_inputs = NNInput::encode(&game_state);
        let black_pos = loc_to_spatial(game_state.board(), black_stone);
        assert_eq!(white_turn_inputs.spatial[MAX_BOARD_POINTS + black_pos], 1.0);
        assert_eq!(white_turn_inputs.global, [7.5, 0.0]);

        assert!(game_state.play(white_stone));
        assert_eq!(game_state.next_player(), Player::Black);

        let inputs = NNInput::encode(&game_state);
        let white_pos = loc_to_spatial(game_state.board(), white_stone);

        assert_eq!(inputs.spatial[black_pos], 1.0);
        assert_eq!(inputs.spatial[MAX_BOARD_POINTS + white_pos], 1.0);
        assert_eq!(inputs.spatial[MAX_BOARD_POINTS + black_pos], 0.0);
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
        let recapture_pos = loc_to_spatial(game_state.board(), recapture);

        assert!(game_state.is_superko_banned(recapture));
        assert_eq!(inputs.spatial[2 * MAX_BOARD_POINTS + recapture_pos], 1.0);
    }
}
