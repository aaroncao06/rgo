use crate::game::{
    board::Loc,
    game_state::GameState,
    hash::{Hash128, nasam, splitmix64},
};

const REPETITION_BOUND: usize = 11;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphKey(Hash128);

fn mix(previous: Hash128, state: Hash128) -> Hash128 {
    let prev0 = previous as u64;
    let prev1 = (previous >> 64) as u64;
    let state0 = state as u64;
    let state1 = (state >> 64) as u64;

    let mixed0 = splitmix64(prev0 ^ prev1);
    let mixed1 = nasam(prev1).wrapping_add(mixed0);

    let result0 = mixed0.wrapping_add(state0);
    let result1 = mixed1.wrapping_add(state1);

    ((result1 as u128) << 64) | result0 as u128
}

impl GraphKey {
    pub(super) fn new(game_state: &GameState) -> Self {
        Self(game_state.current_state_hash())
    }
    pub(super) fn advance(&mut self, game_state: &GameState, last_move: Loc) {
        let state_key = game_state.current_state_hash();
        self.0 = if last_move != Loc::NULL
            && game_state
                .board()
                .repetition_region_is_small(last_move, REPETITION_BOUND)
        {
            mix(self.0, state_key)
        } else {
            state_key
        };
    }
    pub(super) fn raw(&self) -> Hash128 {
        self.0
    }

    #[cfg(test)]
    pub(super) const fn from_raw(raw: Hash128) -> Self {
        Self(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::rules::Rules;

    fn loc(x: usize, y: usize) -> Loc {
        Loc::new(x, y).expect("test coordinates must be on the board")
    }

    #[test]
    fn new_key_is_deterministic_and_changes_with_state() {
        let initial = GameState::new(Rules::TROMP_TAYLORISH);
        let equivalent = GameState::new(Rules::TROMP_TAYLORISH);
        assert_eq!(GraphKey::new(&initial), GraphKey::new(&equivalent));

        let mut moved = GameState::new(Rules::TROMP_TAYLORISH);
        assert!(moved.play(loc(4, 4)));
        assert_ne!(GraphKey::new(&initial), GraphKey::new(&moved));
    }

    #[test]
    fn advance_folds_history_for_a_pass() {
        let mut game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let mut key = GraphKey::new(&game_state);
        let previous = key.raw();

        assert!(game_state.play(Loc::PASS));
        let state_key = game_state.current_state_hash();
        key.advance(&game_state, Loc::PASS);

        assert_eq!(key.raw(), mix(previous, state_key));
        assert_ne!(key.raw(), state_key);
    }

    #[test]
    fn advance_drops_history_for_a_large_repetition_region() {
        let mut game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let mut key = GraphKey::new(&game_state);
        let played = loc(4, 4);

        assert!(game_state.play(played));
        assert!(
            !game_state
                .board()
                .repetition_region_is_small(played, REPETITION_BOUND)
        );
        key.advance(&game_state, played);

        assert_eq!(key, GraphKey::new(&game_state));
    }

    #[test]
    fn state_key_includes_the_current_superko_ban() {
        let recapture = loc(4, 4);
        let capture = loc(4, 5);

        let mut with_repetition = GameState::new(Rules::TROMP_TAYLORISH);
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
            assert!(with_repetition.play(move_loc));
        }

        let mut without_repetition = GameState::new(Rules::TROMP_TAYLORISH);
        for move_loc in [
            loc(4, 3),
            loc(4, 6),
            loc(3, 4),
            loc(3, 5),
            loc(5, 4),
            loc(5, 5),
            loc(0, 0),
            Loc::PASS,
            capture,
        ] {
            assert!(without_repetition.play(move_loc));
        }

        assert_eq!(
            with_repetition.board().position_hash(),
            without_repetition.board().position_hash()
        );
        assert!(with_repetition.is_superko_banned(recapture));
        assert!(!without_repetition.is_superko_banned(recapture));
        assert_ne!(
            GraphKey::new(&with_repetition),
            GraphKey::new(&without_repetition)
        );
    }

    #[test]
    fn null_move_starts_from_the_current_state_only() {
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let mut key = GraphKey::from_raw(123);

        key.advance(&game_state, Loc::NULL);

        assert_eq!(key, GraphKey::new(&game_state));
    }
}
