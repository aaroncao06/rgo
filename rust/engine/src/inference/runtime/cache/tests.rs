use super::*;
use crate::{
    game::{board::Loc, rules::Rules},
    inference::policy::{MAX_POLICY_SIZE, legal_mask},
};

fn test_output() -> Arc<NNOutput> {
    Arc::new(NNOutput::from_raw(
        [0.0; MAX_POLICY_SIZE].into(),
        0.0,
        0.0,
        0.0,
    ))
}

fn test_processed_output() -> Arc<NNOutput> {
    let game_state = GameState::new(Rules::TROMP_TAYLORISH_9);
    let mut output = test_output();
    Arc::get_mut(&mut output).unwrap().process_in_place(
        game_state.next_player(),
        &legal_mask(&game_state),
        game_state.board().size(),
    );
    output
}

fn loc(x: usize, y: usize) -> Loc {
    crate::game::board::Board::new(9)
        .loc(x, y)
        .expect("test coordinates must be on the board")
}

#[test]
fn evaluation_key_is_deterministic_and_changes_with_position_and_passes() {
    let initial = GameState::new(Rules::TROMP_TAYLORISH_9);
    let equivalent = GameState::new(Rules::TROMP_TAYLORISH_9);
    assert_eq!(
        EvaluationKey::new(&initial),
        EvaluationKey::new(&equivalent)
    );

    let mut moved = GameState::new(Rules::TROMP_TAYLORISH_9);
    assert!(moved.play(loc(4, 4)));
    assert_ne!(EvaluationKey::new(&initial), EvaluationKey::new(&moved));

    let mut passed_twice = GameState::new(Rules::TROMP_TAYLORISH_9);
    assert!(passed_twice.play(Loc::PASS));
    assert!(passed_twice.play(Loc::PASS));
    assert_eq!(
        initial.board().position_hash(),
        passed_twice.board().position_hash()
    );
    assert_eq!(initial.next_player(), passed_twice.next_player());
    assert_ne!(
        EvaluationKey::new(&initial),
        EvaluationKey::new(&passed_twice)
    );
}

#[test]
fn evaluation_key_changes_with_komi_and_suicide_rule() {
    let baseline = GameState::new(Rules::TROMP_TAYLORISH_9);
    let different_komi = GameState::new(Rules {
        board_size: 9,
        komi: 6.5,
        multi_stone_suicide_legal: true,
    });
    let different_suicide_rule = GameState::new(Rules::OGS_CHINESE_9);

    assert_ne!(
        EvaluationKey::new(&baseline),
        EvaluationKey::new(&different_komi)
    );
    assert_ne!(
        EvaluationKey::new(&baseline),
        EvaluationKey::new(&different_suicide_rule)
    );
}

#[test]
fn evaluation_identity_distinguishes_board_sizes_with_the_same_stones() {
    let mut larger = GameState::new(Rules::default());
    let mut smaller = GameState::new(Rules {
        board_size: 5,
        ..Rules::default()
    });
    let point = larger.board().loc(0, 0).unwrap();
    assert_ne!(
        larger.board().position_hash(),
        smaller.board().position_hash()
    );
    assert_ne!(EvaluationKey::new(&larger), EvaluationKey::new(&smaller));
    assert!(larger.play(point));
    assert!(smaller.play(point));
    assert_ne!(EvaluationKey::new(&larger), EvaluationKey::new(&smaller));
}

#[test]
fn evaluation_key_includes_superko_history_for_the_same_position() {
    let recapture = loc(4, 4);
    let capture = loc(4, 5);

    let mut with_repetition = GameState::new(Rules::TROMP_TAYLORISH_9);
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

    let mut without_repetition = GameState::new(Rules::TROMP_TAYLORISH_9);
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
    assert_eq!(
        with_repetition.next_player(),
        without_repetition.next_player()
    );
    assert_eq!(
        with_repetition.consecutive_ending_passes(),
        without_repetition.consecutive_ending_passes()
    );
    assert!(with_repetition.is_superko_banned(recapture));
    assert!(!without_repetition.is_superko_banned(recapture));
    assert_ne!(
        EvaluationKey::new(&with_repetition),
        EvaluationKey::new(&without_repetition)
    );
}

#[test]
fn evaluation_cache_misses_then_returns_the_inserted_output() {
    let cache = EvaluationCache::new(8, 2);
    let key = EvaluationKey(3);
    let output = test_processed_output();

    assert!(cache.lookup(key).is_none());
    cache.insert(key, output.clone());

    let cached = cache.lookup(key).unwrap();
    assert!(Arc::ptr_eq(&cached, &output));
}

#[test]
fn evaluation_cache_replaces_collisions_without_invalidating_old_outputs() {
    let cache = EvaluationCache::new(8, 2);
    let first_key = EvaluationKey(3);
    let colliding_key = EvaluationKey(11);
    assert_eq!(cache.indices(first_key), cache.indices(colliding_key));

    let first_output = test_processed_output();
    cache.insert(first_key, first_output.clone());
    let retained = cache.lookup(first_key).unwrap();

    let second_output = test_processed_output();
    cache.insert(colliding_key, second_output.clone());

    assert!(cache.lookup(first_key).is_none());
    let cached = cache.lookup(colliding_key).unwrap();
    assert!(Arc::ptr_eq(&cached, &second_output));
    assert!(Arc::ptr_eq(&retained, &first_output));
    assert!(retained.is_processed());
}
