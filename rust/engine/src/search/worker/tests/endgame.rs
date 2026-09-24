use super::*;
use crate::inference::policy::BOARD_POLICY_SIZE;

fn loc(x: usize, y: usize) -> Loc {
    Loc::new(x, y).unwrap()
}

fn root_worker(
    state: &GameState,
    ownership: [f32; BOARD_POLICY_SIZE],
) -> SearchWorker<FixedArenaNodeStore> {
    let mut worker = worker();
    // Explicit White-perspective fixture; normal inference orients by next_player.
    let mut output = NNOutput::from_raw([0.0; POLICY_SIZE], 0.0, 0.0, -20.0)
        .with_ownership_logits(ownership.map(|v| v.clamp(-0.999999, 0.999999).atanh()));
    output.process_in_place(Player::White, &[true; POLICY_SIZE]);
    worker.search_graph.reset(state, Arc::new(output), 0.0, 16);
    worker
}

fn bonus(worker: &SearchWorker<FixedArenaNodeStore>, loc: Loc) -> f64 {
    worker
        .search_graph
        .root
        .as_ref()
        .unwrap()
        .ending_white_score_bonus(loc, worker.params)
}

#[test]
fn ending_bonus_thresholds_perspectives_and_disabled_cases() {
    for player in [Player::Black, Player::White] {
        let mut state = GameState::new(Rules::TROMP_TAYLORISH);
        if player == Player::White {
            assert!(state.play(Loc::PASS));
        }
        for ownership in [-0.975_f32, -0.95, 0.0, 0.95, 0.975] {
            let mut worker = root_worker(&state, [ownership; BOARD_POLICY_SIZE]);
            let actual_ownership = worker
                .search_graph
                .root
                .as_ref()
                .unwrap()
                .node
                .nn_output()
                .white_ownership()
                .unwrap()[0];
            let magnitude = 0.5 * (f64::from(actual_ownership).abs() - 0.95).max(0.0) / 0.05;
            let expected = if player == Player::White {
                -magnitude
            } else {
                magnitude
            };
            assert!((bonus(&worker, loc(4, 4)) - expected).abs() < 1e-12);
            assert_eq!(bonus(&worker, Loc::PASS), 0.0);
            assert_eq!(bonus(&worker, Loc::NULL), 0.0);
            worker.params.root_ending_bonus_points = 0.0;
            assert_eq!(bonus(&worker, loc(4, 4)), 0.0);
        }
    }
    let mut worker = worker();
    worker.search_graph.reset(
        &GameState::new(Rules::TROMP_TAYLORISH),
        processed_output([0.0; POLICY_SIZE], 0.0),
        0.0,
        16,
    );
    assert_eq!(bonus(&worker, loc(4, 4)), 0.0); // Missing map, like KataGo's helper.
}

#[test]
fn ending_bonus_preserves_captures_cleanup_and_unsettled_connections() {
    let mut state = GameState::new(Rules::TROMP_TAYLORISH);
    assert!(state.play(loc(1, 0)));
    assert!(state.play(loc(0, 0)));
    let opponent_owned = root_worker(&state, [0.99; BOARD_POLICY_SIZE]);
    assert_eq!(bonus(&opponent_owned, loc(0, 1)), 0.0); // Captures White.
    assert!(bonus(&opponent_owned, loc(8, 8)) > 0.0);
    let self_owned = root_worker(&state, [-0.99; BOARD_POLICY_SIZE]);
    assert_eq!(bonus(&self_owned, loc(0, 1)), 0.0); // Adjacent opponent cleanup.

    let mut state = GameState::new(Rules::TROMP_TAYLORISH);
    for point in [loc(3, 4), loc(5, 4)] {
        assert!(state.play(point));
        assert!(state.play(Loc::PASS));
    }
    let worker = root_worker(&state, [-0.99; BOARD_POLICY_SIZE]);
    assert_eq!(bonus(&worker, loc(4, 4)), 0.0); // Connects two unsettled groups.
    assert!(bonus(&worker, loc(3, 5)) > 0.0); // Just one group.
    for point in [loc(3, 3), loc(4, 3), loc(5, 3)] {
        assert!(state.play(point));
        assert!(state.play(Loc::PASS));
    }
    let worker = root_worker(&state, [-0.99; BOARD_POLICY_SIZE]);
    assert!(bonus(&worker, loc(4, 4)) > 0.0); // Same group touches on several sides.
}

#[test]
fn active_simple_ko_disables_the_ending_bonus_for_all_moves() {
    let mut state = GameState::new(Rules::TROMP_TAYLORISH);
    for point in [
        loc(4, 3),
        loc(4, 4),
        loc(3, 4),
        loc(4, 6),
        loc(5, 4),
        loc(3, 5),
        loc(0, 0),
        loc(5, 5),
        loc(4, 5),
    ] {
        assert!(state.play(point));
    }
    assert!(state.board().simple_ko().is_some());
    let worker = root_worker(&state, [0.99; BOARD_POLICY_SIZE]);
    assert_eq!(bonus(&worker, loc(8, 8)), 0.0);
    assert!(state.play(Loc::PASS));
    assert!(state.board().simple_ko().is_none());
    let worker = root_worker(&state, [0.99; BOARD_POLICY_SIZE]);
    assert!(bonus(&worker, loc(8, 8)) > 0.0);
}

// Black has two eyes; White has a remote unsettled group and then passes four
// times. Using real legal moves exercises history and pass-alive calculation.
fn pass_alive_position() -> GameState {
    let mut state = GameState::new(Rules::TROMP_TAYLORISH);
    for (i, (x, y)) in [
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 3),
        (3, 4),
        (5, 4),
        (7, 4),
        (3, 5),
        (4, 5),
        (5, 5),
        (6, 5),
        (7, 5),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(state.play(loc(x, y)));
        assert!(state.play(if i < 9 { loc(i, 0) } else { Loc::PASS }));
    }
    state
}

#[test]
fn root_pruning_filters_search_and_direct_policy_but_not_legality_or_interior() {
    let state = pass_alive_position();
    assert!(state.opponent_passed_last_four_turns());
    let mut worker = root_worker(&state, [0.0; BOARD_POLICY_SIZE]);
    let eye = loc(4, 4);
    let root = worker.search_graph.root.as_mut().unwrap();
    assert_eq!(root.safe_area[eye.index()], Color::Black);
    assert_eq!(root.safe_area[loc(0, 0).index()], Color::Empty); // Unsettled stone is not pass-alive.
    assert_eq!(root.safe_area[loc(0, 8).index()], Color::Empty);
    assert!(state.is_legal(eye));
    root.node.policy_probs_mut().fill(0.0);
    root.node.policy_probs_mut()[loc_to_policy(eye)] = 0.9;
    root.node.policy_probs_mut()[loc_to_policy(Loc::PASS)] = 0.1;
    assert!(!root.is_allowed_move(eye, worker.params));
    assert!(root.is_allowed_move(Loc::PASS, worker.params));
    let root = worker.search_graph.root.as_ref().unwrap();
    assert_eq!(worker.select_child(&root.node, &state, true).0, Loc::PASS);
    let root = worker.search_graph.root.as_ref().unwrap();
    assert_eq!(worker.select_child(&root.node, &state, false).0, eye);
    let (moves, _) = worker.root_selection_weights().unwrap();
    assert!(!moves.contains(&eye));
    assert!(moves.contains(&Loc::PASS));
    worker.params.root_prune_useless_moves = false;
    let root = worker.search_graph.root.as_ref().unwrap();
    assert_eq!(worker.select_child(&root.node, &state, true).0, eye);
    assert!(worker.root_selection_weights().unwrap().0.contains(&eye));
}

#[test]
fn ending_bonus_changes_root_selection_and_lcb_without_changing_graph_stats() {
    let state = GameState::new(Rules::TROMP_TAYLORISH);
    let bad = loc(3, 3);
    let good = loc(4, 4);
    let mut ownership = [0.0; BOARD_POLICY_SIZE];
    ownership[loc_to_policy(bad)] = -0.99;
    let mut worker = root_worker(&state, ownership);
    worker.params.root_desired_per_child_visits_coeff = 0.0;
    worker.params.chosen_move_prune = 0.0;
    let root = worker.search_graph.root.as_mut().unwrap();
    root.node.policy_probs_mut().fill(0.0);
    for point in [bad, good] {
        root.node.policy_probs_mut()[loc_to_policy(point)] = 0.5;
    }
    for (i, point) in [bad, good].into_iter().enumerate() {
        let mut ptr = worker
            .search_graph
            .node_store
            .insert(GraphKey::from_raw(i as u128 + 1))
            .unwrap();
        let child = unsafe { ptr.as_mut() };
        child.initialize_from_nn_eval(processed_output([0.0; POLICY_SIZE], 0.0), 0.0);
        for _ in 1..100 {
            child.record_visit(0.5, 0.0, 0.0, 0.0);
        }
        let edge = root.node.add_child(point, 0.5, ptr);
        for _ in 0..100 {
            root.node.edge_mut(edge).record_visit();
        }
    }
    let before = graph_snapshot(&worker);
    let root = worker.search_graph.root.as_ref().unwrap();
    assert_eq!(worker.select_child(&root.node, &state, false).0, bad); // Insertion tie.
    assert_eq!(worker.select_child(&root.node, &state, true).0, good);
    let mut actual = [100.0, 100.0];
    worker.adjust_root_weights_by_lcb(&mut actual, 100.0);
    let child = unsafe { root.node.edge_for_move(bad).unwrap().1.child().as_ref() };
    let delta = worker.root_ending_utility_bonus(child, bad);
    let radius_limit = worker.params.win_loss_utility_factor
        + worker.params.static_score_utility_factor
        + worker.params.dynamic_score_utility_factor;
    let (lcb, radius) = move_selection::lcb_and_radius(
        0.0,
        0.0,
        100.0,
        100.0,
        radius_limit,
        worker.params.lcb_stdevs,
    );
    let mut expected = [100.0, 100.0];
    move_selection::adjust_lcb(
        &mut expected,
        &[(lcb - delta, radius), (lcb, radius)],
        100.0,
        worker.params.min_visit_prop_for_lcb,
    );
    assert_eq!(actual, expected);
    assert!(actual[1] > actual[0]);
    assert_eq!(graph_snapshot(&worker), before);

    // Retrospective reduction uses a visit/policy-based reference child, not
    // the best utility. Give the good child one extra visit to make it the reference.
    let root = worker.search_graph.root.as_mut().unwrap();
    let (edge_index, edge) = root.node.edge_for_move(good).unwrap();
    let mut ptr = edge.child();
    unsafe { ptr.as_mut() }.record_visit(0.5, 0.0, 0.0, 0.0);
    root.node.edge_mut(edge_index).record_visit();
    let before = graph_snapshot(&worker);
    worker.params.use_lcb_for_selection = false;
    let (_, weights) = worker.root_selection_weights().unwrap();
    assert!(weights[0] < 100.0); // Retrospective PUCT reduction includes the bonus.
    assert_eq!(graph_snapshot(&worker), before);
    worker.params.root_ending_bonus_points = 0.0;
    assert_eq!(
        worker.root_selection_weights().unwrap().1,
        vec![100.0, 101.0]
    );
}

#[tokio::test]
async fn search_prunes_pass_alive_eyes_with_and_without_playouts() {
    let state = pass_alive_position();
    let eye = loc(4, 4);
    let mut logits = [-1000.0; POLICY_SIZE];
    logits[loc_to_policy(eye)] = 5.0;
    logits[loc_to_policy(Loc::PASS)] = 0.0;
    for budget in [0, 1, 4] {
        let mut worker = worker();
        worker.params.root_noise_enabled = false;
        let mut client = inference_client_with_policy(false, logits);
        let mut rng = SmallRng::seed_from_u64(123);
        let chosen = worker
            .search(&state, node_budget(budget), &mut client, &mut rng)
            .await
            .unwrap()
            .selected_move;
        if budget <= 1 {
            assert_eq!(chosen, Loc::PASS);
        }
        let root = worker.search_graph.root.as_ref().unwrap();
        // With more playouts, search may discover that passing loses and try
        // zero-policy moves elsewhere. Pruning must not force a pass.
        assert!(state.is_legal(chosen));
        assert!(root.is_allowed_move(chosen, worker.params));
        assert!(root.node.edge_for_move(eye).is_none());
        assert!(root.node.nn_output().has_ownership());
    }
}

#[tokio::test]
async fn zero_budget_with_all_policy_mass_pruned_returns_an_error() {
    let state = pass_alive_position();
    let mut logits = [-1000.0; POLICY_SIZE];
    logits[loc_to_policy(loc(4, 4))] = 1000.0;
    let mut worker = worker();
    worker.params.root_noise_enabled = false;
    let mut client = inference_client_with_policy(false, logits);
    let mut rng = SmallRng::seed_from_u64(123);
    assert!(matches!(
        worker
            .search(&state, node_budget(0), &mut client, &mut rng)
            .await,
        Err(SearchError::NoSelectableMove)
    ));
}
