use super::*;
use crate::{
    game::rules::Rules,
    inference::{
        backend::InferenceBackend,
        inputs::NNInput,
        policy::POLICY_SIZE,
        runtime::{ModelRuntime, test_backend_factory},
    },
    search::{move_selection, node::SearchStats, node_store::FixedArenaNodeStore},
};
use rand::{SeedableRng, rngs::SmallRng};

mod endgame;

struct TestBackend {
    fail: bool,
    policy_logits: [f32; POLICY_SIZE],
}

impl InferenceBackend for TestBackend {
    fn evaluate_batch(
        &mut self,
        inputs: &[NNInput],
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        if self.fail {
            return Err(InferenceError::ExecutionFailed);
        }
        for input in inputs {
            let mut output = NNOutput::from_raw(self.policy_logits, 0.0, 0.0, 0.0);
            if input.include_ownership {
                output = output
                    .with_ownership_logits([0.0; crate::inference::policy::BOARD_POLICY_SIZE]);
            }
            outputs.push(Arc::new(output));
        }
        Ok(())
    }
}

fn inference_client(fail: bool) -> InferenceClient {
    inference_client_with_policy(fail, [0.0; POLICY_SIZE])
}

fn inference_client_with_policy(fail: bool, policy_logits: [f32; POLICY_SIZE]) -> InferenceClient {
    let model_handle = ModelRuntime::start(
        0,
        vec![test_backend_factory(TestBackend {
            fail,
            policy_logits,
        })],
        1,
        1,
        16,
        1,
    )
    .unwrap();
    InferenceClient::new(model_handle, None)
}

struct OneShotBackend {
    policy_logits: [f32; POLICY_SIZE],
    evaluated: bool,
}

impl InferenceBackend for OneShotBackend {
    fn evaluate_batch(
        &mut self,
        inputs: &[NNInput],
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        if self.evaluated {
            return Err(InferenceError::ExecutionFailed);
        }
        self.evaluated = true;
        for _ in inputs {
            // This fixture also supports cache-first tests that later request a root.
            let output = NNOutput::from_raw(self.policy_logits, 0.0, 0.0, 0.0)
                .with_ownership_logits([0.0; crate::inference::policy::BOARD_POLICY_SIZE]);
            outputs.push(Arc::new(output));
        }
        Ok(())
    }
}

fn one_shot_inference_client(policy_logits: [f32; POLICY_SIZE]) -> InferenceClient {
    let model_handle = ModelRuntime::start(
        0,
        vec![test_backend_factory(OneShotBackend {
            policy_logits,
            evaluated: false,
        })],
        1,
        1,
        16,
        1,
    )
    .unwrap();
    InferenceClient::new(model_handle, None)
}

fn worker() -> SearchWorker<FixedArenaNodeStore> {
    SearchWorker::new(
        FixedArenaNodeStore::new(16),
        SearchParams::KATAGO_SELFPLAY8_MAIN_B18,
    )
}

fn node_budget(max_nodes: usize) -> SearchBudget {
    SearchBudget::new(max_nodes, max_nodes.saturating_mul(4))
}

fn processed_output(policy_logits: [f32; POLICY_SIZE], white_win_logit: f32) -> Arc<NNOutput> {
    let mut output = Arc::new(NNOutput::from_raw(
        policy_logits,
        white_win_logit,
        0.0,
        -20.0,
    ));
    Arc::get_mut(&mut output)
        .unwrap()
        .process_in_place(Player::White, &[true; POLICY_SIZE]);
    output
}

fn initialized_node(output: Arc<NNOutput>, utility: f64) -> Box<SearchNode> {
    let mut node = Box::new(SearchNode::new());
    node.initialize_from_nn_eval(output, utility);
    node
}

#[derive(Debug, PartialEq)]
struct NodeSnapshot {
    visits: i32,
    values: [f64; 7],
    edges: Vec<(Loc, i32, f32, NonNull<SearchNode>)>,
}

fn node_snapshot(node: &SearchNode) -> NodeSnapshot {
    NodeSnapshot {
        visits: node.visits(),
        values: [
            node.white_win_rate(),
            node.white_score_mean(),
            node.white_score_mean_sq(),
            node.white_utility(),
            node.white_utility_mean_sq(),
            node.weight_sum(),
            node.weight_sq_sum(),
        ],
        edges: node
            .edges()
            .map(|edge| {
                (
                    edge.move_loc(),
                    edge.visits(),
                    edge.policy_prior(),
                    edge.child(),
                )
            })
            .collect(),
    }
}

// Snapshot every reachable node, including edge identity and all aggregate
// values, so failure tests detect partial backup or partial attachment.
fn graph_snapshot(worker: &SearchWorker<FixedArenaNodeStore>) -> Vec<NodeSnapshot> {
    let root = &worker.search_graph.root.as_ref().unwrap().node;
    let mut snapshots = vec![node_snapshot(root)];
    let mut pending: Vec<_> = root.edges().map(|edge| edge.child()).collect();
    let mut seen = HashSet::new();
    while let Some(pointer) = pending.pop() {
        if !seen.insert(pointer) {
            continue;
        }
        // SAFETY: the worker owns these nodes and is only shared-borrowed.
        let node = unsafe { pointer.as_ref() };
        snapshots.push(node_snapshot(node));
        pending.extend(node.edges().map(|edge| edge.child()));
    }
    snapshots
}

fn focused_policy() -> [f32; POLICY_SIZE] {
    let mut logits = [-20.0; POLICY_SIZE];
    logits[loc_to_policy(Loc::new(4, 4).unwrap())] = 20.0;
    logits[loc_to_policy(Loc::new(3, 3).unwrap())] = 10.0;
    logits
}

#[tokio::test]
async fn root_preprocessing_preserves_shared_cached_output() {
    let state = GameState::new(Rules::TROMP_TAYLORISH);
    // A second backend evaluation would fail, so the final evaluation must
    // retrieve the original shared output from the cache.
    let mut client = one_shot_inference_client(focused_policy());
    let cached = client.evaluate(&state, false).await.unwrap();
    let original_policy = *cached.policy_probs();
    let mut worker = worker();
    let mut rng = SmallRng::seed_from_u64(17);

    worker
        .search(&state, node_budget(0), &mut client, &mut rng)
        .await
        .unwrap();

    let root = &worker.search_graph.root.as_ref().unwrap().node;
    assert_ne!(root.policy_probs(), &original_policy);
    assert_eq!(cached.policy_probs(), &original_policy);
    assert_eq!(root.nn_output().white_win_prob(), cached.white_win_prob());
    assert_eq!(
        root.nn_output().white_score_mean(),
        cached.white_score_mean()
    );
    assert_eq!(
        root.nn_output().white_score_mean_sq(),
        cached.white_score_mean_sq()
    );
    let cached_again = client.evaluate(&state, false).await.unwrap();
    assert!(Arc::ptr_eq(&cached, &cached_again));
    assert_eq!(cached_again.policy_probs(), &original_policy);
}

#[tokio::test]
async fn search_returns_move_and_policy_target_without_mutating_the_position() {
    let preferred = Loc::new(4, 4).unwrap();
    let mut state = GameState::new(Rules::TROMP_TAYLORISH);
    assert!(state.play(Loc::PASS));
    let key = GraphKey::new(&state);
    for budget in [0, 1, 8] {
        let mut worker = worker();
        worker.params.chosen_move_temperature_early = 0.0;
        worker.params.chosen_move_temperature = 0.0;
        let mut client = inference_client_with_policy(false, focused_policy());
        let mut rng = SmallRng::seed_from_u64(7);
        let result = worker
            .search(&state, node_budget(budget), &mut client, &mut rng)
            .await
            .unwrap();
        let chosen = result.selected_move;
        assert_eq!(chosen, preferred);
        assert!(state.is_legal(chosen));
        assert!((result.policy_target.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(result.policy_target[loc_to_policy(chosen)] > 0.0);
        assert_eq!(GraphKey::new(&state), key);
        assert_eq!(state.turn_number(), 1);
        assert_eq!(
            worker.search_graph.root.as_ref().unwrap().node.visits(),
            budget as i32 + 1
        );
        assert_eq!(worker.search_graph.node_store.len(), budget);
        assert_eq!(worker.search_graph.node_store.capacity(), budget);
        assert_eq!(
            worker
                .search_graph
                .root
                .as_ref()
                .unwrap()
                .game_state
                .turn_number(),
            1
        );
        let before = graph_snapshot(&worker);
        let (_, weights) = worker.root_selection_weights().unwrap();
        assert!(weights.iter().any(|&weight| weight > 0.0));
        assert_eq!(graph_snapshot(&worker), before);
    }
}

#[tokio::test]
async fn search_stops_at_playout_guard_when_terminal_revisits_cannot_fill_the_store() {
    let mut state = GameState::new(Rules::TROMP_TAYLORISH);
    assert!(state.play(Loc::PASS));

    let mut pass_policy = [-1000.0; POLICY_SIZE];
    pass_policy[loc_to_policy(Loc::PASS)] = 0.0;
    let mut worker = worker();
    worker.params.root_noise_enabled = false;
    let mut client = inference_client_with_policy(false, pass_policy);
    let mut rng = SmallRng::seed_from_u64(5);

    worker
        .search(&state, SearchBudget::new(2, 3), &mut client, &mut rng)
        .await
        .unwrap();

    assert_eq!(worker.search_graph.node_store.capacity(), 2);
    assert_eq!(worker.search_graph.node_store.len(), 1);
    assert_eq!(worker.search_graph.root.as_ref().unwrap().node.visits(), 4);
}

#[tokio::test]
async fn search_value_target_uses_current_player_and_converts_moments_to_stdev() {
    for player in [Player::Black, Player::White] {
        let mut state = GameState::new(Rules::TROMP_TAYLORISH);
        if player == Player::White {
            assert!(state.play(Loc::PASS));
        }
        let mut worker = worker();
        let mut client = inference_client(false);
        worker.start_game(&state, &mut client).await.unwrap();
        worker
            .search_graph
            .root
            .as_mut()
            .unwrap()
            .node
            .replace_stats(SearchStats {
                visits: 4,
                white_win_sum: 3.0,
                white_score_sum: 12.0,
                white_score_mean_sq_sum: 52.0,
                white_utility_sum: 0.0,
                white_utility_sq_sum: 0.0,
                weight_sum: 4.0,
                weight_sq_sum: 4.0,
            });

        let mut rng = SmallRng::seed_from_u64(0);
        let target = worker.build_search_result(&mut rng).unwrap().value_target;
        let expected = match player {
            Player::White => SearchValueTarget {
                win_probability: 0.75,
                score_mean: 3.0,
                score_stdev: 2.0,
            },
            Player::Black => SearchValueTarget {
                win_probability: 0.25,
                score_mean: -3.0,
                score_stdev: 2.0,
            },
        };
        assert_eq!(target, expected);
    }
}

#[tokio::test]
async fn search_propagates_inference_failure_without_sampling() {
    let mut worker = worker();
    // Root Dirichlet noise is intentionally sampled before search in the
    // normal self-play path; disable it here to isolate failure behavior.
    worker.params.root_noise_enabled = false;
    let state = GameState::new(Rules::TROMP_TAYLORISH);
    let mut client = one_shot_inference_client(focused_policy());
    let mut rng = SmallRng::seed_from_u64(11);
    let mut untouched_rng = rng.clone();
    assert!(matches!(
        worker
            .search(&state, node_budget(1), &mut client, &mut rng)
            .await,
        Err(SearchError::InferenceError(_))
    ));
    assert_eq!(rng.next_u64(), untouched_rng.next_u64());
    assert_eq!(worker.search_graph.root.as_ref().unwrap().node.visits(), 1);
}

#[tokio::test]
async fn final_lcb_weights_use_edge_sample_size_for_transpositions() {
    let mut worker = worker();
    worker.params.chosen_move_temperature_early = 0.0;
    worker.params.chosen_move_temperature = 0.0;
    let moves = [Loc::new(3, 3).unwrap(), Loc::new(4, 4).unwrap()];
    let mut logits = [-1000.0; POLICY_SIZE];
    for loc in moves {
        logits[loc_to_policy(loc)] = 0.0;
    }
    let mut client = inference_client_with_policy(false, logits);
    let state = GameState::new(Rules::TROMP_TAYLORISH);
    worker.start_game(&state, &mut client).await.unwrap();
    for (i, loc) in moves.into_iter().enumerate() {
        let mut pointer = worker
            .search_graph
            .node_store
            .insert(GraphKey::from_raw(i as u128 + 1))
            .unwrap();
        // SAFETY: fresh live arena node, initialized before linking it.
        let child = unsafe { pointer.as_mut() };
        child.initialize_from_nn_eval(processed_output(logits, 0.0), 0.0);
        child.replace_stats(SearchStats {
            visits: 100,
            white_win_sum: 50.0,
            white_score_sum: 0.0,
            white_score_mean_sq_sum: 0.0,
            white_utility_sum: if i == 0 { 20.0 } else { -20.0 },
            white_utility_sq_sum: 29.0,
            weight_sum: 100.0,
            weight_sq_sum: 160.0,
        });
        let root = &mut worker.search_graph.root.as_mut().unwrap().node;
        let edge = root.add_child(loc, 0.5, pointer);
        for _ in 0..25 {
            root.edge_mut(edge).record_visit();
        }
    }
    let before = graph_snapshot(&worker);
    let (_, weights) = worker.root_selection_weights().unwrap();
    // Native KataGo fixture: radius=.632893836585088, LCB gap=.4.
    // The second child is better for Black, despite equal edge weights.
    assert_eq!(weights[0], 25.0);
    assert!((weights[1] - 52.480946693222528).abs() < 1e-10);
    assert_eq!(graph_snapshot(&worker), before);

    let mut rng = SmallRng::seed_from_u64(0);
    let result = worker.build_search_result(&mut rng).unwrap();
    // Self-play samples without LCB, so the equal unadjusted weights retain
    // insertion order at zero temperature. The recorded target still uses LCB.
    assert_eq!(result.selected_move, moves[0]);
    assert!(
        result.policy_target[loc_to_policy(moves[1])]
            > result.policy_target[loc_to_policy(moves[0])]
    );
}

#[tokio::test]
async fn final_weights_reduce_overexploration_without_changing_graph_stats() {
    let mut worker = worker();
    worker.params.use_lcb_for_selection = false;
    worker.params.cpuct_exploration = 1.0;
    worker.params.cpuct_exploration_log = 0.0;
    let first_move = Loc::new(3, 3).unwrap();
    let second_move = Loc::new(4, 4).unwrap();
    // Finite logits whose softmax underflows to zero on the other moves.
    let mut logits = [-1000.0; POLICY_SIZE];
    logits[loc_to_policy(first_move)] = 0.0;
    logits[loc_to_policy(second_move)] = 0.0;
    let mut client = inference_client_with_policy(false, logits);
    worker
        .start_game(&GameState::new(Rules::TROMP_TAYLORISH), &mut client)
        .await
        .unwrap();
    for (i, (loc, visits, utility)) in [(first_move, 60, -0.5), (second_move, 40, 0.5)]
        .into_iter()
        .enumerate()
    {
        let mut ptr = worker
            .search_graph
            .node_store
            .insert(GraphKey::from_raw(i as u128 + 1))
            .unwrap();
        // SAFETY: fresh arena node; no other references to this node are live.
        let child = unsafe { ptr.as_mut() };
        child.initialize_from_nn_eval(processed_output(logits, 0.0), utility);
        for _ in 1..visits {
            child.record_visit(0.5, 0.0, 0.0, utility);
        }
        let root = &mut worker.search_graph.root.as_mut().unwrap().node;
        let edge = root.add_child(loc, 0.5, ptr);
        for _ in 0..visits {
            root.edge_mut(edge).record_visit();
        }
    }
    let before = graph_snapshot(&worker);
    let (moves, weights) = worker.root_selection_weights().unwrap();
    assert_eq!(moves, vec![first_move, second_move]);
    // Black prefers the first child. Inverting its PUCT score for the
    // second gives ~3.62 visits; KataGo rounds that reduced weight UP.
    assert_eq!(weights, vec![60.0, 4.0]);
    assert_eq!(graph_snapshot(&worker), before);
}

#[test]
fn new_worker_has_no_active_graph_or_scratch_state() {
    let worker = worker();

    assert!(worker.search_graph.root.is_none());
    assert_eq!(worker.search_graph.node_store.len(), 0);
    assert!(worker.scratch_game_state.is_none());
    assert!(worker.playout_path.is_empty());
}

#[test]
fn selection_chooses_the_highest_policy_unexpanded_move() {
    let mut worker = worker();
    let game_state = GameState::new(Rules::TROMP_TAYLORISH);
    let expected_move = Loc::new(4, 4).unwrap();
    let mut logits = [0.0; POLICY_SIZE];
    logits[crate::inference::policy::loc_to_policy(expected_move)] = 5.0;
    let output = processed_output(logits, 0.0);
    worker
        .search_graph
        .reset(&game_state, output.clone(), 0.0, 16);
    let node = initialized_node(output, 0.0);

    assert_eq!(
        worker.select_child(&node, &game_state, true),
        (expected_move, None)
    );
}

#[test]
fn selection_orients_child_utility_for_the_player_to_move() {
    let mut worker = worker();
    let game_state = GameState::new(Rules::TROMP_TAYLORISH);
    assert_eq!(game_state.next_player(), Player::Black);
    let parent_output = processed_output([0.0; POLICY_SIZE], 0.0);
    worker
        .search_graph
        .reset(&game_state, parent_output.clone(), 0.0, 16);
    let mut parent = initialized_node(parent_output, 0.0);

    let white_favored_move = Loc::new(3, 3).unwrap();
    let black_favored_move = Loc::new(4, 4).unwrap();
    let mut white_favored_child =
        initialized_node(processed_output([0.0; POLICY_SIZE], 9.0_f32.ln()), 0.8);
    let mut black_favored_child = initialized_node(
        processed_output([0.0; POLICY_SIZE], (1.0_f32 / 9.0).ln()),
        -0.8,
    );
    parent.add_child(
        white_favored_move,
        parent.policy_probs()[crate::inference::policy::loc_to_policy(white_favored_move)],
        NonNull::from(white_favored_child.as_mut()),
    );
    parent.add_child(
        black_favored_move,
        parent.policy_probs()[crate::inference::policy::loc_to_policy(black_favored_move)],
        NonNull::from(black_favored_child.as_mut()),
    );

    assert_eq!(
        worker.select_child(&parent, &game_state, true).0,
        black_favored_move
    );
}

#[test]
fn root_selection_forces_an_existing_child_below_its_desired_visits() {
    let mut worker = worker();
    let game_state = GameState::new(Rules::TROMP_TAYLORISH);
    let forced_move = Loc::new(3, 3).unwrap();
    let otherwise_best_move = Loc::new(4, 4).unwrap();
    let mut logits = [-20.0; POLICY_SIZE];
    logits[loc_to_policy(forced_move)] = 4.0;
    logits[loc_to_policy(otherwise_best_move)] = 0.0;
    let parent_output = processed_output(logits, 0.0);
    worker
        .search_graph
        .reset(&game_state, parent_output.clone(), 0.0, 16);
    let mut parent = initialized_node(parent_output, 0.0);
    let mut forced_child =
        initialized_node(processed_output([0.0; POLICY_SIZE], 9.0_f32.ln()), 1.0);
    let mut otherwise_best_child = initialized_node(
        processed_output([0.0; POLICY_SIZE], (1.0_f32 / 9.0).ln()),
        -1.0,
    );
    let forced_edge = parent.add_child(
        forced_move,
        parent.policy_probs()[loc_to_policy(forced_move)],
        NonNull::from(forced_child.as_mut()),
    );
    let otherwise_best_edge = parent.add_child(
        otherwise_best_move,
        parent.policy_probs()[loc_to_policy(otherwise_best_move)],
        NonNull::from(otherwise_best_child.as_mut()),
    );
    for _ in 0..10 {
        parent.edge_mut(forced_edge).record_visit();
    }
    for _ in 0..100 {
        parent.edge_mut(otherwise_best_edge).record_visit();
    }

    assert_eq!(
        worker.select_child(&parent, &game_state, false).0,
        otherwise_best_move
    );
    assert_eq!(
        worker.select_child(&parent, &game_state, true).0,
        forced_move
    );
}

#[test]
fn forced_root_visit_ties_follow_child_insertion_order() {
    let mut worker = worker();
    let game_state = GameState::new(Rules::TROMP_TAYLORISH);
    let first_move = Loc::new(4, 4).unwrap();
    let second_move = Loc::new(3, 3).unwrap();
    assert!(loc_to_policy(first_move) > loc_to_policy(second_move));
    let mut logits = [-20.0; POLICY_SIZE];
    logits[loc_to_policy(first_move)] = 0.0;
    logits[loc_to_policy(second_move)] = 0.0;
    let mut parent = initialized_node(processed_output(logits, 0.0), 0.0);
    worker
        .search_graph
        .reset(&game_state, processed_output(logits, 0.0), 0.0, 16);
    let mut first_child = initialized_node(processed_output(logits, 0.0), 0.0);
    let mut second_child = initialized_node(processed_output(logits, 0.0), 0.0);
    let first_edge = parent.add_child(first_move, 0.5, NonNull::from(first_child.as_mut()));
    let second_edge = parent.add_child(second_move, 0.5, NonNull::from(second_child.as_mut()));
    parent.edge_mut(first_edge).record_visit();
    parent.edge_mut(second_edge).record_visit();
    // Each child has weight 1, below sqrt(0.5 * 2 * 2).
    assert_eq!(
        worker.select_child(&parent, &game_state, true),
        (first_move, Some(first_edge))
    );
}

#[test]
fn existing_child_wins_exact_tie_against_unexpanded_move() {
    let mut worker = worker();
    worker.params.cpuct_exploration = 0.0;
    worker.params.cpuct_exploration_log = 0.0;
    worker.params.fpu_reduction_max = 0.0;
    let game_state = GameState::new(Rules::TROMP_TAYLORISH);
    let existing_move = Loc::new(4, 4).unwrap();
    let output = processed_output([0.0; POLICY_SIZE], 0.0);
    let mut parent = initialized_node(output.clone(), 0.0);
    let mut child = initialized_node(output, 0.0);
    let edge = parent.add_child(existing_move, 0.5, NonNull::from(child.as_mut()));
    parent.edge_mut(edge).record_visit();
    assert_eq!(
        worker.select_child(&parent, &game_state, false).0,
        existing_move
    );
}

#[test]
fn value_weighting_favors_the_better_child_and_preserves_total_weight() {
    let mut worker = worker();
    let mut parent = initialized_node(processed_output([0.0; POLICY_SIZE], 0.0), 0.0);
    let mut good_child = initialized_node(processed_output([0.0; POLICY_SIZE], 9.0_f32.ln()), 1.0);
    let mut bad_child = initialized_node(
        processed_output([0.0; POLICY_SIZE], (1.0_f32 / 9.0).ln()),
        -1.0,
    );
    for _ in 0..9 {
        good_child.record_visit(1.0, 0.0, 0.0, 1.0);
        bad_child.record_visit(0.0, 0.0, 0.0, -1.0);
    }

    let good_edge = parent.add_child(
        Loc::new(3, 3).unwrap(),
        0.5,
        NonNull::from(good_child.as_mut()),
    );
    let bad_edge = parent.add_child(
        Loc::new(4, 4).unwrap(),
        0.5,
        NonNull::from(bad_child.as_mut()),
    );
    for _ in 0..10 {
        parent.edge_mut(good_edge).record_visit();
        parent.edge_mut(bad_edge).record_visit();
    }

    worker.recompute_node_stats(&mut parent, Player::White);

    assert_eq!(parent.visits(), 2); // One completed playout, not a sum of child visits.
    assert!((parent.weight_sum() - 21.0).abs() < 1e-12);
    assert!(parent.white_utility() > 0.0);
}

#[test]
fn transposed_child_squared_weights_scale_by_squared_edge_fraction() {
    let mut worker = worker();
    let output = processed_output([0.0; POLICY_SIZE], 0.0);
    let mut child = initialized_node(output.clone(), 0.0);
    for _ in 0..9 {
        child.record_visit(0.5, 0.0, 0.0, 0.0);
    }
    let mut first_parent = initialized_node(output.clone(), 0.0);
    let mut second_parent = initialized_node(output, 0.0);
    let child_ptr = NonNull::from(child.as_mut());
    for parent in [&mut first_parent, &mut second_parent] {
        let edge = parent.add_child(Loc::PASS, 1.0, child_ptr);
        for _ in 0..5 {
            parent.edge_mut(edge).record_visit();
        }
        worker.recompute_node_stats(parent, Player::White);

        assert_eq!(parent.visits(), 2);
        assert!((parent.weight_sum() - 6.0).abs() < 1e-12);
        // Ten child samples scaled by 1/2, plus the parent's unit-weight eval.
        assert!((parent.weight_sq_sum() - 3.5).abs() < 1e-12);
    }
    assert_eq!(child.weight_sq_sum(), 10.0);
}

#[tokio::test]
async fn graph_cycles_end_the_playout_and_back_up_without_inference() {
    for self_loop in [false, true] {
        let root_move = Loc::new(0, 0).unwrap();
        let first_move = Loc::new(1, 0).unwrap();
        let second_move = Loc::new(2, 0).unwrap();
        let policy = |loc| {
            let mut logits = [-20.0; POLICY_SIZE];
            logits[loc_to_policy(loc)] = 20.0;
            logits
        };
        let mut client = one_shot_inference_client(policy(root_move));
        let mut worker = worker();
        worker
            .start_game(&GameState::new(Rules::TROMP_TAYLORISH), &mut client)
            .await
            .unwrap();
        let mut first = worker
            .search_graph
            .node_store
            .insert(GraphKey::from_raw(1))
            .unwrap();
        let mut second = if self_loop {
            first
        } else {
            worker
                .search_graph
                .node_store
                .insert(GraphKey::from_raw(2))
                .unwrap()
        };
        // Synthetic graph: deliberately seed equal edge/node counts to
        // exercise the cycle guard rather than the earlier catch-up check.
        unsafe {
            first
                .as_mut()
                .initialize_from_nn_eval(processed_output(policy(first_move), 0.0), 0.0);
            if !self_loop {
                second
                    .as_mut()
                    .initialize_from_nn_eval(processed_output(policy(second_move), 0.0), 0.0);
            }
        }
        let root = &mut worker.search_graph.root.as_mut().unwrap().node;
        let root_edge = root.add_child(root_move, 1.0, first);
        root.edge_mut(root_edge).record_visit();
        let first_edge = unsafe { first.as_mut() }.add_child(first_move, 1.0, second);
        unsafe { first.as_mut() }
            .edge_mut(first_edge)
            .record_visit();
        let second_edge = if self_loop {
            None
        } else {
            let edge = unsafe { second.as_mut() }.add_child(second_move, 1.0, first);
            unsafe { second.as_mut() }.edge_mut(edge).record_visit();
            Some(edge)
        };

        // Repeat to check that cycle tracking resets between playouts and
        // every closing edge/ancestor receives exactly one update each time.
        for expected_visits in 2..=6 {
            worker.playout(&mut client).await.unwrap();
            let root = &worker.search_graph.root.as_ref().unwrap().node;
            assert_eq!(root.visits(), expected_visits);
            assert_eq!(root.edge(root_edge).visits(), expected_visits);
            let first_node = unsafe { first.as_ref() };
            assert_eq!(first_node.visits(), expected_visits);
            assert_eq!(first_node.edge(first_edge).visits(), expected_visits);
            if let Some(edge) = second_edge {
                let second_node = unsafe { second.as_ref() };
                assert_eq!(second_node.visits(), expected_visits);
                assert_eq!(second_node.edge(edge).visits(), expected_visits);
            }
            assert_eq!(
                worker.search_graph.node_store.len(),
                if self_loop { 1 } else { 2 }
            );
            assert!(
                graph_snapshot(&worker)
                    .iter()
                    .all(|node| node.values.iter().all(|value| value.is_finite()))
            );
            assert!(worker.playout_path.is_empty());
            assert!(worker.visited_nodes.is_empty());
        }
    }
}

#[tokio::test]
async fn transposed_child_catches_up_before_requesting_more_inference() {
    let move_loc = Loc::new(4, 4).unwrap();
    let mut logits = [-20.0; POLICY_SIZE];
    logits[loc_to_policy(move_loc)] = 20.0;
    let mut client = one_shot_inference_client(logits);
    let mut worker = worker();
    let root_state = GameState::new(Rules::TROMP_TAYLORISH);
    let mut child_state = root_state.clone();
    let mut child_key = GraphKey::new(&root_state);
    assert!(child_state.play(move_loc));
    child_key.advance(&child_state, move_loc);
    worker.start_game(&root_state, &mut client).await.unwrap();

    // Model a node already searched three times through another parent.
    let mut child_ptr = worker.search_graph.node_store.insert(child_key).unwrap();
    {
        let child = unsafe { child_ptr.as_mut() };
        child.initialize_from_nn_eval(processed_output([0.0; POLICY_SIZE], 0.0), 0.0);
        for _ in 1..3 {
            child.record_visit(0.5, 0.0, 0.0, 0.0);
        }
    }
    // First playout links a new edge; subsequent ones catch up that edge.
    // The backend rejects any inference after the root's evaluation.
    for expected_visits in 1..=3 {
        worker.playout(&mut client).await.unwrap();
        let root = &worker.search_graph.root.as_ref().unwrap().node;
        assert_eq!(
            root.edge_for_move(move_loc).unwrap().1.visits(),
            expected_visits
        );
        assert_eq!(root.visits(), expected_visits + 1);
        assert_eq!(unsafe { child_ptr.as_ref() }.visits(), 3);
        assert_eq!(worker.search_graph.node_store.len(), 1);
        assert!(worker.playout_path.is_empty());
        if expected_visits > 1 {
            // Existing-edge catch-up stops before changing the board,
            // player, or superko state. New-edge lookup still plays the move.
            let root_state = &worker.search_graph.root.as_ref().unwrap().game_state;
            assert_eq!(
                GraphKey::new(worker.scratch_game_state()),
                GraphKey::new(root_state)
            );
            assert_eq!(
                worker.scratch_game_state().next_player(),
                root_state.next_player()
            );
        }
    }
    // Once caught up, search must descend and request a fresh evaluation.
    assert!(matches!(
        worker.playout(&mut client).await,
        Err(SearchError::InferenceError(_))
    ));
}

#[tokio::test]
async fn start_game_installs_an_evaluated_root_and_scratch_state() {
    let mut worker = worker();
    let mut client = inference_client(false);
    let game_state = GameState::new(Rules::TROMP_TAYLORISH);
    let expected_key = GraphKey::new(&game_state);

    worker.start_game(&game_state, &mut client).await.unwrap();

    let root = worker.search_graph.root.as_ref().unwrap();
    assert_eq!(root.key, expected_key);
    assert_eq!(root.node.visits(), 1);
    assert_eq!(
        GraphKey::new(worker.scratch_game_state.as_ref().unwrap()),
        expected_key
    );
}

#[tokio::test]
async fn playout_expands_and_evaluates_a_missing_child() {
    let mut worker = worker();
    let mut client = inference_client(false);
    worker
        .start_game(&GameState::new(Rules::TROMP_TAYLORISH), &mut client)
        .await
        .unwrap();

    worker.playout(&mut client).await.unwrap();

    assert_eq!(worker.search_graph.node_store.len(), 1);
    assert!(worker.playout_path.is_empty());
    let root = &worker.search_graph.root.as_ref().unwrap().node;
    assert_eq!(root.visits(), 2);
    assert_eq!(root.edge_visit_sum(), 1);
}

#[tokio::test]
async fn consecutive_playouts_descend_and_back_up_through_existing_edges() {
    let preferred_move = Loc::new(4, 4).unwrap();
    let mut logits = [0.0; POLICY_SIZE];
    logits[loc_to_policy(preferred_move)] = 20.0;
    let mut client = inference_client_with_policy(false, logits);
    let mut worker = worker();
    worker
        .start_game(&GameState::new(Rules::TROMP_TAYLORISH), &mut client)
        .await
        .unwrap();

    worker.playout(&mut client).await.unwrap();
    worker.playout(&mut client).await.unwrap();

    assert_eq!(worker.search_graph.node_store.len(), 2);
    assert!(worker.playout_path.is_empty());
    let root = &worker.search_graph.root.as_ref().unwrap().node;
    assert_eq!(root.visits(), 3);
    assert_eq!(root.edge_visit_sum(), 2);
    let root_edge = root.edge_for_move(preferred_move).unwrap().1;
    assert_eq!(root_edge.visits(), 2);
    let child = unsafe { root_edge.child().as_ref() };
    assert_eq!(child.visits(), 2);
    assert_eq!(child.edge_visit_sum(), 1);
}

#[tokio::test]
async fn integer_terminal_scores_use_katago_gridded_second_moment() {
    for komi in [-7.0, 0.0, 7.0] {
        let mut root_state = GameState::new(Rules {
            komi,
            ..Rules::TROMP_TAYLORISH
        });
        assert!(root_state.play(Loc::PASS));
        let mut logits = [-20.0; POLICY_SIZE];
        logits[loc_to_policy(Loc::PASS)] = 20.0;
        let mut client = one_shot_inference_client(logits);
        let mut worker = worker();
        worker.start_game(&root_state, &mut client).await.unwrap();

        for _ in 0..2 {
            worker.playout(&mut client).await.unwrap();
            let root = &worker.search_graph.root.as_ref().unwrap().node;
            let terminal = unsafe { root.edge_for_move(Loc::PASS).unwrap().1.child().as_ref() };
            let score = f64::from(komi);
            let win = if score > 0.0 {
                1.0
            } else if score < 0.0 {
                0.0
            } else {
                0.5
            };
            assert_eq!(terminal.white_win_rate(), win);
            assert_eq!(terminal.white_score_mean(), score);
            assert_eq!(terminal.white_score_mean_sq(), score * score + 0.25);
            let expected_utility = white_utility(
                win,
                score,
                score * score + 0.25,
                worker.recent_score_center,
                worker.params,
            );
            assert!((terminal.white_utility() - expected_utility).abs() < 1e-12);
        }
    }
}

#[tokio::test]
async fn terminal_leaf_uses_exact_score_without_inference() {
    let mut root_state = GameState::new(Rules::TROMP_TAYLORISH);
    assert!(root_state.play(Loc::PASS));
    let mut logits = [0.0; POLICY_SIZE];
    logits[loc_to_policy(Loc::PASS)] = 20.0;
    let mut client = one_shot_inference_client(logits);
    let mut worker = worker();
    worker.start_game(&root_state, &mut client).await.unwrap();

    worker.playout(&mut client).await.unwrap();
    worker.playout(&mut client).await.unwrap();

    assert_eq!(worker.search_graph.node_store.len(), 1);
    assert!(worker.scratch_game_state().is_finished());
    let root = &worker.search_graph.root.as_ref().unwrap().node;
    let pass_edge = root.edge_for_move(Loc::PASS).unwrap().1;
    assert_eq!(pass_edge.visits(), 2);
    let terminal = unsafe { pass_edge.child().as_ref() };
    assert_eq!(terminal.visits(), 2);
    assert_eq!(terminal.white_win_rate(), 1.0);
    assert_eq!(terminal.white_score_mean(), 7.5);
    assert_eq!(terminal.white_score_mean_sq(), 7.5 * 7.5);
}

#[tokio::test]
async fn playout_inference_failure_does_not_attach_an_unevaluated_child() {
    let mut worker = worker();
    let mut working_client = inference_client(false);
    worker
        .start_game(&GameState::new(Rules::TROMP_TAYLORISH), &mut working_client)
        .await
        .unwrap();

    let mut failing_client = inference_client(true);
    let result = worker.playout(&mut failing_client).await;

    assert!(matches!(result, Err(SearchError::InferenceError(_))));
    assert_eq!(worker.search_graph.node_store.len(), 0);
    assert!(worker.playout_path.is_empty());
    assert_eq!(
        worker
            .search_graph
            .root
            .as_ref()
            .unwrap()
            .node
            .visited_policy_mass(),
        0.0
    );
}

#[tokio::test]
async fn deep_inference_failure_preserves_graph_and_retry_backs_up_once() {
    let mut worker = worker();
    let mut client = inference_client_with_policy(false, focused_policy());
    worker
        .start_game(&GameState::new(Rules::TROMP_TAYLORISH), &mut client)
        .await
        .unwrap();
    for _ in 0..2 {
        worker.playout(&mut client).await.unwrap();
    }
    let before = graph_snapshot(&worker);
    assert_eq!(before.len(), 3);
    let mut failing = inference_client(true);
    assert!(matches!(
        worker.playout(&mut failing).await,
        Err(SearchError::InferenceError(_))
    ));
    assert_eq!(
        worker.playout_path.len(),
        2,
        "failure must occur below two existing edges"
    );
    assert_eq!(worker.search_graph.node_store.len(), 2);
    assert_eq!(graph_snapshot(&worker), before);

    worker.playout(&mut client).await.unwrap();
    let after = graph_snapshot(&worker);
    assert_eq!(after.len(), 4);
    assert_eq!(worker.search_graph.node_store.len(), 3);
    for (old, new) in before.iter().zip(&after) {
        assert_eq!(new.visits, old.visits + 1);
        for (old_edge, new_edge) in old.edges.iter().zip(&new.edges) {
            assert_eq!(new_edge.1, old_edge.1 + 1);
            assert_eq!(new_edge.3, old_edge.3);
        }
    }
    assert_eq!(after[2].edges.len(), 1);
    assert_eq!(after[2].edges[0].1, 1);
    assert_eq!(after[3].visits, 1);
    assert!(after[3].edges.is_empty());
    assert!(worker.playout_path.is_empty());
    assert!(worker.visited_nodes.is_empty());
}

#[tokio::test]
async fn full_store_preserves_graph_and_reset_reuses_capacity() {
    let mut worker = SearchWorker::new(
        FixedArenaNodeStore::new(1),
        SearchParams::KATAGO_SELFPLAY8_MAIN_B18,
    );
    let mut client = inference_client_with_policy(false, focused_policy());
    worker
        .start_game(&GameState::new(Rules::TROMP_TAYLORISH), &mut client)
        .await
        .unwrap();
    worker.playout(&mut client).await.unwrap();
    let before = graph_snapshot(&worker);
    // A fresh one-shot client proves the inference before insertion succeeds.
    let mut one_shot = one_shot_inference_client(focused_policy());
    assert!(matches!(
        worker.playout(&mut one_shot).await,
        Err(SearchError::NodeStore(InsertError::StoreFull))
    ));
    assert_eq!(worker.playout_path.len(), 1);
    assert_eq!(worker.search_graph.node_store.len(), 1);
    assert_eq!(graph_snapshot(&worker), before);

    worker
        .reset_graph(&GameState::new(Rules::TROMP_TAYLORISH), 1, &mut client)
        .await
        .unwrap();
    assert_eq!(worker.search_graph.node_store.len(), 0);
    assert_eq!(worker.search_graph.node_store.capacity(), 1);
    assert!(worker.playout_path.is_empty());
    assert!(worker.visited_nodes.is_empty());
    worker.playout(&mut client).await.unwrap();
    assert_eq!(worker.search_graph.node_store.len(), 1);
    let root = &worker.search_graph.root.as_ref().unwrap().node;
    assert_eq!(root.visits(), 2);
    assert_eq!(root.edge_visit_sum(), 1);
}

#[tokio::test]
async fn reset_graph_borrows_the_callers_state_and_keeps_independent_snapshots() {
    let mut worker = worker();
    let mut client = inference_client(false);
    let mut game_state = GameState::new(Rules::TROMP_TAYLORISH);
    let initial_key = GraphKey::new(&game_state);
    worker.start_game(&game_state, &mut client).await.unwrap();

    // The caller retains ownership and can advance without changing the root.
    assert!(game_state.play(Loc::new(4, 4).unwrap()));
    assert_eq!(
        GraphKey::new(&worker.search_graph.root.as_ref().unwrap().game_state),
        initial_key
    );
    worker.playout(&mut client).await.unwrap();
    let next_key = GraphKey::new(&game_state);
    worker
        .reset_graph(&game_state, 16, &mut client)
        .await
        .unwrap();
    let root = worker.search_graph.root.as_ref().unwrap();
    assert_eq!(root.key, next_key);
    assert_eq!(GraphKey::new(&root.game_state), next_key);
    assert_eq!(GraphKey::new(worker.scratch_game_state()), next_key);
    assert_eq!(root.node.visits(), 1);
    assert_eq!(root.node.edges().count(), 0);
    assert_eq!(worker.search_graph.node_store.len(), 0);

    // Scratch traversal and later caller moves must not mutate the snapshot.
    worker.playout(&mut client).await.unwrap();
    assert_eq!(GraphKey::new(&game_state), next_key);
    assert!(game_state.play(Loc::PASS));
    let root = worker.search_graph.root.as_ref().unwrap();
    assert_eq!(GraphKey::new(&root.game_state), next_key);
    assert_eq!(root.game_state.consecutive_ending_passes(), 0);
}

#[tokio::test]
async fn resetting_graph_clears_stored_nodes_and_replaces_the_root() {
    let mut worker = worker();
    let mut client = inference_client(false);
    worker
        .start_game(&GameState::new(Rules::TROMP_TAYLORISH), &mut client)
        .await
        .unwrap();
    worker
        .search_graph
        .node_store
        .insert(GraphKey::from_raw(1))
        .unwrap();

    let mut next_state = GameState::new(Rules::TROMP_TAYLORISH);
    assert!(next_state.play(Loc::new(4, 4).unwrap()));
    let expected_key = GraphKey::new(&next_state);
    worker
        .reset_graph(&next_state, 16, &mut client)
        .await
        .unwrap();

    assert_eq!(worker.search_graph.node_store.len(), 0);
    assert_eq!(worker.search_graph.root.as_ref().unwrap().key, expected_key);
}

#[tokio::test]
async fn inference_failure_preserves_the_existing_graph() {
    let mut worker = worker();
    let mut working_client = inference_client_with_policy(false, focused_policy());
    worker
        .start_game(&GameState::new(Rules::TROMP_TAYLORISH), &mut working_client)
        .await
        .unwrap();
    // Use a nonzero center so an accidental reset to zero is observable.
    // Subsequent playouts back up the graph using this center.
    worker.recent_score_center = 3.0;
    for _ in 0..2 {
        worker.playout(&mut working_client).await.unwrap();
    }
    let before = graph_snapshot(&worker);
    let original_center = worker.recent_score_center;
    let original_scratch_key = GraphKey::new(worker.scratch_game_state());
    let original_key = worker.search_graph.root.as_ref().unwrap().key;

    let mut failing_client = inference_client(true);
    let mut replacement = GameState::new(Rules::TROMP_TAYLORISH);
    assert!(replacement.play(Loc::new(4, 4).unwrap()));
    let result = worker
        .reset_graph(&replacement, 16, &mut failing_client)
        .await;

    assert!(matches!(result, Err(SearchError::InferenceError(_))));
    assert_eq!(worker.search_graph.root.as_ref().unwrap().key, original_key);
    assert_eq!(worker.search_graph.node_store.len(), 2);
    assert_eq!(graph_snapshot(&worker), before);
    assert_eq!(worker.recent_score_center, original_center);
    assert_eq!(
        GraphKey::new(worker.scratch_game_state()),
        original_scratch_key
    );
    worker.playout(&mut working_client).await.unwrap();
    assert_eq!(worker.search_graph.node_store.len(), 3);
    let after = graph_snapshot(&worker);
    for (old, new) in before.iter().zip(&after) {
        assert_eq!(new.visits, old.visits + 1);
    }
    assert!(worker.playout_path.is_empty());
    assert!(worker.visited_nodes.is_empty());
}
