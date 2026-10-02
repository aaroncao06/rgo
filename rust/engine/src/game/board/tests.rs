use super::*;

const TEST_BOARD_DIM: usize = 9;

fn loc(x: usize, y: usize) -> Loc {
    crate::game::board::Board::new(9)
        .loc(x, y)
        .expect("test coordinates must be on the board")
}

fn play(board: &mut Board, x: usize, y: usize, player: Player) {
    board.play_move_assume_legal(loc(x, y), player);
}

fn assert_position_hash_matches_recomputation(board: &Board) {
    let mut recomputed = super::board_dim_hash(board.dim());

    for i in 0..BOARD_STORAGE_LEN {
        let color = board.colors[i];
        if color == Color::Black || color == Color::White {
            recomputed ^= super::stone_hash(Board::loc_from_index(i), color);
        }
    }

    assert_eq!(board.position_hash(), recomputed);
}

fn assert_chain_metadata_matches_recomputation(board: &Board) {
    let mut visited = [false; BOARD_STORAGE_LEN];

    for start in board.locs() {
        let start_i = start.index();
        let color = board.colors[start_i];
        if visited[start_i] || (color != Color::Black && color != Color::White) {
            continue;
        }

        let mut component = [false; BOARD_STORAGE_LEN];
        let mut liberties = [false; BOARD_STORAGE_LEN];
        let mut queue = [Loc::NULL; BOARD_STORAGE_LEN];
        let mut queue_head = 0;
        let mut queue_tail = 1;
        queue[0] = start;
        visited[start_i] = true;

        while queue_head < queue_tail {
            let current = queue[queue_head];
            queue_head += 1;
            let current_i = current.index();
            component[current_i] = true;

            for adj_i in Board::adjacent_indices(current_i) {
                if board.colors[adj_i] == color && !visited[adj_i] {
                    visited[adj_i] = true;
                    queue[queue_tail] = Board::loc_from_index(adj_i);
                    queue_tail += 1;
                } else if board.colors[adj_i] == Color::Empty {
                    liberties[adj_i] = true;
                }
            }
        }

        let expected_size = component.iter().filter(|&&present| present).count() as u16;
        let expected_liberties = liberties.iter().filter(|&&present| present).count() as u16;
        let head = board.chain_head[start_i];

        assert_eq!(board.get_chain_size(start), expected_size);
        assert_eq!(board.get_num_liberties(start), expected_liberties);
        assert_eq!(board.colors[head.index()], color);

        let mut linked_list = [false; BOARD_STORAGE_LEN];
        let mut linked_list_size = 0;
        for chain_loc in board.chain_iter(head).take(BOARD_STORAGE_LEN) {
            let chain_i = chain_loc.index();
            assert!(!linked_list[chain_i], "chain list repeated before closing");
            assert_eq!(board.colors[chain_i], color);
            assert_eq!(board.chain_head[chain_i], head);
            linked_list[chain_i] = true;
            linked_list_size += 1;
        }
        assert_eq!(linked_list_size, expected_size as usize);
        assert_eq!(linked_list, component);
    }
}

fn board_from_ascii(rows: [&str; TEST_BOARD_DIM]) -> Board {
    let mut board = Board::new(9);

    for (stone, player) in [('x', Player::Black), ('o', Player::White)] {
        for (y, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), TEST_BOARD_DIM);
            for (x, cell) in row.bytes().enumerate() {
                if cell == stone as u8 {
                    board.play_move_assume_legal(loc(x, y), player);
                }
            }
        }
    }

    for (y, row) in rows.iter().enumerate() {
        for (x, cell) in row.bytes().enumerate() {
            let expected = match cell {
                b'x' => Color::Black,
                b'o' => Color::White,
                b'.' => Color::Empty,
                _ => panic!("invalid board character"),
            };
            assert_eq!(board.colors[loc(x, y).index()], expected);
        }
    }
    assert_chain_metadata_matches_recomputation(&board);
    assert_position_hash_matches_recomputation(&board);
    board
}

fn assert_area_rows(area: &[Color; BOARD_STORAGE_LEN], expected: [&str; TEST_BOARD_DIM]) {
    for (y, row) in expected.iter().enumerate() {
        assert_eq!(row.len(), TEST_BOARD_DIM);
        for (x, cell) in row.bytes().enumerate() {
            let expected_color = match cell {
                b'X' => Color::Black,
                b'O' => Color::White,
                b'.' => Color::Empty,
                _ => panic!("invalid area character"),
            };
            assert_eq!(area[loc(x, y).index()], expected_color, "at ({x}, {y})");
        }
    }
}

#[test]
fn board_round_trips_coordinates_and_checks_bounds() {
    let board = Board::new(9);
    let point = board.loc(4, 7).unwrap();
    assert_eq!(board.coords(point), Some((4, 7)));
    assert!(board.loc(9, 0).is_none());
    assert!(board.loc(0, 9).is_none());
    assert!(board.loc(usize::MAX, 0).is_none());
    assert_eq!(board.coords(Loc::PASS), None);
    assert_eq!(board.coords(Loc::NULL), None);
}

#[test]
fn active_geometry_and_wall_padding_follow_the_board_dim() {
    for dim in 1..=MAX_BOARD_DIM {
        let board = Board::new(dim);
        assert_eq!(board.dim(), dim);
        assert_eq!(board.locs().count(), dim * dim);
        for (i, &color) in board.colors.iter().enumerate() {
            let point = Board::loc_from_index(i);
            let active = (1..=dim).contains(&(i % STRIDE)) && (1..=dim).contains(&(i / STRIDE));
            assert_eq!(board.is_on_board(point), active);
            assert_eq!(color, if active { Color::Empty } else { Color::Wall });
        }
        for point in board.locs() {
            let (x, y) = board.coords(point).unwrap();
            assert_eq!(board.loc(x, y), Some(point));
            for neighbor in Board::adjacent_indices(point.index()) {
                assert!(neighbor < BOARD_STORAGE_LEN);
            }
        }
        assert!(board.loc(dim, 0).is_none());
        assert!(board.loc(0, dim).is_none());
    }
}

#[test]
fn capture_at_a_smaller_board_edge_preserves_chain_metadata() {
    let mut board = Board::new(2);
    let corner = board.loc(0, 0).unwrap();
    board.play_move_assume_legal(corner, Player::Black);
    assert_eq!(board.get_num_liberties(corner), 2);
    for (x, y) in [(1, 0), (0, 1)] {
        let point = board.loc(x, y).unwrap();
        assert!(board.is_legal_ignoring_ko(point, Player::White, true));
        board.play_move_assume_legal(point, Player::White);
    }
    assert_eq!(board.color_at(corner), Color::Empty);
    assert_chain_metadata_matches_recomputation(&board);
    assert_position_hash_matches_recomputation(&board);
    let area = board.calculate_area(true);
    assert!(
        board
            .locs()
            .all(|point| area[point.index()] == Color::White)
    );
}

#[test]
fn new_board_is_empty_and_a_center_stone_has_four_liberties() {
    let mut board = Board::new(9);
    let center = loc(4, 4);

    assert!(board.is_empty());

    play(&mut board, 4, 4, Player::Black);

    assert!(!board.is_empty());
    assert_eq!(board.get_chain_size(center), 1);
    assert_eq!(board.get_num_liberties(center), 4);
    assert_eq!(board.get_num_immediate_liberties(center), 4);
}

#[test]
fn corner_stone_has_two_liberties() {
    let mut board = Board::new(9);
    let corner = loc(0, 0);

    play(&mut board, 0, 0, Player::Black);

    assert_eq!(board.get_chain_size(corner), 1);
    assert_eq!(board.get_num_liberties(corner), 2);
}

#[test]
fn repetition_region_counts_a_stone_chain_and_its_adjacent_empty_regions() {
    let board = board_from_ascii([
        ".........",
        ".........",
        ".........",
        "....oo...",
        "...ox.o..",
        "....oo...",
        ".........",
        ".........",
        ".........",
    ]);
    let played = loc(4, 4);

    // One Black stone plus its enclosed one-point liberty.
    assert!(!board.repetition_region_is_small(played, 1));
    assert!(board.repetition_region_is_small(played, 2));
}

#[test]
fn repetition_region_counts_the_empty_component_after_suicide() {
    let board = board_from_ascii([
        ".........",
        ".........",
        ".........",
        "....oo...",
        "...o..o..",
        "....oo...",
        ".........",
        ".........",
        ".........",
    ]);
    let played = loc(4, 4);

    // The move location belongs to an enclosed two-point empty region.
    assert!(!board.repetition_region_is_small(played, 1));
    assert!(board.repetition_region_is_small(played, 2));
    assert!(board.repetition_region_is_small(Loc::PASS, 0));
}

#[test]
fn adjacent_friendly_stones_merge_into_one_chain() {
    let mut board = Board::new(9);
    let left = loc(4, 4);
    let right = loc(5, 4);

    play(&mut board, 4, 4, Player::Black);
    play(&mut board, 5, 4, Player::Black);

    assert_eq!(
        board.chain_head[left.index()],
        board.chain_head[right.index()]
    );
    assert_eq!(board.get_chain_size(left), 2);
    assert_eq!(board.get_chain_size(right), 2);
    assert_eq!(board.get_num_liberties(left), 6);
}

#[test]
fn one_move_merges_two_preexisting_friendly_chains() {
    let mut board = Board::new(9);
    let left = loc(3, 4);
    let bridge = loc(4, 4);
    let right = loc(5, 4);

    play(&mut board, 3, 4, Player::Black);
    play(&mut board, 5, 4, Player::Black);
    assert_ne!(
        board.chain_head[left.index()],
        board.chain_head[right.index()]
    );

    play(&mut board, 4, 4, Player::Black);

    assert_eq!(
        board.chain_head[left.index()],
        board.chain_head[bridge.index()]
    );
    assert_eq!(
        board.chain_head[right.index()],
        board.chain_head[bridge.index()]
    );
    assert_eq!(board.get_chain_size(bridge), 3);
    assert_eq!(board.get_num_liberties(bridge), 8);
    assert_chain_metadata_matches_recomputation(&board);
}

#[test]
fn touching_one_chain_from_two_directions_updates_its_liberty_once() {
    let mut board = Board::new(9);
    let played = loc(4, 4);
    let chain_stone = loc(3, 4);

    for (x, y) in [(3, 4), (3, 3), (4, 3)] {
        play(&mut board, x, y, Player::White);
    }
    assert_eq!(board.get_num_liberties(chain_stone), 7);

    play(&mut board, 4, 4, Player::Black);

    assert_eq!(board.get_num_liberties(chain_stone), 6);
    assert_chain_metadata_matches_recomputation(&board);
    assert_eq!(board.colors[played.index()], Color::Black);
}

#[test]
fn surrounding_a_stone_captures_it_and_frees_liberties() {
    let mut board = Board::new(9);
    let captured = loc(4, 4);
    let top = loc(4, 3);
    let capture = loc(4, 5);

    play(&mut board, 4, 4, Player::White);
    play(&mut board, 4, 3, Player::Black);
    play(&mut board, 3, 4, Player::Black);
    play(&mut board, 5, 4, Player::Black);

    // The final Black move has no directly empty neighbor, but captures
    // White's one-liberty chain and is therefore not suicide.
    assert!(!board.is_suicide(capture, Player::Black));
    assert!(board.is_legal(capture, Player::Black, false));

    play(&mut board, 4, 5, Player::Black);

    assert_eq!(board.colors[captured.index()], Color::Empty);
    assert_eq!(board.get_num_liberties(top), 4);
}

#[test]
fn single_stone_ko_marks_the_captured_point() {
    let mut board = Board::new(9);
    let captured = loc(4, 4);
    let capture = loc(4, 5);

    // Surround the White stone except at `capture`.
    play(&mut board, 4, 3, Player::Black);
    play(&mut board, 3, 4, Player::Black);
    play(&mut board, 5, 4, Player::Black);
    play(&mut board, 4, 4, Player::White);

    // Surround the eventual Black capturing stone with White, so it stays
    // an isolated one-liberty stone after the capture.
    play(&mut board, 4, 6, Player::White);
    play(&mut board, 3, 5, Player::White);
    play(&mut board, 5, 5, Player::White);

    play(&mut board, 4, 5, Player::Black);

    assert_eq!(board.colors[captured.index()], Color::Empty);
    assert_eq!(board.get_chain_size(capture), 1);
    assert_eq!(board.get_num_liberties(capture), 1);
    assert_eq!(board.simple_ko, Some(captured));
    assert!(board.is_ko_banned(captured));
    assert!(!board.is_legal(captured, Player::White, true));
    assert!(board.is_legal_ignoring_ko(captured, Player::White, true));
}

#[test]
fn assume_legal_applies_suicide_after_captures_are_resolved() {
    let mut board = Board::new(9);
    let suicide_point = loc(4, 4);

    play(&mut board, 4, 3, Player::White);
    play(&mut board, 5, 4, Player::White);
    play(&mut board, 4, 5, Player::White);
    play(&mut board, 3, 4, Player::White);

    play(&mut board, 4, 4, Player::Black);

    assert_eq!(board.colors[suicide_point.index()], Color::Empty);
}

#[test]
fn legality_rejects_occupied_and_wall_locations() {
    let mut board = Board::new(9);
    let occupied = loc(4, 4);

    play(&mut board, 4, 4, Player::Black);

    assert!(!board.is_legal(occupied, Player::White, true));
    assert!(!board.is_legal_ignoring_ko(occupied, Player::White, true));
    assert!(!board.is_legal(Loc::NULL, Player::Black, true));
    assert!(!board.is_legal_ignoring_ko(Loc::NULL, Player::Black, true));
}

#[test]
fn pass_is_legal_and_clears_simple_ko_without_changing_stones() {
    let mut board = Board::new(9);
    let stone = loc(4, 4);

    play(&mut board, 4, 4, Player::Black);
    board.simple_ko = Some(loc(3, 3));

    assert!(board.is_legal(Loc::PASS, Player::White, true));
    assert!(board.is_legal_ignoring_ko(Loc::PASS, Player::White, true));

    board.play_move_assume_legal(Loc::PASS, Player::White);

    assert_eq!(board.simple_ko, None);
    assert_eq!(board.colors[stone.index()], Color::Black);
}

#[test]
fn single_stone_suicide_is_illegal_under_both_suicide_settings() {
    let mut board = Board::new(9);
    let suicide_point = loc(4, 4);

    play(&mut board, 4, 3, Player::White);
    play(&mut board, 5, 4, Player::White);
    play(&mut board, 4, 5, Player::White);
    play(&mut board, 3, 4, Player::White);

    assert!(board.is_suicide(suicide_point, Player::Black));
    assert!(board.is_illegal_suicide(suicide_point, Player::Black, false));
    assert!(board.is_illegal_suicide(suicide_point, Player::Black, true));
    assert!(!board.is_legal(suicide_point, Player::Black, false));
    assert!(!board.is_legal(suicide_point, Player::Black, true));
}

#[test]
fn multi_stone_suicide_rule_changes_legality_and_removes_the_chain() {
    let mut board = Board::new(9);
    let existing_stone = loc(4, 4);
    let suicide_move = loc(4, 5);

    play(&mut board, 4, 4, Player::Black);
    for (x, y) in [(4, 3), (3, 4), (5, 4), (3, 5), (5, 5), (4, 6)] {
        play(&mut board, x, y, Player::White);
    }

    assert_eq!(board.get_num_liberties(existing_stone), 1);
    assert!(!board.is_legal_ignoring_ko(suicide_move, Player::Black, false));
    assert!(board.is_legal_ignoring_ko(suicide_move, Player::Black, true));

    board.play_move_assume_legal(suicide_move, Player::Black);

    assert_eq!(board.colors[existing_stone.index()], Color::Empty);
    assert_eq!(board.colors[suicide_move.index()], Color::Empty);
    assert_chain_metadata_matches_recomputation(&board);
    assert_position_hash_matches_recomputation(&board);
}

#[test]
fn randomized_play_preserves_incremental_chain_and_hash_invariants() {
    for dim in 1..=MAX_BOARD_DIM {
        let mut board = Board::new(dim);
        let mut player = Player::Black;
        let mut random_state = 0x7267_6f5f_7465_7374_u64;

        for _ in 0..2_000 {
            random_state = random_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let action = (random_state as usize) % (dim * dim + 1);
            let move_loc = if action == dim * dim {
                Loc::PASS
            } else {
                board.loc(action % dim, action / dim).unwrap()
            };

            if !board.is_legal(move_loc, player, true) {
                continue;
            }

            let predicted_hash = board.get_position_hash_after_move(move_loc, player);
            board.play_move_assume_legal(move_loc, player);

            assert_eq!(board.position_hash(), predicted_hash);
            assert_position_hash_matches_recomputation(&board);
            assert_chain_metadata_matches_recomputation(&board);
            player = player.opponent();
        }
    }
}

#[test]
fn position_hash_matches_recomputation_after_moves_and_pass() {
    let mut board = Board::new(9);
    let empty_hash = board.position_hash();

    assert_position_hash_matches_recomputation(&board);

    board.play_move_assume_legal(Loc::PASS, Player::Black);
    assert_eq!(board.position_hash(), empty_hash);
    assert_position_hash_matches_recomputation(&board);

    play(&mut board, 4, 4, Player::Black);
    assert_position_hash_matches_recomputation(&board);

    play(&mut board, 0, 0, Player::White);
    assert_position_hash_matches_recomputation(&board);
}

#[test]
fn position_hash_matches_recomputation_after_multi_stone_capture_and_suicide() {
    let mut capture_board = Board::new(9);

    play(&mut capture_board, 4, 4, Player::Black);
    play(&mut capture_board, 4, 5, Player::Black);
    for (x, y) in [(3, 4), (3, 5), (5, 4), (5, 5), (4, 3), (4, 6)] {
        play(&mut capture_board, x, y, Player::White);
        assert_position_hash_matches_recomputation(&capture_board);
    }
    assert_eq!(capture_board.colors[loc(4, 4).index()], Color::Empty);
    assert_eq!(capture_board.colors[loc(4, 5).index()], Color::Empty);

    let mut suicide_board = Board::new(9);
    for (x, y) in [(4, 3), (5, 4), (4, 5), (3, 4)] {
        play(&mut suicide_board, x, y, Player::White);
    }
    let before_suicide = suicide_board.position_hash();

    play(&mut suicide_board, 4, 4, Player::Black);

    assert_eq!(suicide_board.colors[loc(4, 4).index()], Color::Empty);
    assert_eq!(suicide_board.position_hash(), before_suicide);
    assert_position_hash_matches_recomputation(&suicide_board);
}

#[test]
fn position_hash_after_move_matches_played_position() {
    let mut capture_board = Board::new(9);

    play(&mut capture_board, 4, 4, Player::Black);
    play(&mut capture_board, 4, 5, Player::Black);
    for (x, y) in [(3, 4), (3, 5), (5, 4), (5, 5), (4, 3)] {
        play(&mut capture_board, x, y, Player::White);
    }

    let capture = loc(4, 6);
    let capture_hash = capture_board.get_position_hash_after_move(capture, Player::White);
    let mut captured = capture_board.clone();
    captured.play_move_assume_legal(capture, Player::White);
    assert_eq!(capture_hash, captured.position_hash());

    let mut suicide_board = Board::new(9);
    for (x, y) in [(4, 3), (5, 4), (4, 5), (3, 4)] {
        play(&mut suicide_board, x, y, Player::White);
    }

    let suicide = loc(4, 4);
    let suicide_hash = suicide_board.get_position_hash_after_move(suicide, Player::Black);
    let mut suicided = suicide_board.clone();
    suicided.play_move_assume_legal(suicide, Player::Black);
    assert_eq!(suicide_hash, suicided.position_hash());

    assert_eq!(
        suicide_board.get_position_hash_after_move(Loc::PASS, Player::Black),
        suicide_board.position_hash(),
    );
}

#[test]
fn area_of_an_empty_board_is_neutral() {
    let board = Board::new(9);
    let area = board.calculate_area(true);

    for point in board.locs() {
        assert_eq!(area[point.index()], Color::Empty);
    }
}

#[test]
fn strict_area_rejects_a_chain_without_two_vital_regions() {
    let mut board = Board::new(9);
    let stone = loc(4, 4);
    play(&mut board, 4, 4, Player::Black);

    let mut strict_area = [Color::Empty; BOARD_STORAGE_LEN];
    board.calculate_area_for_player(Player::Black, false, false, true, &mut strict_area);

    assert_eq!(strict_area[stone.index()], Color::Empty);
    assert_eq!(strict_area[loc(0, 0).index()], Color::Empty);
}

#[test]
fn strict_area_keeps_a_two_eye_group_and_its_eyes() {
    let mut board = Board::new(9);

    // A connected Black group surrounding two separate one-point eyes.
    for (x, y) in [
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
    ] {
        play(&mut board, x, y, Player::Black);
    }

    let mut strict_area = [Color::Empty; BOARD_STORAGE_LEN];
    board.calculate_area_for_player(Player::Black, false, false, true, &mut strict_area);

    assert_eq!(strict_area[loc(4, 4).index()], Color::Black);
    assert_eq!(strict_area[loc(6, 4).index()], Color::Black);
    assert_eq!(strict_area[loc(5, 4).index()], Color::Black);
    assert_eq!(strict_area[loc(0, 0).index()], Color::Empty);
}

#[test]
fn default_area_leaves_shared_exterior_neutral() {
    let mut board = Board::new(9);

    // Same two-eye Black group, plus a distant White stone. The exterior
    // region contains both colors, so neither side can claim it.
    for (x, y) in [
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
    ] {
        play(&mut board, x, y, Player::Black);
    }
    play(&mut board, 0, 0, Player::White);

    let area = board.calculate_area(true);

    assert_eq!(area[loc(4, 4).index()], Color::Black);
    assert_eq!(area[loc(6, 4).index()], Color::Black);
    assert_eq!(area[loc(5, 4).index()], Color::Black);
    assert_eq!(area[loc(0, 0).index()], Color::White);
    assert_eq!(area[loc(0, 1).index()], Color::Empty);
}

#[test]
fn surrounded_pass_dead_stone_is_scored_for_the_surrounding_player() {
    let mut board = Board::new(9);

    // One connected Black chain surrounds two one-point eyes and a third
    // chamber containing a White stone with one remaining liberty.
    for x in 1..=8 {
        play(&mut board, x, 3, Player::Black);
        play(&mut board, x, 5, Player::Black);
    }
    for x in [1, 3, 5, 8] {
        play(&mut board, x, 4, Player::Black);
    }
    let dead_white = loc(6, 4);
    play(&mut board, 6, 4, Player::White);

    let area = board.calculate_area(true);

    assert_eq!(board.colors[dead_white.index()], Color::White);
    assert_eq!(area[dead_white.index()], Color::Black);
    assert_eq!(area[loc(7, 4).index()], Color::Black);
}

#[test]
fn pass_alive_analysis_respects_multi_stone_suicide_legality() {
    // Ported from KataGo's "Area 2" regression position. Treating
    // multi-stone suicide as legal changes which chains satisfy Benson's
    // vital-region condition.
    let board = board_from_ascii([
        "x.oooooo.",
        "oox..xx.o",
        "o...xox.o",
        "o...x.x.o",
        "oxxx.xx.o",
        "ox..x...o",
        "o.xox...o",
        "o.xxx...o",
        ".ooooooo.",
    ]);

    let suicide_illegal_area = board.calculate_area(false);
    assert_area_rows(
        &suicide_illegal_area,
        [
            "OOOOOOOOO",
            "OOX..XX.O",
            "O...XXX.O",
            "O...XXX.O",
            "OXXXXXX.O",
            "OXXXX...O",
            "O.XXX...O",
            "O.XXX...O",
            "OOOOOOOOO",
        ],
    );

    let suicide_legal_area = board.calculate_area(true);
    assert_area_rows(
        &suicide_legal_area,
        [
            "X.OOOOOOO",
            "OOX..XX.O",
            "O...XOX.O",
            "O...X.X.O",
            "OXXXXXX.O",
            "OX..X...O",
            "O.XOX...O",
            "O.XXX...O",
            "OOOOOOOOO",
        ],
    );
}
