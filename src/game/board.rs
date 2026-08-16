use super::hash::{PositionHash, stone_hash};

// can make these runtime-configurable in the future
const BOARD_SIZE: usize = 9;
const STRIDE: usize = BOARD_SIZE + 1; // first element of each row is the wall
pub const ARRAY_LEN: usize = STRIDE * STRIDE + STRIDE + 1; //need bottom row of walls and bottom corner

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Empty = 0,
    Black = 1,
    White = 2,
    Wall = 3,
}

impl Color {
    fn is_empty(self) -> bool {
        match self {
            Self::Empty => true,
            _ => false,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    Black = 1,
    White = 2,
}

impl Player {
    pub fn opponent(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
}

impl From<Player> for Color {
    fn from(player: Player) -> Color {
        match player {
            Player::Black => Color::Black,
            Player::White => Color::White,
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loc(u16);

impl Loc {
    const NULL: Self = Self(0);
    pub const PASS: Self = Self(1);
    pub fn new(x: usize, y: usize) -> Option<Loc> {
        if x < BOARD_SIZE && y < BOARD_SIZE {
            Some(Self(((x + 1) + (y + 1) * STRIDE) as u16))
        } else {
            None
        }
    }
    pub fn index(self) -> usize {
        self.0 as usize // cant have u16
    }
    fn from_index(index: usize) -> Self {
        debug_assert!(index < ARRAY_LEN);
        Self(index as u16)
    }
    fn x(self) -> usize {
        (self.0 as usize) % STRIDE - 1
    }
    fn y(self) -> usize {
        (self.0 as usize) / STRIDE - 1
    }
    fn is_adjacent(loc1: Self, loc2: Self) -> bool {
        let difference = loc1.index().abs_diff(loc2.index());
        difference == 1 || difference == STRIDE
    }
    fn adjacent_indices(i: usize) -> [usize; 4] {
        [i + 1, i - STRIDE, i - 1, i + STRIDE] // unit circle direction lol
    }
}

#[derive(Debug, Clone, Copy)]
struct ChainData {
    num_locs: u16,
    num_liberties: u16,
}

#[derive(Clone)]
pub struct Board {
    colors: [Color; ARRAY_LEN], // flat board array
    chain_data: [ChainData; ARRAY_LEN],
    chain_head: [Loc; ARRAY_LEN],
    next_in_chain: [Loc; ARRAY_LEN],
    simple_ko: Option<Loc>,
    position_hash: PositionHash,
}

impl Board {
    pub fn new() -> Self {
        let mut colors = [Color::Empty; ARRAY_LEN];
        for i in 0..STRIDE {
            colors[i] = Color::Wall;
            colors[i + STRIDE * STRIDE] = Color::Wall;
            colors[i * STRIDE] = Color::Wall;
        }
        colors[ARRAY_LEN - 1] = Color::Wall;

        let init_data = ChainData {
            num_locs: 0,
            num_liberties: 0,
        };
        Self {
            colors,
            chain_data: [init_data; ARRAY_LEN],
            chain_head: [Loc(0); ARRAY_LEN],
            next_in_chain: [Loc(0); ARRAY_LEN],
            simple_ko: None,
            position_hash: 0,
        }
    }
    fn is_empty(&self) -> bool {
        self.colors
            .iter()
            .all(|&color| color != Color::Black && color != Color::White)
    }
    fn get_chain_size(&self, loc: Loc) -> u16 {
        self.chain_data[self.chain_head[loc.index()].index()].num_locs
    }
    fn get_num_liberties(&self, loc: Loc) -> u16 {
        self.chain_data[self.chain_head[loc.index()].index()].num_liberties
    }
    fn get_num_immediate_liberties(&self, loc: Loc) -> u16 {
        let i = loc.index();
        u16::from(self.colors[i + 1].is_empty())
            + u16::from(self.colors[i - STRIDE].is_empty())
            + u16::from(self.colors[i - 1].is_empty())
            + u16::from(self.colors[i + STRIDE].is_empty())
    }
    fn is_liberty_of(&self, loc: Loc, head: Loc) -> bool {
        let i = loc.index();
        let owner = self.colors[head.index()];
        Loc::adjacent_indices(i)
            .into_iter()
            .any(|adj| self.colors[adj] == owner && self.chain_head[adj] == head)
    }
    fn change_surrounding_liberties(&mut self, loc: Loc, player: Player, delta: i16) {
        //delta is only ever -1 or +1
        // can hand unroll later

        let i = loc.index();
        let owner = Color::from(player);
        let mut seen_heads = [Loc(0); 4]; //update distinct chains once
        let mut seen_len = 0;
        for adj in Loc::adjacent_indices(i) {
            if self.colors[adj] != owner {
                continue;
            }
            let adj_head = self.chain_head[adj];
            if seen_heads[..seen_len].contains(&adj_head) {
                continue;
            }
            let data = &mut self.chain_data[adj_head.index()];
            data.num_liberties = data
                .num_liberties
                .checked_add_signed(delta)
                .expect("liberty count underflow/overflow");

            seen_heads[seen_len] = adj_head;
            seen_len += 1;
        }
    }

    fn merge_chains(&mut self, loc1: Loc, loc2: Loc) {
        let mut small_head = self.chain_head[loc1.index()];
        let mut large_head = self.chain_head[loc2.index()];
        if self.chain_data[small_head.index()].num_locs
            > self.chain_data[large_head.index()].num_locs
        {
            std::mem::swap(&mut small_head, &mut large_head);
        }

        let mut cur = small_head;

        loop {
            // need to add new liberty in before adding it to that chain group
            let i = cur.index();
            for adj in Loc::adjacent_indices(i) {
                if self.colors[adj].is_empty()
                    && !self.is_liberty_of(Loc::from_index(adj), large_head)
                {
                    self.chain_data[large_head.index()].num_liberties += 1;
                }
            }

            //update the heads
            self.chain_head[i] = large_head;

            let next = self.next_in_chain[i];
            if next == small_head {
                break;
            }
            cur = next;
        }
        //merge the circles
        self.next_in_chain[cur.index()] = self.next_in_chain[large_head.index()];
        self.next_in_chain[large_head.index()] = small_head;

        //add the sizes
        let small_size = self.chain_data[small_head.index()].num_locs;
        self.chain_data[large_head.index()].num_locs += small_size;
    }

    fn remove_chain(&mut self, loc: Loc) -> u16 {
        let mut cur = loc;
        let mut counter = 0_u16;
        let current_color = self.colors[loc.index()];
        let other_player = match current_color {
            Color::Black => Player::White,
            Color::White => Player::Black,
            _ => unreachable!("can only remove stones"),
        };
        loop {
            let i = cur.index();

            counter += 1;
            self.colors[i] = Color::Empty; //clear

            //update position hash
            self.position_hash ^= stone_hash(cur, current_color);

            self.change_surrounding_liberties(cur, other_player, 1); //increment liberties

            let next = self.next_in_chain[i];
            if next == loc {
                break;
            }
            cur = next;
        }
        counter
    }

    pub fn play_move_assume_legal(&mut self, loc: Loc, player: Player) {
        //create new chain, merge with nearby chains, decrement opponent liberties, kill and increment liberties and mark ko
        self.simple_ko = None;
        if loc == Loc::PASS {
            return;
        }
        let i = loc.index();
        let player_color = Color::from(player);
        let opponent_player = player.opponent();
        let opponent_color = Color::from(opponent_player);
        let mut potential_ko_loc = None;
        let mut num_killed = 0_u16;

        //populate new chain
        self.colors[i] = player_color;
        self.chain_data[i] = ChainData {
            num_locs: 1,
            num_liberties: self.get_num_immediate_liberties(loc),
        };
        self.chain_head[i] = loc;
        self.next_in_chain[i] = loc;

        //update hash
        self.position_hash ^= stone_hash(loc, player_color);

        self.change_surrounding_liberties(loc, opponent_player, -1);
        for adj in Loc::adjacent_indices(i) {
            let adj_loc = Loc::from_index(adj);
            //merge chains
            if (self.colors[adj] == player_color) && (self.chain_head[adj] != self.chain_head[i]) {
                self.chain_data[self.chain_head[adj].index()].num_liberties -= 1; // same type so dont need that checked add thing
                self.merge_chains(adj_loc, loc)
            }
            //kill enemies
            else if (self.colors[adj] == opponent_color) && (self.get_num_liberties(adj_loc) == 0)
            {
                num_killed += self.remove_chain(adj_loc);
                potential_ko_loc = Some(adj_loc); //ko only happens when exactly one stone is killed
            }
        }

        // ko
        let new_chain_data = self.chain_data[self.chain_head[i].index()];
        if num_killed == 1 && new_chain_data.num_locs == 1 && new_chain_data.num_liberties == 1 {
            self.simple_ko = potential_ko_loc;
        }

        // suicide
        if new_chain_data.num_liberties == 0 {
            self.remove_chain(loc);
        }
    }
    fn is_ko_banned(&self, loc: Loc) -> bool {
        Some(loc) == self.simple_ko
    }
    fn is_suicide(&self, loc: Loc, player: Player) -> bool {
        let i = loc.index();
        let player_color = Color::from(player);
        let opponent_color = Color::from(player.opponent());
        for adj in Loc::adjacent_indices(i) {
            let adj_color = self.colors[adj];
            if adj_color.is_empty() {
                return false;
            }
            if adj_color == player_color {
                if self.get_num_liberties(Loc::from_index(adj)) > 1 {
                    return false;
                }
            } else if adj_color == opponent_color {
                if self.get_num_liberties(Loc::from_index(adj)) == 1 {
                    return false;
                }
            }
        }
        true
    }
    fn is_illegal_suicide(
        &self,
        loc: Loc,
        player: Player,
        multi_stone_suicide_legal: bool,
    ) -> bool {
        let i = loc.index();
        let player_color = Color::from(player);
        let opponent_color = Color::from(player.opponent());
        for adj in Loc::adjacent_indices(i) {
            let adj_color = self.colors[adj];
            if adj_color.is_empty() {
                return false;
            }
            if adj_color == player_color {
                if multi_stone_suicide_legal {
                    return false;
                }
                if self.get_num_liberties(Loc::from_index(adj)) > 1 {
                    return false;
                }
            } else if adj_color == opponent_color {
                if self.get_num_liberties(Loc::from_index(adj)) == 1 {
                    return false;
                }
            }
        }
        true
    }
    fn is_legal(&self, loc: Loc, player: Player, multi_stone_suicide_legal: bool) -> bool {
        if loc == Loc::PASS {
            return true;
        }
        if !self.colors[loc.index()].is_empty() {
            return false;
        }
        if self.is_ko_banned(loc) {
            return false;
        }
        !self.is_illegal_suicide(loc, player, multi_stone_suicide_legal)
    }
    pub fn is_legal_ignoring_ko(
        &self,
        loc: Loc,
        player: Player,
        multi_stone_suicide_legal: bool,
    ) -> bool {
        if loc == Loc::PASS {
            return true;
        }
        if !self.colors[loc.index()].is_empty() {
            return false;
        }
        !self.is_illegal_suicide(loc, player, multi_stone_suicide_legal)
    }
    pub fn position_hash(&self) -> PositionHash {
        self.position_hash
    }
    fn chain_iter(&self, start: Loc) -> impl Iterator<Item = Loc> + '_ {
        std::iter::successors(Some(start), move |&current| {
            let next = self.next_in_chain[current.index()];
            (next != start).then_some(next) // just an if statement lol
        })
    }
    pub fn get_position_hash_after_move(&self, loc: Loc, player: Player) -> PositionHash {
        if loc == Loc::PASS {
            return self.position_hash;
        }
        let mut new_position_hash = self.position_hash;
        let i = loc.index();
        let current_color = Color::from(player);
        let opponent_color = Color::from(player.opponent());

        let mut seen_heads = [Loc(0); 4]; // update distinct chains once
        let mut seen_len = 0;

        let mut suicide_heads = [Loc(0); 4];
        let mut suicide_len = 0;

        let mut suicide = true;
        for adj in Loc::adjacent_indices(i) {
            let adj_color = self.colors[adj];
            if adj_color == Color::Empty {
                suicide = false;
                continue;
            }
            if adj_color == Color::Wall {
                continue;
            }

            let adj_head = self.chain_head[adj];
            if seen_heads[..seen_len].contains(&adj_head) {
                continue;
            }

            let adj_liberties = self.chain_data[adj_head.index()].num_liberties;

            if adj_color == current_color {
                if adj_liberties > 1 {
                    suicide = false;
                } else {
                    // liberties = 1
                    suicide_heads[suicide_len] = adj_head;
                    suicide_len += 1;
                }
            } else if adj_color == opponent_color && adj_liberties == 1 {
                //kill them
                suicide = false;
                for killed_loc in self.chain_iter(adj_head) {
                    new_position_hash ^= stone_hash(killed_loc, opponent_color);
                }
            }

            seen_heads[seen_len] = adj_head;
            seen_len += 1;
        }
        if suicide {
            // go through each suicide chain and update position hash with each
            for &suicide_head in &suicide_heads[..suicide_len] {
                for killed_loc in self.chain_iter(suicide_head) {
                    new_position_hash ^= stone_hash(killed_loc, current_color);
                }
            }
        } else {
            new_position_hash ^= stone_hash(loc, current_color);
        }
        new_position_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(x: usize, y: usize) -> Loc {
        Loc::new(x, y).expect("test coordinates must be on the board")
    }

    fn play(board: &mut Board, x: usize, y: usize, player: Player) {
        board.play_move_assume_legal(loc(x, y), player);
    }

    fn assert_position_hash_matches_recomputation(board: &Board) {
        let mut recomputed = 0;

        for i in 0..ARRAY_LEN {
            let color = board.colors[i];
            if color == Color::Black || color == Color::White {
                recomputed ^= super::stone_hash(Loc::from_index(i), color);
            }
        }

        assert_eq!(board.position_hash(), recomputed);
    }

    #[test]
    fn loc_round_trips_coordinates_and_knows_adjacency() {
        let point = loc(4, 7);

        assert_eq!(point.x(), 4);
        assert_eq!(point.y(), 7);
        assert!(Loc::is_adjacent(point, loc(4, 6)));
        assert!(Loc::is_adjacent(point, loc(5, 7)));
        assert!(!Loc::is_adjacent(point, loc(5, 6)));
        assert!(!Loc::is_adjacent(point, loc(6, 7)));
        assert!(Loc::new(BOARD_SIZE, 0).is_none());
    }

    #[test]
    fn new_board_is_empty_and_a_center_stone_has_four_liberties() {
        let mut board = Board::new();
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
        let mut board = Board::new();
        let corner = loc(0, 0);

        play(&mut board, 0, 0, Player::Black);

        assert_eq!(board.get_chain_size(corner), 1);
        assert_eq!(board.get_num_liberties(corner), 2);
    }

    #[test]
    fn adjacent_friendly_stones_merge_into_one_chain() {
        let mut board = Board::new();
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
    fn surrounding_a_stone_captures_it_and_frees_liberties() {
        let mut board = Board::new();
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
        let mut board = Board::new();
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
        let mut board = Board::new();
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
        let mut board = Board::new();
        let occupied = loc(4, 4);

        play(&mut board, 4, 4, Player::Black);

        assert!(!board.is_legal(occupied, Player::White, true));
        assert!(!board.is_legal_ignoring_ko(occupied, Player::White, true));
        assert!(!board.is_legal(Loc::NULL, Player::Black, true));
        assert!(!board.is_legal_ignoring_ko(Loc::NULL, Player::Black, true));
    }

    #[test]
    fn pass_is_legal_and_clears_simple_ko_without_changing_stones() {
        let mut board = Board::new();
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
        let mut board = Board::new();
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
    fn position_hash_matches_recomputation_after_moves_and_pass() {
        let mut board = Board::new();
        let empty_hash = board.position_hash();

        assert_eq!(empty_hash, 0);
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
        let mut capture_board = Board::new();

        play(&mut capture_board, 4, 4, Player::Black);
        play(&mut capture_board, 4, 5, Player::Black);
        for (x, y) in [(3, 4), (3, 5), (5, 4), (5, 5), (4, 3), (4, 6)] {
            play(&mut capture_board, x, y, Player::White);
            assert_position_hash_matches_recomputation(&capture_board);
        }
        assert_eq!(capture_board.colors[loc(4, 4).index()], Color::Empty);
        assert_eq!(capture_board.colors[loc(4, 5).index()], Color::Empty);

        let mut suicide_board = Board::new();
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
        let mut capture_board = Board::new();

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

        let mut suicide_board = Board::new();
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
}
