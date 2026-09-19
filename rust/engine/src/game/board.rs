use super::hash::{Hash128, stone_hash};

// can make these runtime-configurable in the future
pub(crate) const BOARD_SIZE: usize = 9;
pub(crate) const STRIDE: usize = BOARD_SIZE + 1; // first element of each row is the wall
pub(crate) const ARRAY_LEN: usize = STRIDE * STRIDE + STRIDE + 1; //need bottom row of walls and bottom corner

mod scoring;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Color {
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
pub(crate) enum Player {
    Black = 1,
    White = 2,
}

impl Player {
    pub(crate) fn opponent(self) -> Self {
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
pub(crate) struct Loc(u16);

impl Loc {
    pub(crate) const NULL: Self = Self(0);
    pub(crate) const PASS: Self = Self(1);
    pub(crate) fn new(x: usize, y: usize) -> Option<Loc> {
        if x < BOARD_SIZE && y < BOARD_SIZE {
            Some(Self(((x + 1) + (y + 1) * STRIDE) as u16))
        } else {
            None
        }
    }
    pub(crate) fn index(self) -> usize {
        self.0 as usize // cant have u16
    }
    pub(crate) fn from_index(index: usize) -> Self {
        debug_assert!(index < ARRAY_LEN);
        Self(index as u16)
    }
    pub(crate) fn x(self) -> usize {
        (self.0 as usize) % STRIDE - 1
    }
    pub(crate) fn y(self) -> usize {
        (self.0 as usize) / STRIDE - 1
    }
    fn is_adjacent(loc1: Self, loc2: Self) -> bool {
        let difference = loc1.index().abs_diff(loc2.index());
        difference == 1 || difference == STRIDE
    }
    fn adjacent_indices(i: usize) -> [usize; 4] {
        [i + 1, i - STRIDE, i - 1, i + STRIDE] // unit circle direction lol
    }
    pub(crate) fn board_iter() -> impl Iterator<Item = Loc> {
        (1..=BOARD_SIZE).flat_map(|y| (1..=BOARD_SIZE).map(move |x| Loc((x + y * STRIDE) as u16)))
    }
    pub(crate) fn is_on_board(self) -> bool {
        let i = self.index();
        let padded_x = i % STRIDE;
        let padded_y = i / STRIDE;
        padded_x > 0 && padded_x <= BOARD_SIZE && padded_y > 0 && padded_y <= BOARD_SIZE
    }
}

#[derive(Debug, Clone, Copy)]
struct ChainData {
    num_locs: u16,
    num_liberties: u16,
}

#[derive(Clone)]
pub(crate) struct Board {
    colors: [Color; ARRAY_LEN], // flat board array
    chain_data: [ChainData; ARRAY_LEN],
    chain_head: [Loc; ARRAY_LEN],
    next_in_chain: [Loc; ARRAY_LEN],
    simple_ko: Option<Loc>,
    position_hash: Hash128,
}

impl Board {
    pub(crate) fn new() -> Self {
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
            chain_head: [Loc::NULL; ARRAY_LEN],
            next_in_chain: [Loc::NULL; ARRAY_LEN],
            simple_ko: None,
            position_hash: 0,
        }
    }
    pub(crate) fn color_at(&self, loc: Loc) -> Color {
        self.colors[loc.index()]
    }
    pub(crate) fn simple_ko(&self) -> Option<Loc> {
        self.simple_ko
    }
    pub(crate) fn is_adjacent_to_player(&self, loc: Loc, player: Player) -> bool {
        Loc::adjacent_indices(loc.index())
            .into_iter()
            .any(|i| self.colors[i] == Color::from(player))
    }
    pub(crate) fn would_capture(&self, loc: Loc, player: Player) -> bool {
        Loc::adjacent_indices(loc.index()).into_iter().any(|i| {
            self.colors[i] == Color::from(player.opponent())
                && self.get_num_liberties(Loc::from_index(i)) == 1
        })
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
            .any(|adj_i| self.colors[adj_i] == owner && self.chain_head[adj_i] == head)
    }
    fn change_surrounding_liberties(&mut self, loc: Loc, player: Player, delta: i16) {
        //delta is only ever -1 or +1
        // can hand unroll later

        let i = loc.index();
        let owner = Color::from(player);
        let mut seen_heads = [Loc::NULL; 4]; //update distinct chains once
        let mut seen_len = 0;
        for adj_i in Loc::adjacent_indices(i) {
            if self.colors[adj_i] != owner {
                continue;
            }
            let adj_head = self.chain_head[adj_i];
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
            for adj_i in Loc::adjacent_indices(i) {
                if self.colors[adj_i].is_empty()
                    && !self.is_liberty_of(Loc::from_index(adj_i), large_head)
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

    pub(crate) fn play_move_assume_legal(&mut self, loc: Loc, player: Player) {
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
        for adj_i in Loc::adjacent_indices(i) {
            let adj_loc = Loc::from_index(adj_i);
            //merge chains
            if (self.colors[adj_i] == player_color)
                && (self.chain_head[adj_i] != self.chain_head[i])
            {
                self.chain_data[self.chain_head[adj_i].index()].num_liberties -= 1; // same type so dont need that checked add thing
                self.merge_chains(adj_loc, loc)
            }
            //kill enemies
            else if (self.colors[adj_i] == opponent_color)
                && (self.get_num_liberties(adj_loc) == 0)
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
        for adj_i in Loc::adjacent_indices(i) {
            let adj_color = self.colors[adj_i];
            if adj_color.is_empty() {
                return false;
            }
            if adj_color == player_color {
                if self.get_num_liberties(Loc::from_index(adj_i)) > 1 {
                    return false;
                }
            } else if adj_color == opponent_color {
                if self.get_num_liberties(Loc::from_index(adj_i)) == 1 {
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
        for adj_i in Loc::adjacent_indices(i) {
            let adj_color = self.colors[adj_i];
            if adj_color.is_empty() {
                return false;
            }
            if adj_color == player_color {
                if multi_stone_suicide_legal {
                    return false;
                }
                if self.get_num_liberties(Loc::from_index(adj_i)) > 1 {
                    return false;
                }
            } else if adj_color == opponent_color {
                if self.get_num_liberties(Loc::from_index(adj_i)) == 1 {
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
    pub(crate) fn is_legal_ignoring_ko(
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
    pub(crate) fn position_hash(&self) -> Hash128 {
        self.position_hash
    }
    fn empty_region_pushes_count_over_bound(
        &self,
        initial_loc: Loc,
        counted: &mut [bool; ARRAY_LEN],
        count: &mut usize,
        bound: usize,
    ) -> bool {
        let initial_i = initial_loc.index();
        if counted[initial_i] {
            return false;
        }

        *count += 1;
        counted[initial_i] = true;
        if *count > bound {
            return true;
        }

        let mut queue = [Loc::NULL; ARRAY_LEN];
        let mut queue_head = 0;
        let mut queue_tail = 1;
        queue[0] = initial_loc;

        while queue_head < queue_tail {
            let current = queue[queue_head];
            queue_head += 1;

            for adj_i in Loc::adjacent_indices(current.index()) {
                if self.colors[adj_i] == Color::Empty && !counted[adj_i] {
                    *count += 1;
                    counted[adj_i] = true;
                    if *count > bound {
                        return true;
                    }

                    queue[queue_tail] = Loc::from_index(adj_i);
                    queue_tail += 1;
                }
            }
        }

        false
    }
    pub(crate) fn repetition_region_is_small(&self, loc: Loc, bound: usize) -> bool {
        if loc == Loc::NULL || loc == Loc::PASS {
            return true;
        }
        debug_assert!(loc.is_on_board());

        let mut count = 0;
        let loc_color = self.colors[loc.index()];

        if loc_color != Color::Empty {
            debug_assert!(loc_color == Color::Black || loc_color == Color::White);
            let data = self.chain_data[self.chain_head[loc.index()].index()];
            count += data.num_locs as usize;

            // Every liberty belongs to one of the empty regions counted below.
            if count + data.num_liberties as usize > bound {
                return false;
            }
        }

        let mut counted = [false; ARRAY_LEN];

        if loc_color == Color::Empty {
            !self.empty_region_pushes_count_over_bound(loc, &mut counted, &mut count, bound)
        } else {
            for chain_loc in self.chain_iter(loc) {
                for adj_i in Loc::adjacent_indices(chain_loc.index()) {
                    if self.colors[adj_i] == Color::Empty
                        && self.empty_region_pushes_count_over_bound(
                            Loc::from_index(adj_i),
                            &mut counted,
                            &mut count,
                            bound,
                        )
                    {
                        return false;
                    }
                }
            }
            true
        }
    }
    fn chain_iter(&self, start: Loc) -> impl Iterator<Item = Loc> + '_ {
        //'_ is tied to self
        std::iter::successors(Some(start), move |&current| {
            let next = self.next_in_chain[current.index()];
            (next != start).then_some(next) // just an if statement lol
        })
    }
    pub(crate) fn get_position_hash_after_move(&self, loc: Loc, player: Player) -> Hash128 {
        //see what stones get removed if you do a move. order of moves doesnt matter for the hash
        if loc == Loc::PASS {
            return self.position_hash;
        }
        let mut new_position_hash = self.position_hash;
        let i = loc.index();
        let current_color = Color::from(player);
        let opponent_color = Color::from(player.opponent());

        let mut seen_heads = [Loc::NULL; 4]; // update distinct chains once
        let mut seen_len = 0;

        let mut suicide_heads = [Loc::NULL; 4];
        let mut suicide_len = 0;

        let mut suicide = true;
        for adj_i in Loc::adjacent_indices(i) {
            let adj_color = self.colors[adj_i];
            if adj_color == Color::Empty {
                suicide = false;
                continue;
            }
            if adj_color == Color::Wall {
                continue;
            }

            let adj_head = self.chain_head[adj_i];
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
mod tests;
