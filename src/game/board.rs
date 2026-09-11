use super::hash::{Hash128, stone_hash};

// can make these runtime-configurable in the future
pub(crate) const BOARD_SIZE: usize = 9;
pub(crate) const STRIDE: usize = BOARD_SIZE + 1; // first element of each row is the wall
pub(crate) const ARRAY_LEN: usize = STRIDE * STRIDE + STRIDE + 1; //need bottom row of walls and bottom corner

const MAX_PLAYER_HEADS: usize = (BOARD_SIZE * BOARD_SIZE + 1) / 2;
const MAX_REGIONS: usize = (BOARD_SIZE * BOARD_SIZE + 1) / 2 + 1;
const VITAL_FOR_CHAIN_HEADS_MAX_LEN: usize = MAX_REGIONS * 4; // max number of (region, chain-head) vital relations for a player

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

    pub(crate) fn calculate_area(&self, multi_stone_suicide_legal: bool) -> [Color; ARRAY_LEN] {
        let mut result = [Color::Empty; ARRAY_LEN];
        self.calculate_area_for_player(
            Player::Black,
            true,
            true,
            multi_stone_suicide_legal,
            &mut result,
        );
        self.calculate_area_for_player(
            Player::White,
            true,
            true,
            multi_stone_suicide_legal,
            &mut result,
        );
        for loc in Loc::board_iter() {
            let i = loc.index();
            if result[i].is_empty() {
                result[i] = self.colors[i];
            }
        }
        result
    }

    fn calculate_area_for_player(
        &self,
        player: Player,
        safe_big: bool,
        unsafe_big: bool,
        multi_stone_suicide_legal: bool,
        result: &mut [Color; ARRAY_LEN],
    ) {
        let player_color = Color::from(player);
        let opponent_color = Color::from(player.opponent());

        let mut region_idx_by_loc = [-1_i16; ARRAY_LEN]; //index regions; -1 means not in region
        let mut next_in_region = [Loc::NULL; ARRAY_LEN]; // basically next_in_chain but for regions. region contains either empty or opponent

        let mut borders_non_pass_alive = [false; MAX_REGIONS]; // if a region head is touching a "dead" group
        let mut vital_for_chain_heads = [Loc::NULL; VITAL_FOR_CHAIN_HEADS_MAX_LEN]; // for each region what chain heads are they vital for
        let mut vital_for_chain_heads_total = 0_usize; // len

        let mut num_regions = 0_usize;
        let mut region_heads = [Loc::NULL; MAX_REGIONS]; // map region index to region head location, starting point for walking next_in_region

        // for each region, the range of indices in vital_for_chain_heads that correspond to it
        let mut vital_start = [0_usize; MAX_REGIONS];
        let mut vital_len = [0_usize; MAX_REGIONS];

        let mut num_internal_spaces_max_2 = [0_u8; MAX_REGIONS]; // how many cells in the region don't touch the current player's stones, cap at 2
        let mut contains_opponent = [false; MAX_REGIONS]; // does the region contain any opponent stones

        let mut build_region_queue = [Loc::NULL; ARRAY_LEN];
        let mut player_has_stones = false;

        //BUILD REGIONS
        for loc in Loc::board_iter() {
            let i = loc.index();
            if region_idx_by_loc[i] != -1 {
                continue;
            }
            if !self.colors[i].is_empty() {
                if self.colors[i] == player_color {
                    player_has_stones = true;
                }
                continue;
            }

            //start work on unassigned empty loc
            let region_idx = num_regions;
            num_regions += 1;
            vital_start[region_idx] = vital_for_chain_heads_total;
            vital_len[region_idx] = 0;
            num_internal_spaces_max_2[region_idx] = 0;
            contains_opponent[region_idx] = false;
            region_heads[region_idx] = loc;

            // initialize candidate vital chain heads
            let mut initial_len = 0;
            for adj_i in Loc::adjacent_indices(i) {
                if self.colors[adj_i] == player_color {
                    let adj_chain_head = self.chain_head[adj_i];

                    if !vital_for_chain_heads
                        [vital_for_chain_heads_total..vital_for_chain_heads_total + initial_len]
                        .contains(&adj_chain_head)
                    {
                        vital_for_chain_heads[vital_for_chain_heads_total + initial_len] =
                            adj_chain_head;
                        initial_len += 1;
                    }
                }
            }
            vital_len[region_idx] = initial_len;

            //BUILD REGION
            let mut queue_head = 0;
            let mut queue_tail = 1;
            build_region_queue[0] = loc;
            let mut tail_loc = loc;

            region_idx_by_loc[i] = region_idx as i16; // need to assign before enqueuing

            while queue_head != queue_tail {
                let current_loc = build_region_queue[queue_head];
                queue_head += 1; //popped
                let current_index = current_loc.index();
                let current_color = self.colors[current_index]; // empty or enemy

                // filter candididate vital chains
                let old_len = vital_len[region_idx];
                if old_len > 0 && (multi_stone_suicide_legal || current_color.is_empty()) {
                    let start = vital_start[region_idx];
                    let mut new_len = 0;
                    for offset in 0..old_len {
                        // go over each of the candidates, keep if they are vital for this loc
                        let chain_head = vital_for_chain_heads[start + offset];
                        if self.is_liberty_of(current_loc, chain_head) {
                            vital_for_chain_heads[start + new_len] = chain_head;
                            new_len += 1;
                        }
                    }
                    vital_len[region_idx] = new_len;
                }

                //count internal cells that dont touch the main player
                if num_internal_spaces_max_2[region_idx] < 2
                    && !Loc::adjacent_indices(current_index)
                        .into_iter()
                        .any(|adj_i| self.colors[adj_i] == player_color)
                {
                    num_internal_spaces_max_2[region_idx] += 1;
                }

                if current_color == opponent_color {
                    contains_opponent[region_idx] = true;
                }

                // wire up circular linked list
                next_in_region[current_index] = tail_loc;
                tail_loc = current_loc;

                //enqueue neighbors if not yet enqued and not curretnn color
                for adj_i in Loc::adjacent_indices(current_index) {
                    if (self.colors[adj_i] == Color::Empty || self.colors[adj_i] == opponent_color)
                        && region_idx_by_loc[adj_i] == -1
                    {
                        region_idx_by_loc[adj_i] = region_idx as i16;
                        build_region_queue[queue_tail] = Loc::from_index(adj_i);
                        queue_tail += 1;
                    }
                }
            }

            next_in_region[i] = tail_loc;

            vital_for_chain_heads_total += vital_len[region_idx];
        }

        //regions are built, initialize list of all player chain heads
        let mut all_player_heads = [Loc::NULL; MAX_PLAYER_HEADS]; // both alive and dead
        let mut num_player_heads = 0_usize;
        for loc in Loc::board_iter() {
            let i = loc.index();
            if self.colors[i] == player_color && self.chain_head[i] == loc {
                all_player_heads[num_player_heads] = loc;
                num_player_heads += 1;
            }
        }
        // track elimination state
        let mut chain_killed = [false; MAX_PLAYER_HEADS]; // whether that player head is killed
        let mut vital_count_by_head = [0_u16; ARRAY_LEN]; // map chain head loc index to number of vital regions

        // count (region, chain-head) vital relations
        for region_idx in 0..num_regions {
            let start = vital_start[region_idx];
            let len = vital_len[region_idx];

            for offset in 0..len {
                let head = vital_for_chain_heads[start + offset];
                vital_count_by_head[head.index()] += 1;
            }
        }

        // chain kill loop
        loop {
            let mut killed_any_chain = false;
            //go through each player chain, and check if any dont have enough vital regions
            for player_head_idx in 0..num_player_heads {
                if chain_killed[player_head_idx] {
                    continue;
                }
                let head = all_player_heads[player_head_idx];
                if vital_count_by_head[head.index()] >= 2 {
                    // safe
                    continue;
                }

                // fewer than 2 vital regions
                chain_killed[player_head_idx] = true;
                killed_any_chain = true;

                //walk chain
                for killed_loc in self.chain_iter(head) {
                    for adj_i in Loc::adjacent_indices(killed_loc.index()) {
                        let stored_region_idx = region_idx_by_loc[adj_i];
                        // not a real region, can be walls or current player's stones
                        if stored_region_idx < 0 {
                            continue;
                        }
                        let region_idx = stored_region_idx as usize;

                        // check so that you only process a loc once for a given killed chain
                        if borders_non_pass_alive[region_idx] {
                            continue;
                        }
                        borders_non_pass_alive[region_idx] = true;

                        // decrement vital region count for every chain that counted this as vital, since one side of it has died
                        let start = vital_start[region_idx];
                        let len = vital_len[region_idx];
                        for offset in 0..len {
                            let dependent_head = vital_for_chain_heads[start + offset];
                            vital_count_by_head[dependent_head.index()] -= 1;
                        }
                    }
                }
            }

            if !killed_any_chain {
                break;
            }
        }

        //mark surviving chains in the result array
        for player_head_idx in 0..num_player_heads {
            if chain_killed[player_head_idx] {
                continue;
            }
            let head = all_player_heads[player_head_idx];
            for chain_loc in self.chain_iter(head) {
                result[chain_loc.index()] = player_color;
            }
        }
        //mark owned regions
        for region_idx in 0..num_regions {
            let strict_territory = num_internal_spaces_max_2[region_idx] <= 1
                && !borders_non_pass_alive[region_idx]
                && player_has_stones;
            let safe_big_territory = safe_big
                && !contains_opponent[region_idx]
                && !borders_non_pass_alive[region_idx]
                && player_has_stones;
            let unsafe_big_territory =
                unsafe_big && !contains_opponent[region_idx] && player_has_stones;

            let region_head = region_heads[region_idx];
            if strict_territory || safe_big_territory {
                let mut cur_i = region_head.index();
                loop {
                    result[cur_i] = player_color;
                    let next = next_in_region[cur_i];
                    if next == region_head {
                        break;
                    }
                    cur_i = next.index();
                }
            } else if unsafe_big_territory {
                let mut cur_i = region_head.index();
                loop {
                    if result[cur_i].is_empty() {
                        result[cur_i] = player_color;
                    }
                    let next = next_in_region[cur_i];
                    if next == region_head {
                        break;
                    }
                    cur_i = next.index();
                }
            }
        }
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

    fn assert_chain_metadata_matches_recomputation(board: &Board) {
        let mut visited = [false; ARRAY_LEN];

        for start in Loc::board_iter() {
            let start_i = start.index();
            let color = board.colors[start_i];
            if visited[start_i] || (color != Color::Black && color != Color::White) {
                continue;
            }

            let mut component = [false; ARRAY_LEN];
            let mut liberties = [false; ARRAY_LEN];
            let mut queue = [Loc::NULL; ARRAY_LEN];
            let mut queue_head = 0;
            let mut queue_tail = 1;
            queue[0] = start;
            visited[start_i] = true;

            while queue_head < queue_tail {
                let current = queue[queue_head];
                queue_head += 1;
                let current_i = current.index();
                component[current_i] = true;

                for adj_i in Loc::adjacent_indices(current_i) {
                    if board.colors[adj_i] == color && !visited[adj_i] {
                        visited[adj_i] = true;
                        queue[queue_tail] = Loc::from_index(adj_i);
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

            let mut linked_list = [false; ARRAY_LEN];
            let mut linked_list_size = 0;
            for chain_loc in board.chain_iter(head).take(ARRAY_LEN) {
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

    fn board_from_ascii(rows: [&str; BOARD_SIZE]) -> Board {
        let mut board = Board::new();

        for (stone, player) in [('x', Player::Black), ('o', Player::White)] {
            for (y, row) in rows.iter().enumerate() {
                assert_eq!(row.len(), BOARD_SIZE);
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

    fn assert_area_rows(area: &[Color; ARRAY_LEN], expected: [&str; BOARD_SIZE]) {
        for (y, row) in expected.iter().enumerate() {
            assert_eq!(row.len(), BOARD_SIZE);
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
    fn one_move_merges_two_preexisting_friendly_chains() {
        let mut board = Board::new();
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
        let mut board = Board::new();
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
    fn multi_stone_suicide_rule_changes_legality_and_removes_the_chain() {
        let mut board = Board::new();
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
        let mut board = Board::new();
        let mut player = Player::Black;
        let mut random_state = 0x7267_6f5f_7465_7374_u64;

        for _ in 0..2_000 {
            random_state = random_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let action = (random_state as usize) % (BOARD_SIZE * BOARD_SIZE + 1);
            let move_loc = if action == BOARD_SIZE * BOARD_SIZE {
                Loc::PASS
            } else {
                loc(action % BOARD_SIZE, action / BOARD_SIZE)
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

    #[test]
    fn area_of_an_empty_board_is_neutral() {
        let board = Board::new();
        let area = board.calculate_area(true);

        for point in Loc::board_iter() {
            assert_eq!(area[point.index()], Color::Empty);
        }
    }

    #[test]
    fn strict_area_rejects_a_chain_without_two_vital_regions() {
        let mut board = Board::new();
        let stone = loc(4, 4);
        play(&mut board, 4, 4, Player::Black);

        let mut strict_area = [Color::Empty; ARRAY_LEN];
        board.calculate_area_for_player(Player::Black, false, false, true, &mut strict_area);

        assert_eq!(strict_area[stone.index()], Color::Empty);
        assert_eq!(strict_area[loc(0, 0).index()], Color::Empty);
    }

    #[test]
    fn strict_area_keeps_a_two_eye_group_and_its_eyes() {
        let mut board = Board::new();

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

        let mut strict_area = [Color::Empty; ARRAY_LEN];
        board.calculate_area_for_player(Player::Black, false, false, true, &mut strict_area);

        assert_eq!(strict_area[loc(4, 4).index()], Color::Black);
        assert_eq!(strict_area[loc(6, 4).index()], Color::Black);
        assert_eq!(strict_area[loc(5, 4).index()], Color::Black);
        assert_eq!(strict_area[loc(0, 0).index()], Color::Empty);
    }

    #[test]
    fn default_area_leaves_shared_exterior_neutral() {
        let mut board = Board::new();

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
        let mut board = Board::new();

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
}
