//! Read-only area scoring and pass-alive analysis.

use super::{ARRAY_LEN, BOARD_SIZE, Board, Color, Loc, Player};

const MAX_PLAYER_HEADS: usize = (BOARD_SIZE * BOARD_SIZE + 1) / 2;
const MAX_REGIONS: usize = (BOARD_SIZE * BOARD_SIZE + 1) / 2 + 1;
const VITAL_FOR_CHAIN_HEADS_MAX_LEN: usize = MAX_REGIONS * 4; // max number of (region, chain-head) vital relations for a player

impl Board {
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

    pub(super) fn calculate_area_for_player(
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
