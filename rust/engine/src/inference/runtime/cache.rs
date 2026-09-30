//! Model-scoped evaluation identity and striped, direct-mapped cache.

use crate::{
    game::{
        game_state::GameState,
        hash::{Hash128, komi_hash, suicide_hash},
    },
    inference::outputs::NNOutput,
};
use std::sync::{Arc, Mutex};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct EvaluationKey(Hash128); // for the eval cache
// Striped eval cache between hash(input * mask * player) and an Arc of the processed output.

struct CacheEntry {
    key: EvaluationKey,
    output: Arc<NNOutput>,
}
struct CacheShard {
    entries: Box<[Option<CacheEntry>]>, //dyn array of entries
}

pub(super) struct EvaluationCache {
    //indexing: xxxx{shard bits}{entry bits}
    shards: Box<[Mutex<CacheShard>]>, //dyn array of shards
    shard_bits: u32,                  // upper
    entry_bits: u32,                  // lower
}
impl CacheShard {
    fn new(entries_per_shard: usize) -> Self {
        let mut entries = Vec::with_capacity(entries_per_shard);
        entries.resize_with(entries_per_shard, || None);
        Self {
            entries: entries.into_boxed_slice(),
        }
    }
}
impl EvaluationCache {
    pub(super) fn validate_dimensions(
        capacity: usize,
        num_shards: usize,
    ) -> Result<(), &'static str> {
        if !capacity.is_power_of_two() {
            return Err("cache capacity must be a power of two");
        }
        if !num_shards.is_power_of_two() {
            return Err("cache shard count must be a power of two");
        }
        if num_shards > capacity {
            return Err("cache cannot have more shards than entries");
        }
        Ok(())
    }

    pub(super) fn new(capacity: usize, num_shards: usize) -> Self {
        Self::validate_dimensions(capacity, num_shards)
            .expect("invalid evaluation cache dimensions");

        let entries_per_shard = capacity / num_shards;
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(Mutex::new(CacheShard::new(entries_per_shard)));
        }
        let shards = shards.into_boxed_slice();
        Self {
            shards,
            shard_bits: num_shards.ilog2(),
            entry_bits: entries_per_shard.ilog2(),
        }
    }
    fn indices(&self, key: EvaluationKey) -> (usize, usize) {
        let bits = key.0 as usize;
        let entry_mask = (1_usize << self.entry_bits) - 1;
        let shard_mask = (1_usize << self.shard_bits) - 1;
        let entry_index = bits & entry_mask;
        let shard_index = (bits >> self.entry_bits) & shard_mask;
        (shard_index, entry_index)
    }
    pub(super) fn lookup(&self, key: EvaluationKey) -> Option<Arc<NNOutput>> {
        let (shard_index, entry_index) = self.indices(key);
        //take guard, check if it matches our key
        let shard = self.shards[shard_index]
            .lock()
            .expect("evaluation cache mutex poisoned");
        match &shard.entries[entry_index] {
            Some(entry) if entry.key == key => Some(entry.output.clone()),
            _ => None,
        }
    }
    pub(super) fn insert(&self, key: EvaluationKey, output: Arc<NNOutput>) {
        debug_assert!(
            output.is_processed(),
            "evaluation cache only takes processed outputs"
        );
        let (shard_index, entry_index) = self.indices(key);
        let old_entry = {
            let mut shard = self.shards[shard_index]
                .lock()
                .expect("evaluation cache mutex poisoned");
            shard.entries[entry_index].replace(CacheEntry { key, output })
            //mutex drops
        };
        drop(old_entry); // outside of the mutex
    }
}

impl EvaluationKey {
    pub(super) fn new(game_state: &GameState) -> Self {
        let mut key = game_state.current_state_hash();

        //state hash done, hash rules now
        key ^= komi_hash(game_state.rules().komi); //randomized komi will cause more cache misses for similar positions. should be discrete

        if game_state.rules().multi_stone_suicide_legal {
            key ^= suicide_hash();
        }
        Self(key)
    }
}

#[cfg(test)]
mod tests;
