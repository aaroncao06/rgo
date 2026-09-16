use std::ptr::NonNull;

use crate::search::{graph_key::GraphKey, node::SearchNode};

const EMPTY_INDEX: u32 = u32::MAX; //max index is empty_index-1

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StoreFull;

// can have different types of storages for different uses (dynamic vs static search budgets)
/// Storage whose nodes may be referenced through raw pointers.
///
/// # Safety
///
/// Every `NonNull<SearchNode>` returned by an implementation must point to a
/// valid, initialized node and remain at the same address until `clear` is
/// called or the store is dropped. Inserting or looking up other nodes must not
/// move or invalidate any previously returned node.
pub(crate) unsafe trait NodeStore {
    fn find(&mut self, key: GraphKey) -> Option<NonNull<SearchNode>>;
    fn find_or_insert(&mut self, key: GraphKey) -> Result<(NonNull<SearchNode>, bool), StoreFull>; // returns pointer to the slot followed by whether it was just inserted or not
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn clear(&mut self);
}

struct NodeEntry {
    key: GraphKey,
    next_in_bucket: u32,
    node: SearchNode,
}
pub(crate) struct FixedArenaNodeStore {
    entries: Vec<NodeEntry>,
    bucket_heads: Box<[u32]>, //basically hash -> index of head of linked list of node entries
    node_capacity: usize,
}
impl FixedArenaNodeStore {
    pub(crate) fn new(node_capacity: usize) -> Self {
        assert!(
            node_capacity > 0 && node_capacity <= u32::MAX as usize,
            "node capacity out of bounds"
        );
        let entries = Vec::with_capacity(node_capacity);
        let bucket_capacity = {
            let value = node_capacity
                .checked_mul(4)
                .expect("bucket-count target overflow");
            let lower = 1_usize << value.ilog2();
            let upper = lower
                .checked_mul(2)
                .expect("bucket-count power of two overflow");

            if value - lower < upper - value {
                lower
            } else {
                upper
            }
        }; // nearest power of two lol
        let bucket_heads = vec![EMPTY_INDEX; bucket_capacity].into_boxed_slice();
        Self {
            entries,
            bucket_heads,
            node_capacity,
        }
    }
}
// SAFETY: `entries` reserves `node_capacity` slots during construction, and
// `find_or_insert` refuses to push once that capacity is reached. It therefore
// never reallocates while nodes are live. Entries are not individually moved
// or removed, so returned node pointers remain stable until `clear` or drop.
unsafe impl NodeStore for FixedArenaNodeStore {
    fn find(&mut self, key: GraphKey) -> Option<NonNull<SearchNode>> {
        let bucket_mask = self.bucket_heads.len() - 1; // bucket capacity is already a power of two
        let bucket_index = (key.raw() as usize) & bucket_mask;
        let mut entry_index = self.bucket_heads[bucket_index];
        while entry_index != EMPTY_INDEX {
            debug_assert!((entry_index as usize) < self.entries.len());
            let entry = &mut self.entries[entry_index as usize];
            if entry.key == key {
                return Some(NonNull::from(&mut entry.node));
            }
            entry_index = entry.next_in_bucket;
        }
        None
    }
    fn find_or_insert(&mut self, key: GraphKey) -> Result<(NonNull<SearchNode>, bool), StoreFull> {
        //if returns true, you need to populate the newly returned pointer, otherwise it was found and already populated
        let bucket_mask = self.bucket_heads.len() - 1; // bucket capacity is already a power of two
        let bucket_index = (key.raw() as usize) & bucket_mask;
        let bucket_head = self.bucket_heads[bucket_index];

        //search for if it already exists
        let mut entry_index = bucket_head;
        while entry_index != EMPTY_INDEX {
            debug_assert!((entry_index as usize) < self.entries.len());
            let entry = &mut self.entries[entry_index as usize];
            if entry.key == key {
                return Ok((NonNull::from(&mut entry.node), false));
            }
            entry_index = entry.next_in_bucket;
        }
        //check should be an internal invariant, as long as you only use this when you bound search budget strictly
        let new_entry_index = self.entries.len();
        if new_entry_index >= self.node_capacity {
            return Err(StoreFull);
        }
        //insert new entry to head of the chain
        let new_entry = NodeEntry {
            key,
            next_in_bucket: bucket_head,
            node: SearchNode::new(),
        };
        self.entries.push(new_entry);
        self.bucket_heads[bucket_index] = new_entry_index as u32;
        Ok((NonNull::from(&mut self.entries[new_entry_index].node), true))
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn capacity(&self) -> usize {
        self.node_capacity
    }
    fn clear(&mut self) {
        self.entries.clear();
        self.bucket_heads.fill(EMPTY_INDEX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(value: u128) -> GraphKey {
        GraphKey::from_raw(value)
    }

    #[test]
    #[should_panic(expected = "node capacity out of bounds")]
    fn store_rejects_zero_capacity() {
        let _ = FixedArenaNodeStore::new(0);
    }

    #[test]
    fn insertion_and_lookup_return_the_same_node() {
        let mut store = FixedArenaNodeStore::new(4);

        let (inserted_node, inserted) = store.find_or_insert(key(3)).expect("store has capacity");
        assert!(inserted);
        assert_eq!(store.len(), 1);
        assert_eq!(store.capacity(), 4);
        assert_eq!(store.find(key(3)), Some(inserted_node));

        let (existing_node, inserted) = store
            .find_or_insert(key(3))
            .expect("existing key does not consume capacity");
        assert!(!inserted);
        assert_eq!(existing_node, inserted_node);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn colliding_keys_remain_independently_findable() {
        let mut store = FixedArenaNodeStore::new(2);
        let bucket_count = store.bucket_heads.len();
        let first_key = key(1);
        let colliding_key = key(1 + bucket_count as u128);

        let (first_node, first_inserted) = store
            .find_or_insert(first_key)
            .expect("store has capacity for first key");
        let (second_node, second_inserted) = store
            .find_or_insert(colliding_key)
            .expect("store has capacity for colliding key");

        assert!(first_inserted);
        assert!(second_inserted);
        assert_ne!(first_node, second_node);
        assert_eq!(store.find(first_key), Some(first_node));
        assert_eq!(store.find(colliding_key), Some(second_node));
    }

    #[test]
    fn inserting_other_nodes_does_not_move_existing_nodes() {
        let mut store = FixedArenaNodeStore::new(4);
        let (first_node, _) = store.find_or_insert(key(1)).expect("store has capacity");

        for value in 2..=4 {
            store
                .find_or_insert(key(value))
                .expect("store has reserved capacity");
        }

        assert_eq!(store.find(key(1)), Some(first_node));
    }

    #[test]
    fn full_store_rejects_only_new_keys() {
        let mut store = FixedArenaNodeStore::new(2);
        let (first_node, _) = store
            .find_or_insert(key(1))
            .expect("store has capacity for first key");
        store
            .find_or_insert(key(2))
            .expect("store has capacity for second key");

        assert!(matches!(store.find_or_insert(key(3)), Err(StoreFull)));
        assert_eq!(store.len(), 2);

        let (existing_node, inserted) = store
            .find_or_insert(key(1))
            .expect("a full store can still find an existing key");
        assert_eq!(existing_node, first_node);
        assert!(!inserted);
    }

    #[test]
    fn clear_resets_lookup_state_and_preserves_capacity_for_reuse() {
        let mut store = FixedArenaNodeStore::new(2);
        store.find_or_insert(key(1)).expect("store has capacity");
        store.find_or_insert(key(2)).expect("store has capacity");

        store.clear();

        assert_eq!(store.len(), 0);
        assert_eq!(store.capacity(), 2);
        assert_eq!(store.find(key(1)), None);
        assert!(store.bucket_heads.iter().all(|&index| index == EMPTY_INDEX));

        let (_, inserted) = store
            .find_or_insert(key(3))
            .expect("cleared storage can be reused");
        assert!(inserted);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn bucket_count_is_nearest_power_of_two_to_four_times_capacity() {
        assert_eq!(FixedArenaNodeStore::new(512).bucket_heads.len(), 2048);
        assert_eq!(FixedArenaNodeStore::new(5).bucket_heads.len(), 16);
        assert_eq!(FixedArenaNodeStore::new(6).bucket_heads.len(), 32);
    }
}
