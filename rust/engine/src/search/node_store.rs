use std::ptr::NonNull;

use crate::search::{graph_key::GraphKey, node::SearchNode};

const EMPTY_INDEX: u32 = u32::MAX; //max index is empty_index-1

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InsertError {
    KeyAlreadyExists,
    StoreFull,
}

// can have different types of storages for different uses (dynamic vs static search budgets)
/// Storage whose nodes may be referenced through raw pointers.
///
/// # Safety
///
/// Every `NonNull<SearchNode>` returned by an implementation must point to a
/// valid, initialized node and remain at the same address until `clear` is
/// called or the store is dropped. Inserting or looking up other nodes must not
/// move or invalidate any previously returned node.
pub(super) unsafe trait NodeStore {
    fn find(&mut self, key: GraphKey) -> Option<NonNull<SearchNode>>;
    fn insert(&mut self, key: GraphKey) -> Result<NonNull<SearchNode>, InsertError>;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn clear(&mut self);
}

struct NodeEntry {
    key: GraphKey,
    next_in_bucket: u32,
    node: SearchNode,
}
pub(super) struct FixedArenaNodeStore {
    entries: Vec<NodeEntry>,
    bucket_heads: Box<[u32]>, //basically hash -> index of head of linked list of node entries
    node_capacity: usize,
}
impl FixedArenaNodeStore {
    pub(super) fn new(node_capacity: usize) -> Self {
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
// `insert` refuses to push once that capacity is reached. It therefore
// never reallocates while nodes are live. Entries are not individually moved
// or removed, so returned node pointers remain stable until `clear` or drop.
// Access through `as_mut_ptr` avoids creating slice/entry references that could
// invalidate outstanding node pointers. Lookup reads only the metadata fields.
unsafe impl NodeStore for FixedArenaNodeStore {
    fn find(&mut self, key: GraphKey) -> Option<NonNull<SearchNode>> {
        let bucket_mask = self.bucket_heads.len() - 1; // bucket capacity is already a power of two
        let bucket_index = (key.raw() as usize) & bucket_mask;
        let mut entry_index = self.bucket_heads[bucket_index];
        while entry_index != EMPTY_INDEX {
            debug_assert!((entry_index as usize) < self.entries.len());
            // SAFETY: bucket links index initialized entries. Use raw field
            // projections so other live node pointers retain their permissions.
            unsafe {
                let entry = self.entries.as_mut_ptr().add(entry_index as usize);
                if (*entry).key == key {
                    return Some(NonNull::new_unchecked(&raw mut (*entry).node));
                }
                entry_index = (*entry).next_in_bucket;
            }
        }
        None
    }
    fn insert(&mut self, key: GraphKey) -> Result<NonNull<SearchNode>, InsertError> {
        let bucket_mask = self.bucket_heads.len() - 1; // bucket capacity is already a power of two
        let bucket_index = (key.raw() as usize) & bucket_mask;
        let bucket_head = self.bucket_heads[bucket_index];

        //search for if it already exists
        let mut entry_index = bucket_head;
        while entry_index != EMPTY_INDEX {
            debug_assert!((entry_index as usize) < self.entries.len());
            // SAFETY: bucket links index initialized entries; only metadata is
            // read, without borrowing the entries or their nodes.
            unsafe {
                let entry = self.entries.as_mut_ptr().add(entry_index as usize);
                if (*entry).key == key {
                    return Err(InsertError::KeyAlreadyExists);
                }
                entry_index = (*entry).next_in_bucket;
            }
        }
        //check should be an internal invariant, as long as you only use this when you bound search budget strictly
        let new_entry_index = self.entries.len();
        if new_entry_index >= self.node_capacity {
            return Err(InsertError::StoreFull);
        }
        //insert new entry to head of the chain
        let new_entry = NodeEntry {
            key,
            next_in_bucket: bucket_head,
            node: SearchNode::new(),
        };
        self.entries.push(new_entry);
        self.bucket_heads[bucket_index] = new_entry_index as u32;
        // SAFETY: the push initialized this slot without reallocating. Project
        // its node pointer without a mutable borrow of the backing slice.
        unsafe {
            let entry = self.entries.as_mut_ptr().add(new_entry_index);
            Ok(NonNull::new_unchecked(&raw mut (*entry).node))
        }
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

        let inserted_node = store.insert(key(3)).expect("store has capacity");
        assert_eq!(store.len(), 1);
        assert_eq!(store.capacity(), 4);
        assert_eq!(store.find(key(3)), Some(inserted_node));

        assert_eq!(store.insert(key(3)), Err(InsertError::KeyAlreadyExists));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn colliding_keys_remain_independently_findable() {
        let mut store = FixedArenaNodeStore::new(2);
        let bucket_count = store.bucket_heads.len();
        let first_key = key(1);
        let colliding_key = key(1 + bucket_count as u128);

        let first_node = store
            .insert(first_key)
            .expect("store has capacity for first key");
        let second_node = store
            .insert(colliding_key)
            .expect("store has capacity for colliding key");

        assert_ne!(first_node, second_node);
        assert_eq!(store.find(first_key), Some(first_node));
        assert_eq!(store.find(colliding_key), Some(second_node));
    }

    #[test]
    fn inserting_other_nodes_does_not_move_existing_nodes() {
        let mut store = FixedArenaNodeStore::new(4);
        let first_node = store.insert(key(1)).expect("store has capacity");

        for value in 2..=4 {
            store
                .insert(key(value))
                .expect("store has reserved capacity");
        }

        assert_eq!(store.find(key(1)), Some(first_node));
    }

    #[test]
    fn old_pointers_remain_usable_after_insertions_and_lookups() {
        let mut store = FixedArenaNodeStore::new(3);
        let stride = store.bucket_heads.len() as u128;
        let first_key = key(1);
        let second_key = key(1 + stride);
        let mut first = store.insert(first_key).unwrap();
        let mut second = store.insert(second_key).unwrap();
        store.insert(key(1 + 2 * stride)).unwrap();

        // Address equality alone does not check pointer validity under Miri.
        // Exercise collision traversal, repeated lookup, and failed insertions.
        let mut found = store.find(first_key).unwrap();
        assert_eq!(store.find(key(1 + 3 * stride)), None);
        assert_eq!(store.insert(first_key), Err(InsertError::KeyAlreadyExists));
        assert_eq!(
            store.insert(key(1 + 3 * stride)),
            Err(InsertError::StoreFull)
        );
        // SAFETY: all pointers refer to live nodes, and each borrow ends before
        // another pointer is used. The store has not been cleared or dropped.
        unsafe {
            first.as_mut().record_visit(0.5, 0.0, 0.0, 0.0);
            second.as_mut().record_visit(0.5, 0.0, 0.0, 0.0);
            found.as_mut().record_visit(0.5, 0.0, 0.0, 0.0);
            assert_eq!(first.as_ref().visits(), 2);
            assert_eq!(second.as_ref().visits(), 1);
        }
    }

    #[test]
    fn insert_distinguishes_full_store_from_existing_key() {
        let mut store = FixedArenaNodeStore::new(2);
        store
            .insert(key(1))
            .expect("store has capacity for first key");
        store
            .insert(key(2))
            .expect("store has capacity for second key");

        assert_eq!(store.insert(key(3)), Err(InsertError::StoreFull));
        assert_eq!(store.len(), 2);
        assert_eq!(store.insert(key(1)), Err(InsertError::KeyAlreadyExists));
    }

    #[test]
    fn clear_resets_lookup_state_and_preserves_capacity_for_reuse() {
        let mut store = FixedArenaNodeStore::new(2);
        store.insert(key(1)).expect("store has capacity");
        store.insert(key(2)).expect("store has capacity");

        store.clear();

        assert_eq!(store.len(), 0);
        assert_eq!(store.capacity(), 2);
        assert_eq!(store.find(key(1)), None);
        assert!(store.bucket_heads.iter().all(|&index| index == EMPTY_INDEX));

        store.insert(key(3)).expect("cleared storage can be reused");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn bucket_count_is_nearest_power_of_two_to_four_times_capacity() {
        assert_eq!(FixedArenaNodeStore::new(512).bucket_heads.len(), 2048);
        assert_eq!(FixedArenaNodeStore::new(5).bucket_heads.len(), 16);
        assert_eq!(FixedArenaNodeStore::new(6).bucket_heads.len(), 32);
    }
}
