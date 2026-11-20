//! Deterministic generational slot arena without `HashMap`.
//!
//! Guarantees cross-platform reproducible handle allocation, iteration order,
//! and O(1) lookup/insertion/removal.

use std::marker::PhantomData;

/// Generational handle identifying a rigid body in a `PhysicsWorld`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BodyId {
    pub index: u32,
    pub generation: u32,
}

impl BodyId {
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: 0,
    };

    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.generation != 0 && self.index != u32::MAX
    }
}

/// Generational handle identifying a joint constraint in a `PhysicsWorld`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct JointId {
    pub index: u32,
    pub generation: u32,
}

impl JointId {
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: 0,
    };

    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.generation != 0 && self.index != u32::MAX
    }
}

/// Trait unifying strongly-typed generational handles stored in [`SlotArena`].
pub trait SlotKey: Copy + Eq + Ord {
    fn from_raw(index: u32, generation: u32) -> Self;
    fn index(self) -> u32;
    fn generation(self) -> u32;
}

impl SlotKey for BodyId {
    #[inline]
    fn from_raw(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    #[inline]
    fn index(self) -> u32 {
        self.index
    }

    #[inline]
    fn generation(self) -> u32 {
        self.generation
    }
}

impl SlotKey for JointId {
    #[inline]
    fn from_raw(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    #[inline]
    fn index(self) -> u32 {
        self.index
    }

    #[inline]
    fn generation(self) -> u32 {
        self.generation
    }
}

impl SlotKey for (u32, u32) {
    #[inline]
    fn from_raw(index: u32, generation: u32) -> Self {
        (index, generation)
    }

    #[inline]
    fn index(self) -> u32 {
        self.0
    }

    #[inline]
    fn generation(self) -> u32 {
        self.1
    }
}

#[derive(Clone, Debug)]
struct Slot<T> {
    generation: u32,
    dense_pos: usize,
    value: Option<T>,
}

/// Deterministic generational slot pool backed by dense vectors and a LIFO free list.
#[derive(Clone, Debug)]
pub struct SlotArena<T, K: SlotKey = BodyId> {
    slots: Vec<Slot<T>>,
    free_list: Vec<u32>,
    dense_ids: Vec<K>,
    _marker: PhantomData<K>,
}

impl<T, K: SlotKey> Default for SlotArena<T, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, K: SlotKey> SlotArena<T, K> {
    #[inline]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            dense_ids: Vec::new(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            free_list: Vec::new(),
            dense_ids: Vec::with_capacity(cap),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dense_ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dense_ids.is_empty()
    }

    /// Dense slice of all currently live handles in deterministic insertion/swap order.
    #[inline]
    pub fn dense_ids(&self) -> &[K] {
        &self.dense_ids
    }

    /// Inserts `value` and returns its generational handle.
    pub fn insert(&mut self, value: T) -> K {
        if let Some(index) = self.free_list.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(slot.value.is_none());
            let dense_pos = self.dense_ids.len();
            let id = K::from_raw(index, slot.generation);
            slot.dense_pos = dense_pos;
            slot.value = Some(value);
            self.dense_ids.push(id);
            id
        } else {
            let index = self.slots.len() as u32;
            let generation = 1u32;
            let dense_pos = self.dense_ids.len();
            let id = K::from_raw(index, generation);
            self.slots.push(Slot {
                generation,
                dense_pos,
                value: Some(value),
            });
            self.dense_ids.push(id);
            id
        }
    }

    /// Removes the element addressed by `id` if its generation matches.
    pub fn remove(&mut self, id: K) -> Option<T> {
        let idx = id.index() as usize;
        let slot = self.slots.get_mut(idx)?;
        if slot.generation != id.generation() || slot.value.is_none() {
            return None;
        }
        let value = slot.value.take();
        let dense_pos = slot.dense_pos;
        slot.generation = slot.generation.wrapping_add(1);
        if slot.generation == 0 {
            slot.generation = 1;
        }
        self.free_list.push(id.index());

        let last_idx = self.dense_ids.len() - 1;
        self.dense_ids.swap_remove(dense_pos);
        if dense_pos < last_idx {
            let moved_id = self.dense_ids[dense_pos];
            self.slots[moved_id.index() as usize].dense_pos = dense_pos;
        }

        value
    }

    #[inline]
    pub fn contains(&self, id: K) -> bool {
        self.slots
            .get(id.index() as usize)
            .is_some_and(|s| s.generation == id.generation() && s.value.is_some())
    }

    #[inline]
    pub fn get(&self, id: K) -> Option<&T> {
        let slot = self.slots.get(id.index() as usize)?;
        if slot.generation == id.generation() {
            slot.value.as_ref()
        } else {
            None
        }
    }

    #[inline]
    pub fn get_mut(&mut self, id: K) -> Option<&mut T> {
        let slot = self.slots.get_mut(id.index() as usize)?;
        if slot.generation == id.generation() {
            slot.value.as_mut()
        } else {
            None
        }
    }

    /// Returns the dense position (`0..self.len()`) of a valid handle.
    #[inline]
    pub fn dense_index_of(&self, id: K) -> Option<usize> {
        let slot = self.slots.get(id.index() as usize)?;
        if slot.generation == id.generation() && slot.value.is_some() {
            Some(slot.dense_pos)
        } else {
            None
        }
    }

    /// Safe simultaneous mutable borrow of two distinct live slots.
    pub fn get_pair_mut(&mut self, a: K, b: K) -> Option<(&mut T, &mut T)> {
        let ia = a.index() as usize;
        let ib = b.index() as usize;
        if ia == ib || ia >= self.slots.len() || ib >= self.slots.len() {
            return None;
        }
        if ia < ib {
            let (left, right) = self.slots.split_at_mut(ib);
            let sa = &mut left[ia];
            let sb = &mut right[0];
            if sa.generation == a.generation() && sb.generation == b.generation() {
                Some((sa.value.as_mut()?, sb.value.as_mut()?))
            } else {
                None
            }
        } else {
            let (left, right) = self.slots.split_at_mut(ia);
            let sb = &mut left[ib];
            let sa = &mut right[0];
            if sa.generation == a.generation() && sb.generation == b.generation() {
                Some((sa.value.as_mut()?, sb.value.as_mut()?))
            } else {
                None
            }
        }
    }

    /// Iterates over all live `(id, &T)` pairs in deterministic dense order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (K, &T)> {
        self.dense_ids.iter().filter_map(|&id| {
            let slot = &self.slots[id.index() as usize];
            slot.value.as_ref().map(|v| (id, v))
        })
    }

    /// Clears all elements while preserving deterministic reset state.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_list.clear();
        self.dense_ids.clear();
    }
}
