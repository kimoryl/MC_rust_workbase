//! Broadphase spatial acceleration structures:
//! - [`DynamicBvh`]: SAH-guided dynamic AABB tree with fat margins and AVL balancing.
//! - [`SpatialHash16`]: deterministic 16-meter chunk-bucketed spatial partition.

use std::collections::BTreeMap;

use crate::math::{Aabbd, Vec3d};

const NULL_NODE: usize = usize::MAX;

#[derive(Clone, Debug)]
struct BvhNode {
    aabb: Aabbd,
    user_id: u32,
    parent: usize,
    child1: usize,
    child2: usize,
    height: i32,
    live: bool,
}

impl BvhNode {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.child1 == NULL_NODE
    }
}

/// Surface Area Heuristic (SAH) dynamic AABB tree with fat AABB margins.
#[derive(Clone, Debug)]
pub struct DynamicBvh {
    nodes: Vec<BvhNode>,
    free_list: Vec<usize>,
    live_leaves: Vec<usize>,
    root: usize,
    fat_margin: f64,
}

impl Default for DynamicBvh {
    fn default() -> Self {
        Self::new(0.15)
    }
}

impl DynamicBvh {
    pub fn new(fat_margin: f64) -> Self {
        Self {
            nodes: Vec::new(),
            free_list: Vec::new(),
            live_leaves: Vec::new(),
            root: NULL_NODE,
            fat_margin: fat_margin.max(0.0),
        }
    }

    #[inline]
    pub fn fat_margin(&self) -> f64 {
        self.fat_margin
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.live_leaves.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live_leaves.is_empty()
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free_list.clear();
        self.live_leaves.clear();
        self.root = NULL_NODE;
    }

    fn allocate_node(&mut self) -> usize {
        if let Some(idx) = self.free_list.pop() {
            let n = &mut self.nodes[idx];
            n.aabb = Aabbd::EMPTY;
            n.user_id = 0;
            n.parent = NULL_NODE;
            n.child1 = NULL_NODE;
            n.child2 = NULL_NODE;
            n.height = 0;
            n.live = true;
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(BvhNode {
                aabb: Aabbd::EMPTY,
                user_id: 0,
                parent: NULL_NODE,
                child1: NULL_NODE,
                child2: NULL_NODE,
                height: 0,
                live: true,
            });
            idx
        }
    }

    fn free_node(&mut self, idx: usize) {
        let n = &mut self.nodes[idx];
        n.parent = NULL_NODE;
        n.child1 = NULL_NODE;
        n.child2 = NULL_NODE;
        n.height = -1;
        n.live = false;
        self.free_list.push(idx);
    }

    /// Inserts a leaf proxy and returns its proxy handle.
    pub fn insert(&mut self, user_id: u32, tight_aabb: Aabbd) -> usize {
        let leaf = self.allocate_node();
        self.nodes[leaf].aabb = tight_aabb.expand_margin(self.fat_margin);
        self.nodes[leaf].user_id = user_id;
        self.nodes[leaf].height = 0;
        self.insert_leaf(leaf);
        self.live_leaves.push(leaf);
        leaf
    }

    /// Removes a previously inserted leaf proxy.
    pub fn remove(&mut self, proxy_id: usize) {
        if proxy_id >= self.nodes.len()
            || !self.nodes[proxy_id].live
            || !self.nodes[proxy_id].is_leaf()
        {
            return;
        }
        self.remove_leaf(proxy_id);
        self.free_node(proxy_id);
        if let Some(pos) = self.live_leaves.iter().position(|&id| id == proxy_id) {
            self.live_leaves.swap_remove(pos);
        }
    }

    /// Updates a leaf proxy with a new tight AABB and velocity displacement.
    ///
    /// Returns `true` if the fat AABB was exceeded and the proxy was reinserted.
    pub fn update(&mut self, proxy_id: usize, tight_aabb: Aabbd, displacement: Vec3d) -> bool {
        if proxy_id >= self.nodes.len() || !self.nodes[proxy_id].live {
            return false;
        }
        let fat = self.nodes[proxy_id].aabb;
        let extended = tight_aabb.expand_margin(self.fat_margin * 2.0);
        if fat.contains_aabb(&tight_aabb) && extended.contains_aabb(&fat) {
            return false;
        }

        self.remove_leaf(proxy_id);

        let mut new_fat = tight_aabb.expand_margin(self.fat_margin);
        let d = displacement * 2.0;
        if d.x < 0.0 {
            new_fat.min.x += d.x;
        } else {
            new_fat.max.x += d.x;
        }
        if d.y < 0.0 {
            new_fat.min.y += d.y;
        } else {
            new_fat.max.y += d.y;
        }
        if d.z < 0.0 {
            new_fat.min.z += d.z;
        } else {
            new_fat.max.z += d.z;
        }

        self.nodes[proxy_id].aabb = new_fat;
        self.insert_leaf(proxy_id);
        true
    }

    #[inline]
    pub fn fat_aabb(&self, proxy_id: usize) -> Option<Aabbd> {
        let n = self.nodes.get(proxy_id)?;
        if n.live && n.is_leaf() {
            Some(n.aabb)
        } else {
            None
        }
    }

    /// Appends all `user_id`s whose fat AABB intersects `query_aabb` into `out` (sorted).
    pub fn query_aabb(&self, query_aabb: Aabbd, out: &mut Vec<u32>) {
        if self.root == NULL_NODE || query_aabb.is_empty() {
            return;
        }
        let start_len = out.len();
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            if !node.aabb.intersects(&query_aabb) {
                continue;
            }
            if node.is_leaf() {
                out.push(node.user_id);
            } else {
                stack.push(node.child1);
                stack.push(node.child2);
            }
        }
        out[start_len..].sort_unstable();
    }

    /// Finds all overlapping distinct leaf pairs `(a, b)` with `a < b`, sorted deterministically.
    pub fn query_pairs(&self, out: &mut Vec<(u32, u32)>) {
        out.clear();
        if self.root == NULL_NODE || self.live_leaves.len() < 2 {
            return;
        }
        let mut stack = Vec::with_capacity(64);
        for &leaf in &self.live_leaves {
            let leaf_aabb = self.nodes[leaf].aabb;
            let leaf_uid = self.nodes[leaf].user_id;
            stack.clear();
            stack.push(self.root);
            while let Some(idx) = stack.pop() {
                let node = &self.nodes[idx];
                if !node.aabb.intersects(&leaf_aabb) {
                    continue;
                }
                if node.is_leaf() {
                    if leaf_uid < node.user_id {
                        out.push((leaf_uid, node.user_id));
                    }
                } else {
                    stack.push(node.child1);
                    stack.push(node.child2);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// Finds all overlapping distinct leaf pairs involving at least one proxy in `query_leaves`,
    /// canonicalized as `(min_uid, max_uid)` and sorted deterministically.
    pub fn query_pairs_for_leaves(&self, query_leaves: &[usize], out: &mut Vec<(u32, u32)>) {
        out.clear();
        if self.root == NULL_NODE || query_leaves.is_empty() || self.live_leaves.len() < 2 {
            return;
        }
        let mut stack = Vec::with_capacity(64);
        for &leaf in query_leaves {
            if leaf >= self.nodes.len() || !self.nodes[leaf].live {
                continue;
            }
            let leaf_aabb = self.nodes[leaf].aabb;
            let leaf_uid = self.nodes[leaf].user_id;
            stack.clear();
            stack.push(self.root);
            while let Some(idx) = stack.pop() {
                let node = &self.nodes[idx];
                if !node.aabb.intersects(&leaf_aabb) {
                    continue;
                }
                if node.is_leaf() {
                    let other_uid = node.user_id;
                    if leaf_uid != other_uid {
                        let pair = if leaf_uid < other_uid {
                            (leaf_uid, other_uid)
                        } else {
                            (other_uid, leaf_uid)
                        };
                        out.push(pair);
                    }
                } else {
                    stack.push(node.child1);
                    stack.push(node.child2);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// Traverses all leaves hit by ray `origin + t * dir` (`0 <= t <= max_dist`).
    ///
    /// `callback(user_id, current_max_dist)` may return a tighter `new_max_dist`
    /// to prune farther subtrees.
    pub fn raycast<F>(&self, origin: Vec3d, dir: Vec3d, max_dist: f64, mut callback: F)
    where
        F: FnMut(u32, f64) -> f64,
    {
        if self.root == NULL_NODE || max_dist < 0.0 {
            return;
        }
        let mut best_dist = max_dist;
        let mut stack: Vec<(usize, f64)> = Vec::with_capacity(64);
        if let Some((t, _)) = self.nodes[self.root]
            .aabb
            .ray_intersect(origin, dir, best_dist)
        {
            stack.push((self.root, t));
        }

        while let Some((idx, t_entry)) = stack.pop() {
            if t_entry > best_dist {
                continue;
            }
            let node = &self.nodes[idx];
            if node.is_leaf() {
                best_dist = callback(node.user_id, best_dist).min(best_dist);
            } else {
                let c1 = node.child1;
                let c2 = node.child2;
                let h1 = self.nodes[c1].aabb.ray_intersect(origin, dir, best_dist);
                let h2 = self.nodes[c2].aabb.ray_intersect(origin, dir, best_dist);
                match (h1, h2) {
                    (Some((t1, _)), Some((t2, _))) => {
                        // Push farther child first so nearer child is popped first.
                        if t1 <= t2 {
                            stack.push((c2, t2));
                            stack.push((c1, t1));
                        } else {
                            stack.push((c1, t1));
                            stack.push((c2, t2));
                        }
                    }
                    (Some((t1, _)), None) => stack.push((c1, t1)),
                    (None, Some((t2, _))) => stack.push((c2, t2)),
                    (None, None) => {}
                }
            }
        }
    }

    fn insert_leaf(&mut self, leaf: usize) {
        if self.root == NULL_NODE {
            self.root = leaf;
            self.nodes[leaf].parent = NULL_NODE;
            return;
        }

        let leaf_aabb = self.nodes[leaf].aabb;
        let mut index = self.root;
        while !self.nodes[index].is_leaf() {
            let child1 = self.nodes[index].child1;
            let child2 = self.nodes[index].child2;

            let area = self.nodes[index].aabb.surface_area();
            let combined_aabb = self.nodes[index].aabb.union(leaf_aabb);
            let combined_area = combined_aabb.surface_area();

            let cost = 2.0 * combined_area;
            let inheritance_cost = 2.0 * (combined_area - area);

            let cost1 = if self.nodes[child1].is_leaf() {
                leaf_aabb.union(self.nodes[child1].aabb).surface_area() + inheritance_cost
            } else {
                let old_area = self.nodes[child1].aabb.surface_area();
                let new_area = leaf_aabb.union(self.nodes[child1].aabb).surface_area();
                (new_area - old_area) + inheritance_cost
            };

            let cost2 = if self.nodes[child2].is_leaf() {
                leaf_aabb.union(self.nodes[child2].aabb).surface_area() + inheritance_cost
            } else {
                let old_area = self.nodes[child2].aabb.surface_area();
                let new_area = leaf_aabb.union(self.nodes[child2].aabb).surface_area();
                (new_area - old_area) + inheritance_cost
            };

            if cost < cost1 && cost < cost2 {
                break;
            }
            index = if cost1 <= cost2 { child1 } else { child2 };
        }

        let sibling = index;
        let old_parent = self.nodes[sibling].parent;
        let new_parent = self.allocate_node();
        self.nodes[new_parent].parent = old_parent;
        self.nodes[new_parent].aabb = leaf_aabb.union(self.nodes[sibling].aabb);
        self.nodes[new_parent].height = self.nodes[sibling].height + 1;
        self.nodes[new_parent].child1 = sibling;
        self.nodes[new_parent].child2 = leaf;

        self.nodes[sibling].parent = new_parent;
        self.nodes[leaf].parent = new_parent;

        if old_parent != NULL_NODE {
            if self.nodes[old_parent].child1 == sibling {
                self.nodes[old_parent].child1 = new_parent;
            } else {
                self.nodes[old_parent].child2 = new_parent;
            }
        } else {
            self.root = new_parent;
        }

        self.fix_upwards(self.nodes[leaf].parent);
    }

    fn remove_leaf(&mut self, leaf: usize) {
        if leaf == self.root {
            self.root = NULL_NODE;
            return;
        }

        let parent = self.nodes[leaf].parent;
        let grand_parent = self.nodes[parent].parent;
        let sibling = if self.nodes[parent].child1 == leaf {
            self.nodes[parent].child2
        } else {
            self.nodes[parent].child1
        };

        if grand_parent != NULL_NODE {
            if self.nodes[grand_parent].child1 == parent {
                self.nodes[grand_parent].child1 = sibling;
            } else {
                self.nodes[grand_parent].child2 = sibling;
            }
            self.nodes[sibling].parent = grand_parent;
            self.free_node(parent);
            self.fix_upwards(grand_parent);
        } else {
            self.root = sibling;
            self.nodes[sibling].parent = NULL_NODE;
            self.free_node(parent);
        }
    }

    fn fix_upwards(&mut self, mut index: usize) {
        while index != NULL_NODE {
            index = self.balance(index);
            let c1 = self.nodes[index].child1;
            let c2 = self.nodes[index].child2;
            self.nodes[index].height = 1 + self.nodes[c1].height.max(self.nodes[c2].height);
            self.nodes[index].aabb = self.nodes[c1].aabb.union(self.nodes[c2].aabb);
            index = self.nodes[index].parent;
        }
    }

    fn balance(&mut self, i_a: usize) -> usize {
        if self.nodes[i_a].is_leaf() || self.nodes[i_a].height < 2 {
            return i_a;
        }
        let i_b = self.nodes[i_a].child1;
        let i_c = self.nodes[i_a].child2;
        let balance = self.nodes[i_c].height - self.nodes[i_b].height;

        // Rotate C up
        if balance > 1 {
            let i_f = self.nodes[i_c].child1;
            let i_g = self.nodes[i_c].child2;
            let parent_a = self.nodes[i_a].parent;

            self.nodes[i_c].child1 = i_a;
            self.nodes[i_c].parent = parent_a;
            self.nodes[i_a].parent = i_c;

            if parent_a != NULL_NODE {
                if self.nodes[parent_a].child1 == i_a {
                    self.nodes[parent_a].child1 = i_c;
                } else {
                    self.nodes[parent_a].child2 = i_c;
                }
            } else {
                self.root = i_c;
            }

            if self.nodes[i_f].height > self.nodes[i_g].height {
                self.nodes[i_c].child2 = i_f;
                self.nodes[i_a].child2 = i_g;
                self.nodes[i_g].parent = i_a;
                self.nodes[i_a].aabb = self.nodes[i_b].aabb.union(self.nodes[i_g].aabb);
                self.nodes[i_c].aabb = self.nodes[i_a].aabb.union(self.nodes[i_f].aabb);
                self.nodes[i_a].height = 1 + self.nodes[i_b].height.max(self.nodes[i_g].height);
                self.nodes[i_c].height = 1 + self.nodes[i_a].height.max(self.nodes[i_f].height);
            } else {
                self.nodes[i_c].child2 = i_g;
                self.nodes[i_a].child2 = i_f;
                self.nodes[i_f].parent = i_a;
                self.nodes[i_a].aabb = self.nodes[i_b].aabb.union(self.nodes[i_f].aabb);
                self.nodes[i_c].aabb = self.nodes[i_a].aabb.union(self.nodes[i_g].aabb);
                self.nodes[i_a].height = 1 + self.nodes[i_b].height.max(self.nodes[i_f].height);
                self.nodes[i_c].height = 1 + self.nodes[i_a].height.max(self.nodes[i_g].height);
            }
            return i_c;
        }

        // Rotate B up
        if balance < -1 {
            let i_d = self.nodes[i_b].child1;
            let i_e = self.nodes[i_b].child2;
            let parent_a = self.nodes[i_a].parent;

            self.nodes[i_b].child1 = i_a;
            self.nodes[i_b].parent = parent_a;
            self.nodes[i_a].parent = i_b;

            if parent_a != NULL_NODE {
                if self.nodes[parent_a].child1 == i_a {
                    self.nodes[parent_a].child1 = i_b;
                } else {
                    self.nodes[parent_a].child2 = i_b;
                }
            } else {
                self.root = i_b;
            }

            if self.nodes[i_d].height > self.nodes[i_e].height {
                self.nodes[i_b].child2 = i_d;
                self.nodes[i_a].child1 = i_e;
                self.nodes[i_e].parent = i_a;
                self.nodes[i_a].aabb = self.nodes[i_c].aabb.union(self.nodes[i_e].aabb);
                self.nodes[i_b].aabb = self.nodes[i_a].aabb.union(self.nodes[i_d].aabb);
                self.nodes[i_a].height = 1 + self.nodes[i_c].height.max(self.nodes[i_e].height);
                self.nodes[i_b].height = 1 + self.nodes[i_a].height.max(self.nodes[i_d].height);
            } else {
                self.nodes[i_b].child2 = i_e;
                self.nodes[i_a].child1 = i_d;
                self.nodes[i_d].parent = i_a;
                self.nodes[i_a].aabb = self.nodes[i_c].aabb.union(self.nodes[i_d].aabb);
                self.nodes[i_b].aabb = self.nodes[i_a].aabb.union(self.nodes[i_e].aabb);
                self.nodes[i_a].height = 1 + self.nodes[i_c].height.max(self.nodes[i_d].height);
                self.nodes[i_b].height = 1 + self.nodes[i_a].height.max(self.nodes[i_e].height);
            }
            return i_b;
        }

        i_a
    }
}

/// Deterministic 16-meter chunk-keyed spatial partition backed by `BTreeMap` and
/// sorted `Vec<u32>` buckets (zero `HashMap`).
#[derive(Clone, Debug, Default)]
pub struct SpatialHash16 {
    buckets: BTreeMap<(i32, i32, i32), Vec<u32>>,
}

impl SpatialHash16 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.buckets.clear();
    }

    #[inline]
    pub fn chunk_coord(v: f64) -> i32 {
        (v / 16.0).floor() as i32
    }

    #[inline]
    pub fn chunk_keys_for_aabb(aabb: &Aabbd) -> ((i32, i32, i32), (i32, i32, i32)) {
        (
            (
                Self::chunk_coord(aabb.min.x),
                Self::chunk_coord(aabb.min.y),
                Self::chunk_coord(aabb.min.z),
            ),
            (
                Self::chunk_coord(aabb.max.x),
                Self::chunk_coord(aabb.max.y),
                Self::chunk_coord(aabb.max.z),
            ),
        )
    }

    pub fn insert(&mut self, id: u32, aabb: &Aabbd) {
        if aabb.is_empty() {
            return;
        }
        let ((x0, y0, z0), (x1, y1, z1)) = Self::chunk_keys_for_aabb(aabb);
        for cx in x0..=x1 {
            for cy in y0..=y1 {
                for cz in z0..=z1 {
                    let bucket = self.buckets.entry((cx, cy, cz)).or_default();
                    if let Err(pos) = bucket.binary_search(&id) {
                        bucket.insert(pos, id);
                    }
                }
            }
        }
    }

    pub fn remove(&mut self, id: u32, aabb: &Aabbd) {
        if aabb.is_empty() {
            return;
        }
        let ((x0, y0, z0), (x1, y1, z1)) = Self::chunk_keys_for_aabb(aabb);
        for cx in x0..=x1 {
            for cy in y0..=y1 {
                for cz in z0..=z1 {
                    let key = (cx, cy, cz);
                    let empty = if let Some(bucket) = self.buckets.get_mut(&key) {
                        if let Ok(pos) = bucket.binary_search(&id) {
                            bucket.remove(pos);
                        }
                        bucket.is_empty()
                    } else {
                        false
                    };
                    if empty {
                        self.buckets.remove(&key);
                    }
                }
            }
        }
    }

    #[inline]
    pub fn query_chunk(&self, chunk_key: (i32, i32, i32)) -> &[u32] {
        self.buckets
            .get(&chunk_key)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn query_aabb(&self, aabb: &Aabbd, out: &mut Vec<u32>) {
        if aabb.is_empty() {
            return;
        }
        let start = out.len();
        let ((x0, y0, z0), (x1, y1, z1)) = Self::chunk_keys_for_aabb(aabb);
        for cx in x0..=x1 {
            for cy in y0..=y1 {
                for cz in z0..=z1 {
                    if let Some(bucket) = self.buckets.get(&(cx, cy, cz)) {
                        out.extend_from_slice(bucket);
                    }
                }
            }
        }
        out[start..].sort_unstable();
        // Deduplicate only the newly appended range.
        let mut write = start;
        for read in start..out.len() {
            if write == start || out[read] != out[write - 1] {
                out[write] = out[read];
                write += 1;
            }
        }
        out.truncate(write);
    }

    #[inline]
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }
}
