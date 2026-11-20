//! Non-recursive Union-Find island partitioning for collective sleeping and waking.

/// Iterative Disjoint Set Union (Union-Find) with path halving and union by rank.
#[derive(Clone, Debug, Default)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    pub fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    pub fn reset(&mut self, len: usize) {
        self.parent.clear();
        self.parent.extend(0..len);
        self.rank.clear();
        self.rank.resize(len, 0);
    }

    /// Iterative path-halving `find` (guaranteed stack-safe for arbitrary island sizes).
    #[inline]
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let p = self.parent[x];
            let gp = self.parent[p];
            self.parent[x] = gp;
            x = gp;
        }
        x
    }

    /// Unions the sets containing `a` and `b`.
    #[inline]
    pub fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }
}

/// Connected component of dynamic rigid bodies coupled by contacts or joints.
#[derive(Clone, Debug, Default)]
pub struct Island {
    pub body_indices: Vec<usize>,
}

/// Deterministic builder grouping dynamic bodies into connected islands.
#[derive(Clone, Debug, Default)]
pub struct IslandBuilder {
    dsu: DisjointSet,
    root_to_island: Vec<usize>,
}

impl IslandBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Partitions dynamic bodies (`is_dynamic[i] == true`) connected by `edges`
    /// into deterministic sorted islands.
    pub fn build(
        &mut self,
        body_count: usize,
        is_dynamic: &[bool],
        edges: &[(usize, usize)],
        out_islands: &mut Vec<Island>,
    ) {
        out_islands.clear();
        if body_count == 0 {
            return;
        }
        self.dsu.reset(body_count);
        for &(a, b) in edges {
            if a < body_count && b < body_count && is_dynamic[a] && is_dynamic[b] {
                self.dsu.union(a, b);
            }
        }

        self.root_to_island.clear();
        self.root_to_island.resize(body_count, usize::MAX);

        for (i, &dyn_flag) in is_dynamic.iter().enumerate().take(body_count) {
            if !dyn_flag {
                continue;
            }
            let root = self.dsu.find(i);
            let island_idx = self.root_to_island[root];
            if island_idx == usize::MAX {
                let new_idx = out_islands.len();
                self.root_to_island[root] = new_idx;
                out_islands.push(Island {
                    body_indices: vec![i],
                });
            } else {
                out_islands[island_idx].body_indices.push(i);
            }
        }
    }
}
