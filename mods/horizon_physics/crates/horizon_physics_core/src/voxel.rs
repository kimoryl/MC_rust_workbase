//! Sparse 16x16x16 voxel sub-grids, material registry, incremental greedy 3D
//! box merging, O(1) Parallel Axis Theorem mass/inertia updates, and
//! 6-neighborhood connected-component splitting.

use std::collections::{BTreeMap, VecDeque};

use crate::math::{Aabbd, Mat3d, Vec3d};

/// Exposed face bitflags for [`LocalBox::exposed_faces`].
pub const FACE_NEG_X: u8 = 1 << 0;
pub const FACE_POS_X: u8 = 1 << 1;
pub const FACE_NEG_Y: u8 = 1 << 2;
pub const FACE_POS_Y: u8 = 1 << 3;
pub const FACE_NEG_Z: u8 = 1 << 4;
pub const FACE_POS_Z: u8 = 1 << 5;
pub const ALL_FACES: u8 = 0x3F;

/// Compact material identifier for voxels and terrain blocks.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MaterialId(pub u16);

impl MaterialId {
    pub const STONE: Self = Self(0);
    pub const WOOD: Self = Self(1);
    pub const IRON: Self = Self(2);
    pub const SLIME: Self = Self(3);
    pub const ICE: Self = Self(4);
    pub const WOOL: Self = Self(5);
}

/// Physical properties of a voxel material (SI units; 1 voxel = 1 m^3).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VoxelMaterial {
    /// Density in kg/m^3.
    pub density: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
    /// Coefficient of restitution in `[0.0, 1.0]`.
    pub restitution: f64,
    /// Linear air drag coefficient (1/s).
    pub linear_drag: f64,
    /// Angular air drag coefficient (1/s).
    pub angular_drag: f64,
}

impl VoxelMaterial {
    pub const STONE: Self = Self {
        density: 2500.0,
        friction: 0.65,
        restitution: 0.10,
        linear_drag: 0.02,
        angular_drag: 0.05,
    };

    pub const WOOD: Self = Self {
        density: 600.0,
        friction: 0.55,
        restitution: 0.15,
        linear_drag: 0.02,
        angular_drag: 0.05,
    };

    pub const IRON: Self = Self {
        density: 7850.0,
        friction: 0.45,
        restitution: 0.20,
        linear_drag: 0.01,
        angular_drag: 0.03,
    };

    pub const SLIME: Self = Self {
        density: 1100.0,
        friction: 0.80,
        restitution: 0.85,
        linear_drag: 0.03,
        angular_drag: 0.06,
    };

    pub const ICE: Self = Self {
        density: 917.0,
        friction: 0.04,
        restitution: 0.08,
        linear_drag: 0.005,
        angular_drag: 0.02,
    };

    pub const WOOL: Self = Self {
        density: 250.0,
        friction: 0.85,
        restitution: 0.05,
        linear_drag: 0.05,
        angular_drag: 0.08,
    };
}

impl Default for VoxelMaterial {
    #[inline]
    fn default() -> Self {
        Self::STONE
    }
}

/// Deterministic registry mapping [`MaterialId`] to [`VoxelMaterial`].
#[derive(Clone, Debug)]
pub struct MaterialRegistry {
    materials: BTreeMap<MaterialId, VoxelMaterial>,
    fallback: VoxelMaterial,
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialRegistry {
    pub fn new() -> Self {
        let mut materials = BTreeMap::new();
        materials.insert(MaterialId::STONE, VoxelMaterial::STONE);
        materials.insert(MaterialId::WOOD, VoxelMaterial::WOOD);
        materials.insert(MaterialId::IRON, VoxelMaterial::IRON);
        materials.insert(MaterialId::SLIME, VoxelMaterial::SLIME);
        materials.insert(MaterialId::ICE, VoxelMaterial::ICE);
        materials.insert(MaterialId::WOOL, VoxelMaterial::WOOL);
        Self {
            materials,
            fallback: VoxelMaterial::STONE,
        }
    }

    pub fn register(&mut self, id: MaterialId, mat: VoxelMaterial) {
        self.materials.insert(id, mat);
    }

    #[inline]
    pub fn get(&self, id: MaterialId) -> VoxelMaterial {
        self.materials.get(&id).copied().unwrap_or(self.fallback)
    }

    #[inline]
    pub fn try_get(&self, id: MaterialId) -> Option<&VoxelMaterial> {
        self.materials.get(&id)
    }
}

/// Integer voxel coordinates in local sub-grid or world space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl VoxelCoord {
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn brick_key(self) -> (i16, i16, i16) {
        (
            self.x.div_euclid(16) as i16,
            self.y.div_euclid(16) as i16,
            self.z.div_euclid(16) as i16,
        )
    }

    #[inline]
    pub const fn local_in_brick(self) -> (usize, usize, usize) {
        (
            self.x.rem_euclid(16) as usize,
            self.y.rem_euclid(16) as usize,
            self.z.rem_euclid(16) as usize,
        )
    }

    /// Center of this unit voxel cube `[x, x+1] x [y, y+1] x [z, z+1]`.
    #[inline]
    pub fn center_local(self) -> Vec3d {
        Vec3d::new(
            self.x as f64 + 0.5,
            self.y as f64 + 0.5,
            self.z as f64 + 0.5,
        )
    }

    #[inline]
    pub const fn neighbors_6(self) -> [Self; 6] {
        [
            Self::new(self.x - 1, self.y, self.z),
            Self::new(self.x + 1, self.y, self.z),
            Self::new(self.x, self.y - 1, self.z),
            Self::new(self.x, self.y + 1, self.z),
            Self::new(self.x, self.y, self.z - 1),
            Self::new(self.x, self.y, self.z + 1),
        ]
    }
}

/// Per-voxel payload stored inside a [`VoxelSubGrid`].
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VoxelEntry {
    pub state_id: u32,
    pub material: MaterialId,
}

impl VoxelEntry {
    #[inline]
    pub const fn new(state_id: u32, material: MaterialId) -> Self {
        Self { state_id, material }
    }

    #[inline]
    pub const fn from_material(material: MaterialId) -> Self {
        Self {
            state_id: 1,
            material,
        }
    }
}

/// Axis-aligned merged collider box in grid-local integer coordinates `[min, max)`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LocalBox {
    /// Inclusive lower integer corner `[x0, y0, z0]`.
    pub min: [i32; 3],
    /// Exclusive upper integer corner `[x1, y1, z1]`.
    pub max: [i32; 3],
    pub material: MaterialId,
    /// Exposed face bitmask: bit 0..5 = `-X, +X, -Y, +Y, -Z, +Z`.
    pub exposed_faces: u8,
}

impl LocalBox {
    #[inline]
    pub fn center_local(&self) -> Vec3d {
        Vec3d::new(
            0.5 * (self.min[0] + self.max[0]) as f64,
            0.5 * (self.min[1] + self.max[1]) as f64,
            0.5 * (self.min[2] + self.max[2]) as f64,
        )
    }

    #[inline]
    pub fn half_extents(&self) -> Vec3d {
        Vec3d::new(
            0.5 * (self.max[0] - self.min[0]) as f64,
            0.5 * (self.max[1] - self.min[1]) as f64,
            0.5 * (self.max[2] - self.min[2]) as f64,
        )
    }

    #[inline]
    pub fn aabb_local(&self) -> Aabbd {
        Aabbd::new(
            Vec3d::new(self.min[0] as f64, self.min[1] as f64, self.min[2] as f64),
            Vec3d::new(self.max[0] as f64, self.max[1] as f64, self.max[2] as f64),
        )
    }

    #[inline]
    pub const fn voxel_volume(&self) -> i64 {
        (self.max[0] - self.min[0]) as i64
            * (self.max[1] - self.min[1]) as i64
            * (self.max[2] - self.min[2]) as i64
    }

    /// Returns whether face `(axis in 0..3, positive_sign)` is exposed.
    #[inline]
    pub const fn is_face_exposed(&self, axis: usize, positive: bool) -> bool {
        let bit = match (axis, positive) {
            (0, false) => FACE_NEG_X,
            (0, true) => FACE_POS_X,
            (1, false) => FACE_NEG_Y,
            (1, true) => FACE_POS_Y,
            (2, false) => FACE_NEG_Z,
            (2, true) => FACE_POS_Z,
            _ => 0,
        };
        (self.exposed_faces & bit) != 0
    }
}

/// Sparse 16x16x16 brick of voxels with cached brick-local merged colliders.
#[derive(Clone, Debug)]
pub struct VoxelBrick16 {
    voxels: Box<[Option<VoxelEntry>; 4096]>,
    count: u16,
    min_local: [u8; 3],
    max_local: [u8; 3],
    pub dirty_collider: bool,
    pub local_boxes: Vec<LocalBox>,
    pub local_aabb: Aabbd,
}

impl VoxelBrick16 {
    #[inline]
    pub(crate) const fn linear_idx(lx: usize, ly: usize, lz: usize) -> usize {
        (lz << 8) | (ly << 4) | lx
    }

    pub fn new(brick_key: (i16, i16, i16)) -> Self {
        let _ = brick_key;
        Self {
            voxels: Box::new([None; 4096]),
            count: 0,
            min_local: [16, 16, 16],
            max_local: [0, 0, 0],
            dirty_collider: true,
            local_boxes: Vec::new(),
            local_aabb: Aabbd::EMPTY,
        }
    }

    #[inline]
    pub fn get(&self, lx: usize, ly: usize, lz: usize) -> Option<VoxelEntry> {
        self.voxels[Self::linear_idx(lx, ly, lz)]
    }

    #[inline]
    pub fn count(&self) -> u16 {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn update_aabb_from_bounds(&mut self, brick_key: (i16, i16, i16)) {
        if self.count == 0 {
            self.local_aabb = Aabbd::EMPTY;
            return;
        }
        let ox = i32::from(brick_key.0) * 16;
        let oy = i32::from(brick_key.1) * 16;
        let oz = i32::from(brick_key.2) * 16;
        self.local_aabb = Aabbd::new(
            Vec3d::new(
                (ox + i32::from(self.min_local[0])) as f64,
                (oy + i32::from(self.min_local[1])) as f64,
                (oz + i32::from(self.min_local[2])) as f64,
            ),
            Vec3d::new(
                (ox + i32::from(self.max_local[0])) as f64,
                (oy + i32::from(self.max_local[1])) as f64,
                (oz + i32::from(self.max_local[2])) as f64,
            ),
        );
    }

    fn recompute_bounds(&mut self, brick_key: (i16, i16, i16)) {
        if self.count == 0 {
            self.min_local = [16, 16, 16];
            self.max_local = [0, 0, 0];
            self.local_aabb = Aabbd::EMPTY;
            return;
        }
        let mut min_x = 16u8;
        let mut min_y = 16u8;
        let mut min_z = 16u8;
        let mut max_x = 0u8;
        let mut max_y = 0u8;
        let mut max_z = 0u8;
        for lz in 0..16usize {
            for ly in 0..16usize {
                let row_base = (lz << 8) | (ly << 4);
                for lx in 0..16usize {
                    if self.voxels[row_base | lx].is_some() {
                        min_x = min_x.min(lx as u8);
                        min_y = min_y.min(ly as u8);
                        min_z = min_z.min(lz as u8);
                        max_x = max_x.max(lx as u8 + 1);
                        max_y = max_y.max(ly as u8 + 1);
                        max_z = max_z.max(lz as u8 + 1);
                    }
                }
            }
        }
        self.min_local = [min_x, min_y, min_z];
        self.max_local = [max_x, max_y, max_z];
        self.update_aabb_from_bounds(brick_key);
    }

    fn set(
        &mut self,
        brick_key: (i16, i16, i16),
        lx: usize,
        ly: usize,
        lz: usize,
        entry: Option<VoxelEntry>,
    ) -> Option<VoxelEntry> {
        let idx = Self::linear_idx(lx, ly, lz);
        let old = self.voxels[idx];
        if old == entry {
            return old;
        }
        self.voxels[idx] = entry;
        self.dirty_collider = true;

        let x = lx as u8;
        let y = ly as u8;
        let z = lz as u8;

        match (old, entry) {
            (None, Some(_)) => {
                self.count += 1;
                self.min_local[0] = self.min_local[0].min(x);
                self.min_local[1] = self.min_local[1].min(y);
                self.min_local[2] = self.min_local[2].min(z);
                self.max_local[0] = self.max_local[0].max(x + 1);
                self.max_local[1] = self.max_local[1].max(y + 1);
                self.max_local[2] = self.max_local[2].max(z + 1);
                self.update_aabb_from_bounds(brick_key);
            }
            (Some(_), None) => {
                self.count -= 1;
                let on_boundary = x == self.min_local[0]
                    || y == self.min_local[1]
                    || z == self.min_local[2]
                    || x + 1 == self.max_local[0]
                    || y + 1 == self.max_local[1]
                    || z + 1 == self.max_local[2];
                if on_boundary {
                    self.recompute_bounds(brick_key);
                }
            }
            _ => {}
        }

        old
    }
}

/// Computes the exact inertia tensor about the local origin `(0, 0, 0)` of a
/// unit cube (`1x1x1`) centered at `r` with mass `m` via the Parallel Axis Theorem.
#[inline]
fn unit_voxel_inertia_about_origin(r: Vec3d, mass: f64) -> Mat3d {
    let diag_self = mass / 6.0;
    let ixx = diag_self + mass * (r.y * r.y + r.z * r.z);
    let iyy = diag_self + mass * (r.x * r.x + r.z * r.z);
    let izz = diag_self + mass * (r.x * r.x + r.y * r.y);
    let ixy = -mass * r.x * r.y;
    let ixz = -mass * r.x * r.z;
    let iyz = -mass * r.y * r.z;
    Mat3d::from_cols(
        Vec3d::new(ixx, ixy, ixz),
        Vec3d::new(ixy, iyy, iyz),
        Vec3d::new(ixz, iyz, izz),
    )
}

/// Computes the parallel-axis shift tensor `M * ((c.c) I - c c^T)`.
#[inline]
fn parallel_axis_shift(c: Vec3d, mass: f64) -> Mat3d {
    let ixx = mass * (c.y * c.y + c.z * c.z);
    let iyy = mass * (c.x * c.x + c.z * c.z);
    let izz = mass * (c.x * c.x + c.y * c.y);
    let ixy = -mass * c.x * c.y;
    let ixz = -mass * c.x * c.z;
    let iyz = -mass * c.y * c.z;
    Mat3d::from_cols(
        Vec3d::new(ixx, ixy, ixz),
        Vec3d::new(ixy, iyy, iyz),
        Vec3d::new(ixz, iyz, izz),
    )
}

/// Sparse deterministic voxel sub-grid backing a rigid body.
///
/// Maintains O(1) incremental mass, center of mass, and inertia tensor via the
/// Parallel Axis Theorem, incremental per-brick greedy 3D box merging with
/// cross-brick exposed-face masks, and 6-connected fracture splitting.
#[derive(Clone, Debug, Default)]
pub struct VoxelSubGrid {
    bricks: BTreeMap<(i16, i16, i16), VoxelBrick16>,
    total_voxels: usize,
    total_mass: f64,
    first_moment: Vec3d,
    inertia_origin: Mat3d,
    weighted_linear_drag: f64,
    weighted_angular_drag: f64,
    cached_boxes: Vec<LocalBox>,
    boxes_dirty: bool,
}

impl VoxelSubGrid {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience builder filling an integer box `[min[0]..max[0], min[1]..max[1], min[2]..max[2])`.
    pub fn solid_box(
        min: [i32; 3],
        max: [i32; 3],
        entry: VoxelEntry,
        reg: &MaterialRegistry,
    ) -> Self {
        let mut grid = Self::new();
        for z in min[2]..max[2] {
            for y in min[1]..max[1] {
                for x in min[0]..max[0] {
                    grid.set_voxel(VoxelCoord::new(x, y, z), Some(entry), reg);
                }
            }
        }
        grid.rebuild_dirty_colliders();
        grid
    }

    /// Convenience builder from an iterator of `(VoxelCoord, VoxelEntry)`.
    pub fn from_voxels(
        iter: impl IntoIterator<Item = (VoxelCoord, VoxelEntry)>,
        reg: &MaterialRegistry,
    ) -> Self {
        let mut grid = Self::new();
        for (coord, entry) in iter {
            grid.set_voxel(coord, Some(entry), reg);
        }
        grid.rebuild_dirty_colliders();
        grid
    }

    #[inline]
    pub fn voxel_count(&self) -> usize {
        self.total_voxels
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_voxels == 0
    }

    #[inline]
    pub fn mass(&self) -> f64 {
        self.total_mass
    }

    /// Center of mass in grid-local coordinates.
    #[inline]
    pub fn com_local(&self) -> Vec3d {
        if self.total_mass > 0.0 {
            self.first_moment / self.total_mass
        } else {
            Vec3d::ZERO
        }
    }

    /// Inertia tensor about [`Self::com_local`] in grid-local axes.
    #[inline]
    pub fn inertia_com(&self) -> Mat3d {
        if self.total_mass <= 0.0 || self.total_voxels == 0 {
            return Mat3d::ZERO;
        }
        let c = self.com_local();
        let mut i_com =
            (self.inertia_origin - parallel_axis_shift(c, self.total_mass)).symmetrize();
        // Clamp tiny negative numerical residue on diagonal for degenerate inputs.
        i_com.cols[0].x = i_com.cols[0].x.max(0.0);
        i_com.cols[1].y = i_com.cols[1].y.max(0.0);
        i_com.cols[2].z = i_com.cols[2].z.max(0.0);
        i_com
    }

    /// Mass-weighted average linear and angular drag across occupied voxels.
    #[inline]
    pub fn average_drag(&self) -> (f64, f64) {
        if self.total_mass > 0.0 {
            (
                (self.weighted_linear_drag / self.total_mass).max(0.0),
                (self.weighted_angular_drag / self.total_mass).max(0.0),
            )
        } else {
            (0.0, 0.0)
        }
    }

    /// Tight grid-local AABB enclosing all occupied voxels.
    pub fn local_aabb(&self) -> Aabbd {
        let mut aabb = Aabbd::EMPTY;
        for brick in self.bricks.values() {
            if !brick.is_empty() {
                aabb = aabb.union(brick.local_aabb);
            }
        }
        aabb
    }

    /// Deterministic sorted list of occupied brick coordinates `(bx, by, bz)`.
    pub fn brick_keys(&self) -> Vec<(i16, i16, i16)> {
        self.bricks.keys().copied().collect()
    }

    /// Read-only access to a brick by key.
    #[inline]
    pub fn brick(&self, key: (i16, i16, i16)) -> Option<&VoxelBrick16> {
        self.bricks.get(&key)
    }

    /// Returns true if any brick needs collider merging.
    #[inline]
    pub fn has_dirty_colliders(&self) -> bool {
        self.boxes_dirty
    }

    /// Cached merged local collider boxes. Call [`Self::rebuild_dirty_colliders`]
    /// (or [`Self::ensure_boxes`]) after direct [`Self::set_voxel`] batches.
    #[inline]
    pub fn boxes(&self) -> &[LocalBox] {
        &self.cached_boxes
    }

    /// Rebuilds any dirty bricks if needed and returns all merged [`LocalBox`]es.
    #[inline]
    pub fn ensure_boxes(&mut self) -> &[LocalBox] {
        if self.boxes_dirty {
            self.rebuild_dirty_colliders();
        }
        &self.cached_boxes
    }

    #[inline]
    pub fn get_voxel(&self, coord: VoxelCoord) -> Option<VoxelEntry> {
        let bk = coord.brick_key();
        let (lx, ly, lz) = coord.local_in_brick();
        self.bricks.get(&bk)?.get(lx, ly, lz)
    }

    #[inline]
    pub fn has_voxel(&self, coord: VoxelCoord) -> bool {
        self.get_voxel(coord).is_some()
    }

    /// Iterates over all occupied `(VoxelCoord, VoxelEntry)` in deterministic order.
    pub fn iter_voxels(&self) -> impl Iterator<Item = (VoxelCoord, VoxelEntry)> + '_ {
        self.bricks.iter().flat_map(|(&(bx, by, bz), brick)| {
            let ox = i32::from(bx) * 16;
            let oy = i32::from(by) * 16;
            let oz = i32::from(bz) * 16;
            (0..4096usize).filter_map(move |idx| {
                let entry = brick.voxels[idx]?;
                let lx = (idx & 15) as i32;
                let ly = ((idx >> 4) & 15) as i32;
                let lz = ((idx >> 8) & 15) as i32;
                Some((VoxelCoord::new(ox + lx, oy + ly, oz + lz), entry))
            })
        })
    }

    /// Mutates a single voxel in O(1), updating mass, center of mass, and
    /// inertia tensor via the Parallel Axis Theorem.
    ///
    /// Returns `Some(delta_com)` when the voxel actually changed, where
    /// `delta_com = new_com_local - old_com_local` (or `Vec3d::ZERO` when
    /// transitioning between empty and non-empty), allowing the owning rigid
    /// body to shift its world position by `rotation.rotate_vec3(delta_com)`
    /// for zero visual jump. Returns `None` if the voxel was unchanged.
    pub fn set_voxel(
        &mut self,
        coord: VoxelCoord,
        entry: Option<VoxelEntry>,
        reg: &MaterialRegistry,
    ) -> Option<Vec3d> {
        let bk = coord.brick_key();
        let (lx, ly, lz) = coord.local_in_brick();

        let existing = self.bricks.get(&bk).and_then(|b| b.get(lx, ly, lz));
        if existing == entry {
            return None;
        }

        let had_voxels = self.total_voxels > 0;
        let old_com = self.com_local();
        let r = coord.center_local();

        if let Some(old_entry) = existing {
            let mat = reg.get(old_entry.material);
            let m = mat.density.max(0.0);
            self.total_voxels -= 1;
            self.total_mass -= m;
            self.first_moment -= r * m;
            self.inertia_origin -= unit_voxel_inertia_about_origin(r, m);
            self.weighted_linear_drag -= mat.linear_drag * m;
            self.weighted_angular_drag -= mat.angular_drag * m;
        }

        if let Some(new_entry) = entry {
            let mat = reg.get(new_entry.material);
            let m = mat.density.max(0.0);
            self.total_voxels += 1;
            self.total_mass += m;
            self.first_moment += r * m;
            self.inertia_origin += unit_voxel_inertia_about_origin(r, m);
            self.weighted_linear_drag += mat.linear_drag * m;
            self.weighted_angular_drag += mat.angular_drag * m;

            let brick = self
                .bricks
                .entry(bk)
                .or_insert_with(|| VoxelBrick16::new(bk));
            brick.set(bk, lx, ly, lz, Some(new_entry));
        } else if let Some(brick) = self.bricks.get_mut(&bk) {
            brick.set(bk, lx, ly, lz, None);
            if brick.is_empty() {
                self.bricks.remove(&bk);
            }
        }

        // Exact zero reset when grid becomes completely empty.
        if self.total_voxels == 0 {
            self.total_mass = 0.0;
            self.first_moment = Vec3d::ZERO;
            self.inertia_origin = Mat3d::ZERO;
            self.weighted_linear_drag = 0.0;
            self.weighted_angular_drag = 0.0;
        }

        // Mark adjacent bricks dirty if the changed voxel lies on a brick boundary
        // so cross-brick `exposed_faces` stay exact.
        let (bx, by, bz) = bk;
        if lx == 0
            && let Some(nb) = self.bricks.get_mut(&(bx - 1, by, bz))
        {
            nb.dirty_collider = true;
        }
        if lx == 15
            && let Some(nb) = self.bricks.get_mut(&(bx + 1, by, bz))
        {
            nb.dirty_collider = true;
        }
        if ly == 0
            && let Some(nb) = self.bricks.get_mut(&(bx, by - 1, bz))
        {
            nb.dirty_collider = true;
        }
        if ly == 15
            && let Some(nb) = self.bricks.get_mut(&(bx, by + 1, bz))
        {
            nb.dirty_collider = true;
        }
        if lz == 0
            && let Some(nb) = self.bricks.get_mut(&(bx, by, bz - 1))
        {
            nb.dirty_collider = true;
        }
        if lz == 15
            && let Some(nb) = self.bricks.get_mut(&(bx, by, bz + 1))
        {
            nb.dirty_collider = true;
        }

        self.boxes_dirty = true;
        let new_com = self.com_local();
        let delta_com = if had_voxels && self.total_voxels > 0 {
            new_com - old_com
        } else {
            Vec3d::ZERO
        };
        Some(delta_com)
    }

    /// Incrementally rebuilds greedy 3D merged boxes ONLY for bricks marked
    /// `dirty_collider == true`, computing exact cross-brick `exposed_faces`.
    ///
    /// Returns the number of bricks rebuilt.
    pub fn rebuild_dirty_colliders(&mut self) -> usize {
        if !self.boxes_dirty {
            return 0;
        }

        let dirty_keys: Vec<(i16, i16, i16)> = self
            .bricks
            .iter()
            .filter_map(|(&k, b)| if b.dirty_collider { Some(k) } else { None })
            .collect();

        let rebuilt_count = dirty_keys.len();

        // Compute new local boxes for each dirty brick using read access to all bricks
        // (needed for exact cross-brick exposed-face checks).
        let mut rebuilt_boxes: Vec<((i16, i16, i16), Vec<LocalBox>)> =
            Vec::with_capacity(rebuilt_count);

        for bk in dirty_keys {
            let Some(brick) = self.bricks.get(&bk) else {
                continue;
            };
            let boxes = self.greedy_merge_brick(bk, brick);
            rebuilt_boxes.push((bk, boxes));
        }

        for (bk, boxes) in rebuilt_boxes {
            if let Some(brick) = self.bricks.get_mut(&bk) {
                brick.local_boxes = boxes;
                brick.dirty_collider = false;
            }
        }

        self.cached_boxes.clear();
        for brick in self.bricks.values() {
            self.cached_boxes.extend_from_slice(&brick.local_boxes);
        }
        self.boxes_dirty = false;
        rebuilt_count
    }

    fn greedy_merge_brick(&self, bk: (i16, i16, i16), brick: &VoxelBrick16) -> Vec<LocalBox> {
        let mut visited = [false; 4096];
        let mut out = Vec::new();
        let ox = i32::from(bk.0) * 16;
        let oy = i32::from(bk.1) * 16;
        let oz = i32::from(bk.2) * 16;

        for lz in 0..16usize {
            for ly in 0..16usize {
                for lx in 0..16usize {
                    let start_idx = VoxelBrick16::linear_idx(lx, ly, lz);
                    if visited[start_idx] {
                        continue;
                    }
                    let Some(entry) = brick.voxels[start_idx] else {
                        continue;
                    };
                    let mat = entry.material;

                    // 1. Extrude along +X.
                    let mut x_end = lx + 1;
                    while x_end < 16 {
                        let idx = VoxelBrick16::linear_idx(x_end, ly, lz);
                        if visited[idx] {
                            break;
                        }
                        match brick.voxels[idx] {
                            Some(e) if e.material == mat => x_end += 1,
                            _ => break,
                        }
                    }

                    // 2. Extrude along +Y.
                    let mut y_end = ly + 1;
                    'y_loop: while y_end < 16 {
                        for x in lx..x_end {
                            let idx = VoxelBrick16::linear_idx(x, y_end, lz);
                            if visited[idx] {
                                break 'y_loop;
                            }
                            match brick.voxels[idx] {
                                Some(e) if e.material == mat => {}
                                _ => break 'y_loop,
                            }
                        }
                        y_end += 1;
                    }

                    // 3. Extrude along +Z.
                    let mut z_end = lz + 1;
                    'z_loop: while z_end < 16 {
                        for y in ly..y_end {
                            for x in lx..x_end {
                                let idx = VoxelBrick16::linear_idx(x, y, z_end);
                                if visited[idx] {
                                    break 'z_loop;
                                }
                                match brick.voxels[idx] {
                                    Some(e) if e.material == mat => {}
                                    _ => break 'z_loop,
                                }
                            }
                        }
                        z_end += 1;
                    }

                    // Mark merged box visited.
                    for z in lz..z_end {
                        for y in ly..y_end {
                            for x in lx..x_end {
                                visited[VoxelBrick16::linear_idx(x, y, z)] = true;
                            }
                        }
                    }

                    let min = [ox + lx as i32, oy + ly as i32, oz + lz as i32];
                    let max = [ox + x_end as i32, oy + y_end as i32, oz + z_end as i32];
                    let exposed_faces = self.compute_exposed_faces(min, max);

                    out.push(LocalBox {
                        min,
                        max,
                        material: mat,
                        exposed_faces,
                    });
                }
            }
        }

        out
    }

    /// Checks all 6 outer slabs of `[min, max)` across brick boundaries to
    /// determine which faces have at least one adjacent empty voxel.
    fn compute_exposed_faces(&self, min: [i32; 3], max: [i32; 3]) -> u8 {
        let mut mask = 0u8;

        // -X face
        'neg_x: for z in min[2]..max[2] {
            for y in min[1]..max[1] {
                if !self.has_voxel(VoxelCoord::new(min[0] - 1, y, z)) {
                    mask |= FACE_NEG_X;
                    break 'neg_x;
                }
            }
        }
        // +X face
        'pos_x: for z in min[2]..max[2] {
            for y in min[1]..max[1] {
                if !self.has_voxel(VoxelCoord::new(max[0], y, z)) {
                    mask |= FACE_POS_X;
                    break 'pos_x;
                }
            }
        }
        // -Y face
        'neg_y: for z in min[2]..max[2] {
            for x in min[0]..max[0] {
                if !self.has_voxel(VoxelCoord::new(x, min[1] - 1, z)) {
                    mask |= FACE_NEG_Y;
                    break 'neg_y;
                }
            }
        }
        // +Y face
        'pos_y: for z in min[2]..max[2] {
            for x in min[0]..max[0] {
                if !self.has_voxel(VoxelCoord::new(x, max[1], z)) {
                    mask |= FACE_POS_Y;
                    break 'pos_y;
                }
            }
        }
        // -Z face
        'neg_z: for y in min[1]..max[1] {
            for x in min[0]..max[0] {
                if !self.has_voxel(VoxelCoord::new(x, y, min[2] - 1)) {
                    mask |= FACE_NEG_Z;
                    break 'neg_z;
                }
            }
        }
        // +Z face
        'pos_z: for y in min[1]..max[1] {
            for x in min[0]..max[0] {
                if !self.has_voxel(VoxelCoord::new(x, y, max[2])) {
                    mask |= FACE_POS_Z;
                    break 'pos_z;
                }
            }
        }

        mask
    }

    /// Detects 6-connected voxel components. If more than one component exists,
    /// retains the largest component in `self` (tie-broken deterministically by
    /// lexicographically smallest coordinate) and splits off all remaining
    /// components.
    ///
    /// Returns a list of `(detached_grid, local_com_offset)` pairs where
    /// `local_com_offset = detached_grid.com_local() - pre_split_com_local`,
    /// enabling exact conservation of linear and angular momentum during rigid
    /// body fracturing.
    pub fn split_disconnected(&mut self, reg: &MaterialRegistry) -> Vec<(VoxelSubGrid, Vec3d)> {
        if self.total_voxels <= 1 {
            return Vec::new();
        }

        // Per-brick 4096-bit visited mask in deterministic BTreeMap order.
        let mut visited: BTreeMap<(i16, i16, i16), [u64; 64]> = BTreeMap::new();
        for &bk in self.bricks.keys() {
            visited.insert(bk, [0u64; 64]);
        }

        let mut components: Vec<Vec<(VoxelCoord, VoxelEntry)>> = Vec::new();
        let mut queue: VecDeque<VoxelCoord> = VecDeque::new();

        let brick_keys: Vec<(i16, i16, i16)> = self.bricks.keys().copied().collect();
        for bk in brick_keys {
            let ox = i32::from(bk.0) * 16;
            let oy = i32::from(bk.1) * 16;
            let oz = i32::from(bk.2) * 16;

            for idx in 0..4096usize {
                let word = idx >> 6;
                let bit = 1u64 << (idx & 63);
                if (visited[&bk][word] & bit) != 0 {
                    continue;
                }
                let Some(start_entry) = self.bricks[&bk].voxels[idx] else {
                    continue;
                };

                let lx = (idx & 15) as i32;
                let ly = ((idx >> 4) & 15) as i32;
                let lz = ((idx >> 8) & 15) as i32;
                let start_coord = VoxelCoord::new(ox + lx, oy + ly, oz + lz);

                visited.get_mut(&bk).expect("brick exists")[word] |= bit;
                queue.clear();
                queue.push_back(start_coord);

                let mut comp = Vec::new();
                comp.push((start_coord, start_entry));

                while let Some(curr) = queue.pop_front() {
                    for nb in curr.neighbors_6() {
                        let nb_bk = nb.brick_key();
                        let Some(nb_brick) = self.bricks.get(&nb_bk) else {
                            continue;
                        };
                        let (nlx, nly, nlz) = nb.local_in_brick();
                        let nidx = VoxelBrick16::linear_idx(nlx, nly, nlz);
                        let nword = nidx >> 6;
                        let nbit = 1u64 << (nidx & 63);
                        let vbits = visited.get_mut(&nb_bk).expect("brick exists");
                        if (vbits[nword] & nbit) != 0 {
                            continue;
                        }
                        if let Some(nb_entry) = nb_brick.voxels[nidx] {
                            vbits[nword] |= nbit;
                            comp.push((nb, nb_entry));
                            queue.push_back(nb);
                        }
                    }
                }

                components.push(comp);
            }
        }

        if components.len() <= 1 {
            return Vec::new();
        }

        // Select the largest component to keep in place (tie-break: smallest min coord).
        let mut keep_idx = 0usize;
        let mut keep_len = components[0].len();
        let mut keep_min = components[0]
            .iter()
            .map(|(c, _)| *c)
            .min()
            .unwrap_or_default();

        for (i, comp) in components.iter().enumerate().skip(1) {
            let cmin = comp.iter().map(|(c, _)| *c).min().unwrap_or_default();
            if comp.len() > keep_len || (comp.len() == keep_len && cmin < keep_min) {
                keep_idx = i;
                keep_len = comp.len();
                keep_min = cmin;
            }
        }

        let orig_com = self.com_local();
        let mut detached = Vec::with_capacity(components.len() - 1);

        for (i, comp) in components.into_iter().enumerate() {
            if i == keep_idx {
                continue;
            }
            let mut sub = VoxelSubGrid::new();
            for (coord, entry) in comp {
                self.set_voxel(coord, None, reg);
                sub.set_voxel(coord, Some(entry), reg);
            }
            sub.rebuild_dirty_colliders();
            let offset = sub.com_local() - orig_com;
            detached.push((sub, offset));
        }

        self.rebuild_dirty_colliders();
        detached
    }
}
