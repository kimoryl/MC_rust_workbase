//! 16x16 chunk collision snapshot cache and block-state material mapping
//! implementing [`horizon_physics_core::WorldVoxelProvider`].

use std::collections::{BTreeMap, BTreeSet};

use horizon_physics_core::{
    Aabbd, MaterialId, MaterialRegistry, Vec3d, VoxelMaterial, WorldBlockCollision,
    WorldFluidSample, WorldVoxelProvider,
};
use mod_api::{Host, LevelRef};

/// Embedded default vanilla block and fluid physical properties.
pub const DEFAULT_BLOCKS_JSON: &str =
    include_str!("../data/horizon_physics/materials/default_blocks.json");

/// One pre-sampled terrain cell record: `(block_xyz, Option<(block_id, local_aabb)>)`.
pub type PrefetchedTerrainSample = ([i32; 3], Option<(String, Option<Aabbd>)>);

/// Fluid physical parameters mapped from a block/fluid identifier.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FluidMaterialSpec {
    pub density: f64,
    pub linear_drag: f64,
    pub angular_drag: f64,
}

impl FluidMaterialSpec {
    #[inline]
    pub const fn water() -> Self {
        Self {
            density: 1000.0,
            linear_drag: 2.5,
            angular_drag: 3.0,
        }
    }

    #[inline]
    pub fn to_sample(self, surface_y: f64) -> WorldFluidSample {
        WorldFluidSample {
            density: self.density,
            surface_y,
            flow_velocity: Vec3d::ZERO,
            linear_drag: self.linear_drag,
            angular_drag: self.angular_drag,
        }
    }
}

/// Deterministic block-name to [`MaterialId`] / [`VoxelMaterial`] registry loaded
/// from `data/horizon_physics/materials/default_blocks.json` plus runtime custom entries.
#[derive(Clone, Debug)]
pub struct BlockMaterialTable {
    block_ids: BTreeMap<String, MaterialId>,
    materials: BTreeMap<MaterialId, VoxelMaterial>,
    fluids: BTreeMap<String, FluidMaterialSpec>,
    default_material: VoxelMaterial,
    next_id: u16,
}

impl Default for BlockMaterialTable {
    fn default() -> Self {
        Self::from_embedded_defaults()
    }
}

impl BlockMaterialTable {
    /// Creates a table pre-populated from the embedded `default_blocks.json`.
    pub fn from_embedded_defaults() -> Self {
        let mut table = Self::empty();
        let _ = table.load_from_json(DEFAULT_BLOCKS_JSON);
        table
    }

    /// Creates a minimal table with only the six built-in core materials.
    pub fn empty() -> Self {
        let mut materials = BTreeMap::new();
        materials.insert(MaterialId::STONE, VoxelMaterial::STONE);
        materials.insert(MaterialId::WOOD, VoxelMaterial::WOOD);
        materials.insert(MaterialId::IRON, VoxelMaterial::IRON);
        materials.insert(MaterialId::SLIME, VoxelMaterial::SLIME);
        materials.insert(MaterialId::ICE, VoxelMaterial::ICE);
        materials.insert(MaterialId::WOOL, VoxelMaterial::WOOL);

        let mut fluids = BTreeMap::new();
        fluids.insert("minecraft:water".to_owned(), FluidMaterialSpec::water());
        fluids.insert(
            "minecraft:lava".to_owned(),
            FluidMaterialSpec {
                density: 3100.0,
                linear_drag: 8.0,
                angular_drag: 10.0,
            },
        );

        Self {
            block_ids: BTreeMap::new(),
            materials,
            fluids,
            default_material: VoxelMaterial::STONE,
            next_id: 16,
        }
    }

    /// Parses and merges block and fluid material entries from JSON text.
    pub fn load_from_json(&mut self, json: &str) -> Result<usize, String> {
        let root = parse_json(json)?;
        let JsonValue::Object(map) = root else {
            return Err("materials root must be a JSON object".to_owned());
        };

        if let Some(def_val) = map.get("default_material")
            && let Some(mat) = parse_voxel_material(def_val, self.default_material)
        {
            self.default_material = mat;
        }

        let mut loaded = 0usize;
        if let Some(JsonValue::Object(blocks)) = map.get("blocks") {
            for (name, val) in blocks {
                if let Some(mat) = parse_voxel_material(val, self.default_material) {
                    self.register_block_material(name, mat);
                    loaded += 1;
                }
            }
        }

        if let Some(JsonValue::Object(fluids)) = map.get("fluids") {
            for (name, val) in fluids {
                if let Some(spec) = parse_fluid_spec(val) {
                    self.fluids.insert(canonical_block_name(name), spec);
                }
            }
        }

        Ok(loaded)
    }

    /// Registers or replaces a block material mapping and returns its [`MaterialId`].
    pub fn register_block_material(&mut self, block_id: &str, mat: VoxelMaterial) -> MaterialId {
        let key = canonical_block_name(block_id);
        if let Some(&existing) = self.block_ids.get(&key) {
            self.materials.insert(existing, mat);
            return existing;
        }

        let mat_id = match key.as_str() {
            "minecraft:stone" => MaterialId::STONE,
            "minecraft:oak_planks" => MaterialId::WOOD,
            "minecraft:iron_block" => MaterialId::IRON,
            "minecraft:slime_block" => MaterialId::SLIME,
            "minecraft:ice" => MaterialId::ICE,
            "minecraft:white_wool" => MaterialId::WOOL,
            _ => {
                let id = MaterialId(self.next_id);
                self.next_id = self.next_id.saturating_add(1).max(16);
                id
            }
        };
        self.materials.insert(mat_id, mat);
        self.block_ids.insert(key, mat_id);
        mat_id
    }

    /// Registers a custom fluid mapping.
    pub fn register_fluid_material(&mut self, fluid_id: &str, spec: FluidMaterialSpec) {
        self.fluids.insert(canonical_block_name(fluid_id), spec);
    }

    /// Synchronizes all material definitions into a core [`MaterialRegistry`].
    pub fn sync_into_registry(&self, reg: &mut MaterialRegistry) {
        for (&id, &mat) in &self.materials {
            reg.register(id, mat);
        }
    }

    /// Iterates over all `(MaterialId, VoxelMaterial)` definitions.
    pub fn iter_materials(&self) -> impl Iterator<Item = (MaterialId, VoxelMaterial)> + '_ {
        self.materials.iter().map(|(&id, &mat)| (id, mat))
    }

    /// Number of explicitly mapped block IDs.
    pub fn block_count(&self) -> usize {
        self.block_ids.len()
    }

    /// Resolves `(MaterialId, VoxelMaterial)` for a block ID using exact table
    /// lookup first, followed by deterministic category suffix rules.
    pub fn resolve_block(&self, block_id: &str) -> (MaterialId, VoxelMaterial) {
        let key = canonical_block_name(block_id);
        if let Some(&id) = self.block_ids.get(&key) {
            let mat = self
                .materials
                .get(&id)
                .copied()
                .unwrap_or(self.default_material);
            return (id, mat);
        }

        let path = key.split_once(':').map_or(key.as_str(), |(_, p)| p);
        let fallback_id = if path.contains("planks")
            || path.contains("log")
            || path.contains("wood")
            || path.contains("timber")
            || path.contains("slab_oak")
        {
            MaterialId::WOOD
        } else if path.contains("iron")
            || path.contains("steel")
            || path.contains("anvil")
            || path.contains("metal")
        {
            MaterialId::IRON
        } else if path.contains("slime") {
            MaterialId::SLIME
        } else if path.contains("ice") {
            MaterialId::ICE
        } else if path.contains("wool") || path.contains("carpet") || path.contains("fabric") {
            MaterialId::WOOL
        } else {
            MaterialId::STONE
        };
        let mat = self
            .materials
            .get(&fallback_id)
            .copied()
            .unwrap_or(self.default_material);
        (fallback_id, mat)
    }

    /// Resolves fluid physical properties if `block_id` represents a known fluid.
    pub fn resolve_fluid(&self, block_id: &str) -> Option<FluidMaterialSpec> {
        let key = canonical_block_name(block_id);
        if let Some(&spec) = self.fluids.get(&key) {
            return Some(spec);
        }
        let path = key.split_once(':').map_or(key.as_str(), |(_, p)| p);
        if path == "flowing_water" {
            self.fluids.get("minecraft:water").copied()
        } else if path == "flowing_lava" {
            self.fluids.get("minecraft:lava").copied()
        } else {
            None
        }
    }

    /// Returns whether a block ID is non-solid (air, fluid, foliage, or pass-through).
    pub fn is_non_solid(&self, block_id: &str) -> bool {
        let key = canonical_block_name(block_id);
        if self.resolve_fluid(&key).is_some() {
            return true;
        }
        let path = key.split_once(':').map_or(key.as_str(), |(_, p)| p);
        matches!(
            path,
            "air"
                | "cave_air"
                | "void_air"
                | "short_grass"
                | "tall_grass"
                | "fern"
                | "large_fern"
                | "torch"
                | "wall_torch"
                | "soul_torch"
                | "redstone_torch"
                | "lever"
                | "rail"
                | "powered_rail"
                | "detector_rail"
                | "activator_rail"
                | "structure_void"
                | "light"
        ) || path.ends_with("_air")
            || path.ends_with("_flower")
            || path.ends_with("_sapling")
    }
}

/// Normalizes a block state string (`"minecraft:oak_stairs[facing=east]"` or `"stone"`)
/// into its canonical `namespace:path` form.
pub fn canonical_block_name(raw: &str) -> String {
    let base = raw.split('[').next().unwrap_or(raw).trim();
    if base.contains(':') {
        base.to_ascii_lowercase()
    } else {
        format!("minecraft:{}", base.to_ascii_lowercase())
    }
}

/// Cached collision/fluid classification for a single world voxel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CachedBlockCell {
    Air,
    Solid {
        collision: WorldBlockCollision,
        full_cube: bool,
    },
    Fluid(WorldFluidSample),
}

impl CachedBlockCell {
    #[inline]
    pub fn from_block_name(
        y: i32,
        block_name: &str,
        materials: &BlockMaterialTable,
        local_aabb: Option<Aabbd>,
    ) -> Self {
        if let Some(fluid) = materials.resolve_fluid(block_name) {
            return Self::Fluid(fluid.to_sample(f64::from(y) + 1.0));
        }
        if materials.is_non_solid(block_name) {
            return Self::Air;
        }
        let (mat_id, _) = materials.resolve_block(block_name);
        match local_aabb {
            Some(aabb) if !aabb.is_empty() => {
                let full_cube = aabb.min.x <= 1e-6
                    && aabb.min.y <= 1e-6
                    && aabb.min.z <= 1e-6
                    && aabb.max.x >= 1.0 - 1e-6
                    && aabb.max.y >= 1.0 - 1e-6
                    && aabb.max.z >= 1.0 - 1e-6;
                Self::Solid {
                    collision: WorldBlockCollision {
                        material: mat_id,
                        local_aabb: aabb,
                    },
                    full_cube,
                }
            }
            Some(_) => Self::Air,
            None => Self::Solid {
                collision: WorldBlockCollision::full_block(mat_id),
                full_cube: true,
            },
        }
    }
}

/// Per-chunk 16x16 horizontal collision cache column.
#[derive(Clone, Debug, Default)]
pub struct ChunkColumnCache {
    cells: BTreeMap<(u8, i32, u8), CachedBlockCell>,
    revision: u64,
}

impl ChunkColumnCache {
    #[inline]
    pub fn get(&self, lx: u8, y: i32, lz: u8) -> Option<CachedBlockCell> {
        self.cells.get(&(lx, y, lz)).copied()
    }

    #[inline]
    pub fn insert(&mut self, lx: u8, y: i32, lz: u8, cell: CachedBlockCell) {
        self.cells.insert((lx, y, lz), cell);
    }

    #[inline]
    pub fn remove(&mut self, lx: u8, y: i32, lz: u8) -> bool {
        let removed = self.cells.remove(&(lx, y, lz)).is_some();
        if removed {
            self.revision = self.revision.wrapping_add(1);
        }
        removed
    }

    #[inline]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// Diagnostic metrics for [`TerrainCache`].
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct TerrainCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub block_invalidations: u64,
    pub chunk_evictions: u64,
    pub cached_chunks: usize,
    pub cached_cells: usize,
}

/// Host terrain sampling abstraction used when a cache miss occurs.
pub trait HostTerrainSampler {
    /// Samples block name and optional custom local collision box at world `(x, y, z)`.
    /// Returns `None` if the block is air or unloaded.
    fn sample_block(&self, x: i32, y: i32, z: i32) -> Option<(String, Option<Aabbd>)>;
}

/// Live Horizon [`LevelRef`] terrain sampler backed by `&dyn Host`.
pub struct LiveLevelSampler<'a> {
    host: &'a dyn Host,
    level: LevelRef,
}

impl<'a> LiveLevelSampler<'a> {
    pub fn new(host: &'a dyn Host, level: LevelRef) -> Self {
        Self { host, level }
    }
}

impl HostTerrainSampler for LiveLevelSampler<'_> {
    fn sample_block(&self, x: i32, y: i32, z: i32) -> Option<(String, Option<Aabbd>)> {
        let block_key = self.level.block(self.host, [x, y, z])?;
        let name = block_key.id(self.host);
        let boxes = self.level.collision_boxes(
            self.host,
            [
                f64::from(x),
                f64::from(y),
                f64::from(z),
                f64::from(x + 1),
                f64::from(y + 1),
                f64::from(z + 1),
            ],
        );
        let local_aabb = boxes.map(|list| {
            if list.is_empty() {
                return Aabbd::EMPTY;
            }
            let mut merged = Aabbd::EMPTY;
            for b in list {
                let min = Vec3d::new(
                    (b[0] - f64::from(x)).clamp(0.0, 1.0),
                    (b[1] - f64::from(y)).clamp(0.0, 1.0),
                    (b[2] - f64::from(z)).clamp(0.0, 1.0),
                );
                let max = Vec3d::new(
                    (b[3] - f64::from(x)).clamp(0.0, 1.0),
                    (b[4] - f64::from(y)).clamp(0.0, 1.0),
                    (b[5] - f64::from(z)).clamp(0.0, 1.0),
                );
                if min.x < max.x && min.y < max.y && min.z < max.z {
                    merged = merged.union(Aabbd::new(min, max));
                }
            }
            merged
        });
        Some((name, local_aabb))
    }
}

/// 16x16 chunk-indexed terrain collision cache with targeted invalidation and
/// built-in material resolution.
#[derive(Clone, Debug)]
pub struct TerrainCache {
    chunks: BTreeMap<(i32, i32), ChunkColumnCache>,
    overrides: BTreeMap<(i32, i32, i32), CachedBlockCell>,
    materials: BlockMaterialTable,
    flat_floor_y: Option<i32>,
    flat_water_y: Option<f64>,
    hits: u64,
    misses: u64,
    block_invalidations: u64,
    chunk_evictions: u64,
}

impl Default for TerrainCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TerrainCache {
    pub fn new() -> Self {
        Self {
            chunks: BTreeMap::new(),
            overrides: BTreeMap::new(),
            materials: BlockMaterialTable::from_embedded_defaults(),
            flat_floor_y: None,
            flat_water_y: None,
            hits: 0,
            misses: 0,
            block_invalidations: 0,
            chunk_evictions: 0,
        }
    }

    #[inline]
    pub fn materials(&self) -> &BlockMaterialTable {
        &self.materials
    }

    #[inline]
    pub fn materials_mut(&mut self) -> &mut BlockMaterialTable {
        &mut self.materials
    }

    /// Configures an optional synthetic flat ground plane (`y < ground_y` solid)
    /// for flat test worlds or fallback terrain.
    pub fn set_flat_floor(&mut self, ground_y: Option<i32>) {
        self.flat_floor_y = ground_y;
        self.chunks.clear();
    }

    /// Configures an optional flat water surface elevation for buoyancy testing.
    pub fn set_flat_water_level(&mut self, water_y: Option<f64>) {
        self.flat_water_y = water_y;
        self.chunks.clear();
    }

    /// Explicitly populates a block into both the authoritative override table and
    /// its 16x16 chunk cache entry.
    pub fn set_block(&mut self, pos: [i32; 3], block_name: &str) {
        let cell = CachedBlockCell::from_block_name(pos[1], block_name, &self.materials, None);
        self.overrides.insert((pos[0], pos[1], pos[2]), cell);
        let (ck, lx, lz) = chunk_and_local(pos[0], pos[2]);
        self.chunks
            .entry(ck)
            .or_default()
            .insert(lx, pos[1], lz, cell);
    }

    /// Clears a single block override and evicts its chunk cache slot.
    pub fn clear_block(&mut self, pos: [i32; 3]) {
        self.overrides.remove(&(pos[0], pos[1], pos[2]));
        let (ck, lx, lz) = chunk_and_local(pos[0], pos[2]);
        if let Some(chunk) = self.chunks.get_mut(&ck) {
            chunk.remove(lx, pos[1], lz);
        }
    }

    /// Precisely invalidates the cached cell at `pos` (optionally replacing it
    /// with `new_block_name` if already known from `block_changed`/`break_block`/
    /// `entity_place`), leaving unrelated chunks untouched.
    ///
    /// Returns the expanded world [`Aabbd`] that should be passed to
    /// [`horizon_physics_core::PhysicsWorld::wake_in_aabb`].
    pub fn invalidate_block(&mut self, pos: [i32; 3], new_block_name: Option<&str>) -> Aabbd {
        let (ck, lx, lz) = chunk_and_local(pos[0], pos[2]);
        self.block_invalidations += 1;

        if let Some(name) = new_block_name {
            let cell = CachedBlockCell::from_block_name(pos[1], name, &self.materials, None);
            self.overrides.insert((pos[0], pos[1], pos[2]), cell);
            let chunk = self.chunks.entry(ck).or_default();
            chunk.revision = chunk.revision.wrapping_add(1);
            chunk.insert(lx, pos[1], lz, cell);
        } else {
            self.overrides.remove(&(pos[0], pos[1], pos[2]));
            if let Some(chunk) = self.chunks.get_mut(&ck) {
                chunk.remove(lx, pos[1], lz);
            }
        }

        Aabbd::new(
            Vec3d::new(
                f64::from(pos[0]) - 0.25,
                f64::from(pos[1]) - 0.25,
                f64::from(pos[2]) - 0.25,
            ),
            Vec3d::new(
                f64::from(pos[0] + 1) + 0.25,
                f64::from(pos[1] + 1) + 0.25,
                f64::from(pos[2] + 1) + 0.25,
            ),
        )
    }

    /// Evicts an entire 16x16 chunk column on `chunk_unload`.
    pub fn invalidate_chunk(&mut self, cx: i32, cz: i32) -> bool {
        if self.chunks.remove(&(cx, cz)).is_some() {
            self.chunk_evictions += 1;
            true
        } else {
            false
        }
    }

    /// Clears all cached chunks (retaining material definitions and overrides).
    pub fn clear_cache(&mut self) {
        self.chunks.clear();
    }

    /// Returns whether a cell is currently resident in its 16x16 chunk cache.
    pub fn is_cell_cached(&self, pos: [i32; 3]) -> bool {
        let (ck, lx, lz) = chunk_and_local(pos[0], pos[2]);
        self.chunks
            .get(&ck)
            .and_then(|c| c.get(lx, pos[1], lz))
            .is_some()
    }

    /// Returns the revision counter of chunk `(cx, cz)`, if cached.
    pub fn chunk_revision(&self, cx: i32, cz: i32) -> Option<u64> {
        self.chunks.get(&(cx, cz)).map(ChunkColumnCache::revision)
    }

    /// Returns diagnostic statistics for telemetry and tests.
    pub fn stats(&self) -> TerrainCacheStats {
        let cached_cells = self.chunks.values().map(ChunkColumnCache::len).sum();
        TerrainCacheStats {
            hits: self.hits,
            misses: self.misses,
            block_invalidations: self.block_invalidations,
            chunk_evictions: self.chunk_evictions,
            cached_chunks: self.chunks.len(),
            cached_cells,
        }
    }

    /// Borrows this cache coupled with a live host sampler as a [`WorldVoxelProvider`].
    pub fn with_sampler<'a>(
        &'a mut self,
        sampler: &'a dyn HostTerrainSampler,
    ) -> TerrainCacheView<'a> {
        TerrainCacheView {
            cache: self,
            sampler: Some(sampler),
        }
    }

    /// Returns `true` when `(x, y, z)` already resides in a cached chunk or override map.
    #[inline]
    pub fn has_cached_cell(&self, x: i32, y: i32, z: i32) -> bool {
        let (ck, lx, lz) = chunk_and_local(x, z);
        self.chunks
            .get(&ck)
            .and_then(|c| c.get(lx, y, lz))
            .is_some()
            || self.overrides.contains_key(&(x, y, z))
    }

    /// Collects all uncached integer block coordinates intersecting `aabbs` (capped at `max_cells`),
    /// allowing callers to prefetch via [`HostTerrainSampler`] without holding a write lock.
    pub fn collect_uncached_cells(&self, aabbs: &[Aabbd], max_cells: usize) -> Vec<[i32; 3]> {
        let empty = BTreeSet::new();
        self.collect_uncached_cells_skipping(aabbs, max_cells, &empty)
    }

    /// Variant of [`Self::collect_uncached_cells`] skipping coordinates already attempted
    /// within the same lock-free multi-batch prefetch pass (e.g. unloaded chunks returning `None`).
    pub fn collect_uncached_cells_skipping(
        &self,
        aabbs: &[Aabbd],
        max_cells: usize,
        skip: &BTreeSet<[i32; 3]>,
    ) -> Vec<[i32; 3]> {
        let mut seen: BTreeSet<[i32; 3]> = BTreeSet::new();
        let mut out = Vec::new();
        for aabb in aabbs {
            if aabb.is_empty() || !aabb.min.is_finite() || !aabb.max.is_finite() {
                continue;
            }
            let min_x = aabb.min.x.floor() as i32;
            let min_y = aabb.min.y.floor() as i32;
            let min_z = aabb.min.z.floor() as i32;
            let max_x = aabb.max.x.ceil() as i32;
            let max_y = aabb.max.y.ceil() as i32;
            let max_z = aabb.max.z.ceil() as i32;
            for z in min_z..=max_z {
                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        if out.len() >= max_cells {
                            return out;
                        }
                        let pos = [x, y, z];
                        if !skip.contains(&pos)
                            && !self.has_cached_cell(x, y, z)
                            && seen.insert(pos)
                        {
                            out.push(pos);
                        }
                    }
                }
            }
        }
        out
    }

    /// Inserts authoritative prefetched terrain samples into chunk storage before stepping.
    /// `None` samples (unloaded chunks or out-of-bounds world reads) are NOT written as air.
    pub fn insert_prefetched_samples(&mut self, samples: Vec<PrefetchedTerrainSample>) {
        for ([x, y, z], sampled) in samples {
            if self.has_cached_cell(x, y, z) {
                continue;
            }
            let Some((name, local_aabb)) = sampled else {
                continue;
            };
            let cell = CachedBlockCell::from_block_name(y, &name, &self.materials, local_aabb);
            let (ck, lx, lz) = chunk_and_local(x, z);
            self.chunks.entry(ck).or_default().insert(lx, y, lz, cell);
        }
    }

    /// Resolves a cell in read-only mode (used by immutable raycasts).
    pub fn peek_cell(&self, x: i32, y: i32, z: i32) -> CachedBlockCell {
        let (ck, lx, lz) = chunk_and_local(x, z);
        if let Some(cell) = self.chunks.get(&ck).and_then(|c| c.get(lx, y, lz)) {
            return cell;
        }
        if let Some(&cell) = self.overrides.get(&(x, y, z)) {
            return cell;
        }
        self.synthetic_cell(y)
    }

    fn query_cell(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        sampler: Option<&dyn HostTerrainSampler>,
    ) -> CachedBlockCell {
        let (ck, lx, lz) = chunk_and_local(x, z);
        if let Some(cell) = self.chunks.get(&ck).and_then(|c| c.get(lx, y, lz)) {
            self.hits += 1;
            return cell;
        }

        self.misses += 1;
        let cell = if let Some(&ov) = self.overrides.get(&(x, y, z)) {
            ov
        } else if let Some(s) = sampler {
            if let Some((name, local_aabb)) = s.sample_block(x, y, z) {
                CachedBlockCell::from_block_name(y, &name, &self.materials, local_aabb)
            } else {
                // Chunk is unloaded / world boundary returned None: synthesize transiently,
                // never poison permanent chunk storage with air.
                return self.synthetic_cell(y);
            }
        } else {
            // Do NOT poison permanent chunk cache with synthetic air/flat guesses when
            // stepping without a live sampler; return the transient synthetic cell only.
            return self.synthetic_cell(y);
        };

        self.chunks.entry(ck).or_default().insert(lx, y, lz, cell);
        cell
    }

    #[inline]
    fn synthetic_cell(&self, y: i32) -> CachedBlockCell {
        if let Some(floor_y) = self.flat_floor_y
            && y < floor_y
        {
            return CachedBlockCell::Solid {
                collision: WorldBlockCollision::full_block(MaterialId::STONE),
                full_cube: true,
            };
        }
        if let Some(water_y) = self.flat_water_y {
            let min_water_y = self.flat_floor_y.unwrap_or(i32::MIN);
            if y >= min_water_y && f64::from(y) < water_y {
                return CachedBlockCell::Fluid(WorldFluidSample::water(water_y));
            }
        }
        CachedBlockCell::Air
    }
}

impl WorldVoxelProvider for TerrainCache {
    fn voxel_collision(&mut self, x: i32, y: i32, z: i32) -> Option<WorldBlockCollision> {
        match self.query_cell(x, y, z, None) {
            CachedBlockCell::Solid { collision, .. } => Some(collision),
            CachedBlockCell::Air | CachedBlockCell::Fluid(_) => None,
        }
    }

    fn is_full_solid(&mut self, x: i32, y: i32, z: i32) -> bool {
        matches!(
            self.query_cell(x, y, z, None),
            CachedBlockCell::Solid {
                full_cube: true,
                ..
            }
        )
    }

    fn fluid_at(&mut self, x: i32, y: i32, z: i32) -> Option<WorldFluidSample> {
        match self.query_cell(x, y, z, None) {
            CachedBlockCell::Fluid(sample) => Some(sample),
            CachedBlockCell::Air | CachedBlockCell::Solid { .. } => None,
        }
    }
}

/// Short-lived [`WorldVoxelProvider`] view binding [`TerrainCache`] to an optional
/// live [`HostTerrainSampler`] during a server physics step.
pub struct TerrainCacheView<'a> {
    cache: &'a mut TerrainCache,
    sampler: Option<&'a dyn HostTerrainSampler>,
}

impl WorldVoxelProvider for TerrainCacheView<'_> {
    fn voxel_collision(&mut self, x: i32, y: i32, z: i32) -> Option<WorldBlockCollision> {
        match self.cache.query_cell(x, y, z, self.sampler) {
            CachedBlockCell::Solid { collision, .. } => Some(collision),
            CachedBlockCell::Air | CachedBlockCell::Fluid(_) => None,
        }
    }

    fn is_full_solid(&mut self, x: i32, y: i32, z: i32) -> bool {
        matches!(
            self.cache.query_cell(x, y, z, self.sampler),
            CachedBlockCell::Solid {
                full_cube: true,
                ..
            }
        )
    }

    fn fluid_at(&mut self, x: i32, y: i32, z: i32) -> Option<WorldFluidSample> {
        match self.cache.query_cell(x, y, z, self.sampler) {
            CachedBlockCell::Fluid(sample) => Some(sample),
            CachedBlockCell::Air | CachedBlockCell::Solid { .. } => None,
        }
    }
}

#[inline]
pub const fn chunk_and_local(x: i32, z: i32) -> ((i32, i32), u8, u8) {
    (
        (x.div_euclid(16), z.div_euclid(16)),
        x.rem_euclid(16) as u8,
        z.rem_euclid(16) as u8,
    )
}

// ---------------------------------------------------------------------------
// Minimal, zero-dependency, safe JSON parser for material data tables.
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum JsonValue {
    Ignored,
    Number(f64),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(v) if v.is_finite() => Some(*v),
            _ => None,
        }
    }
}

fn parse_voxel_material(val: &JsonValue, fallback: VoxelMaterial) -> Option<VoxelMaterial> {
    let JsonValue::Object(obj) = val else {
        return None;
    };
    let density = obj
        .get("density")
        .and_then(JsonValue::as_f64)
        .unwrap_or(fallback.density)
        .max(1.0);
    let friction = obj
        .get("friction")
        .and_then(JsonValue::as_f64)
        .unwrap_or(fallback.friction)
        .clamp(0.0, 2.0);
    let restitution = obj
        .get("restitution")
        .and_then(JsonValue::as_f64)
        .unwrap_or(fallback.restitution)
        .clamp(0.0, 1.0);
    let linear_drag = obj
        .get("linear_drag")
        .and_then(JsonValue::as_f64)
        .unwrap_or(fallback.linear_drag)
        .max(0.0);
    let angular_drag = obj
        .get("angular_drag")
        .and_then(JsonValue::as_f64)
        .unwrap_or(fallback.angular_drag)
        .max(0.0);
    Some(VoxelMaterial {
        density,
        friction,
        restitution,
        linear_drag,
        angular_drag,
    })
}

fn parse_fluid_spec(val: &JsonValue) -> Option<FluidMaterialSpec> {
    let JsonValue::Object(obj) = val else {
        return None;
    };
    let density = obj
        .get("density")
        .and_then(JsonValue::as_f64)
        .unwrap_or(1000.0)
        .max(1.0);
    let linear_drag = obj
        .get("linear_drag")
        .and_then(JsonValue::as_f64)
        .unwrap_or(2.5)
        .max(0.0);
    let angular_drag = obj
        .get("angular_drag")
        .and_then(JsonValue::as_f64)
        .unwrap_or(3.0)
        .max(0.0);
    Some(FluidMaterialSpec {
        density,
        linear_drag,
        angular_drag,
    })
}

fn parse_json(src: &str) -> Result<JsonValue, String> {
    let mut parser = JsonParser {
        bytes: src.as_bytes(),
        pos: 0,
    };
    let val = parser.parse_value(0)?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(format!("trailing bytes at offset {}", parser.pos));
    }
    Ok(val)
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, String> {
        if depth > 64 {
            return Err("JSON nesting depth exceeded".to_owned());
        }
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(depth + 1),
            Some(b'[') => self.parse_array(depth + 1),
            Some(b'"') => self.parse_string().map(|_| JsonValue::Ignored),
            Some(b't') => {
                self.expect_literal(b"true")?;
                Ok(JsonValue::Ignored)
            }
            Some(b'f') => {
                self.expect_literal(b"false")?;
                Ok(JsonValue::Ignored)
            }
            Some(b'n') => {
                self.expect_literal(b"null")?;
                Ok(JsonValue::Ignored)
            }
            Some(b'-' | b'0'..=b'9') => self.parse_number().map(JsonValue::Number),
            Some(other) => Err(format!(
                "unexpected byte '{}' at offset {}",
                other as char, self.pos
            )),
            None => Err("unexpected end of JSON input".to_owned()),
        }
    }

    fn expect_literal(&mut self, lit: &[u8]) -> Result<(), String> {
        if self.bytes.get(self.pos..self.pos + lit.len()) == Some(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(format!("invalid literal at offset {}", self.pos))
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.pos += 1; // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at offset {}", self.pos));
            }
            self.pos += 1;
            let value = self.parse_value(depth)?;
            map.insert(key, value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                _ => return Err(format!("expected ',' or '}}' at offset {}", self.pos)),
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.pos += 1; // '['
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Ignored);
        }
        loop {
            let _ = self.parse_value(depth)?;
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Ignored);
                }
                _ => return Err(format!("expected ',' or ']' at offset {}", self.pos)),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected '\"' at offset {}", self.pos));
        }
        self.pos += 1;
        let mut out = String::new();
        while let Some(&b) = self.bytes.get(self.pos) {
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = *self
                        .bytes
                        .get(self.pos)
                        .ok_or_else(|| "unterminated escape".to_owned())?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        _ => return Err(format!("unsupported escape at offset {}", self.pos)),
                    }
                }
                other => out.push(other as char),
            }
        }
        Err("unterminated JSON string".to_owned())
    }

    fn parse_number(&mut self) -> Result<f64, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid UTF-8 in number".to_owned())?;
        let val: f64 = slice
            .parse()
            .map_err(|_| format!("invalid number '{slice}' at offset {start}"))?;
        if val.is_finite() {
            Ok(val)
        } else {
            Err(format!("non-finite number '{slice}'"))
        }
    }
}
