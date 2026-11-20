//! High-level voxel rigid-body physics world orchestrating broadphase,
//! narrowphase, CCD, buoyancy, hybrid TGS+XPBD solving, island sleeping,
//! raycasting, fracture splitting, and entity deck carry/sweep.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use crate::arena::{BodyId, JointId, SlotArena};
use crate::broadphase::{DynamicBvh, SpatialHash16};
use crate::ccd::{SweptObb, ToiHit, conservative_advancement_obb_pair};
use crate::island::{Island, IslandBuilder};
use crate::math::{Aabbd, Mat3d, Obbd, Quatd, Transformd, Vec3d};
use crate::narrowphase::{
    ContactManifold, ContactPoint, FeatureKey, collide_obb_aabb, collide_obb_obb,
    reduce_contacts_to_4,
};
use crate::solver::{
    ActiveJoint, JointKind, JointSpec, PreparedManifold, SoftContactParams, SolverSoA,
    SubstepSolverParams, prepare_contacts, solve_joints_xpbd_positions, solve_velocities,
    write_back_contacts,
};
use crate::voxel::{
    ALL_FACES, FACE_NEG_X, FACE_NEG_Y, FACE_NEG_Z, FACE_POS_X, FACE_POS_Y, FACE_POS_Z, MaterialId,
    MaterialRegistry, VoxelCoord, VoxelEntry, VoxelMaterial, VoxelSubGrid,
};

/// Fluid properties sampled at a world voxel coordinate for Archimedes buoyancy.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WorldFluidSample {
    /// Fluid density in kg/m^3 (e.g. `1000.0` for water).
    pub density: f64,
    /// Top fluid surface height in world Y coordinates.
    pub surface_y: f64,
    /// Fluid flow velocity in m/s.
    pub flow_velocity: Vec3d,
    /// Linear hydrodynamic drag coefficient.
    pub linear_drag: f64,
    /// Angular hydrodynamic drag coefficient.
    pub angular_drag: f64,
}

impl WorldFluidSample {
    #[inline]
    pub fn water(surface_y: f64) -> Self {
        Self {
            density: 1000.0,
            surface_y,
            flow_velocity: Vec3d::ZERO,
            linear_drag: 2.5,
            angular_drag: 3.0,
        }
    }
}

/// Static world voxel collision shape within unit cell `[0, 1]^3`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WorldBlockCollision {
    pub material: MaterialId,
    /// Local AABB inside `[0, 1]^3` relative to integer block origin `(x, y, z)`.
    pub local_aabb: Aabbd,
}

impl WorldBlockCollision {
    #[inline]
    pub const fn full_block(material: MaterialId) -> Self {
        Self {
            material,
            local_aabb: Aabbd::new(Vec3d::ZERO, Vec3d::ONE),
        }
    }
}

/// Terrain query interface implemented by the host voxel world.
pub trait WorldVoxelProvider {
    /// Returns collision bounds and material if world block `(x, y, z)` is solid.
    fn voxel_collision(&mut self, x: i32, y: i32, z: i32) -> Option<WorldBlockCollision>;

    /// Fast predicate indicating whether `(x, y, z)` is a full solid cube
    /// (used for internal-edge exposed-face masking).
    fn is_full_solid(&mut self, x: i32, y: i32, z: i32) -> bool {
        self.voxel_collision(x, y, z).is_some()
    }

    /// Returns fluid properties if world block `(x, y, z)` contains fluid.
    fn fluid_at(&mut self, _x: i32, _y: i32, _z: i32) -> Option<WorldFluidSample> {
        None
    }
}

/// Empty world provider containing neither solid terrain nor fluids.
#[derive(Copy, Clone, Debug, Default)]
pub struct EmptyWorldProvider;

impl WorldVoxelProvider for EmptyWorldProvider {
    #[inline]
    fn voxel_collision(&mut self, _x: i32, _y: i32, _z: i32) -> Option<WorldBlockCollision> {
        None
    }

    #[inline]
    fn is_full_solid(&mut self, _x: i32, _y: i32, _z: i32) -> bool {
        false
    }
}

/// Simple flat-ground (and optional water-plane) world provider for testing and prototyping.
#[derive(Copy, Clone, Debug)]
pub struct FlatWorldProvider {
    /// All blocks with `y < ground_y` are solid (so ground surface sits at `y = ground_y as f64`).
    pub ground_y: i32,
    pub ground_material: MaterialId,
    /// Optional water plane surface height; cells with `ground_y <= y < water_level` are water.
    pub water_level: Option<f64>,
}

impl FlatWorldProvider {
    pub const fn new(ground_y: i32) -> Self {
        Self {
            ground_y,
            ground_material: MaterialId::STONE,
            water_level: None,
        }
    }

    pub const fn with_water(ground_y: i32, water_level: f64) -> Self {
        Self {
            ground_y,
            ground_material: MaterialId::STONE,
            water_level: Some(water_level),
        }
    }
}

impl WorldVoxelProvider for FlatWorldProvider {
    #[inline]
    fn voxel_collision(&mut self, _x: i32, y: i32, _z: i32) -> Option<WorldBlockCollision> {
        if y < self.ground_y {
            Some(WorldBlockCollision::full_block(self.ground_material))
        } else {
            None
        }
    }

    #[inline]
    fn is_full_solid(&mut self, _x: i32, y: i32, _z: i32) -> bool {
        y < self.ground_y
    }

    #[inline]
    fn fluid_at(&mut self, _x: i32, y: i32, _z: i32) -> Option<WorldFluidSample> {
        let wl = self.water_level?;
        if y >= self.ground_y && (y as f64) < wl {
            Some(WorldFluidSample::water(wl))
        } else {
            None
        }
    }
}

/// Rigid-body motion category.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BodyKind {
    Dynamic,
    Static,
    Kinematic,
}

/// Descriptor used to spawn a voxel rigid body via [`PhysicsWorld::spawn_body`].
#[derive(Clone, Debug)]
pub struct BodyDesc {
    pub kind: BodyKind,
    /// World transform of the body's center of mass.
    pub transform: Transformd,
    pub linear_velocity: Vec3d,
    pub angular_velocity: Vec3d,
    pub grid: VoxelSubGrid,
    pub gravity_scale: f64,
    pub allow_sleep: bool,
    pub enable_ccd: bool,
    pub linear_damping_override: Option<f64>,
    pub angular_damping_override: Option<f64>,
    pub user_data: u64,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            kind: BodyKind::Dynamic,
            transform: Transformd::IDENTITY,
            linear_velocity: Vec3d::ZERO,
            angular_velocity: Vec3d::ZERO,
            grid: VoxelSubGrid::new(),
            gravity_scale: 1.0,
            allow_sleep: true,
            enable_ccd: false,
            linear_damping_override: None,
            angular_damping_override: None,
            user_data: 0,
        }
    }
}

impl BodyDesc {
    pub fn dynamic(grid: VoxelSubGrid, position: Vec3d) -> Self {
        Self {
            kind: BodyKind::Dynamic,
            transform: Transformd::from_position(position),
            grid,
            ..Self::default()
        }
    }

    pub fn static_body(grid: VoxelSubGrid, position: Vec3d) -> Self {
        Self {
            kind: BodyKind::Static,
            transform: Transformd::from_position(position),
            grid,
            allow_sleep: false,
            ..Self::default()
        }
    }

    pub fn kinematic(grid: VoxelSubGrid, position: Vec3d) -> Self {
        Self {
            kind: BodyKind::Kinematic,
            transform: Transformd::from_position(position),
            grid,
            allow_sleep: false,
            ..Self::default()
        }
    }
}

/// Immutable snapshot of a rigid body's state for rendering, networking, or queries.
#[derive(Clone, Debug, PartialEq)]
pub struct BodySnapshot {
    pub id: BodyId,
    pub kind: BodyKind,
    pub transform: Transformd,
    pub prev_transform: Transformd,
    pub linear_velocity: Vec3d,
    pub angular_velocity: Vec3d,
    pub mass: f64,
    pub com_local: Vec3d,
    pub world_aabb: Aabbd,
    pub sleeping: bool,
    pub enable_ccd: bool,
    pub gravity_scale: f64,
    pub geometry_version: u64,
    pub voxel_count: usize,
    pub user_data: u64,
}

/// Hit information returned by [`PhysicsWorld::raycast`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RaycastHit {
    pub body_id: BodyId,
    pub distance: f64,
    pub point: Vec3d,
    pub normal: Vec3d,
    pub voxel_coord: Option<VoxelCoord>,
    pub box_index: usize,
}

/// Global tuning configuration for [`PhysicsWorld`].
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicsConfig {
    pub gravity: Vec3d,
    pub substeps: u32,
    pub velocity_iterations: u32,
    pub contact_hertz: f64,
    pub contact_damping_ratio: f64,
    pub contact_slop: f64,
    pub max_depenetration_velocity: f64,
    pub warm_start_factor: f64,
    pub restitution_threshold: f64,
    pub linear_sleep_threshold: f64,
    pub angular_sleep_threshold: f64,
    pub sleep_time_threshold: f64,
    pub broadphase_fat_margin: f64,
    pub max_linear_speed: f64,
    pub max_angular_speed: f64,
    pub ccd_speed_threshold: f64,
    pub entity_step_height: f64,
    pub enable_buoyancy: bool,
    pub enable_sleeping: bool,
    pub enable_grid_splitting: bool,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3d::new(0.0, -9.81, 0.0),
            substeps: 4,
            velocity_iterations: 4,
            contact_hertz: 30.0,
            contact_damping_ratio: 1.0,
            contact_slop: 0.002,
            max_depenetration_velocity: 10.0,
            warm_start_factor: 0.85,
            restitution_threshold: 0.5,
            linear_sleep_threshold: 0.05,
            angular_sleep_threshold: 0.05,
            sleep_time_threshold: 0.5,
            broadphase_fat_margin: 0.15,
            max_linear_speed: 1000.0,
            max_angular_speed: 100.0,
            ccd_speed_threshold: 12.0,
            entity_step_height: 1.15,
            enable_buoyancy: true,
            enable_sleeping: true,
            enable_grid_splitting: true,
        }
    }
}

/// Per-step diagnostic counters and wall-clock execution duration.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct StepMetrics {
    pub step_index: u64,
    pub dt: f64,
    pub total_bodies: usize,
    pub awake_bodies: usize,
    pub sleeping_bodies: usize,
    pub island_count: usize,
    pub broadphase_pairs: usize,
    pub contact_manifolds: usize,
    pub contact_points: usize,
    pub active_joints: usize,
    pub toi_events: usize,
    pub step_time_us: u64,
}

/// Simulation lifecycle and collision events emitted into the world queue.
#[derive(Clone, Debug, PartialEq)]
pub enum PhysicsEvent {
    BodySlept {
        body: BodyId,
    },
    BodyWoke {
        body: BodyId,
    },
    Collision {
        body_a: BodyId,
        body_b: Option<BodyId>,
        point: Vec3d,
        normal: Vec3d,
        normal_impulse: f64,
    },
    JointBroken {
        joint: JointId,
        body_a: BodyId,
        body_b: Option<BodyId>,
        force: f64,
        torque: f64,
    },
    GridSplit {
        parent: BodyId,
        children: Vec<BodyId>,
    },
}

/// Outcome of entity AABB deck-carry, collision sweep, and stair step-up against physics bodies.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EntitySweepResult {
    /// Final resolved world-space AABB of the entity.
    pub resolved_aabb: Aabbd,
    /// Net world-space displacement applied (`resolved_center - initial_center`).
    pub applied_delta: Vec3d,
    /// Rigid-body carry displacement contributed by the moving/rotating ground deck.
    pub carry_delta: Vec3d,
    /// Yaw rotation delta (radians about +Y) imparted by a rotating ground deck.
    pub carry_yaw_delta: f64,
    /// Whether the entity is supported on a walkable physics body deck.
    pub on_ground: bool,
    /// Supporting body handle and entity foot anchor in that body's local frame.
    pub ground_body: Option<(BodyId, Vec3d)>,
    /// Primary contact normal encountered during resolution.
    pub hit_normal: Option<Vec3d>,
    /// True if horizontal motion was obstructed.
    pub collided_horizontally: bool,
    /// True if vertical motion hit a floor or ceiling.
    pub collided_vertically: bool,
    /// Vertical height climbed via step-up (`0.0` if none).
    pub stepped_up_height: f64,
}

/// Alias matching [`PhysicsWorld::sweep_entity_aabb`] return type.
pub type EntityCarryOutcome = EntitySweepResult;

#[derive(Clone, Debug)]
struct BodyRecord {
    kind: BodyKind,
    transform: Transformd,
    prev_transform: Transformd,
    linear_velocity: Vec3d,
    angular_velocity: Vec3d,
    accum_force: Vec3d,
    accum_torque: Vec3d,
    grid: VoxelSubGrid,
    mass: f64,
    inv_mass: f64,
    inv_inertia_local: Mat3d,
    inv_inertia_world: Mat3d,
    gravity_scale: f64,
    linear_damping_override: Option<f64>,
    angular_damping_override: Option<f64>,
    allow_sleep: bool,
    enable_ccd: bool,
    sleeping: bool,
    sleep_timer: f64,
    world_aabb: Aabbd,
    bvh_proxy: usize,
    min_box_extent: f64,
    geometry_version: u64,
    user_data: u64,
}

impl BodyRecord {
    fn recompute_mass_properties(&mut self) {
        self.grid.ensure_boxes();
        let mut min_ext = f64::INFINITY;
        for b in self.grid.boxes() {
            let e = b.half_extents() * 2.0;
            min_ext = min_ext.min(e.min_element());
        }
        self.min_box_extent = if min_ext.is_finite() { min_ext } else { 1.0 };

        if self.kind == BodyKind::Dynamic && self.grid.mass() > 1e-12 && self.grid.voxel_count() > 0
        {
            self.mass = self.grid.mass();
            self.inv_mass = 1.0 / self.mass;
            let i_com = self.grid.inertia_com();
            self.inv_inertia_local = i_com.try_inverse().unwrap_or(Mat3d::ZERO).symmetrize();
            let r = self.transform.rotation.to_mat3();
            self.inv_inertia_world =
                Mat3d::similarity_transform(r, self.inv_inertia_local).symmetrize();
        } else {
            self.mass = if self.kind == BodyKind::Dynamic {
                0.0
            } else {
                self.grid.mass()
            };
            self.inv_mass = 0.0;
            self.inv_inertia_local = Mat3d::ZERO;
            self.inv_inertia_world = Mat3d::ZERO;
        }
    }

    fn compute_world_aabb(&self) -> Aabbd {
        let local = self.grid.local_aabb();
        if local.is_empty() {
            return Aabbd::from_center_half_extents(self.transform.position, Vec3d::ZERO);
        }
        let com = self.grid.com_local();
        let local_center = local.center() - com;
        let obb = Obbd::new(
            self.transform.transform_point(local_center),
            local.half_extents(),
            self.transform.rotation,
        );
        obb.to_aabb()
    }

    #[inline]
    fn box_to_world_obb(&self, lb: &crate::voxel::LocalBox) -> Obbd {
        let com = self.grid.com_local();
        let offset = lb.center_local() - com;
        Obbd::new(
            self.transform.transform_point(offset),
            lb.half_extents(),
            self.transform.rotation,
        )
    }

    #[inline]
    fn box_to_world_obb_at(&self, lb: &crate::voxel::LocalBox, pos: Vec3d, rot: Quatd) -> Obbd {
        let com = self.grid.com_local();
        let offset = lb.center_local() - com;
        Obbd::new(pos + rot.rotate_vec3(offset), lb.half_extents(), rot)
    }

    #[inline]
    fn damping(&self) -> (f64, f64) {
        let (avg_lin, avg_ang) = self.grid.average_drag();
        (
            self.linear_damping_override.unwrap_or(avg_lin).max(0.0),
            self.angular_damping_override.unwrap_or(avg_ang).max(0.0),
        )
    }
}

#[derive(Clone, Debug)]
struct JointRecord {
    spec: JointSpec,
    ref_a: Vec3d,
    ref_b: Vec3d,
    slider_rel_rot: Quatd,
}

type WarmStartKey = (BodyId, Option<BodyId>, FeatureKey);

/// Deterministic pure-Rust voxel rigid-body physics world.
#[derive(Clone, Debug)]
pub struct PhysicsWorld {
    config: PhysicsConfig,
    materials: MaterialRegistry,
    bodies: SlotArena<BodyRecord, BodyId>,
    joints: SlotArena<JointRecord, JointId>,
    bvh: DynamicBvh,
    spatial_hash: SpatialHash16,
    soa: SolverSoA,
    island_builder: IslandBuilder,
    islands: Vec<Island>,
    contact_cache: BTreeMap<WarmStartKey, (f64, [f64; 2])>,
    events: Vec<PhysicsEvent>,
    step_index: u64,
    // Scratch buffers reused across solver steps
    bp_pairs: Vec<(u32, u32)>,
    manifolds: Vec<ContactManifold>,
    prepared: Vec<PreparedManifold>,
}

impl PhysicsWorld {
    pub fn new(config: PhysicsConfig) -> Self {
        let fat_margin = config.broadphase_fat_margin;
        Self {
            config,
            materials: MaterialRegistry::new(),
            bodies: SlotArena::new(),
            joints: SlotArena::new(),
            bvh: DynamicBvh::new(fat_margin),
            spatial_hash: SpatialHash16::new(),
            soa: SolverSoA::new(),
            island_builder: IslandBuilder::new(),
            islands: Vec::new(),
            contact_cache: BTreeMap::new(),
            events: Vec::new(),
            step_index: 0,
            bp_pairs: Vec::new(),
            manifolds: Vec::new(),
            prepared: Vec::new(),
        }
    }

    #[inline]
    pub fn config(&self) -> &PhysicsConfig {
        &self.config
    }

    #[inline]
    pub fn materials(&self) -> &MaterialRegistry {
        &self.materials
    }

    #[inline]
    pub fn register_material(&mut self, id: MaterialId, mat: VoxelMaterial) {
        self.materials.register(id, mat);
    }

    #[inline]
    pub fn step_index(&self) -> u64 {
        self.step_index
    }

    #[inline]
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    #[inline]
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    #[inline]
    pub fn body_grid(&self, id: BodyId) -> Option<&VoxelSubGrid> {
        self.bodies.get(id).map(|b| &b.grid)
    }

    pub fn spawn_body(&mut self, mut desc: BodyDesc) -> BodyId {
        desc.grid.rebuild_dirty_colliders();
        desc.transform.rotation = desc.transform.rotation.normalize();

        let mut rec = BodyRecord {
            kind: desc.kind,
            transform: desc.transform,
            prev_transform: desc.transform,
            linear_velocity: desc.linear_velocity,
            angular_velocity: desc.angular_velocity,
            accum_force: Vec3d::ZERO,
            accum_torque: Vec3d::ZERO,
            grid: desc.grid,
            mass: 0.0,
            inv_mass: 0.0,
            inv_inertia_local: Mat3d::ZERO,
            inv_inertia_world: Mat3d::ZERO,
            gravity_scale: desc.gravity_scale,
            linear_damping_override: desc.linear_damping_override,
            angular_damping_override: desc.angular_damping_override,
            allow_sleep: desc.allow_sleep && desc.kind == BodyKind::Dynamic,
            enable_ccd: desc.enable_ccd,
            sleeping: false,
            sleep_timer: 0.0,
            world_aabb: Aabbd::EMPTY,
            bvh_proxy: usize::MAX,
            min_box_extent: 1.0,
            geometry_version: 1,
            user_data: desc.user_data,
        };
        rec.recompute_mass_properties();
        rec.world_aabb = rec.compute_world_aabb();
        let aabb = rec.world_aabb;

        let id = self.bodies.insert(rec);
        let proxy = self.bvh.insert(id.index, aabb);
        self.spatial_hash.insert(id.index, &aabb);
        if let Some(b) = self.bodies.get_mut(id) {
            b.bvh_proxy = proxy;
        }
        id
    }

    pub fn despawn_body(&mut self, id: BodyId) -> bool {
        let Some(rec) = self.bodies.remove(id) else {
            return false;
        };
        self.bvh.remove(rec.bvh_proxy);
        self.spatial_hash.remove(id.index, &rec.world_aabb);

        // Remove all joints referencing this body.
        let attached_joints: Vec<JointId> = self
            .joints
            .iter()
            .filter_map(|(jid, j)| {
                if j.spec.body_a == id || j.spec.body_b == Some(id) {
                    Some(jid)
                } else {
                    None
                }
            })
            .collect();
        for jid in attached_joints {
            self.joints.remove(jid);
        }

        // Prune warm-start cache entries for this body.
        self.contact_cache
            .retain(|&(ba, bb, _), _| ba != id && bb != Some(id));
        true
    }

    pub fn body_snapshot(&self, id: BodyId) -> Option<BodySnapshot> {
        let b = self.bodies.get(id)?;
        Some(BodySnapshot {
            id,
            kind: b.kind,
            transform: b.transform,
            prev_transform: b.prev_transform,
            linear_velocity: b.linear_velocity,
            angular_velocity: b.angular_velocity,
            mass: b.mass,
            com_local: b.grid.com_local(),
            world_aabb: b.world_aabb,
            sleeping: b.sleeping,
            enable_ccd: b.enable_ccd,
            gravity_scale: b.gravity_scale,
            geometry_version: b.geometry_version,
            voxel_count: b.grid.voxel_count(),
            user_data: b.user_data,
        })
    }

    pub fn snapshots(&self) -> Vec<BodySnapshot> {
        self.bodies
            .iter()
            .map(|(id, b)| BodySnapshot {
                id,
                kind: b.kind,
                transform: b.transform,
                prev_transform: b.prev_transform,
                linear_velocity: b.linear_velocity,
                angular_velocity: b.angular_velocity,
                mass: b.mass,
                com_local: b.grid.com_local(),
                world_aabb: b.world_aabb,
                sleeping: b.sleeping,
                enable_ccd: b.enable_ccd,
                gravity_scale: b.gravity_scale,
                geometry_version: b.geometry_version,
                voxel_count: b.grid.voxel_count(),
                user_data: b.user_data,
            })
            .collect()
    }

    pub fn set_body_transform(&mut self, id: BodyId, mut transform: Transformd) -> bool {
        transform.rotation = transform.rotation.normalize();
        let Some(b) = self.bodies.get_mut(id) else {
            return false;
        };
        let old_aabb = b.world_aabb;
        let disp = transform.position - b.transform.position;
        b.transform = transform;
        if b.kind == BodyKind::Dynamic && b.inv_mass > 0.0 {
            let r = b.transform.rotation.to_mat3();
            b.inv_inertia_world = Mat3d::similarity_transform(r, b.inv_inertia_local).symmetrize();
        }
        let new_aabb = b.compute_world_aabb();
        b.world_aabb = new_aabb;
        let proxy = b.bvh_proxy;
        self.bvh.update(proxy, new_aabb, disp);
        self.spatial_hash.remove(id.index, &old_aabb);
        self.spatial_hash.insert(id.index, &new_aabb);
        true
    }

    pub fn set_body_velocity(&mut self, id: BodyId, linear: Vec3d, angular: Vec3d) -> bool {
        let Some(b) = self.bodies.get_mut(id) else {
            return false;
        };
        b.linear_velocity = linear;
        b.angular_velocity = angular;
        if b.sleeping && (linear.length_sq() > 1e-12 || angular.length_sq() > 1e-12) {
            b.sleeping = false;
            b.sleep_timer = 0.0;
            self.events.push(PhysicsEvent::BodyWoke { body: id });
        }
        true
    }

    pub fn apply_force_at_world_point(
        &mut self,
        id: BodyId,
        force: Vec3d,
        world_point: Vec3d,
    ) -> bool {
        if !force.is_finite() || !world_point.is_finite() {
            return false;
        }
        let Some(b) = self.bodies.get_mut(id) else {
            return false;
        };
        if b.kind != BodyKind::Dynamic || b.inv_mass <= 0.0 {
            return false;
        }
        if b.sleeping && force.length_sq() > 1e-18 {
            b.sleeping = false;
            b.sleep_timer = 0.0;
            self.events.push(PhysicsEvent::BodyWoke { body: id });
        }
        let r = world_point - b.transform.position;
        b.accum_force += force;
        b.accum_torque += r.cross(force);
        true
    }

    pub fn apply_impulse_at_world_point(
        &mut self,
        id: BodyId,
        impulse: Vec3d,
        world_point: Vec3d,
    ) -> bool {
        if !impulse.is_finite() || !world_point.is_finite() {
            return false;
        }
        let Some(b) = self.bodies.get_mut(id) else {
            return false;
        };
        if b.kind != BodyKind::Dynamic || b.inv_mass <= 0.0 {
            return false;
        }
        if b.sleeping && impulse.length_sq() > 1e-18 {
            b.sleeping = false;
            b.sleep_timer = 0.0;
            self.events.push(PhysicsEvent::BodyWoke { body: id });
        }
        let r = world_point - b.transform.position;
        b.linear_velocity += impulse * b.inv_mass;
        b.angular_velocity += b.inv_inertia_world.mul_vec3(r.cross(impulse));
        true
    }

    pub fn apply_torque(&mut self, id: BodyId, torque: Vec3d) -> bool {
        if !torque.is_finite() {
            return false;
        }
        let Some(b) = self.bodies.get_mut(id) else {
            return false;
        };
        if b.kind != BodyKind::Dynamic || b.inv_mass <= 0.0 {
            return false;
        }
        if b.sleeping && torque.length_sq() > 1e-18 {
            b.sleeping = false;
            b.sleep_timer = 0.0;
            self.events.push(PhysicsEvent::BodyWoke { body: id });
        }
        b.accum_torque += torque;
        true
    }

    pub fn wake_body(&mut self, id: BodyId) -> bool {
        let Some(b) = self.bodies.get_mut(id) else {
            return false;
        };
        b.sleep_timer = 0.0;
        if b.sleeping {
            b.sleeping = false;
            self.events.push(PhysicsEvent::BodyWoke { body: id });
        }
        true
    }

    pub fn sleep_body(&mut self, id: BodyId) -> bool {
        let Some(b) = self.bodies.get_mut(id) else {
            return false;
        };
        if b.kind != BodyKind::Dynamic {
            return false;
        }
        b.linear_velocity = Vec3d::ZERO;
        b.angular_velocity = Vec3d::ZERO;
        b.accum_force = Vec3d::ZERO;
        b.accum_torque = Vec3d::ZERO;
        b.sleep_timer = self.config.sleep_time_threshold;
        if !b.sleeping {
            b.sleeping = true;
            self.events.push(PhysicsEvent::BodySlept { body: id });
        }
        true
    }

    /// Wakes all sleeping bodies intersecting `world_aabb` (plus a small support margin).
    pub fn wake_in_aabb(&mut self, world_aabb: Aabbd) {
        let expanded = world_aabb.expand_margin(0.15);
        let mut hits = Vec::new();
        self.bvh.query_aabb(expanded, &mut hits);

        // Map slot indices back to live BodyIds deterministically.
        let ids: Vec<BodyId> = self.bodies.dense_ids().to_vec();
        for id in ids {
            if hits.binary_search(&id.index).is_ok()
                && let Some(b) = self.bodies.get_mut(id)
                && b.world_aabb.intersects(&expanded)
            {
                b.sleep_timer = 0.0;
                if b.sleeping {
                    b.sleeping = false;
                    self.events.push(PhysicsEvent::BodyWoke { body: id });
                }
            }
        }
    }

    /// Modifies a single voxel on a body's sub-grid in O(1), updates its inertia,
    /// shifts world position to compensate CoM drift (zero visual jump), wakes
    /// nearby islands, and automatically splits disconnected components while
    /// strictly conserving linear and angular momentum.
    pub fn set_body_voxel(
        &mut self,
        id: BodyId,
        coord: VoxelCoord,
        entry: Option<VoxelEntry>,
    ) -> bool {
        let (old_aabb, new_aabb, wake_box, detached_pieces, parent_snap) = {
            let Some(b) = self.bodies.get_mut(id) else {
                return false;
            };
            let old_aabb = b.world_aabb;
            let Some(delta_com) = b.grid.set_voxel(coord, entry, &self.materials) else {
                return false;
            };
            b.geometry_version = b.geometry_version.wrapping_add(1).max(1);

            // Compensate CoM shift so every remaining voxel stays at the exact same world coordinates.
            if delta_com.length_sq() > 0.0 {
                let world_shift = b.transform.rotation.rotate_vec3(delta_com);
                b.transform.position += world_shift;
                b.prev_transform.position += world_shift;
                // Preserve exact velocity field of the rigid body at the new CoM:
                // v(new_com) = v(old_com) + omega x world_shift.
                b.linear_velocity += b.angular_velocity.cross(world_shift);
            }

            // Capture pre-split state before checking connectivity.
            let pre_split_com = b.grid.com_local();
            let pre_split_pos = b.transform.position;
            let pre_split_rot = b.transform.rotation;
            let pre_split_vel = b.linear_velocity;
            let pre_split_omega = b.angular_velocity;
            let parent_kind = b.kind;
            let parent_gravity_scale = b.gravity_scale;
            let parent_allow_sleep = b.allow_sleep;
            let parent_enable_ccd = b.enable_ccd;
            let parent_lin_damp = b.linear_damping_override;
            let parent_ang_damp = b.angular_damping_override;
            let parent_user_data = b.user_data;

            let detached = if self.config.enable_grid_splitting && entry.is_none() {
                b.grid.split_disconnected(&self.materials)
            } else {
                Vec::new()
            };

            if !detached.is_empty() {
                // Shift surviving parent body to its new component's CoM with exact velocity!
                let post_split_com = b.grid.com_local();
                let r_main_world = pre_split_rot.rotate_vec3(post_split_com - pre_split_com);
                b.transform.position = pre_split_pos + r_main_world;
                b.prev_transform.position = b.transform.position;
                b.linear_velocity = pre_split_vel + pre_split_omega.cross(r_main_world);
                b.angular_velocity = pre_split_omega;
            }

            b.recompute_mass_properties();
            b.world_aabb = b.compute_world_aabb();
            b.sleep_timer = 0.0;
            if b.sleeping {
                b.sleeping = false;
                self.events.push(PhysicsEvent::BodyWoke { body: id });
            }

            let new_aabb = b.world_aabb;
            let proxy = b.bvh_proxy;
            self.bvh.update(proxy, new_aabb, Vec3d::ZERO);
            self.spatial_hash.remove(id.index, &old_aabb);
            self.spatial_hash.insert(id.index, &new_aabb);

            let wake_box = old_aabb.union(new_aabb);
            let parent_state = (
                pre_split_pos,
                pre_split_rot,
                pre_split_vel,
                pre_split_omega,
                parent_kind,
                parent_gravity_scale,
                parent_allow_sleep,
                parent_enable_ccd,
                parent_lin_damp,
                parent_ang_damp,
                parent_user_data,
            );
            (old_aabb, new_aabb, wake_box, detached, parent_state)
        };
        let _ = (old_aabb, new_aabb);

        // Wake neighboring bodies in the affected envelope.
        self.wake_in_aabb(wake_box);

        // Spawn detached sub-bodies preserving exact linear & angular momentum.
        if !detached_pieces.is_empty() {
            let (
                pre_pos,
                pre_rot,
                pre_vel,
                pre_omega,
                kind,
                gravity_scale,
                allow_sleep,
                enable_ccd,
                lin_damp,
                ang_damp,
                user_data,
            ) = parent_snap;

            let mut children = Vec::with_capacity(detached_pieces.len());
            for (sub_grid, local_com_offset) in detached_pieces {
                let r_piece = pre_rot.rotate_vec3(local_com_offset);
                let child_pos = pre_pos + r_piece;
                let child_vel = pre_vel + pre_omega.cross(r_piece);
                let child_desc = BodyDesc {
                    kind,
                    transform: Transformd::new(child_pos, pre_rot),
                    linear_velocity: child_vel,
                    angular_velocity: pre_omega,
                    grid: sub_grid,
                    gravity_scale,
                    allow_sleep,
                    enable_ccd,
                    linear_damping_override: lin_damp,
                    angular_damping_override: ang_damp,
                    user_data,
                };
                let child_id = self.spawn_body(child_desc);
                children.push(child_id);
            }
            self.events.push(PhysicsEvent::GridSplit {
                parent: id,
                children,
            });
        }

        true
    }

    pub fn add_joint(&mut self, spec: JointSpec) -> Option<JointId> {
        let body_a = self.bodies.get(spec.body_a)?;
        let qa = body_a.transform.rotation;
        let qb = match spec.body_b {
            Some(bid) => {
                if bid == spec.body_a {
                    return None;
                }
                self.bodies.get(bid)?.transform.rotation
            }
            None => Quatd::IDENTITY,
        };

        let (ref_a, ref_b) = match &spec.kind {
            JointKind::Hinge { axis_a, .. } => {
                let ra = axis_a.normalize_or(Vec3d::Y).orthonormal_basis().0;
                let rb = qb.inverse_rotate_vec3(qa.rotate_vec3(ra)).normalize_or(ra);
                (ra, rb)
            }
            _ => (Vec3d::X, Vec3d::X),
        };
        let slider_rel_rot = (qa.conjugate() * qb).normalize();

        let body_a_id = spec.body_a;
        let body_b_id = spec.body_b;
        let jid = self.joints.insert(JointRecord {
            spec,
            ref_a,
            ref_b,
            slider_rel_rot,
        });

        self.wake_body(body_a_id);
        if let Some(bid) = body_b_id {
            self.wake_body(bid);
        }
        Some(jid)
    }

    pub fn remove_joint(&mut self, id: JointId) -> bool {
        let Some(rec) = self.joints.remove(id) else {
            return false;
        };
        self.wake_body(rec.spec.body_a);
        if let Some(bid) = rec.spec.body_b {
            self.wake_body(bid);
        }
        true
    }

    /// Casts a ray against all physics bodies' merged OBB colliders using the BVH.
    pub fn raycast(&self, origin: Vec3d, dir: Vec3d, max_dist: f64) -> Option<RaycastHit> {
        let unit_dir = dir.try_normalize()?;
        if max_dist < 0.0 || !origin.is_finite() {
            return None;
        }

        // Map slot index -> BodyId for deterministic lookup during BVH traversal.
        let mut slot_to_id: BTreeMap<u32, BodyId> = BTreeMap::new();
        for &id in self.bodies.dense_ids() {
            slot_to_id.insert(id.index, id);
        }

        let mut best_hit: Option<RaycastHit> = None;

        self.bvh
            .raycast(origin, unit_dir, max_dist, |uid, cur_max| {
                let Some(&bid) = slot_to_id.get(&uid) else {
                    return cur_max;
                };
                let Some(body) = self.bodies.get(bid) else {
                    return cur_max;
                };
                let mut local_best = cur_max;
                let com = body.grid.com_local();
                for (box_idx, lb) in body.grid.boxes().iter().enumerate() {
                    let obb = body.box_to_world_obb(lb);
                    if let Some((t, normal)) = obb.ray_intersect(origin, unit_dir, local_best) {
                        let better = match best_hit {
                            None => true,
                            Some(prev) => {
                                t < prev.distance - 1e-12
                                    || ((t - prev.distance).abs() <= 1e-12 && bid < prev.body_id)
                            }
                        };
                        if better {
                            local_best = t;
                            let hit_point = origin + unit_dir * t;
                            // Compute struck voxel coordinate in grid-local frame (nudged slightly inward).
                            let inside_world = hit_point - normal * 1e-3;
                            let local_grid_pt =
                                body.transform.inverse_transform_point(inside_world) + com;
                            let vc = VoxelCoord::new(
                                local_grid_pt.x.floor() as i32,
                                local_grid_pt.y.floor() as i32,
                                local_grid_pt.z.floor() as i32,
                            );
                            let voxel_coord = if body.grid.has_voxel(vc) {
                                Some(vc)
                            } else {
                                None
                            };
                            best_hit = Some(RaycastHit {
                                body_id: bid,
                                distance: t,
                                point: hit_point,
                                normal,
                                voxel_coord,
                                box_index: box_idx,
                            });
                        }
                    }
                }
                local_best
            });

        best_hit
    }

    /// Sweeps an entity AABB against all physics bodies with moving/rotating deck
    /// carry, tilted-deck traversal, and automatic stair step-up.
    pub fn sweep_entity_aabb(
        &self,
        entity_aabb: Aabbd,
        delta_world: Vec3d,
        prev_ground_body: Option<(BodyId, Vec3d)>,
    ) -> EntityCarryOutcome {
        self.sweep_entity_against_bodies(
            entity_aabb,
            delta_world,
            prev_ground_body,
            self.config.entity_step_height,
        )
    }

    /// Customizable variant of [`Self::sweep_entity_aabb`] accepting an explicit `max_step_height`.
    pub fn sweep_entity_against_bodies(
        &self,
        entity_aabb: Aabbd,
        delta_world: Vec3d,
        prev_ground_body: Option<(BodyId, Vec3d)>,
        max_step_height: f64,
    ) -> EntitySweepResult {
        let initial_center = entity_aabb.center();

        // 1. Compute exact local-frame carry displacement & yaw delta from previous ground body.
        let (carry_delta, carry_yaw_delta) = if let Some((gid, anchor_local)) = prev_ground_body
            && let Some(gb) = self.bodies.get(gid)
        {
            let old_world = gb.prev_transform.transform_point(anchor_local);
            let new_world = gb.transform.transform_point(anchor_local);
            let dq = (gb.transform.rotation * gb.prev_transform.rotation.conjugate()).normalize();
            let fwd = dq.rotate_vec3(Vec3d::X);
            let yaw = if fwd.x * fwd.x + fwd.z * fwd.z > 1e-12 {
                (-fwd.z).atan2(fwd.x)
            } else {
                0.0
            };
            (new_world - old_world, yaw)
        } else {
            (Vec3d::ZERO, 0.0)
        };

        let mut cur_aabb = entity_aabb.translate(carry_delta);

        // Gather candidate body OBBs overlapping the swept query region.
        let end_guess = cur_aabb.translate(delta_world);
        let query_aabb = cur_aabb
            .union(end_guess)
            .expand_margin(max_step_height.max(0.0) + 0.5);

        let mut candidate_obbs: Vec<(BodyId, Obbd, u8)> = Vec::new();
        for (bid, body) in self.bodies.iter() {
            if !body.world_aabb.intersects(&query_aabb) {
                continue;
            }
            for lb in body.grid.boxes() {
                let obb = body.box_to_world_obb(lb);
                if obb.to_aabb().intersects(&query_aabb) {
                    candidate_obbs.push((bid, obb, lb.exposed_faces));
                }
            }
        }

        // First depenetrate any residual overlap from deck rotation/translation carry.
        let (depen_aabb, _, _) = resolve_entity_overlaps(cur_aabb, &candidate_obbs, true);
        cur_aabb = depen_aabb;

        let mut collided_horizontally = false;
        let mut collided_vertically = false;
        let mut stepped_up_height = 0.0;
        let mut primary_normal: Option<Vec3d> = None;

        // 2. Horizontal motion with slope riding and automatic stair step-up.
        let delta_xz = Vec3d::new(delta_world.x, 0.0, delta_world.z);
        let horiz_len = delta_xz.length();
        if horiz_len > 1e-9 {
            let base_aabb = cur_aabb;
            let (slide_aabb, slide_hit_wall, slide_normal) =
                sweep_entity_segment(base_aabb, delta_xz, &candidate_obbs, false);

            let slide_disp = slide_aabb.center() - base_aabb.center();
            let horiz_dir = delta_xz / horiz_len;
            let slide_progress = slide_disp.dot(horiz_dir);

            let mut chosen_aabb = slide_aabb;
            let mut chosen_wall = slide_hit_wall;
            if let Some(n) = slide_normal {
                primary_normal = Some(n);
            }

            // If horizontally blocked (by a stair riser or step) and step-up is enabled:
            if (slide_hit_wall || slide_progress < horiz_len - 1e-4) && max_step_height > 1e-4 {
                // a) Lift upward up to `max_step_height` (respecting ceilings).
                let (lifted_aabb, _, _) = sweep_entity_segment(
                    base_aabb,
                    Vec3d::new(0.0, max_step_height, 0.0),
                    &candidate_obbs,
                    false,
                );
                let lift_achieved = lifted_aabb.min.y - base_aabb.min.y;
                if lift_achieved > 1e-4 {
                    // b) Move horizontally at lifted elevation.
                    let (high_moved, _, _) =
                        sweep_entity_segment(lifted_aabb, delta_xz, &candidate_obbs, false);
                    // c) Settle downward onto the step top.
                    let (settled, _, drop_normal) = sweep_entity_segment(
                        high_moved,
                        Vec3d::new(0.0, -lift_achieved - 0.05, 0.0),
                        &candidate_obbs,
                        true,
                    );
                    let step_progress = (settled.center() - base_aabb.center()).dot(horiz_dir);
                    let actual_rise = settled.min.y - base_aabb.min.y;
                    let landed_walkable = drop_normal.is_some_and(|n| n.y >= 0.55);

                    if step_progress > slide_progress + 1e-4
                        && actual_rise > 1e-4
                        && actual_rise <= max_step_height + 1e-3
                        && landed_walkable
                    {
                        chosen_aabb = settled;
                        chosen_wall = false;
                        stepped_up_height = actual_rise;
                        primary_normal = drop_normal;
                    }
                }
            }

            cur_aabb = chosen_aabb;
            if chosen_wall {
                collided_horizontally = true;
            }
        }

        // 3. Vertical motion (gravity / jumping).
        if delta_world.y.abs() > 1e-9 {
            let before_y = cur_aabb.min.y;
            let (vert_aabb, _, vert_normal) = sweep_entity_segment(
                cur_aabb,
                Vec3d::new(0.0, delta_world.y, 0.0),
                &candidate_obbs,
                delta_world.y < 0.0,
            );
            let actual_dy = vert_aabb.min.y - before_y;
            if (actual_dy - delta_world.y).abs() > 1e-5 {
                collided_vertically = true;
                if let Some(n) = vert_normal {
                    primary_normal = Some(n);
                }
            }
            cur_aabb = vert_aabb;
        }

        // 4. Ground support detection via downward probe (`0.08` m below feet).
        let probe_dist = 0.08;
        let probe_aabb = cur_aabb.translate(Vec3d::new(0.0, -probe_dist, 0.0));
        let mut best_ground: Option<(BodyId, Vec3d, f64)> = None;
        for &(bid, ref obb, exposed) in &candidate_obbs {
            if let Some(res) = collide_obb_aabb(obb, exposed, 0, &probe_aabb, ALL_FACES, 0) {
                // `res.normal` points from probe_aabb (B) -> obb (A), so outward deck normal pushing entity UP is `-res.normal`.
                let deck_normal = -res.normal;
                if deck_normal.y >= 0.55 {
                    let max_pen = res.points[..res.point_count]
                        .iter()
                        .map(|p| p.penetration)
                        .fold(0.0_f64, f64::max);
                    if best_ground.is_none_or(|(_, _, prev_p)| max_pen > prev_p) {
                        best_ground = Some((bid, deck_normal, max_pen));
                    }
                }
            }
        }

        let (on_ground, ground_body) = if let Some((gid, gnorm, _)) = best_ground {
            if primary_normal.is_none() {
                primary_normal = Some(gnorm);
            }
            let foot_world = Vec3d::new(cur_aabb.center().x, cur_aabb.min.y, cur_aabb.center().z);
            let anchor_local = self
                .bodies
                .get(gid)
                .map(|b| b.transform.inverse_transform_point(foot_world))
                .unwrap_or(Vec3d::ZERO);
            (true, Some((gid, anchor_local)))
        } else {
            (false, None)
        };

        EntitySweepResult {
            resolved_aabb: cur_aabb,
            applied_delta: cur_aabb.center() - initial_center,
            carry_delta,
            carry_yaw_delta,
            on_ground,
            ground_body,
            hit_normal: primary_normal,
            collided_horizontally,
            collided_vertically,
            stepped_up_height,
        }
    }

    /// Advances the entire physics world by `dt` seconds against static `terrain`.
    pub fn step(&mut self, dt: f64, terrain: &mut dyn WorldVoxelProvider) -> StepMetrics {
        let start_time = Instant::now();
        if dt <= 0.0 || !dt.is_finite() {
            return StepMetrics {
                step_index: self.step_index,
                dt,
                total_bodies: self.bodies.len(),
                ..StepMetrics::default()
            };
        }

        self.step_index += 1;
        let substeps = self.config.substeps.max(1);
        let h = dt / f64::from(substeps);
        let soft = SoftContactParams::from_hertz(
            self.config.contact_hertz,
            self.config.contact_damping_ratio,
            h,
        );

        // 1. Snapshot prev_transform and populate SolverSoA in deterministic dense order.
        let body_count = self.bodies.len();
        self.soa.clear();
        let dense_ids: Vec<BodyId> = self.bodies.dense_ids().to_vec();

        // Slot-index -> dense-index table for O(1) deterministic lookups.
        let max_slot = dense_ids
            .iter()
            .map(|id| id.index as usize + 1)
            .max()
            .unwrap_or(0);
        let mut slot_to_dense = vec![usize::MAX; max_slot];
        let mut body_gravity_scales = Vec::with_capacity(body_count);
        let mut body_enable_ccd = Vec::with_capacity(body_count);
        let mut body_min_extents = Vec::with_capacity(body_count);
        let mut body_accum_forces = Vec::with_capacity(body_count);
        let mut body_accum_torques = Vec::with_capacity(body_count);
        let mut body_kinds = Vec::with_capacity(body_count);

        for (dense_i, &id) in dense_ids.iter().enumerate() {
            slot_to_dense[id.index as usize] = dense_i;
            let b = self.bodies.get_mut(id).expect("live body");
            if b.grid.has_dirty_colliders() {
                b.recompute_mass_properties();
                b.world_aabb = b.compute_world_aabb();
                let proxy = b.bvh_proxy;
                let aabb = b.world_aabb;
                self.bvh.update(proxy, aabb, Vec3d::ZERO);
            }
            b.prev_transform = b.transform;

            let (ld, ad) = b.damping();
            let is_dyn = b.kind == BodyKind::Dynamic && b.inv_mass > 0.0;
            self.soa.body_ids.push(id);
            self.soa.positions.push(b.transform.position);
            self.soa.rotations.push(b.transform.rotation);
            self.soa.linear_velocities.push(b.linear_velocity);
            self.soa.angular_velocities.push(b.angular_velocity);
            self.soa.inv_masses.push(b.inv_mass);
            self.soa.inv_inertias_local.push(b.inv_inertia_local);
            self.soa.inv_inertias_world.push(b.inv_inertia_world);
            self.soa.linear_damping.push(ld);
            self.soa.angular_damping.push(ad);
            self.soa.is_dynamic.push(is_dyn);
            self.soa.sleeping.push(b.sleeping);
            if is_dyn && !b.sleeping {
                self.soa.awake_dense_indices.push(dense_i);
            }
            body_gravity_scales.push(b.gravity_scale);
            body_enable_ccd.push(b.enable_ccd);
            body_min_extents.push(b.min_box_extent);
            body_accum_forces.push(b.accum_force);
            body_accum_torques.push(b.accum_torque);
            body_kinds.push(b.kind);
        }

        // Build non-colliding pairs & ActiveJoint list for this step.
        let mut non_colliding: BTreeSet<(BodyId, BodyId)> = BTreeSet::new();
        let mut active_joints: Vec<ActiveJoint> = Vec::with_capacity(self.joints.len());
        let joint_ids: Vec<JointId> = self.joints.dense_ids().to_vec();
        for jid in joint_ids {
            let Some(jrec) = self.joints.get(jid) else {
                continue;
            };
            let Some(idx_a) = self.bodies.dense_index_of(jrec.spec.body_a) else {
                continue;
            };
            let idx_b = match jrec.spec.body_b {
                Some(bid) => match self.bodies.dense_index_of(bid) {
                    Some(ib) => Some(ib),
                    None => continue,
                },
                None => None,
            };
            if !jrec.spec.collide_connected
                && let Some(bid) = jrec.spec.body_b
            {
                let pair = if jrec.spec.body_a < bid {
                    (jrec.spec.body_a, bid)
                } else {
                    (bid, jrec.spec.body_a)
                };
                non_colliding.insert(pair);
            }
            active_joints.push(ActiveJoint {
                id: jid,
                spec: jrec.spec.clone(),
                idx_a,
                idx_b,
                ref_a: jrec.ref_a,
                ref_b: jrec.ref_b,
                slider_rel_rot: jrec.slider_rel_rot,
                xpbd_pos_lambda: Vec3d::ZERO,
                xpbd_ang_lambda: Vec3d::ZERO,
                vel_pos_impulse: Vec3d::ZERO,
                vel_ang_impulse: Vec3d::ZERO,
                scalar_lambda_pos: [0.0; 3],
                scalar_lambda_ang: [0.0; 3],
                motor_impulse: 0.0,
            });
        }

        let mut total_toi_events = 0usize;
        let mut broken_joints_step: Vec<JointId> = Vec::new();
        let mut max_collision_events: BTreeMap<(BodyId, Option<BodyId>), (Vec3d, Vec3d, f64)> =
            BTreeMap::new();
        let mut last_bp_pairs = 0usize;
        let mut last_manifold_count = 0usize;
        let mut last_contact_points = 0usize;
        let mut skip_integration = vec![false; body_count];

        let solver_params = SubstepSolverParams {
            dt: h,
            velocity_iterations: self.config.velocity_iterations,
            contact_slop: self.config.contact_slop,
            max_depenetration_velocity: self.config.max_depenetration_velocity,
            warm_start_factor: self.config.warm_start_factor,
            restitution_threshold: self.config.restitution_threshold,
            soft,
        };

        // 2. Substep loop.
        for _substep in 0..substeps {
            skip_integration.fill(false);
            for j in &mut active_joints {
                j.xpbd_pos_lambda = Vec3d::ZERO;
                j.xpbd_ang_lambda = Vec3d::ZERO;
                j.vel_pos_impulse = Vec3d::ZERO;
                j.vel_ang_impulse = Vec3d::ZERO;
                j.scalar_lambda_pos = [0.0; 3];
                j.scalar_lambda_ang = [0.0; 3];
                j.motor_impulse = 0.0;
            }

            // (a) Forces, gravity, buoyancy, and damping on awake dynamic bodies.
            self.soa.awake_dense_indices.clear();
            for i in 0..body_count {
                if self.soa.is_dynamic[i] && !self.soa.sleeping[i] {
                    self.soa.awake_dense_indices.push(i);
                }
            }

            for idx_pos in 0..self.soa.awake_dense_indices.len() {
                let i = self.soa.awake_dense_indices[idx_pos];
                let im = self.soa.inv_masses[i];
                // Gravity + external force/torque.
                self.soa.linear_velocities[i] += self.config.gravity * (body_gravity_scales[i] * h)
                    + body_accum_forces[i] * (im * h);
                self.soa.angular_velocities[i] +=
                    self.soa.inv_inertias_world[i].mul_vec3(body_accum_torques[i] * h);

                // Archimedes voxel buoyancy.
                if self.config.enable_buoyancy {
                    self.apply_buoyancy_substep(i, h, terrain);
                }

                // Exponentially stable air drag.
                let lin_damp = self.soa.linear_damping[i];
                let ang_damp = self.soa.angular_damping[i];
                if lin_damp > 0.0 {
                    self.soa.linear_velocities[i] /= 1.0 + lin_damp * h;
                }
                if ang_damp > 0.0 {
                    self.soa.angular_velocities[i] /= 1.0 + ang_damp * h;
                }

                // Speed clamps for numerical safety.
                let v_sq = self.soa.linear_velocities[i].length_sq();
                let max_v = self.config.max_linear_speed;
                if v_sq > max_v * max_v {
                    self.soa.linear_velocities[i] *= max_v / v_sq.sqrt();
                }
                let w_sq = self.soa.angular_velocities[i].length_sq();
                let max_w = self.config.max_angular_speed;
                if w_sq > max_w * max_w {
                    self.soa.angular_velocities[i] *= max_w / w_sq.sqrt();
                }
            }

            // (b) Update BVH for active/kinematic bodies & query broadphase pairs.
            let mut active_proxies: Vec<usize> =
                Vec::with_capacity(self.soa.awake_dense_indices.len());
            for (i, &kind) in body_kinds.iter().enumerate().take(body_count) {
                if (!self.soa.is_dynamic[i] && kind != BodyKind::Kinematic) || self.soa.sleeping[i]
                {
                    continue;
                }
                let id = self.soa.body_ids[i];
                let pos = self.soa.positions[i];
                let rot = self.soa.rotations[i];
                let disp = self.soa.linear_velocities[i] * h;
                let b = self.bodies.get(id).expect("live body");
                let local = b.grid.local_aabb();
                let aabb = if local.is_empty() {
                    Aabbd::from_center_half_extents(pos, Vec3d::ZERO)
                } else {
                    let lc = local.center() - b.grid.com_local();
                    Obbd::new(pos + rot.rotate_vec3(lc), local.half_extents(), rot).to_aabb()
                };
                self.bvh.update(b.bvh_proxy, aabb, disp);
                active_proxies.push(b.bvh_proxy);
            }
            self.bvh
                .query_pairs_for_leaves(&active_proxies, &mut self.bp_pairs);
            last_bp_pairs = self.bp_pairs.len();

            // (c) Continuous Collision Detection (TOI) for fast dynamic bodies.
            for idx_pos in 0..self.soa.awake_dense_indices.len() {
                let i = self.soa.awake_dense_indices[idx_pos];
                let speed = self.soa.linear_velocities[i].length();
                let fast_enough = (body_enable_ccd[i] && speed * h > 0.25 * body_min_extents[i])
                    || speed * h > 0.45 * body_min_extents[i]
                    || (body_enable_ccd[i] && speed >= self.config.ccd_speed_threshold);
                if !fast_enough {
                    continue;
                }
                if let Some(toi) =
                    self.find_earliest_toi(i, h, &slot_to_dense, &non_colliding, terrain)
                {
                    total_toi_events += 1;
                    // Advance body i to exact TOI pose and resolve impact impulse immediately.
                    self.soa.positions[i] += self.soa.linear_velocities[i] * toi.hit.time;
                    self.soa.rotations[i] = self.soa.rotations[i]
                        .integrate_angular_velocity(self.soa.angular_velocities[i], toi.hit.time);
                    self.soa.update_world_inertia(i);
                    skip_integration[i] = true;

                    // Wake other body if sleeping dynamic.
                    if let Some(ob) = toi.other_dense
                        && self.soa.sleeping[ob]
                        && self.soa.is_dynamic[ob]
                    {
                        self.soa.sleeping[ob] = false;
                        self.events.push(PhysicsEvent::BodyWoke {
                            body: self.soa.body_ids[ob],
                        });
                    }

                    let jn = self.resolve_toi_impulse(i, &toi, terrain);
                    if jn > 1e-4 {
                        let key = (
                            self.soa.body_ids[i],
                            toi.other_dense.map(|ob| self.soa.body_ids[ob]),
                        );
                        let entry = max_collision_events.entry(key).or_insert((
                            toi.hit.point_a,
                            toi.hit.normal,
                            0.0,
                        ));
                        if jn > entry.2 {
                            *entry = (toi.hit.point_a, toi.hit.normal, jn);
                        }
                    }
                }
            }

            // (d) Discrete narrowphase (body-body + body-terrain) with warm-start lookup.
            self.manifolds.clear();
            self.generate_body_body_contacts(&slot_to_dense, &non_colliding);
            self.generate_terrain_contacts(&skip_integration, terrain);

            // Wake any sleeping dynamic body contacted by an awake body.
            for m in &self.manifolds {
                let Some(ia) = self.bodies.dense_index_of(m.body_a) else {
                    continue;
                };
                if let Some(bid_b) = m.body_b
                    && let Some(ib) = self.bodies.dense_index_of(bid_b)
                {
                    if !self.soa.sleeping[ia] && self.soa.sleeping[ib] && self.soa.is_dynamic[ib] {
                        self.soa.sleeping[ib] = false;
                        self.events.push(PhysicsEvent::BodyWoke { body: bid_b });
                    } else if !self.soa.sleeping[ib]
                        && self.soa.sleeping[ia]
                        && self.soa.is_dynamic[ia]
                    {
                        self.soa.sleeping[ia] = false;
                        self.events.push(PhysicsEvent::BodyWoke { body: m.body_a });
                    }
                }
            }

            // Attach warm-started impulses from cache.
            for m in &mut self.manifolds {
                for i in 0..m.point_count {
                    let key = (m.body_a, m.body_b, m.points[i].feature);
                    if let Some(&(jn, jt)) = self.contact_cache.get(&key) {
                        m.points[i].normal_impulse = jn;
                        m.points[i].tangent_impulse = jt;
                    }
                }
            }

            last_manifold_count = self.manifolds.len();
            last_contact_points = self.manifolds.iter().map(|m| m.point_count).sum();

            // (e) TGS Velocity Solve (contacts + joint motors/damping).
            let bodies_ref = &self.bodies;
            prepare_contacts(
                &mut self.soa,
                &self.manifolds,
                &|bid| bodies_ref.dense_index_of(bid),
                &solver_params,
                &mut self.prepared,
            );
            solve_velocities(
                &mut self.soa,
                &mut self.prepared,
                &mut active_joints,
                &solver_params,
                true,
            );

            // (f) Position & exponential-map rotation integration.
            for i in 0..body_count {
                if skip_integration[i] {
                    continue;
                }
                if (self.soa.is_dynamic[i] && !self.soa.sleeping[i])
                    || body_kinds[i] == BodyKind::Kinematic
                {
                    self.soa.positions[i] += self.soa.linear_velocities[i] * h;
                    self.soa.rotations[i] = self.soa.rotations[i]
                        .integrate_angular_velocity(self.soa.angular_velocities[i], h);
                    if self.soa.is_dynamic[i] {
                        self.soa.update_world_inertia(i);
                    }
                }
            }

            // (g) XPBD Positional Joint Projection + Breakage detection.
            let broken = solve_joints_xpbd_positions(&mut self.soa, &mut active_joints, h, 4);
            if !broken.is_empty() {
                for bj in broken {
                    broken_joints_step.push(bj.id);
                    self.events.push(PhysicsEvent::JointBroken {
                        joint: bj.id,
                        body_a: bj.body_a,
                        body_b: bj.body_b,
                        force: bj.force,
                        torque: bj.torque,
                    });
                }
                active_joints.retain(|j| !broken_joints_step.contains(&j.id));
            }

            // (h) Relaxation pass (zero bias, skips bouncing contacts) for rock-solid stacking.
            solve_velocities(
                &mut self.soa,
                &mut self.prepared,
                &mut [],
                &solver_params,
                false,
            );

            // (i) Write impulses back & refresh warm-start cache.
            write_back_contacts(&self.prepared, &mut self.manifolds);
            self.contact_cache.clear();
            for m in &self.manifolds {
                let mut best_imp = 0.0;
                let mut best_pt = Vec3d::ZERO;
                for p in m.active_points() {
                    let key = (m.body_a, m.body_b, p.feature);
                    self.contact_cache
                        .insert(key, (p.normal_impulse, p.tangent_impulse));
                    if p.normal_impulse > best_imp {
                        best_imp = p.normal_impulse;
                        best_pt = p.world_point_a;
                    }
                }
                if best_imp > 1e-3 {
                    let entry = max_collision_events
                        .entry((m.body_a, m.body_b))
                        .or_insert((best_pt, m.normal, 0.0));
                    if best_imp > entry.2 {
                        *entry = (best_pt, m.normal, best_imp);
                    }
                }
            }
        }

        // Remove broken joints permanently from world arena.
        for jid in broken_joints_step {
            self.joints.remove(jid);
        }

        // 3. Sync SolverSoA back to BodyRecord + update spatial structures + clear forces.
        for i in 0..body_count {
            let id = self.soa.body_ids[i];
            let b = self.bodies.get_mut(id).expect("live body");
            let old_aabb = b.world_aabb;
            let disp = self.soa.positions[i] - b.transform.position;
            b.transform.position = self.soa.positions[i];
            b.transform.rotation = self.soa.rotations[i];
            b.linear_velocity = self.soa.linear_velocities[i];
            b.angular_velocity = self.soa.angular_velocities[i];
            b.inv_inertia_world = self.soa.inv_inertias_world[i];
            b.sleeping = self.soa.sleeping[i];
            b.accum_force = Vec3d::ZERO;
            b.accum_torque = Vec3d::ZERO;

            if !b.sleeping || disp.length_sq() > 0.0 {
                let new_aabb = b.compute_world_aabb();
                b.world_aabb = new_aabb;
                let proxy = b.bvh_proxy;
                self.bvh.update(proxy, new_aabb, disp);
                if SpatialHash16::chunk_keys_for_aabb(&old_aabb)
                    != SpatialHash16::chunk_keys_for_aabb(&new_aabb)
                {
                    self.spatial_hash.remove(id.index, &old_aabb);
                    self.spatial_hash.insert(id.index, &new_aabb);
                }
            }
        }

        // 4. Island partitioning & collective sleeping.
        let mut edges: Vec<(usize, usize)> =
            Vec::with_capacity(self.manifolds.len() + active_joints.len());
        for m in &self.manifolds {
            if let Some(bid_b) = m.body_b
                && let (Some(ia), Some(ib)) = (
                    self.bodies.dense_index_of(m.body_a),
                    self.bodies.dense_index_of(bid_b),
                )
            {
                edges.push((ia, ib));
            }
        }
        for j in &active_joints {
            if let Some(ib) = j.idx_b {
                edges.push((j.idx_a, ib));
            }
        }
        self.island_builder
            .build(body_count, &self.soa.is_dynamic, &edges, &mut self.islands);

        if self.config.enable_sleeping {
            let lin_sq_thresh =
                self.config.linear_sleep_threshold * self.config.linear_sleep_threshold;
            let ang_sq_thresh =
                self.config.angular_sleep_threshold * self.config.angular_sleep_threshold;
            let time_thresh = self.config.sleep_time_threshold;

            for island in &self.islands {
                let mut island_can_sleep = true;
                let mut min_sleep_timer = f64::INFINITY;
                let mut any_awake_moving = false;

                for &bi in &island.body_indices {
                    let id = self.soa.body_ids[bi];
                    let b = self.bodies.get_mut(id).expect("live body");
                    if !b.allow_sleep {
                        island_can_sleep = false;
                        b.sleep_timer = 0.0;
                        continue;
                    }
                    let slow = b.linear_velocity.length_sq() < lin_sq_thresh
                        && b.angular_velocity.length_sq() < ang_sq_thresh;
                    if slow {
                        if !b.sleeping {
                            b.sleep_timer += dt;
                        }
                        min_sleep_timer = min_sleep_timer.min(b.sleep_timer);
                    } else {
                        island_can_sleep = false;
                        b.sleep_timer = 0.0;
                        any_awake_moving = true;
                    }
                }

                if any_awake_moving {
                    for &bi in &island.body_indices {
                        let id = self.soa.body_ids[bi];
                        let b = self.bodies.get_mut(id).expect("live body");
                        b.sleep_timer = 0.0;
                        if b.sleeping {
                            b.sleeping = false;
                            self.events.push(PhysicsEvent::BodyWoke { body: id });
                        }
                    }
                } else if island_can_sleep && min_sleep_timer >= time_thresh {
                    for &bi in &island.body_indices {
                        let id = self.soa.body_ids[bi];
                        let b = self.bodies.get_mut(id).expect("live body");
                        b.linear_velocity = Vec3d::ZERO;
                        b.angular_velocity = Vec3d::ZERO;
                        if !b.sleeping {
                            b.sleeping = true;
                            self.events.push(PhysicsEvent::BodySlept { body: id });
                        }
                    }
                }
            }
        }

        // 5. Emit deduplicated collision events.
        for ((body_a, body_b), (point, normal, normal_impulse)) in max_collision_events {
            self.events.push(PhysicsEvent::Collision {
                body_a,
                body_b,
                point,
                normal,
                normal_impulse,
            });
        }

        let mut awake_bodies = 0usize;
        let mut sleeping_bodies = 0usize;
        for (_, b) in self.bodies.iter() {
            if b.kind == BodyKind::Dynamic {
                if b.sleeping {
                    sleeping_bodies += 1;
                } else {
                    awake_bodies += 1;
                }
            }
        }

        StepMetrics {
            step_index: self.step_index,
            dt,
            total_bodies: body_count,
            awake_bodies,
            sleeping_bodies,
            island_count: self.islands.len(),
            broadphase_pairs: last_bp_pairs,
            contact_manifolds: last_manifold_count,
            contact_points: last_contact_points,
            active_joints: self.joints.len(),
            toi_events: total_toi_events,
            step_time_us: start_time.elapsed().as_micros() as u64,
        }
    }

    pub fn drain_events(&mut self) -> Vec<PhysicsEvent> {
        std::mem::take(&mut self.events)
    }

    /// Generates color-coded debug wireframe segments `(start, end, rgba_hex)`
    /// for all body merged OBBs, active joints, and current contact normals.
    pub fn debug_wireframes(&self) -> Vec<(Vec3d, Vec3d, u32)> {
        let mut lines = Vec::new();
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];

        for (_, b) in self.bodies.iter() {
            let color = match (b.kind, b.sleeping) {
                (BodyKind::Static, _) => 0x8080_80FF,
                (BodyKind::Kinematic, _) => 0x00CC_FFFF,
                (BodyKind::Dynamic, true) => 0x3366_CCFF,
                (BodyKind::Dynamic, false) => 0x00FF_66FF,
            };
            for lb in b.grid.boxes() {
                let obb = b.box_to_world_obb(lb);
                let axes = obb.axes();
                let hx = axes[0] * obb.half_extents.x;
                let hy = axes[1] * obb.half_extents.y;
                let hz = axes[2] * obb.half_extents.z;
                let corners = [
                    obb.center - hx - hy - hz,
                    obb.center + hx - hy - hz,
                    obb.center - hx + hy - hz,
                    obb.center + hx + hy - hz,
                    obb.center - hx - hy + hz,
                    obb.center + hx - hy + hz,
                    obb.center - hx + hy + hz,
                    obb.center + hx + hy + hz,
                ];
                for &(a, c) in &EDGES {
                    lines.push((corners[a], corners[c], color));
                }
            }
        }

        // Active joint anchor lines (yellow).
        for (_, j) in self.joints.iter() {
            let Some(ba) = self.bodies.get(j.spec.body_a) else {
                continue;
            };
            let pa = ba.transform.transform_point(j.spec.local_anchor_a);
            let pb = match j.spec.body_b {
                Some(bid) => match self.bodies.get(bid) {
                    Some(bb) => bb.transform.transform_point(j.spec.local_anchor_b),
                    None => continue,
                },
                None => j.spec.local_anchor_b,
            };
            lines.push((pa, pb, 0xFFFF_00FF));
        }

        // Contact normals (red).
        for m in &self.manifolds {
            for p in m.active_points() {
                lines.push((
                    p.world_point_a,
                    p.world_point_a + m.normal * 0.4,
                    0xFF33_33FF,
                ));
            }
        }

        lines
    }

    fn apply_buoyancy_substep(
        &mut self,
        dense_i: usize,
        h: f64,
        terrain: &mut dyn WorldVoxelProvider,
    ) {
        let id = self.soa.body_ids[dense_i];
        let pos = self.soa.positions[dense_i];
        let rot = self.soa.rotations[dense_i];
        let vel = self.soa.linear_velocities[dense_i];
        let omega = self.soa.angular_velocities[dense_i];
        let inv_m = self.soa.inv_masses[dense_i];
        let inv_i = self.soa.inv_inertias_world[dense_i];
        if inv_m <= 0.0 {
            return;
        }

        let body = self.bodies.get(id).expect("live body");
        let com = body.grid.com_local();
        let rot_mat = rot.to_mat3();
        let gravity = self.config.gravity;

        let mut total_dv = Vec3d::ZERO;
        let mut total_dw = Vec3d::ZERO;
        let mut total_submerged_vol = 0.0;
        let mut weighted_ang_drag = 0.0;
        let total_box_vol = body.grid.voxel_count().max(1) as f64;

        for lb in body.grid.boxes() {
            let sx = (lb.max[0] - lb.min[0]).max(1);
            let sy = (lb.max[1] - lb.min[1]).max(1);
            let sz = (lb.max[2] - lb.min[2]).max(1);
            let nx = sx.min(4);
            let ny = sy.min(4);
            let nz = sz.min(4);

            let dx = f64::from(sx) / f64::from(nx);
            let dy = f64::from(sy) / f64::from(ny);
            let dz = f64::from(sz) / f64::from(nz);
            let cell_vol = dx * dy * dz;
            // World vertical thickness of an oriented sub-cell.
            let cell_h_y = (rot_mat.cols[0].y.abs() * dx
                + rot_mat.cols[1].y.abs() * dy
                + rot_mat.cols[2].y.abs() * dz)
                .max(1e-4);

            for iz in 0..nz {
                let lz = lb.min[2] as f64 + (iz as f64 + 0.5) * dz;
                for iy in 0..ny {
                    let ly = lb.min[1] as f64 + (iy as f64 + 0.5) * dy;
                    for ix in 0..nx {
                        let lx = lb.min[0] as f64 + (ix as f64 + 0.5) * dx;
                        let local_pt = Vec3d::new(lx, ly, lz) - com;
                        let r_world = rot.rotate_vec3(local_pt);
                        let p_world = pos + r_world;

                        let wx = p_world.x.floor() as i32;
                        let wy = p_world.y.floor() as i32;
                        let wz = p_world.z.floor() as i32;

                        let cell_bottom = p_world.y - 0.5 * cell_h_y;
                        let fluid = terrain.fluid_at(wx, wy, wz).or_else(|| {
                            let bottom_wy = cell_bottom.floor() as i32;
                            if bottom_wy < wy {
                                terrain.fluid_at(wx, bottom_wy, wz)
                            } else {
                                None
                            }
                        });

                        let Some(f) = fluid else {
                            continue;
                        };
                        if f.density <= 0.0 || f.surface_y <= cell_bottom {
                            continue;
                        }

                        let sub_frac = ((f.surface_y - cell_bottom) / cell_h_y).clamp(0.0, 1.0);
                        if sub_frac <= 0.0 {
                            continue;
                        }

                        let sub_vol = cell_vol * sub_frac;
                        total_submerged_vol += sub_vol;
                        weighted_ang_drag += f.angular_drag * sub_vol;

                        let displaced_mass = f.density * sub_vol;
                        // Submerged centroid shifts slightly below cell center when partially submerged.
                        let centroid_shift_y = -0.5 * cell_h_y * (1.0 - sub_frac);
                        let r_buoy = r_world + Vec3d::new(0.0, centroid_shift_y, 0.0);

                        // Archimedes upward lift: -g * m_displaced * h.
                        let buoy_imp = -gravity * (displaced_mass * h);
                        total_dv += buoy_imp * inv_m;
                        total_dw += inv_i.mul_vec3(r_buoy.cross(buoy_imp));

                        // Hydrodynamic drag opposing relative fluid velocity (clamped for stability).
                        let v_pt = vel + omega.cross(r_buoy) - f.flow_velocity;
                        let drag_coeff = (f.linear_drag * displaced_mass * inv_m * h).min(0.45);
                        let drag_imp = -v_pt * (drag_coeff / inv_m);
                        total_dv += drag_imp * inv_m;
                        total_dw += inv_i.mul_vec3(r_buoy.cross(drag_imp));
                    }
                }
            }
        }

        if total_submerged_vol > 0.0 {
            self.soa.linear_velocities[dense_i] += total_dv;
            self.soa.angular_velocities[dense_i] += total_dw;
            let avg_ang_drag = weighted_ang_drag / total_submerged_vol;
            let sub_ratio = (total_submerged_vol / total_box_vol).clamp(0.0, 1.0);
            self.soa.angular_velocities[dense_i] /= 1.0 + avg_ang_drag * sub_ratio * h;
        }
    }

    fn find_earliest_toi(
        &self,
        dense_a: usize,
        h: f64,
        slot_to_dense: &[usize],
        non_colliding: &BTreeSet<(BodyId, BodyId)>,
        terrain: &mut dyn WorldVoxelProvider,
    ) -> Option<ToiCandidate> {
        let id_a = self.soa.body_ids[dense_a];
        let body_a = self.bodies.get(id_a)?;
        let pos_a = self.soa.positions[dense_a];
        let rot_a = self.soa.rotations[dense_a];
        let vel_a = self.soa.linear_velocities[dense_a];
        let omega_a = self.soa.angular_velocities[dense_a];
        let com_a = body_a.grid.com_local();

        let mut best: Option<ToiCandidate> = None;
        let mut best_t = h;

        // 1. Check swept OBBs against terrain voxels.
        for (bi_a, lb) in body_a.grid.boxes().iter().enumerate() {
            let local_off = lb.center_local() - com_a;
            let he = lb.half_extents();
            let swept_a = SweptObb {
                body_pos: pos_a,
                body_rot: rot_a,
                linear_vel: vel_a,
                angular_vel: omega_a,
                local_offset: local_off,
                half_extents: he,
            };

            let aabb0 = swept_a.obb_at(0.0).to_aabb();
            let aabb1 = swept_a.obb_at(best_t).to_aabb();
            let swept_aabb = aabb0.union(aabb1).expand_margin(0.05);

            let min_x = swept_aabb.min.x.floor() as i32;
            let min_y = swept_aabb.min.y.floor() as i32;
            let min_z = swept_aabb.min.z.floor() as i32;
            let max_x = swept_aabb.max.x.floor() as i32;
            let max_y = swept_aabb.max.y.floor() as i32;
            let max_z = swept_aabb.max.z.floor() as i32;

            // Safety cap on voxel sweep span to avoid pathological billion-voxel loops.
            let span = (i64::from(max_x) - i64::from(min_x) + 1)
                * (i64::from(max_y) - i64::from(min_y) + 1)
                * (i64::from(max_z) - i64::from(min_z) + 1);
            if span > 32_768 {
                continue;
            }

            let mat_a = self.materials.get(lb.material);
            for vx in min_x..=max_x {
                for vy in min_y..=max_y {
                    for vz in min_z..=max_z {
                        let Some(block) = terrain.voxel_collision(vx, vy, vz) else {
                            continue;
                        };
                        let origin = Vec3d::new(vx as f64, vy as f64, vz as f64);
                        let b_world = Aabbd::new(
                            origin + block.local_aabb.min,
                            origin + block.local_aabb.max,
                        );
                        let swept_b = SweptObb {
                            body_pos: b_world.center(),
                            body_rot: Quatd::IDENTITY,
                            linear_vel: Vec3d::ZERO,
                            angular_vel: Vec3d::ZERO,
                            local_offset: Vec3d::ZERO,
                            half_extents: b_world.half_extents(),
                        };
                        if let Some(mut hit) =
                            conservative_advancement_obb_pair(&swept_a, &swept_b, best_t, 1e-3)
                        {
                            // Filter out buried internal terrain faces so seam corners never
                            // deflect high-speed impacts at phantom 45-degree angles.
                            let exposed_b = terrain_exposed_mask(terrain, vx, vy, vz);
                            hit.normal = filter_terrain_normal(hit.normal, exposed_b);
                            let closing = -vel_a.dot(hit.normal);
                            if closing <= 1e-6 {
                                continue;
                            }
                            let better = hit.time < best_t - 1e-5
                                || ((hit.time - best_t).abs() <= 1e-5
                                    && best
                                        .as_ref()
                                        .is_none_or(|prev| closing > -vel_a.dot(prev.hit.normal)));
                            if better {
                                best_t = hit.time;
                                let mat_b = self.materials.get(block.material);
                                let rest = mat_a.restitution.max(mat_b.restitution);
                                let fric = (mat_a.friction * mat_b.friction).sqrt();
                                best = Some(ToiCandidate {
                                    hit,
                                    box_idx_a: bi_a,
                                    other_dense: None,
                                    target_shape: ToiTargetShape::Terrain,
                                    restitution: rest,
                                    friction: fric,
                                });
                            }
                        }
                    }
                }
            }
        }

        // 2. Check swept OBBs against overlapping bodies in the BVH.
        let body_aabb0 = body_a.compute_world_aabb();
        let body_swept_aabb = body_aabb0
            .union(body_aabb0.translate(vel_a * best_t))
            .expand_margin(0.25);
        let mut nearby = Vec::new();
        self.bvh.query_aabb(body_swept_aabb, &mut nearby);

        for uid in nearby {
            let slot = uid as usize;
            if slot >= slot_to_dense.len() {
                continue;
            }
            let dense_b = slot_to_dense[slot];
            if dense_b == usize::MAX || dense_b == dense_a {
                continue;
            }
            let id_b = self.soa.body_ids[dense_b];
            let pair_key = if id_a < id_b {
                (id_a, id_b)
            } else {
                (id_b, id_a)
            };
            if non_colliding.contains(&pair_key) {
                continue;
            }
            let Some(body_b) = self.bodies.get(id_b) else {
                continue;
            };
            let com_b = body_b.grid.com_local();
            let pos_b = self.soa.positions[dense_b];
            let rot_b = self.soa.rotations[dense_b];
            let vel_b = self.soa.linear_velocities[dense_b];
            let omega_b = self.soa.angular_velocities[dense_b];

            for (bi_a, lba) in body_a.grid.boxes().iter().enumerate() {
                let mat_a = self.materials.get(lba.material);
                let swept_a = SweptObb {
                    body_pos: pos_a,
                    body_rot: rot_a,
                    linear_vel: vel_a,
                    angular_vel: omega_a,
                    local_offset: lba.center_local() - com_a,
                    half_extents: lba.half_extents(),
                };
                for (bi_b, lbb) in body_b.grid.boxes().iter().enumerate() {
                    let swept_b = SweptObb {
                        body_pos: pos_b,
                        body_rot: rot_b,
                        linear_vel: vel_b,
                        angular_vel: omega_b,
                        local_offset: lbb.center_local() - com_b,
                        half_extents: lbb.half_extents(),
                    };
                    if let Some(hit) =
                        conservative_advancement_obb_pair(&swept_a, &swept_b, best_t, 1e-3)
                        && hit.time < best_t
                    {
                        best_t = hit.time;
                        let mat_b = self.materials.get(lbb.material);
                        let rest = mat_a.restitution.max(mat_b.restitution);
                        let fric = (mat_a.friction * mat_b.friction).sqrt();
                        best = Some(ToiCandidate {
                            hit,
                            box_idx_a: bi_a,
                            other_dense: Some(dense_b),
                            target_shape: ToiTargetShape::BodyBox { box_idx_b: bi_b },
                            restitution: rest,
                            friction: fric,
                        });
                    }
                }
            }
        }

        best
    }

    fn resolve_toi_impulse(
        &mut self,
        idx_a: usize,
        toi: &ToiCandidate,
        terrain: &mut dyn WorldVoxelProvider,
    ) -> f64 {
        let idx_b = toi.other_dense;
        let restitution = toi.restitution;
        let friction = toi.friction;

        // Probe the exact contact patch (1..4 points) at the TOI pose via a tiny 5mm virtual
        // overlap across all touching terrain voxels (or target body box) so flat-face high-speed
        // impacts distribute impulses symmetrically across the full contact face.
        let mut patch_pts: Vec<(Vec3d, Vec3d)> = Vec::with_capacity(4);
        let mut n = toi.hit.normal;

        let id_a = self.soa.body_ids[idx_a];
        if let Some(body_a) = self.bodies.get(id_a)
            && let Some(lba) = body_a.grid.boxes().get(toi.box_idx_a)
        {
            let probe_pos_a = self.soa.positions[idx_a] - n * 0.005;
            let obb_a = body_a.box_to_world_obb_at(lba, probe_pos_a, self.soa.rotations[idx_a]);
            match toi.target_shape {
                ToiTargetShape::Terrain => {
                    let probe_aabb = obb_a.to_aabb();
                    let min_x = probe_aabb.min.x.floor() as i32;
                    let min_y = probe_aabb.min.y.floor() as i32;
                    let min_z = probe_aabb.min.z.floor() as i32;
                    let max_x = probe_aabb.max.x.floor() as i32;
                    let max_y = probe_aabb.max.y.floor() as i32;
                    let max_z = probe_aabb.max.z.floor() as i32;

                    let mut raw_pts = Vec::new();
                    for vx in min_x..=max_x {
                        for vy in min_y..=max_y {
                            for vz in min_z..=max_z {
                                let Some(block) = terrain.voxel_collision(vx, vy, vz) else {
                                    continue;
                                };
                                let origin = Vec3d::new(vx as f64, vy as f64, vz as f64);
                                let block_aabb = Aabbd::new(
                                    origin + block.local_aabb.min,
                                    origin + block.local_aabb.max,
                                );
                                if !probe_aabb.intersects(&block_aabb) {
                                    continue;
                                }
                                let exposed_b = terrain_exposed_mask(terrain, vx, vy, vz);
                                let shape_b = pack_terrain_coord_hash(vx, vy, vz);
                                if let Some(res) = collide_obb_aabb(
                                    &obb_a,
                                    lba.exposed_faces,
                                    toi.box_idx_a as u32,
                                    &block_aabb,
                                    exposed_b,
                                    shape_b,
                                ) && res.normal.dot(toi.hit.normal) > 0.7
                                {
                                    n = res.normal;
                                    raw_pts.extend_from_slice(&res.points[..res.point_count]);
                                }
                            }
                        }
                    }
                    if !raw_pts.is_empty() {
                        let (reduced, cnt) = reduce_contacts_to_4(&raw_pts, n);
                        for p in reduced.iter().take(cnt) {
                            patch_pts.push((p.world_point_a + n * 0.005, p.world_point_b));
                        }
                    }
                }
                ToiTargetShape::BodyBox { box_idx_b } => {
                    if let Some((ib, bb)) = idx_b
                        .and_then(|ib| self.bodies.get(self.soa.body_ids[ib]).map(|bb| (ib, bb)))
                        && let Some(lbb) = bb.grid.boxes().get(box_idx_b)
                    {
                        let obb_b = bb.box_to_world_obb_at(
                            lbb,
                            self.soa.positions[ib],
                            self.soa.rotations[ib],
                        );
                        if let Some(res) = collide_obb_obb(
                            &obb_a,
                            lba.exposed_faces,
                            0,
                            &obb_b,
                            lbb.exposed_faces,
                            0,
                        ) {
                            if res.normal.dot(toi.hit.normal) > 0.5 {
                                n = res.normal;
                            }
                            for p in res.points.iter().take(res.point_count) {
                                patch_pts.push((p.world_point_a + n * 0.005, p.world_point_b));
                            }
                        }
                    }
                }
            }
        }
        if patch_pts.is_empty() {
            patch_pts.push((toi.hit.point_a, toi.hit.point_b));
        }

        let ima = self.soa.inv_masses[idx_a];
        let imb = idx_b.map_or(0.0, |ib| self.soa.inv_masses[ib]);
        let inv_i_a = self.soa.inv_inertias_world[idx_a];
        let inv_i_b = idx_b.map_or(Mat3d::ZERO, |ib| self.soa.inv_inertias_world[ib]);
        let pos_a = self.soa.positions[idx_a];
        let pos_b = idx_b.map_or(Vec3d::ZERO, |ib| self.soa.positions[ib]);

        // Precompute initial bounce target per contact point in the TOI patch.
        let mut target_vns = Vec::with_capacity(patch_pts.len());
        let mut acc_jn = vec![0.0_f64; patch_pts.len()];
        for &(pa, pb) in &patch_pts {
            let r_a = pa - pos_a;
            let r_b = if idx_b.is_some() {
                pb - pos_b
            } else {
                Vec3d::ZERO
            };
            let va =
                self.soa.linear_velocities[idx_a] + self.soa.angular_velocities[idx_a].cross(r_a);
            let vb = idx_b.map_or(Vec3d::ZERO, |ib| {
                self.soa.linear_velocities[ib] + self.soa.angular_velocities[ib].cross(r_b)
            });
            let vn0 = (va - vb).dot(n);
            target_vns.push(if vn0 < 0.0 { -restitution * vn0 } else { 0.0 });
        }

        // Solve multi-point TOI patch with 8 Gauss-Seidel iterations.
        for _ in 0..8 {
            for (pi, &(pa, pb)) in patch_pts.iter().enumerate() {
                let r_a = pa - pos_a;
                let r_b = if idx_b.is_some() {
                    pb - pos_b
                } else {
                    Vec3d::ZERO
                };
                let va = self.soa.linear_velocities[idx_a]
                    + self.soa.angular_velocities[idx_a].cross(r_a);
                let vb = idx_b.map_or(Vec3d::ZERO, |ib| {
                    self.soa.linear_velocities[ib] + self.soa.angular_velocities[ib].cross(r_b)
                });
                let vn = (va - vb).dot(n);
                let ra_x_n = r_a.cross(n);
                let rb_x_n = r_b.cross(n);
                let k_n = ima
                    + imb
                    + ra_x_n.dot(inv_i_a.mul_vec3(ra_x_n))
                    + rb_x_n.dot(inv_i_b.mul_vec3(rb_x_n));
                if k_n <= 1e-18 {
                    continue;
                }
                let dj = (target_vns[pi] - vn) / k_n;
                let old_j = acc_jn[pi];
                let new_j = (old_j + dj).max(0.0);
                let applied = new_j - old_j;
                acc_jn[pi] = new_j;
                if applied.abs() > 0.0 {
                    let imp_n = n * applied;
                    if ima > 0.0 {
                        self.soa.linear_velocities[idx_a] += imp_n * ima;
                        self.soa.angular_velocities[idx_a] += inv_i_a.mul_vec3(r_a.cross(imp_n));
                    }
                    if let Some(ib) = idx_b
                        && imb > 0.0
                    {
                        self.soa.linear_velocities[ib] -= imp_n * imb;
                        self.soa.angular_velocities[ib] -= inv_i_b.mul_vec3(r_b.cross(imp_n));
                    }
                }
            }
        }

        let total_jn: f64 = acc_jn.iter().sum();

        // Tangential friction impulse across patch points.
        if friction > 0.0 && total_jn > 0.0 {
            for (pi, &(pa, pb)) in patch_pts.iter().enumerate() {
                let jn_pt = acc_jn[pi];
                if jn_pt <= 0.0 {
                    continue;
                }
                let r_a = pa - pos_a;
                let r_b = if idx_b.is_some() {
                    pb - pos_b
                } else {
                    Vec3d::ZERO
                };
                let va2 = self.soa.linear_velocities[idx_a]
                    + self.soa.angular_velocities[idx_a].cross(r_a);
                let vb2 = idx_b.map_or(Vec3d::ZERO, |ib| {
                    self.soa.linear_velocities[ib] + self.soa.angular_velocities[ib].cross(r_b)
                });
                let vt = (va2 - vb2) - n * (va2 - vb2).dot(n);
                if let Some(t_dir) = vt.try_normalize() {
                    let ra_x_t = r_a.cross(t_dir);
                    let rb_x_t = r_b.cross(t_dir);
                    let k_t = ima
                        + imb
                        + ra_x_t.dot(inv_i_a.mul_vec3(ra_x_t))
                        + rb_x_t.dot(inv_i_b.mul_vec3(rb_x_t));
                    if k_t > 1e-18 {
                        let jt = (-vt.length() / k_t).clamp(-friction * jn_pt, friction * jn_pt);
                        let imp_t = t_dir * jt;
                        if ima > 0.0 {
                            self.soa.linear_velocities[idx_a] += imp_t * ima;
                            self.soa.angular_velocities[idx_a] +=
                                inv_i_a.mul_vec3(r_a.cross(imp_t));
                        }
                        if let Some(ib) = idx_b
                            && imb > 0.0
                        {
                            self.soa.linear_velocities[ib] -= imp_t * imb;
                            self.soa.angular_velocities[ib] -= inv_i_b.mul_vec3(r_b.cross(imp_t));
                        }
                    }
                }
            }
        }

        total_jn
    }

    fn generate_body_body_contacts(
        &mut self,
        slot_to_dense: &[usize],
        non_colliding: &BTreeSet<(BodyId, BodyId)>,
    ) {
        for &(uid_a, uid_b) in &self.bp_pairs {
            let sa = uid_a as usize;
            let sb = uid_b as usize;
            if sa >= slot_to_dense.len() || sb >= slot_to_dense.len() {
                continue;
            }
            let mut idx_a = slot_to_dense[sa];
            let mut idx_b = slot_to_dense[sb];
            if idx_a == usize::MAX || idx_b == usize::MAX {
                continue;
            }

            let dyn_a = self.soa.is_dynamic[idx_a];
            let dyn_b = self.soa.is_dynamic[idx_b];
            if !dyn_a && !dyn_b {
                continue;
            }
            if (self.soa.sleeping[idx_a] || !dyn_a) && (self.soa.sleeping[idx_b] || !dyn_b) {
                continue;
            }

            // Canonicalize so body A is dynamic (or smaller BodyId if both dynamic).
            if (!dyn_a && dyn_b)
                || (dyn_a && dyn_b && self.soa.body_ids[idx_a] > self.soa.body_ids[idx_b])
            {
                std::mem::swap(&mut idx_a, &mut idx_b);
            }

            let id_a = self.soa.body_ids[idx_a];
            let id_b = self.soa.body_ids[idx_b];
            let pair_key = if id_a < id_b {
                (id_a, id_b)
            } else {
                (id_b, id_a)
            };
            if non_colliding.contains(&pair_key) {
                continue;
            }

            let body_a = self.bodies.get(id_a).expect("live body");
            let body_b = self.bodies.get(id_b).expect("live body");
            let pos_a = self.soa.positions[idx_a];
            let rot_a = self.soa.rotations[idx_a];
            let pos_b = self.soa.positions[idx_b];
            let rot_b = self.soa.rotations[idx_b];

            // Fast tight body-AABB cull.
            let boxes_a = body_a.grid.boxes();
            let boxes_b = body_b.grid.boxes();
            if boxes_a.is_empty() || boxes_b.is_empty() {
                continue;
            }

            let mut clusters: Vec<ContactCluster> = Vec::new();
            for (bi_a, lba) in boxes_a.iter().enumerate() {
                let obb_a = body_a.box_to_world_obb_at(lba, pos_a, rot_a);
                let aabb_a = obb_a.to_aabb();
                let mat_a = self.materials.get(lba.material);

                for (bi_b, lbb) in boxes_b.iter().enumerate() {
                    let obb_b = body_b.box_to_world_obb_at(lbb, pos_b, rot_b);
                    if !aabb_a.intersects(&obb_b.to_aabb()) {
                        continue;
                    }
                    if let Some(res) = collide_obb_obb(
                        &obb_a,
                        lba.exposed_faces,
                        bi_a as u32,
                        &obb_b,
                        lbb.exposed_faces,
                        bi_b as u32,
                    ) {
                        let mat_b = self.materials.get(lbb.material);
                        let friction = (mat_a.friction * mat_b.friction).sqrt();
                        let restitution = mat_a.restitution.max(mat_b.restitution);
                        push_into_clusters(
                            &mut clusters,
                            res.normal,
                            &res.points[..res.point_count],
                            friction,
                            restitution,
                        );
                    }
                }
            }

            emit_clustered_manifolds(
                &clusters,
                id_a,
                Some(id_b),
                Transformd::new(pos_a, rot_a),
                Some(Transformd::new(pos_b, rot_b)),
                &mut self.manifolds,
            );
        }
    }

    fn generate_terrain_contacts(
        &mut self,
        skip_integration: &[bool],
        terrain: &mut dyn WorldVoxelProvider,
    ) {
        let mut clusters: Vec<ContactCluster> = Vec::new();
        for idx_pos in 0..self.soa.awake_dense_indices.len() {
            let i = self.soa.awake_dense_indices[idx_pos];
            if skip_integration[i] {
                continue;
            }
            let id_a = self.soa.body_ids[i];
            let body_a = self.bodies.get(id_a).expect("live body");
            let pos_a = self.soa.positions[i];
            let rot_a = self.soa.rotations[i];

            clusters.clear();
            for (bi_a, lba) in body_a.grid.boxes().iter().enumerate() {
                let obb_a = body_a.box_to_world_obb_at(lba, pos_a, rot_a);
                let box_aabb = obb_a.to_aabb();
                let mat_a = self.materials.get(lba.material);

                let min_x = box_aabb.min.x.floor() as i32;
                let min_y = box_aabb.min.y.floor() as i32;
                let min_z = box_aabb.min.z.floor() as i32;
                let max_x = box_aabb.max.x.floor() as i32;
                let max_y = box_aabb.max.y.floor() as i32;
                let max_z = box_aabb.max.z.floor() as i32;

                let span = (i64::from(max_x) - i64::from(min_x) + 1)
                    * (i64::from(max_y) - i64::from(min_y) + 1)
                    * (i64::from(max_z) - i64::from(min_z) + 1);
                if span > 65_536 {
                    continue;
                }

                for vx in min_x..=max_x {
                    for vy in min_y..=max_y {
                        for vz in min_z..=max_z {
                            let Some(block) = terrain.voxel_collision(vx, vy, vz) else {
                                continue;
                            };
                            let origin = Vec3d::new(vx as f64, vy as f64, vz as f64);
                            let block_aabb = Aabbd::new(
                                origin + block.local_aabb.min,
                                origin + block.local_aabb.max,
                            );
                            if !box_aabb.intersects(&block_aabb) {
                                continue;
                            }

                            let exposed_b = terrain_exposed_mask(terrain, vx, vy, vz);
                            let shape_b = pack_terrain_coord_hash(vx, vy, vz);
                            if let Some(res) = collide_obb_aabb(
                                &obb_a,
                                lba.exposed_faces,
                                bi_a as u32,
                                &block_aabb,
                                exposed_b,
                                shape_b,
                            ) {
                                let mat_b = self.materials.get(block.material);
                                let friction = (mat_a.friction * mat_b.friction).sqrt();
                                let restitution = mat_a.restitution.max(mat_b.restitution);
                                push_into_clusters(
                                    &mut clusters,
                                    res.normal,
                                    &res.points[..res.point_count],
                                    friction,
                                    restitution,
                                );
                            }
                        }
                    }
                }
            }

            emit_clustered_manifolds(
                &clusters,
                id_a,
                None,
                Transformd::new(pos_a, rot_a),
                None,
                &mut self.manifolds,
            );
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum ToiTargetShape {
    Terrain,
    BodyBox { box_idx_b: usize },
}

#[derive(Copy, Clone, Debug)]
struct ToiCandidate {
    hit: ToiHit,
    box_idx_a: usize,
    other_dense: Option<usize>,
    target_shape: ToiTargetShape,
    restitution: f64,
    friction: f64,
}

#[derive(Clone, Debug)]
struct ContactCluster {
    normal: Vec3d,
    plane_offset: f64,
    friction: f64,
    restitution: f64,
    points: Vec<ContactPoint>,
}

fn push_into_clusters(
    clusters: &mut Vec<ContactCluster>,
    normal: Vec3d,
    pts: &[ContactPoint],
    friction: f64,
    restitution: f64,
) {
    if pts.is_empty() {
        return;
    }
    let plane_offset = pts[0].world_point_b.dot(normal);
    for c in clusters.iter_mut() {
        if c.normal.dot(normal) > 0.985 && (c.plane_offset - plane_offset).abs() < 0.06 {
            c.points.extend_from_slice(pts);
            c.friction = 0.5 * (c.friction + friction);
            c.restitution = c.restitution.max(restitution);
            return;
        }
    }
    clusters.push(ContactCluster {
        normal,
        plane_offset,
        friction,
        restitution,
        points: pts.to_vec(),
    });
}

fn emit_clustered_manifolds(
    clusters: &[ContactCluster],
    body_a: BodyId,
    body_b: Option<BodyId>,
    tf_a: Transformd,
    tf_b: Option<Transformd>,
    out: &mut Vec<ContactManifold>,
) {
    for c in clusters {
        let (mut reduced, count) = reduce_contacts_to_4(&c.points, c.normal);
        if count == 0 {
            continue;
        }
        for p in reduced.iter_mut().take(count) {
            p.local_point_a = tf_a.inverse_transform_point(p.world_point_a);
            p.local_point_b = match tf_b {
                Some(tb) => tb.inverse_transform_point(p.world_point_b),
                None => p.world_point_b,
            };
        }
        out.push(ContactManifold {
            body_a,
            body_b,
            normal: c.normal,
            friction: c.friction,
            restitution: c.restitution,
            points: reduced,
            point_count: count,
            is_terrain: body_b.is_none(),
        });
    }
}

#[inline]
fn terrain_exposed_mask(terrain: &mut dyn WorldVoxelProvider, x: i32, y: i32, z: i32) -> u8 {
    let mut mask = 0u8;
    if !terrain.is_full_solid(x - 1, y, z) {
        mask |= FACE_NEG_X;
    }
    if !terrain.is_full_solid(x + 1, y, z) {
        mask |= FACE_POS_X;
    }
    if !terrain.is_full_solid(x, y - 1, z) {
        mask |= FACE_NEG_Y;
    }
    if !terrain.is_full_solid(x, y + 1, z) {
        mask |= FACE_POS_Y;
    }
    if !terrain.is_full_solid(x, y, z - 1) {
        mask |= FACE_NEG_Z;
    }
    if !terrain.is_full_solid(x, y, z + 1) {
        mask |= FACE_POS_Z;
    }
    mask
}

#[inline]
fn filter_terrain_normal(normal: Vec3d, exposed_mask: u8) -> Vec3d {
    let mut n = normal;
    if (n.x < 0.0 && (exposed_mask & FACE_NEG_X) == 0)
        || (n.x > 0.0 && (exposed_mask & FACE_POS_X) == 0)
    {
        n.x = 0.0;
    }
    if (n.y < 0.0 && (exposed_mask & FACE_NEG_Y) == 0)
        || (n.y > 0.0 && (exposed_mask & FACE_POS_Y) == 0)
    {
        n.y = 0.0;
    }
    if (n.z < 0.0 && (exposed_mask & FACE_NEG_Z) == 0)
        || (n.z > 0.0 && (exposed_mask & FACE_POS_Z) == 0)
    {
        n.z = 0.0;
    }
    n.try_normalize().unwrap_or(normal)
}

#[inline]
fn pack_terrain_coord_hash(x: i32, y: i32, z: i32) -> u32 {
    let ux = (x as u32).wrapping_mul(0x9E37_79B1);
    let uy = (y as u32).wrapping_mul(0x85EB_CA77);
    let uz = (z as u32).wrapping_mul(0xC2B2_AE3D);
    ux ^ uy.rotate_left(11) ^ uz.rotate_left(23)
}

/// Resolves overlapping entity AABB against candidate body OBBs iteratively via SAT MTV.
///
/// When `vertical_only_on_walkable` is true (gravity settling / carry depenetration),
/// walkable deck slopes (`push_dir.y >= 0.55`) push strictly upward along `+Y` by
/// `pen / push_dir.y` so standing on a tilted deck suffers zero downhill slip.
fn resolve_entity_overlaps(
    mut aabb: Aabbd,
    obbs: &[(BodyId, Obbd, u8)],
    vertical_only_on_walkable: bool,
) -> (Aabbd, bool, Option<Vec3d>) {
    let mut hit_wall = false;
    let mut best_normal: Option<Vec3d> = None;

    for _ in 0..6 {
        let mut max_pen = 0.0;
        let mut best_push = Vec3d::ZERO;
        let mut step_normal: Option<Vec3d> = None;

        for &(_, ref obb, exposed) in obbs {
            if !obb.to_aabb().intersects(&aabb) {
                continue;
            }
            if let Some(res) = collide_obb_aabb(obb, exposed, 0, &aabb, ALL_FACES, 0) {
                // `res.normal` points from entity (B) -> obb (A); push entity along `-res.normal`.
                let push_dir = -res.normal;
                let pen = res.points[..res.point_count]
                    .iter()
                    .map(|p| p.penetration)
                    .fold(0.0_f64, f64::max);
                if pen > max_pen {
                    max_pen = pen;
                    best_push = if vertical_only_on_walkable && push_dir.y >= 0.55 {
                        Vec3d::new(0.0, (pen + 1e-6) / push_dir.y, 0.0)
                    } else {
                        push_dir * (pen + 1e-6)
                    };
                    step_normal = Some(push_dir);
                }
            }
        }

        if max_pen <= 1e-6 {
            break;
        }
        aabb = aabb.translate(best_push);
        if let Some(n) = step_normal {
            if n.y.abs() < 0.55 {
                hit_wall = true;
            }
            best_normal = Some(n);
        }
    }

    (aabb, hit_wall, best_normal)
}

/// Sweeps an entity AABB along `delta` in fine safe sub-steps so it smoothly
/// rides tilted decks and lands accurately on narrow stair ledges.
fn sweep_entity_segment(
    start: Aabbd,
    delta: Vec3d,
    obbs: &[(BodyId, Obbd, u8)],
    vertical_only_on_walkable: bool,
) -> (Aabbd, bool, Option<Vec3d>) {
    let dist = delta.length();
    if dist <= 1e-9 {
        return (start, false, None);
    }
    let max_step = if vertical_only_on_walkable {
        0.025
    } else {
        0.06
    };
    let steps = ((dist / max_step).ceil() as usize).clamp(1, 96);
    let inc = delta / (steps as f64);

    let mut cur = start;
    let mut any_wall = false;
    let mut last_normal = None;

    for _ in 0..steps {
        cur = cur.translate(inc);
        let (resolved, hit_wall, normal) =
            resolve_entity_overlaps(cur, obbs, vertical_only_on_walkable);
        cur = resolved;
        if hit_wall {
            any_wall = true;
        }
        if let Some(n) = normal {
            last_normal = Some(n);
            // Stop descending immediately once seated on a walkable floor/step.
            if vertical_only_on_walkable && inc.y < 0.0 && n.y >= 0.55 {
                break;
            }
        }
    }

    (cur, any_wall, last_normal)
}
