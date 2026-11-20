//! Public clean-room physics extension API for third-party Horizon 26.3 mods
//! (aviation, maritime, rail, mechanical vehicles, and custom force fields).

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock, RwLock, TryLockError};

use horizon_physics_core::{
    Aabbd, BodyDesc, BodyId, BodyKind, BodySnapshot, JointId, JointKind, JointSpec, MaterialId,
    Obbd, PhysicsConfig, PhysicsEvent, PhysicsWorld, Quatd, StepMetrics, Transformd, Vec3d,
    VoxelCoord, VoxelEntry, VoxelMaterial, VoxelSubGrid, ray_intersect_aabb,
};

use crate::terrain_cache::{
    CachedBlockCell, HostTerrainSampler, PrefetchedTerrainSample, TerrainCache,
    canonical_block_name,
};

/// Opaque generational handle to an assembled voxel sub-grid rigid body.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SubGridHandle(pub BodyId);

impl SubGridHandle {
    #[inline]
    pub const fn from_body_id(id: BodyId) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn body_id(self) -> BodyId {
        self.0
    }

    #[inline]
    pub const fn index(self) -> u32 {
        self.0.index
    }

    #[inline]
    pub const fn generation(self) -> u32 {
        self.0.generation
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0.is_valid()
    }
}

/// Opaque generational handle to a registered mechanical joint constraint.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct JointHandle(pub JointId);

impl JointHandle {
    #[inline]
    pub const fn from_joint_id(id: JointId) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn joint_id(self) -> JointId {
        self.0
    }
}

/// Physical mass and surface parameters for a single block type.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockMassSpec {
    pub mass_kg: f64,
    pub friction: f64,
    pub restitution: f64,
}

impl BlockMassSpec {
    #[inline]
    pub const fn new(mass_kg: f64, friction: f64, restitution: f64) -> Self {
        Self {
            mass_kg,
            friction,
            restitution,
        }
    }

    #[inline]
    pub const fn stone() -> Self {
        Self::new(2500.0, 0.65, 0.10)
    }

    #[inline]
    pub const fn wood() -> Self {
        Self::new(600.0, 0.55, 0.15)
    }

    #[inline]
    pub const fn iron() -> Self {
        Self::new(7850.0, 0.45, 0.20)
    }

    #[inline]
    pub const fn slime() -> Self {
        Self::new(1100.0, 0.80, 0.85)
    }

    #[inline]
    pub const fn ice() -> Self {
        Self::new(917.0, 0.04, 0.08)
    }

    #[inline]
    pub const fn wool() -> Self {
        Self::new(250.0, 0.85, 0.05)
    }
}

/// One voxel entry in a sub-grid assembly or dynamic modification request.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VoxelPlacement {
    pub local_coord: [i32; 3],
    pub mass_kg: f64,
    pub friction: f64,
    pub restitution: f64,
}

impl VoxelPlacement {
    #[inline]
    pub const fn from_spec(local_coord: [i32; 3], spec: BlockMassSpec) -> Self {
        Self {
            local_coord,
            mass_kg: spec.mass_kg,
            friction: spec.friction,
            restitution: spec.restitution,
        }
    }
}

/// Descriptor for assembling a new dynamic voxel sub-grid rigid body.
#[derive(Clone, Debug, PartialEq)]
pub struct SubGridAssemblyDesc {
    /// Initial world-space position of the sub-grid center of mass (or grid origin).
    pub world_origin: [f64; 3],
    /// Initial orientation quaternion `[x, y, z, w]`.
    pub initial_rotation: [f64; 4],
    /// Occupied local voxels and their physical mass/surface properties.
    pub voxels: Vec<VoxelPlacement>,
    /// Enables conservative-advancement continuous collision detection (CCD).
    pub enable_ccd: bool,
}

/// Immutable kinematic snapshot of a live sub-grid.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SubGridState {
    pub handle: SubGridHandle,
    pub world_position: [f64; 3],
    pub world_rotation: [f64; 4],
    pub linear_velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub mass_kg: f64,
    pub com_local: [f64; 3],
    pub voxel_count: usize,
    pub sleeping: bool,
}

impl From<BodySnapshot> for SubGridState {
    fn from(snap: BodySnapshot) -> Self {
        Self {
            handle: SubGridHandle(snap.id),
            world_position: snap.transform.position.to_array(),
            world_rotation: [
                snap.transform.rotation.x,
                snap.transform.rotation.y,
                snap.transform.rotation.z,
                snap.transform.rotation.w,
            ],
            linear_velocity: snap.linear_velocity.to_array(),
            angular_velocity: snap.angular_velocity.to_array(),
            mass_kg: snap.mass,
            com_local: snap.com_local.to_array(),
            voxel_count: snap.voxel_count,
            sleeping: snap.sleeping,
        }
    }
}

/// Disassembled voxel output carrying world-snapped coordinates and original block id.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DisassembledVoxel {
    pub local_coord: [i32; 3],
    pub world_coord: [i32; 3],
    pub block_id: String,
    pub material: MaterialId,
}

/// Registry interface for assembling, modifying, querying, and disassembling sub-grids.
pub trait SubGridRegistry {
    fn assemble_subgrid(&mut self, desc: SubGridAssemblyDesc) -> Result<SubGridHandle, String>;

    fn disassemble_subgrid(&mut self, grid: SubGridHandle) -> Option<Vec<DisassembledVoxel>>;

    fn set_voxel(&mut self, grid: SubGridHandle, placement: VoxelPlacement) -> Result<(), String>;

    fn remove_voxel(&mut self, grid: SubGridHandle, local_coord: [i32; 3]) -> Result<(), String>;

    fn subgrid_state(&self, grid: SubGridHandle) -> Option<SubGridState>;

    fn active_subgrids(&self) -> Vec<SubGridHandle>;

    fn apply_impulse(
        &mut self,
        grid: SubGridHandle,
        impulse_world: [f64; 3],
        world_point: Option<[f64; 3]>,
    ) -> bool;

    fn sleep_subgrid(&mut self, grid: SubGridHandle) -> bool;

    fn wake_subgrid(&mut self, grid: SubGridHandle) -> bool;
}

/// Sub-step kinematic and environmental context supplied to [`ForceProvider`]
/// and [`Propulsion`] callbacks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ForceQueryContext {
    /// Body center-of-mass world position (m).
    pub world_position: [f64; 3],
    /// Body world orientation quaternion `(x, y, z, w)`.
    pub world_rotation: [f64; 4],
    /// Body center-of-mass linear velocity in world frame (m/s).
    pub linear_velocity_world: [f64; 3],
    /// Body angular velocity vector in world frame (rad/s).
    pub angular_velocity_world: [f64; 3],
    /// Current total mass of the sub-grid (kg).
    pub mass_kg: f64,
    /// Center of mass in local sub-grid coordinates (m).
    pub com_local: [f64; 3],
    /// Ambient world wind velocity (m/s).
    pub ambient_wind_world: [f64; 3],
    /// Time step duration of the current physics step (s).
    pub dt: f64,
    /// Monotonic simulation step index.
    pub step_index: u64,
}

/// Alias matching concise extension imports.
pub type ForceContext = ForceQueryContext;

/// Alias for [`Propulsion::evaluate_thrust`] context.
pub type PropulsionContext = ForceQueryContext;

impl ForceQueryContext {
    #[inline]
    fn quat(&self) -> Quatd {
        Quatd::new(
            self.world_rotation[0],
            self.world_rotation[1],
            self.world_rotation[2],
            self.world_rotation[3],
        )
        .normalize()
    }

    /// Computes relative velocity (body point velocity minus ambient wind)
    /// expressed in the body's local frame for a point `local_point` relative to CoM.
    #[inline]
    pub fn local_point_velocity(&self, local_point: [f64; 3]) -> [f64; 3] {
        let q = self.quat();
        let r_world = q.rotate_vec3(Vec3d::from_array(local_point));
        let v_com = Vec3d::from_array(self.linear_velocity_world);
        let omega = Vec3d::from_array(self.angular_velocity_world);
        let wind = Vec3d::from_array(self.ambient_wind_world);
        let v_world = v_com + omega.cross(r_world) - wind;
        q.inverse_rotate_vec3(v_world).to_array()
    }

    /// Scalar component of body linear airspeed along a local unit axis.
    #[inline]
    pub fn local_velocity_along(&self, local_axis: [f64; 3]) -> f64 {
        let q = self.quat();
        let axis = Vec3d::from_array(local_axis).normalize();
        let v_rel_world = Vec3d::from_array(self.linear_velocity_world)
            - Vec3d::from_array(self.ambient_wind_world);
        let v_local = q.inverse_rotate_vec3(v_rel_world);
        v_local.dot(axis)
    }

    /// Transforms a body-local point (relative to CoM) into world space.
    #[inline]
    pub fn local_to_world_point(&self, local_point: [f64; 3]) -> [f64; 3] {
        let pos = Vec3d::from_array(self.world_position);
        (pos + self.quat().rotate_vec3(Vec3d::from_array(local_point))).to_array()
    }

    /// Rotates a local direction vector into world space.
    #[inline]
    pub fn local_to_world_dir(&self, local_dir: [f64; 3]) -> [f64; 3] {
        self.quat()
            .rotate_vec3(Vec3d::from_array(local_dir))
            .to_array()
    }

    /// Rotates a world direction vector into the body's local frame.
    #[inline]
    pub fn world_to_local_dir(&self, world_dir: [f64; 3]) -> [f64; 3] {
        self.quat()
            .inverse_rotate_vec3(Vec3d::from_array(world_dir))
            .to_array()
    }
}

/// Force and/or torque output produced by a [`ForceProvider`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ForceContribution {
    pub force: [f64; 3],
    pub torque: [f64; 3],
    pub application_point: Option<[f64; 3]>,
    pub is_local_space: bool,
}

impl ForceContribution {
    /// Applies a local-space force at a body-local mount point (relative to CoM).
    #[inline]
    pub const fn local_force_at_point(local_force: [f64; 3], local_point: [f64; 3]) -> Self {
        Self {
            force: local_force,
            torque: [0.0; 3],
            application_point: Some(local_point),
            is_local_space: true,
        }
    }

    /// Applies both a local-space force at `local_point` and a pure local torque.
    #[inline]
    pub const fn local_force_and_torque(
        local_force: [f64; 3],
        local_torque: [f64; 3],
        local_point: [f64; 3],
    ) -> Self {
        Self {
            force: local_force,
            torque: local_torque,
            application_point: Some(local_point),
            is_local_space: true,
        }
    }

    /// Applies a world-space force at a world-space application point.
    #[inline]
    pub const fn world_force_at_point(world_force: [f64; 3], world_point: [f64; 3]) -> Self {
        Self {
            force: world_force,
            torque: [0.0; 3],
            application_point: Some(world_point),
            is_local_space: false,
        }
    }

    /// Applies a pure world-space torque without linear force.
    #[inline]
    pub const fn world_torque(world_torque: [f64; 3]) -> Self {
        Self {
            force: [0.0; 3],
            torque: world_torque,
            application_point: None,
            is_local_space: false,
        }
    }
}

/// Distributed external force plugin evaluated on each physics step (aerodynamic
/// wings, control surfaces, custom buoyancy, magnetic fields, etc.).
pub trait ForceProvider: Send + Sync + 'static {
    /// Stable namespaced identifier (`"my_mod:airfoil_wing"`).
    fn provider_id(&self) -> &'static str;

    /// Computes force and/or torque contribution for `grid` under `ctx`.
    fn compute_force(
        &self,
        grid: SubGridHandle,
        ctx: &ForceQueryContext,
    ) -> Option<ForceContribution>;
}

/// Output produced by evaluating a [`Propulsion`] actuator on a physics step.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PropulsionOutput {
    /// Local mount position relative to CoM (m).
    pub local_mount: [f64; 3],
    /// Local thrust force vector (N).
    pub local_force: [f64; 3],
    /// Local reaction torque vector (N*m).
    pub local_torque: [f64; 3],
    /// Instantaneous fuel or energy burn rate (arbitrary mod units/s).
    pub fuel_burn_rate: f64,
}

/// Controllable propulsion unit (turboprop, jet engine, thruster nozzle, paddlewheel).
pub trait Propulsion: Send + Sync + 'static {
    /// Stable namespaced identifier (`"my_mod:main_thruster"`).
    fn propulsion_id(&self) -> &'static str;

    /// Sets normalized throttle command in `[0.0, 1.0]` (or `[-1.0, 1.0]` for reversible units).
    fn set_throttle(&mut self, throttle: f64);

    /// Returns the current throttle setting (defaults to `0.0`).
    fn throttle(&self) -> f64 {
        0.0
    }

    /// Optional thrust vectoring deflection angles in radians `(pitch, yaw)`.
    fn set_vector_angles(&mut self, _pitch_rad: f64, _yaw_rad: f64) {}

    /// Evaluates thrust force, reaction torque, and fuel burn for the current step.
    fn evaluate_thrust(&self, grid: SubGridHandle, ctx: &PropulsionContext) -> PropulsionOutput;
}

/// Active thruster exhaust plume telemetry emitted for client WGSL rendering.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ThrusterPlumeState {
    pub grid: SubGridHandle,
    pub local_mount: [f64; 3],
    pub local_exhaust_dir: [f64; 3],
    pub throttle: f64,
    pub thrust_n: f64,
}

/// Servo motor parameters for angular or linear joints.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct JointMotor {
    pub target_angle_rad: f64,
    pub max_torque_nm: f64,
    pub stiffness: f64,
    pub damping: f64,
}

/// Revolute hinge joint connecting two sub-grids.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HingeJointDesc {
    pub body_a: SubGridHandle,
    pub body_b: SubGridHandle,
    pub anchor_a: [f64; 3],
    pub anchor_b: [f64; 3],
    pub axis_a: [f64; 3],
    pub axis_b: [f64; 3],
    pub min_angle_rad: f64,
    pub max_angle_rad: f64,
    pub compliance: f64,
    pub motor: Option<JointMotor>,
}

/// Prismatic slider joint connecting two sub-grids (or sub-grid to world).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SliderJointDesc {
    pub body_a: SubGridHandle,
    pub body_b: Option<SubGridHandle>,
    pub anchor_a: [f64; 3],
    pub anchor_b: [f64; 3],
    pub axis_a: [f64; 3],
    pub axis_b: [f64; 3],
    pub limits: Option<(f64, f64)>,
    pub compliance: f64,
    pub motor_target_velocity: Option<f64>,
    pub motor_max_force: f64,
}

/// Distance / spring-damper rope or strut joint.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DistanceJointDesc {
    pub body_a: SubGridHandle,
    pub body_b: Option<SubGridHandle>,
    pub anchor_a: [f64; 3],
    pub anchor_b: [f64; 3],
    pub rest_length: f64,
    pub min_length: Option<f64>,
    pub max_length: Option<f64>,
    pub compliance: f64,
    pub damping: f64,
}

/// Rigid 6-DOF weld locking two sub-grids (or anchoring one sub-grid to world).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FixedWeldJointDesc {
    pub body_a: SubGridHandle,
    pub body_b: Option<SubGridHandle>,
    pub anchor_a: [f64; 3],
    pub anchor_b: [f64; 3],
    pub relative_rotation: [f64; 4],
    pub compliance: f64,
}

/// Unified high-level joint constraint description exposed to third-party mods.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum JointConstraint {
    Hinge(HingeJointDesc),
    Slider(SliderJointDesc),
    Distance(DistanceJointDesc),
    FixedWeld(FixedWeldJointDesc),
}

impl JointConstraint {
    /// Converts this public constraint into the core solver's [`JointSpec`].
    pub fn to_core_spec(&self) -> JointSpec {
        match *self {
            Self::Hinge(h) => {
                let mut spec = JointSpec::hinge(
                    h.body_a.body_id(),
                    Some(h.body_b.body_id()),
                    Vec3d::from_array(h.axis_a),
                    Vec3d::from_array(h.axis_b),
                );
                spec.local_anchor_a = Vec3d::from_array(h.anchor_a);
                spec.local_anchor_b = Vec3d::from_array(h.anchor_b);
                spec.compliance = h.compliance.max(0.0);
                if let JointKind::Hinge {
                    limits,
                    motor_max_torque,
                    ..
                } = &mut spec.kind
                {
                    if h.min_angle_rad <= h.max_angle_rad {
                        *limits = Some((h.min_angle_rad, h.max_angle_rad));
                    }
                    if let Some(m) = h.motor {
                        *motor_max_torque = m.max_torque_nm.max(0.0);
                    }
                }
                spec
            }
            Self::Slider(s) => {
                let mut spec = JointSpec::slider(
                    s.body_a.body_id(),
                    s.body_b.map(SubGridHandle::body_id),
                    Vec3d::from_array(s.axis_a),
                    Vec3d::from_array(s.axis_b),
                );
                spec.local_anchor_a = Vec3d::from_array(s.anchor_a);
                spec.local_anchor_b = Vec3d::from_array(s.anchor_b);
                spec.compliance = s.compliance.max(0.0);
                if let JointKind::Slider {
                    limits,
                    motor_target_velocity,
                    motor_max_force,
                    ..
                } = &mut spec.kind
                {
                    *limits = s.limits;
                    *motor_target_velocity = s.motor_target_velocity;
                    *motor_max_force = s.motor_max_force.max(0.0);
                }
                spec
            }
            Self::Distance(d) => {
                let mut spec = JointSpec::distance(
                    d.body_a.body_id(),
                    d.body_b.map(SubGridHandle::body_id),
                    d.rest_length.max(0.0),
                );
                spec.local_anchor_a = Vec3d::from_array(d.anchor_a);
                spec.local_anchor_b = Vec3d::from_array(d.anchor_b);
                spec.compliance = d.compliance.max(0.0);
                spec.damping = d.damping.clamp(0.0, 1.0);
                if let JointKind::Distance {
                    min_length,
                    max_length,
                    ..
                } = &mut spec.kind
                {
                    *min_length = d.min_length;
                    *max_length = d.max_length;
                }
                spec
            }
            Self::FixedWeld(w) => {
                let rel_q = Quatd::new(
                    w.relative_rotation[0],
                    w.relative_rotation[1],
                    w.relative_rotation[2],
                    w.relative_rotation[3],
                )
                .normalize();
                let mut spec = JointSpec::fixed_weld(
                    w.body_a.body_id(),
                    w.body_b.map(SubGridHandle::body_id),
                    rel_q,
                );
                spec.local_anchor_a = Vec3d::from_array(w.anchor_a);
                spec.local_anchor_b = Vec3d::from_array(w.anchor_b);
                spec.compliance = w.compliance.max(0.0);
                spec
            }
        }
    }
}

/// Registry interface for adding and removing mechanical joints.
pub trait JointRegistry {
    fn add_joint(&mut self, constraint: JointConstraint) -> Option<JointHandle>;
    fn remove_joint(&mut self, handle: JointHandle) -> bool;
}

/// Filtering mask controlling which geometry layers a ray query intersects.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct RaycastFilter {
    pub exclude_subgrid: Option<SubGridHandle>,
    pub include_world_blocks: bool,
    pub include_subgrids: bool,
}

impl Default for RaycastFilter {
    fn default() -> Self {
        Self::all()
    }
}

impl RaycastFilter {
    #[inline]
    pub const fn all() -> Self {
        Self {
            exclude_subgrid: None,
            include_world_blocks: true,
            include_subgrids: true,
        }
    }

    #[inline]
    pub const fn exclude_subgrid(grid: SubGridHandle) -> Self {
        Self {
            exclude_subgrid: Some(grid),
            include_world_blocks: true,
            include_subgrids: true,
        }
    }

    #[inline]
    pub const fn world_only() -> Self {
        Self {
            exclude_subgrid: None,
            include_world_blocks: true,
            include_subgrids: false,
        }
    }

    #[inline]
    pub const fn subgrids_only() -> Self {
        Self {
            exclude_subgrid: None,
            include_world_blocks: false,
            include_subgrids: true,
        }
    }
}

/// Double-precision raycast input query.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RayQuery {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
    pub max_distance: f64,
    pub filter: RaycastFilter,
}

/// Classified target struck by a combined world + sub-grid raycast.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RaycastHitTarget {
    WorldBlock {
        block_pos: [i32; 3],
        material: MaterialId,
    },
    SubGridVoxel {
        grid: SubGridHandle,
        local_coord: Option<[i32; 3]>,
        box_index: usize,
    },
}

/// Closest hit returned by [`PhysicsRaycast::cast_ray`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CombinedRaycastHit {
    pub distance: f64,
    pub point: [f64; 3],
    pub normal: [f64; 3],
    pub target: RaycastHitTarget,
}

/// Unified double-precision raycasting across static terrain and dynamic sub-grids.
pub trait PhysicsRaycast {
    fn cast_ray(&self, query: &RayQuery) -> Option<CombinedRaycastHit>;
}

type SharedForceProvider = Arc<dyn ForceProvider>;

#[derive(Clone)]
struct PropulsionSlot {
    id: &'static str,
    unit: Arc<Mutex<Box<dyn Propulsion>>>,
}

#[derive(Clone, Debug)]
enum PropulsionOp {
    SetThrottle(f64),
    SetVector(f64, f64),
}

type PreparedPropulsionOp = (
    SubGridHandle,
    String,
    Arc<Mutex<Box<dyn Propulsion>>>,
    PropulsionOp,
);

/// Central thread-safe coordinator owning the [`PhysicsWorld`], [`TerrainCache`],
/// external [`ForceProvider`] / [`Propulsion`] plugins, and metadata maps.
pub struct PhysicsApi {
    world: PhysicsWorld,
    terrain: TerrainCache,
    body_force_providers: BTreeMap<SubGridHandle, Vec<SharedForceProvider>>,
    global_force_providers: Vec<SharedForceProvider>,
    body_propulsions: BTreeMap<SubGridHandle, Vec<PropulsionSlot>>,
    pending_propulsion_ops: Vec<(SubGridHandle, String, PropulsionOp)>,
    joints: BTreeMap<JointHandle, JointConstraint>,
    voxel_block_names: BTreeMap<SubGridHandle, BTreeMap<[i32; 3], String>>,
    geometry_revisions: BTreeMap<BodyId, u64>,
    pending_step_events: Vec<PhysicsEvent>,
    active_plumes: Vec<ThrusterPlumeState>,
    last_metrics: StepMetrics,
    last_events: Vec<PhysicsEvent>,
    last_fuel_burn_total: f64,
    ambient_wind_world: Vec3d,
    debug_overlay_enabled: bool,
    next_custom_material: u16,
    server_tick_counter: u64,
}

impl Default for PhysicsApi {
    fn default() -> Self {
        Self::new(PhysicsConfig::default())
    }
}

impl PhysicsApi {
    pub fn new(config: PhysicsConfig) -> Self {
        let mut world = PhysicsWorld::new(config);
        let terrain = TerrainCache::new();
        for (id, mat) in terrain.materials().iter_materials() {
            world.register_material(id, mat);
        }
        Self {
            world,
            terrain,
            body_force_providers: BTreeMap::new(),
            global_force_providers: Vec::new(),
            body_propulsions: BTreeMap::new(),
            pending_propulsion_ops: Vec::new(),
            joints: BTreeMap::new(),
            voxel_block_names: BTreeMap::new(),
            geometry_revisions: BTreeMap::new(),
            pending_step_events: Vec::new(),
            active_plumes: Vec::new(),
            last_metrics: StepMetrics::default(),
            last_events: Vec::new(),
            last_fuel_burn_total: 0.0,
            ambient_wind_world: Vec3d::ZERO,
            debug_overlay_enabled: false,
            next_custom_material: 4096,
            server_tick_counter: 0,
        }
    }

    /// Resets all runtime state to a fresh world with default configuration.
    pub fn reset(&mut self) {
        *self = Self::new(PhysicsConfig::default());
    }

    /// Resets the world with a custom [`PhysicsConfig`].
    pub fn reset_with_config(&mut self, config: PhysicsConfig) {
        *self = Self::new(config);
    }

    #[inline]
    pub fn world(&self) -> &PhysicsWorld {
        &self.world
    }

    #[inline]
    pub fn world_mut(&mut self) -> &mut PhysicsWorld {
        &mut self.world
    }

    #[inline]
    pub fn terrain(&self) -> &TerrainCache {
        &self.terrain
    }

    #[inline]
    pub fn terrain_mut(&mut self) -> &mut TerrainCache {
        &mut self.terrain
    }

    #[inline]
    pub fn last_metrics(&self) -> StepMetrics {
        self.last_metrics
    }

    #[inline]
    pub fn last_events(&self) -> &[PhysicsEvent] {
        &self.last_events
    }

    #[inline]
    pub fn active_plumes(&self) -> &[ThrusterPlumeState] {
        &self.active_plumes
    }

    #[inline]
    pub fn last_fuel_burn_total(&self) -> f64 {
        self.last_fuel_burn_total
    }

    #[inline]
    pub fn server_tick_counter(&self) -> u64 {
        self.server_tick_counter
    }

    #[inline]
    pub fn debug_overlay_enabled(&self) -> bool {
        self.debug_overlay_enabled
    }

    #[inline]
    pub fn set_debug_overlay_enabled(&mut self, enabled: bool) {
        self.debug_overlay_enabled = enabled;
    }

    #[inline]
    pub fn toggle_debug_overlay(&mut self) -> bool {
        self.debug_overlay_enabled = !self.debug_overlay_enabled;
        self.debug_overlay_enabled
    }

    #[inline]
    pub fn set_ambient_wind(&mut self, wind_world: [f64; 3]) {
        self.ambient_wind_world = Vec3d::from_array(wind_world);
    }

    #[inline]
    pub fn ambient_wind(&self) -> [f64; 3] {
        self.ambient_wind_world.to_array()
    }

    /// Registers a custom block material in both [`TerrainCache`] and [`PhysicsWorld`].
    pub fn register_block_material(&mut self, block_id: &str, mat: VoxelMaterial) -> MaterialId {
        let id = self
            .terrain
            .materials_mut()
            .register_block_material(block_id, mat);
        self.world.register_material(id, mat);
        id
    }

    /// Loads or reloads material definitions from JSON text into terrain & physics world.
    pub fn load_materials_json(&mut self, json: &str) -> Result<usize, String> {
        let count = self.terrain.materials_mut().load_from_json(json)?;
        for (id, mat) in self.terrain.materials().iter_materials() {
            self.world.register_material(id, mat);
        }
        Ok(count)
    }

    /// Returns a map of `local_coord -> block_id` for a sub-grid.
    pub fn subgrid_block_names(&self, grid: SubGridHandle) -> Option<&BTreeMap<[i32; 3], String>> {
        self.voxel_block_names.get(&grid)
    }

    /// Returns all registered joint descriptors.
    pub fn joints(&self) -> &BTreeMap<JointHandle, JointConstraint> {
        &self.joints
    }

    /// Finds a live [`SubGridHandle`] by its arena slot index (useful for `/hphys` commands).
    pub fn find_subgrid_by_index(&self, index: u32) -> Option<SubGridHandle> {
        self.world
            .snapshots()
            .into_iter()
            .find(|s| s.id.index == index)
            .map(|s| SubGridHandle(s.id))
    }

    /// Spawns a sub-grid from named Minecraft blocks (used by `/hphys assemble`, `/hphys spawn`,
    /// and NBT deserialization).
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_named_subgrid(
        &mut self,
        world_origin: Vec3d,
        initial_rotation: Quatd,
        kind: BodyKind,
        enable_ccd: bool,
        gravity_scale: f64,
        user_data: u64,
        blocks: &[([i32; 3], String)],
    ) -> Result<SubGridHandle, String> {
        if blocks.is_empty() {
            return Err("cannot assemble an empty sub-grid".to_owned());
        }
        let mut grid = VoxelSubGrid::new();
        let mut names = BTreeMap::new();
        for (coord, raw_name) in blocks {
            let block_name = canonical_block_name(raw_name);
            let (mat_id, mat) = self.terrain.materials().resolve_block(&block_name);
            self.world.register_material(mat_id, mat);
            grid.set_voxel(
                VoxelCoord::new(coord[0], coord[1], coord[2]),
                Some(VoxelEntry::from_material(mat_id)),
                self.world.materials(),
            );
            names.insert(*coord, block_name);
        }
        grid.rebuild_dirty_colliders();
        if grid.is_empty() {
            return Err("no solid voxels found for sub-grid assembly".to_owned());
        }
        let mut desc = BodyDesc {
            kind,
            transform: Transformd::new(world_origin, initial_rotation.normalize()),
            grid,
            gravity_scale,
            allow_sleep: kind == BodyKind::Dynamic,
            enable_ccd,
            user_data,
            ..BodyDesc::default()
        };
        desc.transform.rotation = desc.transform.rotation.normalize();
        let id = self.world.spawn_body(desc);
        let handle = SubGridHandle(id);
        self.voxel_block_names.insert(handle, names);
        self.geometry_revisions.insert(id, 1);
        Ok(handle)
    }

    /// Returns the monotonic geometry revision for `id` (combines core `geometry_version`
    /// with bridge metadata revisions so direct `world_mut()` edits never miss sync).
    #[inline]
    pub fn geometry_revision(&self, id: BodyId) -> u64 {
        let core_rev = self
            .world
            .body_snapshot(id)
            .map_or(0, |snap| snap.geometry_version);
        let meta_rev = self.geometry_revisions.get(&id).copied().unwrap_or(0);
        if core_rev == 0 && meta_rev == 0 {
            0
        } else {
            core_rev.wrapping_add(meta_rev)
        }
    }

    fn bump_geometry_revision(&mut self, id: BodyId) {
        let rev = self.geometry_revisions.entry(id).or_insert(0);
        *rev = rev.wrapping_add(1).max(1);
    }

    /// Attaches a [`ForceProvider`] to a specific sub-grid (replaces any existing provider
    /// on that body with the same `provider_id`).
    pub fn attach_force_provider(
        &mut self,
        grid: SubGridHandle,
        provider: Box<dyn ForceProvider>,
    ) -> bool {
        if self.world.body_snapshot(grid.body_id()).is_none() {
            return false;
        }
        let pid = provider.provider_id();
        let shared: Arc<dyn ForceProvider> = Arc::from(provider);
        let list = self.body_force_providers.entry(grid).or_default();
        list.retain(|existing| existing.provider_id() != pid);
        list.push(shared);
        self.world.wake_body(grid.body_id());
        true
    }

    /// Detaches a [`ForceProvider`] by `provider_id` from `grid`.
    pub fn detach_force_provider(&mut self, grid: SubGridHandle, provider_id: &str) -> bool {
        let Some(list) = self.body_force_providers.get_mut(&grid) else {
            return false;
        };
        let before = list.len();
        list.retain(|p| p.provider_id() != provider_id);
        list.len() < before
    }

    /// Registers a global [`ForceProvider`] evaluated against every active dynamic sub-grid.
    pub fn register_global_force_provider(&mut self, provider: Box<dyn ForceProvider>) {
        let pid = provider.provider_id();
        let shared: Arc<dyn ForceProvider> = Arc::from(provider);
        self.global_force_providers
            .retain(|existing| existing.provider_id() != pid);
        self.global_force_providers.push(shared);
    }

    /// Unregisters a global [`ForceProvider`] by `provider_id`.
    pub fn unregister_global_force_provider(&mut self, provider_id: &str) -> bool {
        let before = self.global_force_providers.len();
        self.global_force_providers
            .retain(|p| p.provider_id() != provider_id);
        self.global_force_providers.len() < before
    }

    /// Attaches a [`Propulsion`] unit to `grid` (replaces any unit on `grid` with the same id).
    pub fn attach_propulsion(
        &mut self,
        grid: SubGridHandle,
        propulsion: Box<dyn Propulsion>,
    ) -> bool {
        if self.world.body_snapshot(grid.body_id()).is_none() {
            return false;
        }
        let pid = propulsion.propulsion_id();
        let slot = PropulsionSlot {
            id: pid,
            unit: Arc::new(Mutex::new(propulsion)),
        };
        let list = self.body_propulsions.entry(grid).or_default();
        list.retain(|existing| existing.id != pid);
        list.push(slot);
        self.world.wake_body(grid.body_id());
        true
    }

    /// Detaches a [`Propulsion`] unit by `propulsion_id` from `grid`.
    pub fn detach_propulsion(&mut self, grid: SubGridHandle, propulsion_id: &str) -> bool {
        let Some(list) = self.body_propulsions.get_mut(&grid) else {
            return false;
        };
        let before = list.len();
        list.retain(|slot| slot.id != propulsion_id);
        self.pending_propulsion_ops
            .retain(|(g, pid, _)| *g != grid || pid != propulsion_id);
        list.len() < before
    }

    fn apply_or_queue_propulsion_op(
        &mut self,
        grid: SubGridHandle,
        propulsion_id: &str,
        op: PropulsionOp,
    ) -> bool {
        let Some(list) = self.body_propulsions.get(&grid) else {
            return false;
        };
        for slot in list {
            if slot.id == propulsion_id {
                if let PropulsionOp::SetThrottle(throttle) = op
                    && throttle.abs() > 1e-4
                {
                    self.world.wake_body(grid.body_id());
                }
                // Defer external trait callbacks when inside `with_physics_api` write lock so
                // user `set_throttle`/`set_vector_angles` can freely call `with_physics_api_read`.
                let in_global_write = WRITE_LOCK_DEPTH.with(|d| d.get() > 0);
                if in_global_write {
                    self.pending_propulsion_ops
                        .push((grid, propulsion_id.to_owned(), op));
                    return true;
                }
                match slot.unit.try_lock() {
                    Ok(mut guard) => match op {
                        PropulsionOp::SetThrottle(throttle) => {
                            guard.set_throttle(throttle);
                        }
                        PropulsionOp::SetVector(pitch_rad, yaw_rad) => {
                            guard.set_vector_angles(pitch_rad, yaw_rad);
                        }
                    },
                    Err(TryLockError::WouldBlock) => {
                        self.pending_propulsion_ops
                            .push((grid, propulsion_id.to_owned(), op));
                    }
                    Err(TryLockError::Poisoned(e)) => {
                        let mut guard = e.into_inner();
                        match op {
                            PropulsionOp::SetThrottle(throttle) => guard.set_throttle(throttle),
                            PropulsionOp::SetVector(pitch_rad, yaw_rad) => {
                                guard.set_vector_angles(pitch_rad, yaw_rad);
                            }
                        }
                    }
                }
                return true;
            }
        }
        false
    }

    fn take_pending_propulsion_ops_for_unlock(&mut self) -> Vec<PreparedPropulsionOp> {
        if self.pending_propulsion_ops.is_empty() {
            return Vec::new();
        }
        let queued = std::mem::take(&mut self.pending_propulsion_ops);
        let mut prepared = Vec::with_capacity(queued.len());
        for (grid, pid, op) in queued {
            if let Some(list) = self.body_propulsions.get(&grid)
                && let Some(slot) = list.iter().find(|s| s.id == pid)
            {
                prepared.push((grid, pid, Arc::clone(&slot.unit), op));
            }
        }
        prepared
    }

    fn requeue_deferred_propulsion_ops(
        &mut self,
        deferred: Vec<(SubGridHandle, String, PropulsionOp)>,
    ) {
        if !deferred.is_empty() {
            self.pending_propulsion_ops.extend(deferred);
        }
    }

    fn flush_pending_propulsion_ops(&mut self) {
        if self.pending_propulsion_ops.is_empty() {
            return;
        }
        if WRITE_LOCK_DEPTH.with(|d| d.get() > 0) {
            return;
        }
        let queued = std::mem::take(&mut self.pending_propulsion_ops);
        for (grid, pid, op) in queued {
            let _ = self.apply_or_queue_propulsion_op(grid, &pid, op);
        }
    }

    /// Updates throttle on a mounted [`Propulsion`] unit and wakes the body if thrusting.
    pub fn set_propulsion_throttle(
        &mut self,
        grid: SubGridHandle,
        propulsion_id: &str,
        throttle: f64,
    ) -> bool {
        self.apply_or_queue_propulsion_op(grid, propulsion_id, PropulsionOp::SetThrottle(throttle))
    }

    /// Updates thrust-vectoring deflection angles `(pitch_rad, yaw_rad)` on a propulsion unit.
    pub fn set_propulsion_vector(
        &mut self,
        grid: SubGridHandle,
        propulsion_id: &str,
        pitch_rad: f64,
        yaw_rad: f64,
    ) -> bool {
        self.apply_or_queue_propulsion_op(
            grid,
            propulsion_id,
            PropulsionOp::SetVector(pitch_rad, yaw_rad),
        )
    }

    /// Total count of registered per-body and global force providers.
    pub fn force_provider_count(&self) -> usize {
        self.global_force_providers.len()
            + self
                .body_force_providers
                .values()
                .map(Vec::len)
                .sum::<usize>()
    }

    /// Total count of registered propulsion units across all bodies.
    pub fn propulsion_count(&self) -> usize {
        self.body_propulsions.values().map(Vec::len).sum()
    }

    /// Invalidates a terrain block cell in [`TerrainCache`] and wakes any sleeping
    /// physics islands intersecting that block's support envelope.
    pub fn notify_block_changed(&mut self, pos: [i32; 3], new_block_name: Option<&str>) {
        let wake_aabb = self.terrain.invalidate_block(pos, new_block_name);
        self.world.wake_in_aabb(wake_aabb);
    }

    /// Evicts a 16x16 chunk column from [`TerrainCache`] on chunk unload.
    pub fn notify_chunk_unloaded(&mut self, cx: i32, cz: i32) {
        self.terrain.invalidate_chunk(cx, cz);
    }

    /// Advances simulation by `dt` seconds using cached/synthetic terrain.
    pub fn step(&mut self, dt: f64) -> StepMetrics {
        self.step_internal(dt, None)
    }

    /// Advances simulation by `dt` seconds while pulling uncached terrain blocks
    /// lazily through `sampler` when called on a standalone instance (not holding global locks).
    pub fn step_with_sampler(&mut self, dt: f64, sampler: &dyn HostTerrainSampler) -> StepMetrics {
        if WRITE_LOCK_DEPTH.with(|d| d.get() > 0) || READ_LOCK_DEPTH.with(|d| d.get() > 0) {
            return self.step_internal(dt, None);
        }
        self.step_internal(dt, Some(sampler))
    }

    fn reconcile_events(&mut self, events: &[PhysicsEvent]) {
        for ev in events {
            match ev {
                PhysicsEvent::JointBroken { joint, .. } => {
                    self.joints.remove(&JointHandle(*joint));
                }
                PhysicsEvent::GridSplit { parent, children } => {
                    let parent_handle = SubGridHandle(*parent);
                    let old_names = self
                        .voxel_block_names
                        .remove(&parent_handle)
                        .unwrap_or_default();

                    if let Some(parent_grid) = self.world.body_grid(*parent) {
                        let mut kept = BTreeMap::new();
                        for (vc, _) in parent_grid.iter_voxels() {
                            let c = [vc.x, vc.y, vc.z];
                            if let Some(name) = old_names.get(&c) {
                                kept.insert(c, name.clone());
                            }
                        }
                        self.voxel_block_names.insert(parent_handle, kept);
                        self.bump_geometry_revision(*parent);
                    } else {
                        self.body_force_providers.remove(&parent_handle);
                        self.body_propulsions.remove(&parent_handle);
                        self.geometry_revisions.remove(parent);
                    }

                    for &child_id in children {
                        let child_handle = SubGridHandle(child_id);
                        if let Some(child_grid) = self.world.body_grid(child_id) {
                            let mut child_names = BTreeMap::new();
                            for (vc, _) in child_grid.iter_voxels() {
                                let c = [vc.x, vc.y, vc.z];
                                if let Some(name) = old_names.get(&c) {
                                    child_names.insert(c, name.clone());
                                }
                            }
                            self.voxel_block_names.insert(child_handle, child_names);
                            self.bump_geometry_revision(child_id);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn step_internal(&mut self, dt: f64, sampler: Option<&dyn HostTerrainSampler>) -> StepMetrics {
        let batch = self.prepare_external_forces(dt);
        let evaluated = Self::evaluate_prepared_forces(&batch);
        self.apply_evaluated_forces_and_step(dt, evaluated, sampler)
    }

    /// Snapshots active dynamic bodies and `Arc`-cloned force/propulsion handles
    /// so external plugin callbacks can execute without holding any `PhysicsApi` lock.
    pub fn prepare_external_forces(&self, dt: f64) -> PreparedExternalForceBatch {
        let step_index = self.world.step_index();
        let wind = self.ambient_wind_world.to_array();
        let snapshots = self.world.snapshots();
        let mut items = Vec::with_capacity(snapshots.len());

        for snap in snapshots {
            if snap.kind != BodyKind::Dynamic || snap.mass <= 0.0 {
                continue;
            }
            let handle = SubGridHandle(snap.id);
            let ctx = ForceQueryContext {
                world_position: snap.transform.position.to_array(),
                world_rotation: [
                    snap.transform.rotation.x,
                    snap.transform.rotation.y,
                    snap.transform.rotation.z,
                    snap.transform.rotation.w,
                ],
                linear_velocity_world: snap.linear_velocity.to_array(),
                angular_velocity_world: snap.angular_velocity.to_array(),
                mass_kg: snap.mass,
                com_local: snap.com_local.to_array(),
                ambient_wind_world: wind,
                dt,
                step_index,
            };
            let body_providers = self
                .body_force_providers
                .get(&handle)
                .cloned()
                .unwrap_or_default();
            let propulsions = self
                .body_propulsions
                .get(&handle)
                .cloned()
                .unwrap_or_default();

            if self.global_force_providers.is_empty()
                && body_providers.is_empty()
                && propulsions.is_empty()
            {
                continue;
            }

            items.push(PreparedBodyForceItem {
                body_id: snap.id,
                handle,
                transform: snap.transform,
                ctx,
                body_providers,
                propulsions,
            });
        }

        PreparedExternalForceBatch {
            dt,
            global_providers: self.global_force_providers.clone(),
            items,
        }
    }

    /// Evaluates prepared external `ForceProvider` and `Propulsion` callbacks completely
    /// lock-free relative to `GLOBAL_PHYSICS_API`, preventing reentrant read-lock deadlocks.
    pub fn evaluate_prepared_forces(batch: &PreparedExternalForceBatch) -> EvaluatedExternalForces {
        let mut forces: Vec<(BodyId, Vec3d, Vec3d)> = Vec::new();
        let mut torques: Vec<(BodyId, Vec3d)> = Vec::new();
        let mut plumes: Vec<ThrusterPlumeState> = Vec::new();
        let mut fuel_burn_total = 0.0_f64;

        let collect_contribution =
            |body_id: BodyId,
             transform: Transformd,
             contrib: ForceContribution,
             forces_out: &mut Vec<(BodyId, Vec3d, Vec3d)>,
             torques_out: &mut Vec<(BodyId, Vec3d)>| {
                let f = Vec3d::from_array(contrib.force);
                let tau = Vec3d::from_array(contrib.torque);
                if contrib.is_local_space {
                    let f_world = transform.transform_dir(f);
                    let pt_world = match contrib.application_point {
                        Some(lp) => transform.transform_point(Vec3d::from_array(lp)),
                        None => transform.position,
                    };
                    if f_world.is_finite() && f_world.length_sq() > 1e-18 {
                        forces_out.push((body_id, f_world, pt_world));
                    }
                    let tau_world = transform.transform_dir(tau);
                    if tau_world.is_finite() && tau_world.length_sq() > 1e-18 {
                        torques_out.push((body_id, tau_world));
                    }
                } else {
                    let pt_world = match contrib.application_point {
                        Some(wp) => Vec3d::from_array(wp),
                        None => transform.position,
                    };
                    if f.is_finite() && f.length_sq() > 1e-18 {
                        forces_out.push((body_id, f, pt_world));
                    }
                    if tau.is_finite() && tau.length_sq() > 1e-18 {
                        torques_out.push((body_id, tau));
                    }
                }
            };

        for item in &batch.items {
            for provider in &batch.global_providers {
                if let Some(contrib) = provider.compute_force(item.handle, &item.ctx) {
                    collect_contribution(
                        item.body_id,
                        item.transform,
                        contrib,
                        &mut forces,
                        &mut torques,
                    );
                }
            }
            for provider in &item.body_providers {
                if let Some(contrib) = provider.compute_force(item.handle, &item.ctx) {
                    collect_contribution(
                        item.body_id,
                        item.transform,
                        contrib,
                        &mut forces,
                        &mut torques,
                    );
                }
            }
            for slot in &item.propulsions {
                let Ok(guard) = slot.unit.lock() else {
                    continue;
                };
                let out = guard.evaluate_thrust(item.handle, &item.ctx);
                if out.fuel_burn_rate.is_finite() && out.fuel_burn_rate > 0.0 {
                    fuel_burn_total += out.fuel_burn_rate * batch.dt.max(0.0);
                }
                let f_local = Vec3d::from_array(out.local_force);
                let thrust_n = f_local.length();
                if thrust_n > 1.0 {
                    let exhaust_dir = (-f_local / thrust_n).to_array();
                    let inferred_throttle = guard
                        .throttle()
                        .abs()
                        .max((thrust_n / 20_000.0).clamp(0.15, 1.0));
                    plumes.push(ThrusterPlumeState {
                        grid: item.handle,
                        local_mount: out.local_mount,
                        local_exhaust_dir: exhaust_dir,
                        throttle: inferred_throttle,
                        thrust_n,
                    });
                }
                collect_contribution(
                    item.body_id,
                    item.transform,
                    ForceContribution::local_force_and_torque(
                        out.local_force,
                        out.local_torque,
                        out.local_mount,
                    ),
                    &mut forces,
                    &mut torques,
                );
            }
        }

        EvaluatedExternalForces {
            forces,
            torques,
            plumes,
            fuel_burn_total,
        }
    }

    /// Collects uncached block coordinates in dynamic-body swept trajectories so callers
    /// can prefetch terrain samples outside any `GLOBAL_PHYSICS_API` lock.
    pub fn prepare_uncached_terrain_cells(&self, dt: f64) -> Vec<[i32; 3]> {
        self.prepare_uncached_terrain_cells_with_forces(dt, None)
    }

    /// Trajectory-aware variant of [`Self::prepare_uncached_terrain_cells`] that also accounts
    /// for pre-evaluated external force impulses on dynamic bodies before stepping.
    pub fn prepare_uncached_terrain_cells_with_forces(
        &self,
        dt: f64,
        evaluated: Option<&EvaluatedExternalForces>,
    ) -> Vec<[i32; 3]> {
        let empty = BTreeSet::new();
        self.prepare_uncached_terrain_cells_with_forces_skipping(dt, evaluated, &empty)
    }

    /// Substep-accurate swept trajectory variant skipping cells already queried in the
    /// current lock-free prefetch pass (preserving unloaded chunk `None` responses).
    pub fn prepare_uncached_terrain_cells_with_forces_skipping(
        &self,
        dt: f64,
        evaluated: Option<&EvaluatedExternalForces>,
        skip: &BTreeSet<[i32; 3]>,
    ) -> Vec<[i32; 3]> {
        let clamped_dt = dt.abs().clamp(0.0, 0.25);
        let cfg = self.world.config();
        let max_speed = cfg.max_linear_speed.max(1.0);
        let substeps = cfg.substeps.max(1) as usize;
        let sub_dt = clamped_dt / (substeps as f64);
        let gravity = cfg.gravity;
        let mut net_forces: BTreeMap<BodyId, Vec3d> = BTreeMap::new();
        if let Some(ev) = evaluated {
            for &(bid, force, _) in &ev.forces {
                *net_forces.entry(bid).or_insert(Vec3d::ZERO) += force;
            }
        }
        let mut envelopes = Vec::new();
        for snap in self.world.snapshots() {
            if snap.kind != BodyKind::Dynamic || snap.world_aabb.is_empty() {
                continue;
            }
            let radius = (snap.world_aabb.max - snap.world_aabb.min).length() * 0.5;
            let angular_pad =
                (snap.angular_velocity.length() * radius * clamped_dt).clamp(0.0, 8.0);
            let pad = 2.0 + angular_pad;
            let ext_accel = if snap.mass > 1e-6
                && let Some(&force) = net_forces.get(&snap.id)
            {
                force * (1.0 / snap.mass)
            } else {
                Vec3d::ZERO
            };
            let accel = gravity + ext_accel;
            if clamped_dt <= 1e-9
                || (snap.linear_velocity.length_sq() < 1e-6 && accel.length_sq() < 1e-6)
            {
                envelopes.push(snap.world_aabb.expand_margin(pad));
                continue;
            }
            // Mirror Symplectic Euler substep kinematics so even extreme hard-braking
            // (high initial speed + opposing deceleration) covers its early forward travel.
            let mut vel = snap.linear_velocity;
            let mut cur_shift = Vec3d::ZERO;
            for _ in 0..substeps {
                vel += accel * sub_dt;
                let v_len = vel.length();
                if v_len > max_speed && v_len > 1e-9 {
                    vel *= max_speed / v_len;
                }
                let next_shift = cur_shift + vel * sub_dt;
                let seg_travel = (next_shift - cur_shift).length();
                if seg_travel <= 3.0 {
                    let s0_min = snap.world_aabb.min + cur_shift;
                    let s0_max = snap.world_aabb.max + cur_shift;
                    let s1_min = snap.world_aabb.min + next_shift;
                    let s1_max = snap.world_aabb.max + next_shift;
                    let swept = Aabbd::new(s0_min.min(s1_min), s0_max.max(s1_max));
                    envelopes.push(swept.expand_margin(pad));
                } else {
                    let divs = (seg_travel / 3.0).ceil().clamp(2.0, 32.0) as usize;
                    for d in 0..=divs {
                        let alpha = d as f64 / divs as f64;
                        let shift = cur_shift + (next_shift - cur_shift) * alpha;
                        let seg =
                            Aabbd::new(snap.world_aabb.min + shift, snap.world_aabb.max + shift);
                        envelopes.push(seg.expand_margin(pad));
                    }
                }
                cur_shift = next_shift;
            }
        }
        if envelopes.is_empty() {
            return Vec::new();
        }
        self.terrain
            .collect_uncached_cells_skipping(&envelopes, 131_072, skip)
    }

    /// Applies pre-evaluated external forces, PD joint servo torques, steps the physics world,
    /// and reconciles metadata for broken joints or split sub-grids.
    pub fn apply_evaluated_forces_and_step(
        &mut self,
        dt: f64,
        evaluated: EvaluatedExternalForces,
        sampler: Option<&dyn HostTerrainSampler>,
    ) -> StepMetrics {
        self.flush_pending_propulsion_ops();
        self.server_tick_counter = self.server_tick_counter.wrapping_add(1);
        self.active_plumes = evaluated.plumes;
        self.last_fuel_burn_total = evaluated.fuel_burn_total;

        for (bid, force, world_pt) in evaluated.forces {
            self.world.apply_force_at_world_point(bid, force, world_pt);
        }
        for (bid, torque) in evaluated.torques {
            self.world.apply_torque(bid, torque);
        }
        self.apply_joint_motor_torques();

        let in_global_lock =
            WRITE_LOCK_DEPTH.with(|d| d.get() > 0) || READ_LOCK_DEPTH.with(|d| d.get() > 0);
        let safe_sampler = if in_global_lock { None } else { sampler };
        let metrics = match safe_sampler {
            Some(s) => {
                let mut view = self.terrain.with_sampler(s);
                self.world.step(dt, &mut view)
            }
            None => self.world.step(dt, &mut self.terrain),
        };

        let mut events = std::mem::take(&mut self.pending_step_events);
        let step_events = self.world.drain_events();
        self.reconcile_events(&step_events);
        events.extend(step_events);
        self.last_events = events;
        self.last_metrics = metrics;
        metrics
    }

    fn apply_joint_motor_torques(&mut self) {
        let mut motor_torques: Vec<(BodyId, Vec3d)> = Vec::new();
        for constraint in self.joints.values() {
            if let JointConstraint::Hinge(h) = constraint
                && let Some(motor) = h.motor
                && let (Some(sa), Some(sb)) = (
                    self.world.body_snapshot(h.body_a.body_id()),
                    self.world.body_snapshot(h.body_b.body_id()),
                )
            {
                let axis_world = sa
                    .transform
                    .transform_dir(Vec3d::from_array(h.axis_a))
                    .normalize_or(Vec3d::Y);
                let rel_q = (sa.transform.rotation.conjugate() * sb.transform.rotation).normalize();
                let rot_vec = sa.transform.transform_dir(rel_q.to_rotation_vector());
                let cur_angle = rot_vec.dot(axis_world);
                let rel_omega = (sb.angular_velocity - sa.angular_velocity).dot(axis_world);
                let raw_torque = motor.stiffness * (motor.target_angle_rad - cur_angle)
                    - motor.damping * rel_omega;
                let max_t = motor.max_torque_nm.abs();
                let clamped_t = raw_torque.clamp(-max_t, max_t);
                if clamped_t.is_finite() && clamped_t.abs() > 1e-9 {
                    let tau = axis_world * clamped_t;
                    motor_torques.push((h.body_b.body_id(), tau));
                    motor_torques.push((h.body_a.body_id(), -tau));
                }
            }
        }
        for (bid, torque) in motor_torques {
            self.world.apply_torque(bid, torque);
        }
    }

    fn allocate_custom_material(&mut self, placement: VoxelPlacement) -> MaterialId {
        let mat = VoxelMaterial {
            density: placement.mass_kg.max(1.0),
            friction: placement.friction.clamp(0.0, 2.0),
            restitution: placement.restitution.clamp(0.0, 1.0),
            linear_drag: 0.02,
            angular_drag: 0.05,
        };
        // Reuse standard material slots when parameters match closely.
        for std_id in [
            MaterialId::STONE,
            MaterialId::WOOD,
            MaterialId::IRON,
            MaterialId::SLIME,
            MaterialId::ICE,
            MaterialId::WOOL,
        ] {
            let m = self.world.materials().get(std_id);
            if (m.density - mat.density).abs() < 1e-3
                && (m.friction - mat.friction).abs() < 1e-3
                && (m.restitution - mat.restitution).abs() < 1e-3
            {
                return std_id;
            }
        }
        let id = MaterialId(self.next_custom_material);
        self.next_custom_material = self.next_custom_material.saturating_add(1).max(4096);
        self.world.register_material(id, mat);
        id
    }

    fn cast_ray_world_blocks(
        &self,
        origin: Vec3d,
        unit_dir: Vec3d,
        max_dist: f64,
    ) -> Option<CombinedRaycastHit> {
        // 3D Amanatides-Woo DDA grid traversal over TerrainCache.
        let mut bx = origin.x.floor() as i32;
        let mut by = origin.y.floor() as i32;
        let mut bz = origin.z.floor() as i32;

        let step_x = if unit_dir.x > 0.0 {
            1
        } else if unit_dir.x < 0.0 {
            -1
        } else {
            0
        };
        let step_y = if unit_dir.y > 0.0 {
            1
        } else if unit_dir.y < 0.0 {
            -1
        } else {
            0
        };
        let step_z = if unit_dir.z > 0.0 {
            1
        } else if unit_dir.z < 0.0 {
            -1
        } else {
            0
        };

        let next_boundary = |coord: i32, step: i32| -> f64 {
            if step > 0 {
                f64::from(coord + 1)
            } else {
                f64::from(coord)
            }
        };

        let mut t_max_x = if step_x != 0 {
            (next_boundary(bx, step_x) - origin.x) / unit_dir.x
        } else {
            f64::INFINITY
        };
        let mut t_max_y = if step_y != 0 {
            (next_boundary(by, step_y) - origin.y) / unit_dir.y
        } else {
            f64::INFINITY
        };
        let mut t_max_z = if step_z != 0 {
            (next_boundary(bz, step_z) - origin.z) / unit_dir.z
        } else {
            f64::INFINITY
        };

        let t_delta_x = if step_x != 0 {
            (1.0 / unit_dir.x).abs()
        } else {
            f64::INFINITY
        };
        let t_delta_y = if step_y != 0 {
            (1.0 / unit_dir.y).abs()
        } else {
            f64::INFINITY
        };
        let t_delta_z = if step_z != 0 {
            (1.0 / unit_dir.z).abs()
        } else {
            f64::INFINITY
        };

        let max_steps = ((max_dist * 3.0).ceil() as usize).clamp(16, 8192);
        for _ in 0..max_steps {
            if let CachedBlockCell::Solid { collision, .. } = self.terrain.peek_cell(bx, by, bz) {
                let world_aabb = Aabbd::new(
                    Vec3d::new(
                        f64::from(bx) + collision.local_aabb.min.x,
                        f64::from(by) + collision.local_aabb.min.y,
                        f64::from(bz) + collision.local_aabb.min.z,
                    ),
                    Vec3d::new(
                        f64::from(bx) + collision.local_aabb.max.x,
                        f64::from(by) + collision.local_aabb.max.y,
                        f64::from(bz) + collision.local_aabb.max.z,
                    ),
                );
                if let Some((t, normal)) =
                    ray_intersect_aabb(origin, unit_dir, max_dist, &world_aabb)
                {
                    let pt = origin + unit_dir * t;
                    return Some(CombinedRaycastHit {
                        distance: t,
                        point: pt.to_array(),
                        normal: normal.to_array(),
                        target: RaycastHitTarget::WorldBlock {
                            block_pos: [bx, by, bz],
                            material: collision.material,
                        },
                    });
                }
            }

            let t_next = t_max_x.min(t_max_y).min(t_max_z);
            if t_next > max_dist {
                break;
            }
            if t_max_x <= t_max_y && t_max_x <= t_max_z {
                bx += step_x;
                t_max_x += t_delta_x;
            } else if t_max_y <= t_max_z {
                by += step_y;
                t_max_y += t_delta_y;
            } else {
                bz += step_z;
                t_max_z += t_delta_z;
            }
        }
        None
    }

    fn cast_ray_subgrids(
        &self,
        origin: Vec3d,
        unit_dir: Vec3d,
        max_dist: f64,
        exclude: Option<SubGridHandle>,
    ) -> Option<CombinedRaycastHit> {
        if exclude.is_none() {
            return self
                .world
                .raycast(origin, unit_dir, max_dist)
                .map(|hit| CombinedRaycastHit {
                    distance: hit.distance,
                    point: hit.point.to_array(),
                    normal: hit.normal.to_array(),
                    target: RaycastHitTarget::SubGridVoxel {
                        grid: SubGridHandle(hit.body_id),
                        local_coord: hit.voxel_coord.map(|c| [c.x, c.y, c.z]),
                        box_index: hit.box_index,
                    },
                });
        }

        let mut best: Option<CombinedRaycastHit> = None;
        let mut best_dist = max_dist;

        for snap in self.world.snapshots() {
            let handle = SubGridHandle(snap.id);
            if exclude == Some(handle) {
                continue;
            }
            if snap
                .world_aabb
                .ray_intersect(origin, unit_dir, best_dist)
                .is_none()
            {
                continue;
            }
            let Some(grid) = self.world.body_grid(snap.id) else {
                continue;
            };
            let com = grid.com_local();
            for (box_index, lb) in grid.boxes().iter().enumerate() {
                let obb = Obbd::new(
                    snap.transform.transform_point(lb.center_local() - com),
                    lb.half_extents(),
                    snap.transform.rotation,
                );
                if let Some((t, normal)) = obb.ray_intersect(origin, unit_dir, best_dist) {
                    best_dist = t;
                    let pt = origin + unit_dir * t;
                    let inside_world = pt - normal * 1e-3;
                    let local_pt = snap.transform.inverse_transform_point(inside_world) + com;
                    let vc = VoxelCoord::new(
                        local_pt.x.floor() as i32,
                        local_pt.y.floor() as i32,
                        local_pt.z.floor() as i32,
                    );
                    let local_coord = grid.has_voxel(vc).then_some([vc.x, vc.y, vc.z]);
                    best = Some(CombinedRaycastHit {
                        distance: t,
                        point: pt.to_array(),
                        normal: normal.to_array(),
                        target: RaycastHitTarget::SubGridVoxel {
                            grid: handle,
                            local_coord,
                            box_index,
                        },
                    });
                }
            }
        }
        best
    }
}

impl SubGridRegistry for PhysicsApi {
    fn assemble_subgrid(&mut self, desc: SubGridAssemblyDesc) -> Result<SubGridHandle, String> {
        if desc.voxels.is_empty() {
            return Err("SubGridAssemblyDesc requires at least one voxel".to_owned());
        }
        let origin = Vec3d::from_array(desc.world_origin);
        let rot = Quatd::new(
            desc.initial_rotation[0],
            desc.initial_rotation[1],
            desc.initial_rotation[2],
            desc.initial_rotation[3],
        );
        if !origin.is_finite() || !rot.is_finite() {
            return Err("SubGridAssemblyDesc contains non-finite transform values".to_owned());
        }

        let mut grid = VoxelSubGrid::new();
        let mut names = BTreeMap::new();
        for placement in &desc.voxels {
            let mat_id = self.allocate_custom_material(*placement);
            grid.set_voxel(
                VoxelCoord::new(
                    placement.local_coord[0],
                    placement.local_coord[1],
                    placement.local_coord[2],
                ),
                Some(VoxelEntry::from_material(mat_id)),
                self.world.materials(),
            );
            let default_name = match mat_id {
                MaterialId::WOOD => "minecraft:oak_planks",
                MaterialId::IRON => "minecraft:iron_block",
                MaterialId::SLIME => "minecraft:slime_block",
                MaterialId::ICE => "minecraft:ice",
                MaterialId::WOOL => "minecraft:white_wool",
                _ => "minecraft:stone",
            };
            names.insert(placement.local_coord, default_name.to_owned());
        }
        grid.rebuild_dirty_colliders();
        let body_desc = BodyDesc {
            kind: BodyKind::Dynamic,
            transform: Transformd::new(origin, rot.normalize()),
            grid,
            enable_ccd: desc.enable_ccd,
            ..BodyDesc::default()
        };
        let id = self.world.spawn_body(body_desc);
        let handle = SubGridHandle(id);
        self.voxel_block_names.insert(handle, names);
        self.geometry_revisions.insert(id, 1);
        Ok(handle)
    }

    fn disassemble_subgrid(&mut self, grid: SubGridHandle) -> Option<Vec<DisassembledVoxel>> {
        let snap = self.world.body_snapshot(grid.body_id())?;
        let com = snap.com_local;
        let voxels: Vec<DisassembledVoxel> = {
            let sub = self.world.body_grid(grid.body_id())?;
            let names = self.voxel_block_names.get(&grid);
            sub.iter_voxels()
                .map(|(vc, entry)| {
                    let lc = [vc.x, vc.y, vc.z];
                    let world_pt = snap.transform.transform_point(vc.center_local() - com);
                    let world_coord = [
                        world_pt.x.floor() as i32,
                        world_pt.y.floor() as i32,
                        world_pt.z.floor() as i32,
                    ];
                    let block_id = names
                        .and_then(|m| m.get(&lc))
                        .cloned()
                        .unwrap_or_else(|| "minecraft:stone".to_owned());
                    DisassembledVoxel {
                        local_coord: lc,
                        world_coord,
                        block_id,
                        material: entry.material,
                    }
                })
                .collect()
        };

        self.world.despawn_body(grid.body_id());
        self.body_force_providers.remove(&grid);
        self.body_propulsions.remove(&grid);
        self.pending_propulsion_ops.retain(|(g, _, _)| *g != grid);
        self.voxel_block_names.remove(&grid);
        self.geometry_revisions.remove(&grid.body_id());
        self.joints.retain(|_, c| match c {
            JointConstraint::Hinge(h) => h.body_a != grid && h.body_b != grid,
            JointConstraint::Slider(s) => s.body_a != grid && s.body_b != Some(grid),
            JointConstraint::Distance(d) => d.body_a != grid && d.body_b != Some(grid),
            JointConstraint::FixedWeld(w) => w.body_a != grid && w.body_b != Some(grid),
        });
        Some(voxels)
    }

    fn set_voxel(&mut self, grid: SubGridHandle, placement: VoxelPlacement) -> Result<(), String> {
        if self.world.body_snapshot(grid.body_id()).is_none() {
            return Err("sub-grid handle is stale or missing".to_owned());
        }
        let mat_id = self.allocate_custom_material(placement);
        let changed = self.world.set_body_voxel(
            grid.body_id(),
            VoxelCoord::new(
                placement.local_coord[0],
                placement.local_coord[1],
                placement.local_coord[2],
            ),
            Some(VoxelEntry::from_material(mat_id)),
        );
        if changed {
            self.voxel_block_names
                .entry(grid)
                .or_default()
                .entry(placement.local_coord)
                .or_insert_with(|| "minecraft:stone".to_owned());
            self.bump_geometry_revision(grid.body_id());
            let ev = self.world.drain_events();
            self.reconcile_events(&ev);
            self.pending_step_events.extend(ev);
        }
        Ok(())
    }

    fn remove_voxel(&mut self, grid: SubGridHandle, local_coord: [i32; 3]) -> Result<(), String> {
        if self.world.body_snapshot(grid.body_id()).is_none() {
            return Err("sub-grid handle is stale or missing".to_owned());
        }
        let changed = self.world.set_body_voxel(
            grid.body_id(),
            VoxelCoord::new(local_coord[0], local_coord[1], local_coord[2]),
            None,
        );
        if let Some(map) = self.voxel_block_names.get_mut(&grid) {
            map.remove(&local_coord);
        }
        if changed {
            self.bump_geometry_revision(grid.body_id());
            let ev = self.world.drain_events();
            self.reconcile_events(&ev);
            self.pending_step_events.extend(ev);
        }
        Ok(())
    }

    fn subgrid_state(&self, grid: SubGridHandle) -> Option<SubGridState> {
        self.world
            .body_snapshot(grid.body_id())
            .map(SubGridState::from)
    }

    fn active_subgrids(&self) -> Vec<SubGridHandle> {
        self.world
            .snapshots()
            .into_iter()
            .map(|s| SubGridHandle(s.id))
            .collect()
    }

    fn apply_impulse(
        &mut self,
        grid: SubGridHandle,
        impulse_world: [f64; 3],
        world_point: Option<[f64; 3]>,
    ) -> bool {
        let Some(snap) = self.world.body_snapshot(grid.body_id()) else {
            return false;
        };
        let pt = world_point.map_or(snap.transform.position, Vec3d::from_array);
        self.world.apply_impulse_at_world_point(
            grid.body_id(),
            Vec3d::from_array(impulse_world),
            pt,
        )
    }

    fn sleep_subgrid(&mut self, grid: SubGridHandle) -> bool {
        self.world.sleep_body(grid.body_id())
    }

    fn wake_subgrid(&mut self, grid: SubGridHandle) -> bool {
        self.world.wake_body(grid.body_id())
    }
}

impl JointRegistry for PhysicsApi {
    fn add_joint(&mut self, constraint: JointConstraint) -> Option<JointHandle> {
        let spec = constraint.to_core_spec();
        let id = self.world.add_joint(spec)?;
        let handle = JointHandle(id);
        self.joints.insert(handle, constraint);
        Some(handle)
    }

    fn remove_joint(&mut self, handle: JointHandle) -> bool {
        self.joints.remove(&handle);
        self.world.remove_joint(handle.joint_id())
    }
}

impl PhysicsRaycast for PhysicsApi {
    fn cast_ray(&self, query: &RayQuery) -> Option<CombinedRaycastHit> {
        let origin = Vec3d::from_array(query.origin);
        let unit_dir = Vec3d::from_array(query.direction).try_normalize()?;
        if !origin.is_finite() || !query.max_distance.is_finite() || query.max_distance < 0.0 {
            return None;
        }

        let world_hit = if query.filter.include_world_blocks {
            self.cast_ray_world_blocks(origin, unit_dir, query.max_distance)
        } else {
            None
        };

        let subgrid_limit = world_hit.map_or(query.max_distance, |h| h.distance);
        let subgrid_hit = if query.filter.include_subgrids {
            self.cast_ray_subgrids(
                origin,
                unit_dir,
                subgrid_limit,
                query.filter.exclude_subgrid,
            )
        } else {
            None
        };

        match (world_hit, subgrid_hit) {
            (Some(w), Some(s)) => {
                if s.distance <= w.distance {
                    Some(s)
                } else {
                    Some(w)
                }
            }
            (Some(w), None) => Some(w),
            (None, Some(s)) => Some(s),
            (None, None) => None,
        }
    }
}

/// Prepared snapshot item for lock-free evaluation of one dynamic body's external forces.
pub struct PreparedBodyForceItem {
    body_id: BodyId,
    handle: SubGridHandle,
    transform: Transformd,
    ctx: ForceQueryContext,
    body_providers: Vec<SharedForceProvider>,
    propulsions: Vec<PropulsionSlot>,
}

/// Lock-free batch prepared by [`PhysicsApi::prepare_external_forces`].
pub struct PreparedExternalForceBatch {
    dt: f64,
    global_providers: Vec<SharedForceProvider>,
    items: Vec<PreparedBodyForceItem>,
}

/// Evaluated world-space forces, torques, and plumes produced by
/// [`PhysicsApi::evaluate_prepared_forces`].
pub struct EvaluatedExternalForces {
    forces: Vec<(BodyId, Vec3d, Vec3d)>,
    torques: Vec<(BodyId, Vec3d)>,
    plumes: Vec<ThrusterPlumeState>,
    fuel_burn_total: f64,
}

static GLOBAL_PHYSICS_API: OnceLock<RwLock<PhysicsApi>> = OnceLock::new();

thread_local! {
    static WRITE_LOCK_DEPTH: Cell<usize> = const { Cell::new(0) };
    static READ_LOCK_DEPTH: Cell<usize> = const { Cell::new(0) };
}

struct WriteDepthGuard;

impl WriteDepthGuard {
    fn enter() -> Self {
        READ_LOCK_DEPTH.with(|rd| {
            assert!(
                rd.get() == 0,
                "reentrant `with_physics_api` write lock requested while holding `with_physics_api_read` on the same thread; release the read guard first"
            );
        });
        WRITE_LOCK_DEPTH.with(|d| {
            let cur = d.get();
            assert!(
                cur == 0,
                "reentrant `with_physics_api` write lock detected on the same thread; release the outer write guard before re-entering"
            );
            d.set(cur + 1);
        });
        Self
    }
}

impl Drop for WriteDepthGuard {
    fn drop(&mut self) {
        WRITE_LOCK_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

struct ReadDepthGuard;

impl ReadDepthGuard {
    fn enter() -> Self {
        WRITE_LOCK_DEPTH.with(|d| {
            assert!(
                d.get() == 0,
                "reentrant `with_physics_api_read` called while holding `with_physics_api` write lock; use `step_global_physics*` so external callbacks run lock-free"
            );
        });
        READ_LOCK_DEPTH.with(|rd| rd.set(rd.get() + 1));
        Self
    }
}

impl Drop for ReadDepthGuard {
    fn drop(&mut self) {
        READ_LOCK_DEPTH.with(|rd| rd.set(rd.get().saturating_sub(1)));
    }
}

fn global_lock() -> &'static RwLock<PhysicsApi> {
    GLOBAL_PHYSICS_API.get_or_init(|| RwLock::new(PhysicsApi::default()))
}

fn execute_prepared_propulsion_ops(ops: Vec<PreparedPropulsionOp>) {
    let mut deferred: Vec<(SubGridHandle, String, PropulsionOp)> = Vec::new();
    for (grid, pid, unit, op) in ops {
        match unit.try_lock() {
            Ok(mut guard) => match op {
                PropulsionOp::SetThrottle(throttle) => guard.set_throttle(throttle),
                PropulsionOp::SetVector(pitch_rad, yaw_rad) => {
                    guard.set_vector_angles(pitch_rad, yaw_rad);
                }
            },
            Err(TryLockError::WouldBlock) => {
                deferred.push((grid, pid, op));
            }
            Err(TryLockError::Poisoned(e)) => {
                let mut guard = e.into_inner();
                match op {
                    PropulsionOp::SetThrottle(throttle) => guard.set_throttle(throttle),
                    PropulsionOp::SetVector(pitch_rad, yaw_rad) => {
                        guard.set_vector_angles(pitch_rad, yaw_rad);
                    }
                }
            }
        }
    }
    if !deferred.is_empty() {
        let _depth = WriteDepthGuard::enter();
        let mut guard = global_lock()
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.requeue_deferred_propulsion_ops(deferred);
    }
}

fn flush_global_pending_propulsion_ops() {
    let pending = {
        let _depth = WriteDepthGuard::enter();
        let mut guard = global_lock()
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.take_pending_propulsion_ops_for_unlock()
    };
    if !pending.is_empty() {
        execute_prepared_propulsion_ops(pending);
    }
}

/// Acquires exclusive mutable access to the global [`PhysicsApi`] runtime.
///
/// Designed for third-party Horizon mods (aviation, maritime, vehicles) to register
/// [`ForceProvider`]s, [`Propulsion`] units, [`JointConstraint`]s, and assemble sub-grids.
/// Any deferred propulsion trait mutations queued inside the write lock are executed cleanly
/// after the global write lock is released.
pub fn with_physics_api<R>(f: impl FnOnce(&mut PhysicsApi) -> R) -> R {
    let (result, pending) = {
        let _depth = WriteDepthGuard::enter();
        let mut guard = global_lock()
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let res = f(&mut guard);
        let pending = guard.take_pending_propulsion_ops_for_unlock();
        (res, pending)
    };
    if !pending.is_empty() {
        execute_prepared_propulsion_ops(pending);
    }
    result
}

/// Acquires shared read-only access to the global [`PhysicsApi`] runtime for
/// zero-contention concurrent raycasts, state snapshots, and render extraction.
pub fn with_physics_api_read<R>(f: impl FnOnce(&PhysicsApi) -> R) -> R {
    let _depth = ReadDepthGuard::enter();
    let guard = global_lock()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(&guard)
}

/// Advances the global [`PhysicsApi`] simulation in three deadlock-free phases:
/// 1. Lock-free flush of any previously deferred propulsion commands + read-lock snapshot
/// 2. Lock-free evaluation of external [`ForceProvider`]/[`Propulsion`] callbacks and [`HostTerrainSampler`] prefetch
/// 3. Write lock to commit prefetched terrain and step with cached terrain (`None` sampler so host locks are never touched)
pub fn step_global_physics(dt: f64) -> StepMetrics {
    step_global_physics_with_sampler(dt, None)
}

/// Lock-free three-phase variant of [`step_global_physics`] using a live [`HostTerrainSampler`].
pub fn step_global_physics_with_sampler(
    dt: f64,
    sampler: Option<&dyn HostTerrainSampler>,
) -> StepMetrics {
    flush_global_pending_propulsion_ops();

    let batch = with_physics_api_read(|api| api.prepare_external_forces(dt));
    let evaluated = PhysicsApi::evaluate_prepared_forces(&batch);

    if let Some(s) = sampler {
        let mut attempted: BTreeSet<[i32; 3]> = BTreeSet::new();
        for _ in 0..8 {
            let uncached_cells = with_physics_api_read(|api| {
                api.prepare_uncached_terrain_cells_with_forces_skipping(
                    dt,
                    Some(&evaluated),
                    &attempted,
                )
            });
            if uncached_cells.is_empty() {
                break;
            }
            let reached_cap = uncached_cells.len() >= 131_072;
            let prefetched: Vec<PrefetchedTerrainSample> = uncached_cells
                .into_iter()
                .map(|c| {
                    attempted.insert(c);
                    (c, s.sample_block(c[0], c[1], c[2]))
                })
                .collect();
            with_physics_api(|api| {
                api.terrain_mut().insert_prefetched_samples(prefetched);
            });
            if !reached_cap {
                break;
            }
        }
    }

    with_physics_api(|api| api.apply_evaluated_forces_and_step(dt, evaluated, None))
}
