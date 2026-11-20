//! Clean-room pure-Rust voxel rigid-body & sub-grid physics engine.
//!
//! Built strictly on the Rust standard library (`std`) with zero external
//! dependencies and deterministic data structures (`BTreeMap`, `SlotArena`,
//! `Vec`) for bit-exact cross-platform replay.

#![deny(unsafe_code)]

pub mod arena;
pub mod broadphase;
pub mod ccd;
pub mod island;
pub mod math;
pub mod narrowphase;
pub mod solver;
pub mod voxel;
pub mod world;

pub use arena::{BodyId, JointId, SlotArena, SlotKey};
pub use broadphase::{DynamicBvh, SpatialHash16};
pub use ccd::{
    GjkDistanceResult, MAX_CA_ITERATIONS, SweptObb, ToiHit, conservative_advancement_obb_pair,
    gjk_obb_distance,
};
pub use island::{DisjointSet, Island, IslandBuilder};
pub use math::{
    Aabbd, Mat3d, Obbd, Quatd, Transformd, Vec3d, ray_intersect_aabb, ray_intersect_obb,
};
pub use narrowphase::{
    BoxContactResult, ContactManifold, ContactPoint, FeatureKey, closest_points_segments,
    collide_obb_aabb, collide_obb_obb, reduce_contacts_to_4,
};
pub use solver::{JointKind, JointSpec, SoftContactParams, SolverSoA};
pub use voxel::{
    ALL_FACES, FACE_NEG_X, FACE_NEG_Y, FACE_NEG_Z, FACE_POS_X, FACE_POS_Y, FACE_POS_Z, LocalBox,
    MaterialId, MaterialRegistry, VoxelBrick16, VoxelCoord, VoxelEntry, VoxelMaterial,
    VoxelSubGrid,
};
pub use world::{
    BodyDesc, BodyKind, BodySnapshot, EmptyWorldProvider, EntityCarryOutcome, EntitySweepResult,
    FlatWorldProvider, PhysicsConfig, PhysicsEvent, PhysicsWorld, RaycastHit, StepMetrics,
    WorldBlockCollision, WorldFluidSample, WorldVoxelProvider,
};
