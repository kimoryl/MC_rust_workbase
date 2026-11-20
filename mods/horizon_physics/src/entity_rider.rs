//! Entity riding, local-frame deck carry, stair step-up, proxy collision
//! exclusions, and client velocity-inheritance compensation.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use horizon_physics_core::{Aabbd, BodyId, EntitySweepResult, PhysicsWorld, Vec3d};
use mod_api::{
    ClientHost, CollisionExclusionsError, EntityMoveEvent, EntityRef, Flow, LevelRef, ServerHost,
};

/// Minimum allowed automatic step-up height for riding entities (meters).
pub const MIN_STEP_HEIGHT: f64 = 0.6;
/// Maximum allowed automatic step-up height for riding entities (meters).
pub const MAX_STEP_HEIGHT: f64 = 1.15;
/// Horizon API hard limit on per-player collision exclusion boxes.
pub const MAX_COLLISION_EXCLUSIONS: usize = 64;

/// Persistent foot-anchor record for an entity riding a physics sub-grid deck.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RiderAnchor {
    pub ground_body: BodyId,
    /// Foot contact point expressed in `ground_body`'s local CoM frame.
    pub anchor_local: Vec3d,
    /// Deck surface velocity at the anchor point (m/s).
    pub surface_velocity: Vec3d,
    /// Carry translation applied on the most recent move step (m).
    pub last_carry_delta: Vec3d,
    /// Carry yaw rotation applied on the most recent move step (rad).
    pub last_carry_yaw_delta: f64,
    /// Tick index when this anchor was last refreshed.
    pub last_seen_tick: u64,
}

/// Cached world-space entity pose synchronized from `level_tick_pre` / tests.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TrackedEntityPose {
    pub aabb: Aabbd,
    pub yaw_rad: f64,
    pub pitch_rad: f64,
    pub last_tick: u64,
}

/// Complete outcome of resolving an entity movement step against sub-grids.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityMoveResolution {
    pub outcome: EntitySweepResult,
    pub surface_velocity: Vec3d,
    pub dismount_velocity_kick: Option<[f32; 3]>,
    pub collision_exclusions: Vec<[f64; 6]>,
}

/// Pending host-side carry/exclusion action queued during `entity_move` and
/// flushed during `level_tick_post`.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingRiderAction {
    pub entity_id: u32,
    pub yaw_delta_deg: f32,
    pub collision_exclusions: Vec<[f64; 6]>,
    pub client_velocity_kick: Option<[f32; 3]>,
}

/// Manages entity deck anchors, local-frame movement sweeps, exclusion boxes,
/// and velocity inheritance across server and client ticks.
#[derive(Clone, Debug)]
pub struct EntityRiderManager {
    anchors: BTreeMap<u32, RiderAnchor>,
    tracked: BTreeMap<u32, TrackedEntityPose>,
    pending_actions: BTreeMap<u32, PendingRiderAction>,
    default_step_height: f64,
}

impl Default for EntityRiderManager {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityRiderManager {
    pub fn new() -> Self {
        Self {
            anchors: BTreeMap::new(),
            tracked: BTreeMap::new(),
            pending_actions: BTreeMap::new(),
            default_step_height: MAX_STEP_HEIGHT,
        }
    }

    pub fn clear(&mut self) {
        self.anchors.clear();
        self.tracked.clear();
        self.pending_actions.clear();
    }

    /// Sets the default stair step-up height, clamped to `[0.6, 1.15]` meters.
    pub fn set_default_step_height(&mut self, height: f64) {
        self.default_step_height = height.clamp(MIN_STEP_HEIGHT, MAX_STEP_HEIGHT);
    }

    #[inline]
    pub fn default_step_height(&self) -> f64 {
        self.default_step_height
    }

    /// Registers or refreshes an entity's world AABB and orientation for move resolution.
    pub fn update_entity_pose(
        &mut self,
        entity_id: u32,
        aabb: Aabbd,
        yaw_rad: f64,
        pitch_rad: f64,
        tick: u64,
    ) {
        self.tracked.insert(
            entity_id,
            TrackedEntityPose {
                aabb,
                yaw_rad,
                pitch_rad,
                last_tick: tick,
            },
        );
    }

    #[inline]
    pub fn anchor_of(&self, entity_id: u32) -> Option<RiderAnchor> {
        self.anchors.get(&entity_id).copied()
    }

    #[inline]
    pub fn tracked_pose(&self, entity_id: u32) -> Option<TrackedEntityPose> {
        self.tracked.get(&entity_id).copied()
    }

    #[inline]
    pub fn active_rider_count(&self) -> usize {
        self.anchors.len()
    }

    /// Drains pending actions queued by [`Self::on_entity_move`].
    pub fn drain_pending_actions(&mut self) -> Vec<PendingRiderAction> {
        let keys: Vec<u32> = self.pending_actions.keys().copied().collect();
        let mut out = Vec::with_capacity(keys.len());
        for k in keys {
            if let Some(act) = self.pending_actions.remove(&k) {
                out.push(act);
            }
        }
        out
    }

    /// Computes up to [`MAX_COLLISION_EXCLUSIONS`] finite world AABBs of nearby
    /// physics sub-grids overlapping `entity_aabb` expanded by `margin`, ordered
    /// nearest-first to the entity center so close hulls are never crowded out.
    pub fn compute_collision_exclusions(
        world: &PhysicsWorld,
        entity_aabb: Aabbd,
        margin: f64,
    ) -> Vec<[f64; 6]> {
        let query = entity_aabb.expand_margin(margin.max(0.25));
        let center = entity_aabb.center();
        let mut candidates: Vec<(f64, [f64; 6])> = Vec::new();
        for snap in world.snapshots() {
            if snap.world_aabb.is_empty() || !snap.world_aabb.intersects(&query) {
                continue;
            }
            let b = [
                snap.world_aabb.min.x,
                snap.world_aabb.min.y,
                snap.world_aabb.min.z,
                snap.world_aabb.max.x,
                snap.world_aabb.max.y,
                snap.world_aabb.max.z,
            ];
            if b.iter().all(|v| v.is_finite()) && b[0] <= b[3] && b[1] <= b[4] && b[2] <= b[5] {
                let dx = (snap.world_aabb.min.x - center.x)
                    .max(0.0)
                    .max(center.x - snap.world_aabb.max.x);
                let dy = (snap.world_aabb.min.y - center.y)
                    .max(0.0)
                    .max(center.y - snap.world_aabb.max.y);
                let dz = (snap.world_aabb.min.z - center.z)
                    .max(0.0)
                    .max(center.z - snap.world_aabb.max.z);
                candidates.push((dx * dx + dy * dy + dz * dz, b));
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        candidates
            .into_iter()
            .take(MAX_COLLISION_EXCLUSIONS)
            .map(|(_, b)| b)
            .collect()
    }

    /// Resolves an entity AABB sweep against all active and sleeping sub-grids
    /// with deck carry, rotating-deck yaw inheritance, stair step-up, and proxy
    /// collision exclusions.
    pub fn resolve_entity_move(
        &mut self,
        world: &PhysicsWorld,
        entity_id: u32,
        entity_aabb: Aabbd,
        requested_delta: Vec3d,
        step_height: f64,
        tick: u64,
    ) -> EntityMoveResolution {
        let prev_anchor = self.anchors.get(&entity_id).copied();
        let prev_ground = prev_anchor.map(|a| (a.ground_body, a.anchor_local));
        let clamped_step = step_height.clamp(MIN_STEP_HEIGHT, MAX_STEP_HEIGHT);

        let outcome = world.sweep_entity_against_bodies(
            entity_aabb,
            requested_delta,
            prev_ground,
            clamped_step,
        );

        let mut surface_velocity = Vec3d::ZERO;
        let mut dismount_velocity_kick = None;

        if let Some((ground_body, anchor_local)) = outcome.ground_body {
            if let Some(snap) = world.body_snapshot(ground_body) {
                let r_world = snap.transform.rotation.rotate_vec3(anchor_local);
                surface_velocity = snap.linear_velocity + snap.angular_velocity.cross(r_world);
            }
            self.anchors.insert(
                entity_id,
                RiderAnchor {
                    ground_body,
                    anchor_local,
                    surface_velocity,
                    last_carry_delta: outcome.carry_delta,
                    last_carry_yaw_delta: outcome.carry_yaw_delta,
                    last_seen_tick: tick,
                },
            );
        } else {
            if let Some(prev) = prev_anchor {
                // Entity jumped or stepped off a moving deck: inherit deck surface velocity.
                let kick_blocks_per_tick = prev.surface_velocity / 20.0;
                if kick_blocks_per_tick.is_finite() && kick_blocks_per_tick.length_sq() > 1e-6 {
                    dismount_velocity_kick = Some(clamp_player_kick(kick_blocks_per_tick));
                }
            }
            self.anchors.remove(&entity_id);
        }

        // Update tracked pose with resolved AABB and inherited yaw delta.
        let entry = self.tracked.entry(entity_id).or_insert(TrackedEntityPose {
            aabb: outcome.resolved_aabb,
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            last_tick: tick,
        });
        entry.aabb = outcome.resolved_aabb;
        entry.yaw_rad += outcome.carry_yaw_delta;
        entry.last_tick = tick;

        let collision_exclusions =
            Self::compute_collision_exclusions(world, outcome.resolved_aabb, 1.5);

        EntityMoveResolution {
            outcome,
            surface_velocity,
            dismount_velocity_kick,
            collision_exclusions,
        }
    }

    /// Hook entry point for [`mod_api::Hooks::entity_move`].
    ///
    /// Adjusts `event.delta` using [`PhysicsWorld::sweep_entity_against_bodies`]
    /// and queues yaw/exclusion updates for `level_tick_post`.
    pub fn on_entity_move(
        &mut self,
        world: &PhysicsWorld,
        event: &mut EntityMoveEvent<'_>,
        tick: u64,
    ) -> Flow {
        if world.body_count() == 0 {
            return Flow::Continue;
        }
        let Some(pose) = self.tracked.get(&event.entity).copied() else {
            return Flow::Continue;
        };

        let requested = Vec3d::from_array(*event.delta);
        if !requested.is_finite() {
            return Flow::Continue;
        }

        let res = self.resolve_entity_move(
            world,
            event.entity,
            pose.aabb,
            requested,
            self.default_step_height,
            tick,
        );

        *event.delta = res.outcome.applied_delta.to_array();

        let yaw_delta_deg = res.outcome.carry_yaw_delta.to_degrees() as f32;
        if yaw_delta_deg.abs() > 1e-5
            || !res.collision_exclusions.is_empty()
            || res.dismount_velocity_kick.is_some()
        {
            self.pending_actions.insert(
                event.entity,
                PendingRiderAction {
                    entity_id: event.entity,
                    yaw_delta_deg,
                    collision_exclusions: res.collision_exclusions,
                    client_velocity_kick: res.dismount_velocity_kick,
                },
            );
        }

        Flow::Continue
    }

    /// Synchronizes entity bounding boxes around active sub-grids from a live server level.
    pub fn snapshot_nearby_entities_from_level(
        &mut self,
        world: &PhysicsWorld,
        host: &dyn ServerHost,
        level: LevelRef,
        tick: u64,
    ) {
        if world.body_count() == 0 {
            return;
        }
        let mut combined = Aabbd::EMPTY;
        for snap in world.snapshots() {
            if !snap.world_aabb.is_empty() {
                combined = combined.union(snap.world_aabb);
            }
        }
        if combined.is_empty() {
            return;
        }
        let expanded = combined.expand_margin(4.0);
        let Some(entities) =
            level.entities_in(host, expanded.min.to_array(), expanded.max.to_array())
        else {
            return;
        };
        for entity in entities {
            let Some([min, max]) = entity.bounding_box(host) else {
                continue;
            };
            let aabb = Aabbd::new(Vec3d::from_array(min), Vec3d::from_array(max));
            let [yaw_deg, pitch_deg] = entity.rotation(host).unwrap_or([0.0, 0.0]);
            self.update_entity_pose(
                entity.id(),
                aabb,
                f64::from(yaw_deg).to_radians(),
                f64::from(pitch_deg).to_radians(),
                tick,
            );
        }
    }

    /// Flushes queued server-side yaw carry and proxy collision exclusions.
    pub fn flush_server_actions(&mut self, host: &mut dyn ServerHost) -> usize {
        let actions = self.drain_pending_actions();
        let mut applied = 0usize;
        for act in actions {
            let Some(entity) = host.entity(act.entity_id) else {
                continue;
            };
            if act.yaw_delta_deg.abs() > 1e-5
                && let Some([yaw, pitch]) = entity.rotation(host)
            {
                let _ = entity.set_rotation(host, yaw + act.yaw_delta_deg, pitch);
            }
            if !act.collision_exclusions.is_empty() {
                let _ = apply_server_exclusions(host, entity, &act.collision_exclusions);
            }
            applied += 1;
        }
        applied
    }

    /// Flushes queued client-side collision exclusions and velocity inheritance kick.
    pub fn flush_client_actions(&mut self, host: &mut dyn ClientHost) -> usize {
        let actions = self.drain_pending_actions();
        let mut applied = 0usize;
        for act in actions {
            if apply_client_compensation(host, &act.collision_exclusions, act.client_velocity_kick)
                .is_ok()
            {
                applied += 1;
            }
        }
        applied
    }

    /// Removes stale entity tracking records older than `max_age_ticks`.
    pub fn prune_stale(&mut self, current_tick: u64, max_age_ticks: u64) {
        self.anchors
            .retain(|_, a| current_tick.saturating_sub(a.last_seen_tick) <= max_age_ticks);
        self.tracked
            .retain(|_, p| current_tick.saturating_sub(p.last_tick) <= max_age_ticks);
    }
}

/// Applies collision exclusion boxes to a server player entity via Horizon's
/// [`EntityRef::set_collision_exclusions`] API (truncating safely to 64 boxes).
pub fn apply_server_exclusions(
    host: &mut dyn ServerHost,
    entity: EntityRef,
    boxes: &[[f64; 6]],
) -> Result<(), CollisionExclusionsError> {
    let bounded = &boxes[..boxes.len().min(MAX_COLLISION_EXCLUSIONS)];
    entity.set_collision_exclusions(host, bounded)
}

/// Applies client-side collision exclusions and optional deck velocity inheritance
/// kick via [`ClientHost::set_collision_exclusions`] and [`ClientHost::push_player`].
pub fn apply_client_compensation(
    host: &mut dyn ClientHost,
    boxes: &[[f64; 6]],
    velocity_kick: Option<[f32; 3]>,
) -> Result<(), CollisionExclusionsError> {
    let bounded = &boxes[..boxes.len().min(MAX_COLLISION_EXCLUSIONS)];
    let res = host.set_collision_exclusions(bounded);
    if let Some(kick) = velocity_kick
        && kick.iter().all(|c| c.is_finite())
    {
        host.push_player([
            kick[0].clamp(-4.0, 4.0),
            kick[1].clamp(-4.0, 4.0),
            kick[2].clamp(-4.0, 4.0),
        ]);
    }
    res
}

#[inline]
pub fn clamp_player_kick(vel_blocks_per_tick: Vec3d) -> [f32; 3] {
    [
        (vel_blocks_per_tick.x as f32).clamp(-4.0, 4.0),
        (vel_blocks_per_tick.y as f32).clamp(-4.0, 4.0),
        (vel_blocks_per_tick.z as f32).clamp(-4.0, 4.0),
    ]
}

static GLOBAL_RIDER_MANAGER: OnceLock<Mutex<EntityRiderManager>> = OnceLock::new();

/// Acquires the global [`EntityRiderManager`] lock cleanly.
pub fn with_rider_manager<R>(f: impl FnOnce(&mut EntityRiderManager) -> R) -> R {
    let lock = GLOBAL_RIDER_MANAGER.get_or_init(|| Mutex::new(EntityRiderManager::new()));
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}
