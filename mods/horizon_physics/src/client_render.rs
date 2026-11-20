//! Client-side double-buffered `f64` pose interpolation, 16x16x16 brick
//! instanced batching (`ModelInstance` / `InstanceGroupRef`), WGSL thruster
//! plume pipeline rendering, and `/hphys debug` wireframe overlay.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};

use horizon_physics_core::{
    Aabbd, BodyId, MaterialId, PhysicsEvent, Transformd, Vec3d, VoxelCoord,
};
use mod_api::{
    ClientHost, Identifier, InstanceGroupRef, LightCoords, ModelInstance, ModelRef, Overlay,
    PipelineBlend, PipelineQuad, RenderLevelStage, RenderLevelStageEvent, RenderPipeline,
    RenderPipelineRef, WorldDepth, WorldLine, WorldQuad,
};

use crate::api::{PhysicsApi, SubGridHandle, ThrusterPlumeState};
use crate::sync::PoseSnapshotPacket;

/// Namespaced Horizon `RenderPipeline` identifier for thruster exhaust plumes.
pub const PLUME_PIPELINE_ID: &str = "horizon_physics:fx_plume";

/// Embedded WGSL thruster exhaust plume fragment shader source.
pub const PLUME_SHADER_SOURCE: &str =
    include_str!("../assets/horizon_physics/shaders/fx_plume.wgsl");

/// Maximum model instances submitted per view (matching Horizon's view budget).
pub const MAX_VIEW_INSTANCES: usize = 16_384;

/// Maximum quads (plain + pipeline) submitted per view.
pub const MAX_VIEW_QUADS: usize = 4_096;

/// Maximum fallback quads reserved for opaque sub-grid voxels per frame, leaving
/// headroom for translucent plumes and debug overlays within Horizon's 4,096 quad cap.
const MAX_FALLBACK_QUADS_PER_VIEW: usize = 3_072;

/// Maximum plume billboard quads emitted per view within Horizon's shared quad cap.
const MAX_PLUME_QUADS_PER_VIEW: usize = 512;

/// Maximum debug overlay quads emitted per view within Horizon's shared quad cap.
const MAX_DEBUG_QUADS_PER_VIEW: usize = 512;

/// One renderable voxel inside a 16x16x16 sub-grid brick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderVoxelCell {
    pub coord: VoxelCoord,
    pub block_id: String,
    pub material: MaterialId,
}

/// Double-buffered client kinematic state and brick-grouped geometry for one body.
#[derive(Clone, Debug)]
pub struct ClientBodyRenderState {
    pub body_id: BodyId,
    pub prev_transform: Transformd,
    pub curr_transform: Transformd,
    pub linear_velocity: Vec3d,
    pub angular_velocity: Vec3d,
    pub sleeping: bool,
    pub com_local: Vec3d,
    pub world_aabb: Aabbd,
    pub bricks: BTreeMap<(i16, i16, i16), Vec<RenderVoxelCell>>,
}

impl ClientBodyRenderState {
    /// Interpolates between `prev_transform` and `curr_transform` in `f64` using
    /// linear position interpolation (`lerp`) and shortest-arc quaternion `slerp`.
    #[inline]
    pub fn interpolated_transform(&self, partial_tick: f32) -> Transformd {
        let alpha = f64::from(partial_tick).clamp(0.0, 1.0);
        self.prev_transform.interpolate(&self.curr_transform, alpha)
    }
}

/// Batched instanced draw payload for one `(BodyId, BrickKey, block_id)` group.
#[derive(Clone, Debug, PartialEq)]
pub struct BrickInstanceBatch {
    pub body_id: BodyId,
    pub brick_key: (i16, i16, i16),
    pub block_id: String,
    pub instances: Vec<ModelInstance>,
}

/// Extracted frame submission summary for diagnostics and offline testing.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RenderExtractionReport {
    pub brick_batches: usize,
    pub total_instances: usize,
    pub plume_quads: usize,
    pub debug_lines: usize,
    pub debug_quads: usize,
}

/// Composite map key identifying a retained 16^3 brick instance group.
type BrickGroupKey = (BodyId, (i16, i16, i16), String);

/// Built-in vanilla block models registered via [`mod_api::Hooks::standalone_models`]
/// so `RenderAssets::model` resolves baked block models cleanly on Horizon clients.
pub fn standalone_block_model_ids() -> Vec<String> {
    const STANDARD_MODELS: &[&str] = &[
        "minecraft:block/stone",
        "minecraft:block/cobblestone",
        "minecraft:block/deepslate",
        "minecraft:block/obsidian",
        "minecraft:block/bedrock",
        "minecraft:block/oak_planks",
        "minecraft:block/spruce_planks",
        "minecraft:block/birch_planks",
        "minecraft:block/dark_oak_planks",
        "minecraft:block/oak_log",
        "minecraft:block/iron_block",
        "minecraft:block/copper_block",
        "minecraft:block/gold_block",
        "minecraft:block/netherite_block",
        "minecraft:block/slime_block",
        "minecraft:block/honey_block",
        "minecraft:block/ice",
        "minecraft:block/packed_ice",
        "minecraft:block/blue_ice",
        "minecraft:block/white_wool",
        "minecraft:block/sponge",
        "minecraft:block/hay_block",
        "minecraft:block/glass",
        "minecraft:block/sandstone",
        "minecraft:block/bricks",
        "minecraft:block/dirt",
        "minecraft:block/grass_block",
        "minecraft:block/sand",
        "minecraft:block/gravel",
        "minecraft:block/tnt",
    ];
    STANDARD_MODELS.iter().map(|s| (*s).to_owned()).collect()
}

/// Client-side render state holding interpolated body buffers, resolved handles,
/// persistent brick instance groups, retired group cleanup queue, and debug toggle.
#[derive(Default)]
pub struct ClientRenderState {
    bodies: BTreeMap<BodyId, ClientBodyRenderState>,
    body_geometry_revs: BTreeMap<BodyId, u64>,
    plumes: Vec<ThrusterPlumeState>,
    recent_contact_markers: Vec<(Vec3d, Vec3d)>,
    plume_pipeline: Option<RenderPipelineRef>,
    block_models: BTreeMap<String, ModelRef>,
    extra_model_ids: BTreeSet<String>,
    brick_groups: BTreeMap<BrickGroupKey, InstanceGroupRef>,
    retired_groups: Vec<InstanceGroupRef>,
    debug_enabled: bool,
    last_partial_tick: f32,
}

impl ClientRenderState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.bodies.clear();
        self.body_geometry_revs.clear();
        self.plumes.clear();
        self.recent_contact_markers.clear();
        self.plume_pipeline = None;
        self.block_models.clear();
        self.extra_model_ids.clear();
        self.brick_groups.clear();
        self.retired_groups.clear();
        self.debug_enabled = false;
        self.last_partial_tick = 0.0;
    }

    /// Registers an additional block ID hint so resource baking/reload resolves its `ModelRef`.
    pub fn track_block_model_hint(&mut self, block_id: &str) {
        if !block_id.is_empty() {
            self.extra_model_ids.insert(block_id.to_owned());
        }
    }

    fn retire_groups_matching<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&BrickGroupKey) -> bool,
    {
        let keys: Vec<BrickGroupKey> = self
            .brick_groups
            .keys()
            .filter(|k| predicate(k))
            .cloned()
            .collect();
        for key in keys {
            if let Some(group) = self.brick_groups.remove(&key) {
                self.retired_groups.push(group);
            }
        }
    }

    fn flush_retired_groups(&mut self, host: &dyn ClientHost) {
        for group in self.retired_groups.drain(..) {
            let _ = group.destroy(host);
        }
    }

    #[inline]
    pub fn set_debug_enabled(&mut self, enabled: bool) {
        self.debug_enabled = enabled;
    }

    #[inline]
    pub fn debug_enabled(&self) -> bool {
        self.debug_enabled
    }

    #[inline]
    pub fn body_state(&self, id: BodyId) -> Option<&ClientBodyRenderState> {
        self.bodies.get(&id)
    }

    #[inline]
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    #[inline]
    pub fn plume_pipeline(&self) -> Option<RenderPipelineRef> {
        self.plume_pipeline
    }

    /// Manually registers a resolved [`ModelRef`] for a block ID (useful for both
    /// live `client_setup` model resolution and offline render tests).
    pub fn set_block_model(&mut self, block_id: &str, model: ModelRef) {
        self.block_models.insert(block_id.to_owned(), model);
    }

    /// Manually stores a resolved [`RenderPipelineRef`] for plume rendering.
    pub fn set_plume_pipeline(&mut self, pipeline: Option<RenderPipelineRef>) {
        self.plume_pipeline = pipeline;
    }

    /// Appends the `horizon_physics:fx_plume` [`RenderPipeline`] during
    /// `Hooks::register_render_pipelines`.
    pub fn register_pipelines(pipelines: &mut Vec<RenderPipeline>) {
        if let Ok(id) = Identifier::parse(PLUME_PIPELINE_ID) {
            pipelines.push(
                RenderPipeline::new(id.clone(), id, PipelineBlend::Additive, WorldDepth::Tested)
                    .optional(),
            );
        }
    }

    /// Appends standard physics block models to `Hooks::standalone_models`.
    pub fn register_standalone_models(models: &mut Vec<String>) {
        for id in standalone_block_model_ids() {
            if !models.contains(&id) {
                models.push(id);
            }
        }
    }

    /// Resolves the registered plume [`RenderPipelineRef`] and standalone block models
    /// after client setup, model baking, or resource reload.
    pub fn resolve_resources(&mut self, host: &dyn ClientHost) {
        self.retire_groups_matching(|_| true);
        self.flush_retired_groups(host);
        self.block_models.clear();
        if let Ok(id) = Identifier::parse(PLUME_PIPELINE_ID) {
            self.plume_pipeline = host.render_assets().render_pipeline(&id).ok();
        }
        let assets = host.render_assets();
        for model_id in standalone_block_model_ids() {
            if let Ok(ident) = Identifier::parse(&model_id)
                && let Ok(model_ref) = assets.model(&ident)
            {
                // Map both `minecraft:block/stone` and canonical `minecraft:stone`.
                let short_id = model_id.replacen(":block/", ":", 1);
                self.block_models.insert(short_id, model_ref);
            }
        }
        let mut candidate_blocks: Vec<String> = self.extra_model_ids.iter().cloned().collect();
        for body in self.bodies.values() {
            for cells in body.bricks.values() {
                for cell in cells {
                    if !self.block_models.contains_key(&cell.block_id) {
                        candidate_blocks.push(cell.block_id.clone());
                    }
                }
            }
        }
        for block_id in candidate_blocks {
            if self.block_models.contains_key(&block_id) {
                continue;
            }
            let block_model_str = if let Some((ns, path)) = block_id.split_once(':') {
                if path.starts_with("block/") {
                    block_id.clone()
                } else {
                    format!("{ns}:block/{path}")
                }
            } else {
                format!("minecraft:block/{block_id}")
            };
            let resolved = Identifier::parse(&block_model_str)
                .ok()
                .and_then(|ident| assets.model(&ident).ok())
                .or_else(|| {
                    Identifier::parse(&block_id)
                        .ok()
                        .and_then(|ident| assets.model(&ident).ok())
                });
            if let Some(model_ref) = resolved {
                self.block_models.insert(block_id, model_ref);
            }
        }
    }

    /// Applies an incoming network [`PoseSnapshotPacket`] into the client's
    /// double-buffered `(prev_transform, curr_transform)` and chunked brick geometry tables.
    pub fn apply_pose_packet(&mut self, packet: &PoseSnapshotPacket) {
        for &removed in &packet.removed {
            self.bodies.remove(&removed);
            self.body_geometry_revs.remove(&removed);
            self.retire_groups_matching(|(bid, _, _)| *bid == removed);
        }
        if !packet.is_delta {
            let live: Vec<BodyId> = packet.bodies.iter().map(|b| b.id).collect();
            self.bodies.retain(|id, _| live.contains(id));
            self.body_geometry_revs.retain(|id, _| live.contains(id));
            self.retire_groups_matching(|(bid, _, _)| !live.contains(bid));
        }

        for rec in &packet.bodies {
            if let Some(existing) = self.bodies.get_mut(&rec.id) {
                existing.prev_transform = existing.curr_transform;
                existing.curr_transform = rec.transform;
                existing.linear_velocity = rec.lin_vel;
                existing.angular_velocity = rec.ang_vel;
                existing.sleeping = rec.sleeping;
            } else {
                self.bodies.insert(
                    rec.id,
                    ClientBodyRenderState {
                        body_id: rec.id,
                        prev_transform: rec.transform,
                        curr_transform: rec.transform,
                        linear_velocity: rec.lin_vel,
                        angular_velocity: rec.ang_vel,
                        sleeping: rec.sleeping,
                        com_local: Vec3d::ZERO,
                        world_aabb: Aabbd::from_center_half_extents(
                            rec.transform.position,
                            Vec3d::splat(0.5),
                        ),
                        bricks: BTreeMap::new(),
                    },
                );
            }
        }

        for geom in &packet.geometries {
            let Some(existing) = self.bodies.get_mut(&geom.id) else {
                continue;
            };
            if geom.replace_geometry {
                existing.bricks.clear();
            }
            existing.com_local = geom.com_local;
            for v in &geom.voxels {
                existing
                    .bricks
                    .entry(v.coord.brick_key())
                    .or_default()
                    .push(RenderVoxelCell {
                        coord: v.coord,
                        block_id: v.block_id.clone(),
                        material: v.material,
                    });
            }

            let mut min_pt = Vec3d::splat(f64::INFINITY);
            let mut max_pt = Vec3d::splat(f64::NEG_INFINITY);
            let mut total_cells = 0usize;
            for cells in existing.bricks.values() {
                for cell in cells {
                    total_cells += 1;
                    let c = cell.coord.center_local() - geom.com_local;
                    min_pt = min_pt.min(c - Vec3d::splat(0.5));
                    max_pt = max_pt.max(c + Vec3d::splat(0.5));
                }
            }
            if total_cells > 0 && min_pt.is_finite() && max_pt.is_finite() {
                let radius = ((max_pt - min_pt) * 0.5).length();
                existing.world_aabb = Aabbd::from_center_half_extents(
                    existing.curr_transform.position,
                    Vec3d::splat(radius),
                );
            }

            self.body_geometry_revs
                .insert(geom.id, geom.geometry_revision);
            let gid = geom.id;
            self.retire_groups_matching(|(bid, _, _)| *bid == gid);
        }
    }

    /// Synchronizes full render buffers (transforms, 16^3 voxel bricks, active
    /// thruster plumes, contact markers, and debug toggle) directly from [`PhysicsApi`].
    pub fn sync_from_api(&mut self, api: &PhysicsApi) {
        self.debug_enabled = api.debug_overlay_enabled();
        self.plumes = api.active_plumes().to_vec();

        self.recent_contact_markers.clear();
        for ev in api.last_events() {
            if let PhysicsEvent::Collision { point, normal, .. } = ev {
                self.recent_contact_markers.push((*point, *normal));
            }
        }

        let snapshots = api.world().snapshots();
        let mut live_ids = Vec::with_capacity(snapshots.len());

        for snap in snapshots {
            live_ids.push(snap.id);
            let handle = SubGridHandle(snap.id);
            let block_names = api.subgrid_block_names(handle);
            let mut bricks: BTreeMap<(i16, i16, i16), Vec<RenderVoxelCell>> = BTreeMap::new();

            if let Some(grid) = api.world().body_grid(snap.id) {
                for (vc, entry) in grid.iter_voxels() {
                    let block_id = block_names
                        .and_then(|m| m.get(&[vc.x, vc.y, vc.z]))
                        .cloned()
                        .unwrap_or_else(|| "minecraft:stone".to_owned());
                    bricks
                        .entry(vc.brick_key())
                        .or_default()
                        .push(RenderVoxelCell {
                            coord: vc,
                            block_id,
                            material: entry.material,
                        });
                }
            }

            let new_rev = api.geometry_revision(snap.id);
            let old_rev = self.body_geometry_revs.insert(snap.id, new_rev);
            if old_rev != Some(new_rev) {
                let sid = snap.id;
                self.retire_groups_matching(|(bid, _, _)| *bid == sid);
            }

            self.bodies.insert(
                snap.id,
                ClientBodyRenderState {
                    body_id: snap.id,
                    prev_transform: snap.prev_transform,
                    curr_transform: snap.transform,
                    linear_velocity: snap.linear_velocity,
                    angular_velocity: snap.angular_velocity,
                    sleeping: snap.sleeping,
                    com_local: snap.com_local,
                    world_aabb: snap.world_aabb,
                    bricks,
                },
            );
        }

        self.bodies.retain(|id, _| live_ids.contains(id));
        self.body_geometry_revs
            .retain(|id, _| live_ids.contains(id));
        self.retire_groups_matching(|(bid, _, _)| !live_ids.contains(bid));
    }

    /// Generates camera-relative `f32` [`ModelInstance`] batches grouped by
    /// `(BodyId, 16x16x16 BrickKey, block_id)`.
    ///
    /// Camera subtraction is executed in `f64` via
    /// [`Transformd::to_camera_relative_affine_f32`] before narrowing to `f32`,
    /// eliminating far-world jitter even past `|x| > 10^7`.
    pub fn build_brick_instances(
        &self,
        camera: [f64; 3],
        partial_tick: f32,
    ) -> Vec<BrickInstanceBatch> {
        let cam_world = Vec3d::from_array(camera);
        let mut batches = Vec::new();
        let mut budget_left = MAX_VIEW_INSTANCES;

        for body in self.bodies.values() {
            if budget_left == 0 {
                break;
            }
            let interp = body.interpolated_transform(partial_tick);
            for (&brick_key, cells) in &body.bricks {
                let mut grouped: BTreeMap<&str, Vec<ModelInstance>> = BTreeMap::new();
                for cell in cells {
                    if budget_left == 0 {
                        break;
                    }
                    let local_offset = cell.coord.center_local() - body.com_local;
                    let affine = interp.to_camera_relative_affine_f32(cam_world, local_offset);
                    if !affine.iter().all(|v| v.is_finite()) {
                        continue;
                    }
                    let tint = material_tint_argb(cell.material);
                    let instance = ModelInstance::new(
                        camera,
                        affine,
                        tint,
                        LightCoords::FULL_BRIGHT,
                        Overlay::NONE,
                    );
                    grouped
                        .entry(cell.block_id.as_str())
                        .or_default()
                        .push(instance);
                    budget_left -= 1;
                }
                for (block_id, instances) in grouped {
                    if !instances.is_empty() {
                        batches.push(BrickInstanceBatch {
                            body_id: body.body_id,
                            brick_key,
                            block_id: block_id.to_owned(),
                            instances,
                        });
                    }
                }
            }
        }
        batches
    }

    /// Builds crossed axial [`PipelineQuad`] billboards for all active propulsion plumes.
    pub fn build_plume_quads(&self, partial_tick: f32) -> Vec<PipelineQuad> {
        let mut quads = Vec::new();
        for plume in &self.plumes {
            if quads.len() + 2 > MAX_PLUME_QUADS_PER_VIEW || plume.throttle <= 0.01 {
                continue;
            }
            let Some(body) = self.bodies.get(&plume.grid.body_id()) else {
                continue;
            };
            let interp = body.interpolated_transform(partial_tick);
            let nozzle_world = interp.transform_point(Vec3d::from_array(plume.local_mount));
            let exhaust_dir = interp
                .transform_dir(Vec3d::from_array(plume.local_exhaust_dir))
                .normalize_or(Vec3d::Z);

            let (side_a, side_b) = exhaust_dir.orthonormal_basis();
            let length = (0.75 + plume.throttle * 2.75).clamp(0.5, 6.0);
            let half_w = (0.22 + plume.throttle * 0.28).clamp(0.15, 0.85);
            let tip_world = nozzle_world + exhaust_dir * length;

            let alpha = ((0.45 + 0.50 * plume.throttle).clamp(0.1, 1.0) * 255.0) as u32;
            let color = (alpha << 24) | 0x005C_D6FF;

            for side in [side_a, side_b] {
                let offset = side * half_w;
                let corners = [
                    (nozzle_world - offset).to_array(),
                    (tip_world - offset * 0.55).to_array(),
                    (tip_world + offset * 0.55).to_array(),
                    (nozzle_world + offset).to_array(),
                ];
                quads.push(PipelineQuad::spanning(corners, color));
            }
        }
        quads
    }

    /// Builds `/hphys debug` overlay primitives (`WorldLine` for OBBs, BVH AABBs,
    /// CoM axes, active joints, and contact normals; `WorldQuad` for contact points).
    pub fn build_debug_geometry(
        &self,
        api: &PhysicsApi,
        partial_tick: f32,
    ) -> (Vec<WorldLine>, Vec<WorldQuad>) {
        if !self.debug_enabled {
            return (Vec::new(), Vec::new());
        }

        let mut lines = Vec::new();
        let mut quads = Vec::new();

        // 1. Core merged OBBs, active joints, and solver contact normals.
        for (start, end, rgba) in api.world().debug_wireframes() {
            if start.is_finite() && end.is_finite() {
                lines.push(WorldLine::new(
                    start.to_array(),
                    end.to_array(),
                    rgba_to_argb(rgba),
                    1.8,
                    WorldDepth::Tested,
                ));
            }
        }

        // 2. Broadphase world BVH AABBs (translucent cyan) & interpolated CoM RGB axes.
        for body in self.bodies.values() {
            push_aabb_lines(&mut lines, &body.world_aabb, 0x9940_E0FF, 1.2);
            let interp = body.interpolated_transform(partial_tick);
            let c = interp.position;
            let axes = interp.rotation.to_mat3().cols;
            let axis_len = 0.75;
            lines.push(WorldLine::new(
                c.to_array(),
                (c + axes[0] * axis_len).to_array(),
                0xFFFF_3333,
                2.5,
                WorldDepth::ThroughWalls,
            ));
            lines.push(WorldLine::new(
                c.to_array(),
                (c + axes[1] * axis_len).to_array(),
                0xFF33_FF55,
                2.5,
                WorldDepth::ThroughWalls,
            ));
            lines.push(WorldLine::new(
                c.to_array(),
                (c + axes[2] * axis_len).to_array(),
                0xFF33_88FF,
                2.5,
                WorldDepth::ThroughWalls,
            ));
        }

        // 3. Contact point quads + normal vectors.
        for &(point, normal) in &self.recent_contact_markers {
            if !point.is_finite() || !normal.is_finite() {
                continue;
            }
            let (t1, t2) = normal.orthonormal_basis();
            let r = 0.12;
            let p = point + normal * 0.01;
            let corners = [
                (p - t1 * r + t2 * r).to_array(),
                (p - t1 * r - t2 * r).to_array(),
                (p + t1 * r - t2 * r).to_array(),
                (p + t1 * r + t2 * r).to_array(),
            ];
            quads.push(WorldQuad::new(
                corners,
                0xCCFF_4422,
                WorldDepth::ThroughWalls,
            ));
            lines.push(WorldLine::new(
                point.to_array(),
                (point + normal * 0.5).to_array(),
                0xFFFF_AA22,
                2.2,
                WorldDepth::ThroughWalls,
            ));
        }

        (lines, quads)
    }

    /// Executes custom world rendering for the active [`RenderLevelStage`].
    pub fn render_stage(
        &mut self,
        host: &dyn ClientHost,
        api: &PhysicsApi,
        event: &mut RenderLevelStageEvent<'_>,
    ) -> RenderExtractionReport {
        self.render_stage_inner(host, Some(api), event)
    }

    /// Lock-free variant of [`Self::render_stage`] for opaque/translucent render passes
    /// that do not require querying [`PhysicsApi`] for debug wireframes.
    pub fn render_stage_cached(
        &mut self,
        host: &dyn ClientHost,
        event: &mut RenderLevelStageEvent<'_>,
    ) -> RenderExtractionReport {
        self.render_stage_inner(host, None, event)
    }

    fn render_stage_inner(
        &mut self,
        host: &dyn ClientHost,
        api: Option<&PhysicsApi>,
        event: &mut RenderLevelStageEvent<'_>,
    ) -> RenderExtractionReport {
        self.flush_retired_groups(host);
        self.last_partial_tick = event.view.partial_tick;
        let mut report = RenderExtractionReport::default();

        match event.stage {
            RenderLevelStage::AfterOpaqueBlocks => {
                let batches =
                    self.build_brick_instances(event.view.camera, event.view.partial_tick);
                report.brick_batches = batches.len();
                let mut fallback_quads: Vec<WorldQuad> = Vec::new();
                for batch in &batches {
                    report.total_instances += batch.instances.len();
                    if let Some(model_ref) = self.block_models.get(&batch.block_id).copied() {
                        let key = (batch.body_id, batch.brick_key, batch.block_id.clone());
                        if let Some(&group_ref) = self.brick_groups.get(&key)
                            && group_ref.write(host, 0, &batch.instances).is_ok()
                            && group_ref
                                .set_len(host, batch.instances.len() as u32)
                                .is_ok()
                        {
                            let _ = event.draw.draw_instance_group(group_ref);
                            continue;
                        }
                        if self.brick_groups.len() < MAX_VIEW_INSTANCES
                            && !self.brick_groups.contains_key(&key)
                        {
                            let cap =
                                (batch.instances.len().next_power_of_two()).clamp(16, 4096) as u32;
                            if let Ok(group_ref) =
                                host.instance_groups().create_group(model_ref, cap)
                                && group_ref.write(host, 0, &batch.instances).is_ok()
                                && group_ref
                                    .set_len(host, batch.instances.len() as u32)
                                    .is_ok()
                            {
                                self.brick_groups.insert(key, group_ref);
                                let _ = event.draw.draw_instance_group(group_ref);
                                continue;
                            }
                        }
                        let _ = event.draw.draw_model_instances(model_ref, &batch.instances);
                    } else {
                        self.extra_model_ids.insert(batch.block_id.clone());
                        if fallback_quads.len() < MAX_FALLBACK_QUADS_PER_VIEW
                            && let Some(body) = self.bodies.get(&batch.body_id)
                            && let Some(cells) = body.bricks.get(&batch.brick_key)
                        {
                            let interp = body.interpolated_transform(event.view.partial_tick);
                            for cell in cells {
                                if cell.block_id != batch.block_id {
                                    continue;
                                }
                                if fallback_quads.len() + 6 > MAX_FALLBACK_QUADS_PER_VIEW {
                                    break;
                                }
                                let center_local = cell.coord.center_local() - body.com_local;
                                push_shaded_voxel_quads(
                                    &mut fallback_quads,
                                    &interp,
                                    center_local,
                                    material_tint_argb(cell.material),
                                );
                            }
                        }
                    }
                }
                for chunk in fallback_quads.chunks(256) {
                    if event.draw.quads(chunk).is_err() {
                        break;
                    }
                }
            }
            RenderLevelStage::AfterTranslucentBlocks => {
                let quads = self.build_plume_quads(event.view.partial_tick);
                report.plume_quads = quads.len();
                if !quads.is_empty()
                    && let Some(pipeline_ref) = self.plume_pipeline
                {
                    for chunk in quads.chunks(128) {
                        if event.draw.pipeline_quads(pipeline_ref, chunk).is_err() {
                            break;
                        }
                    }
                }
            }
            RenderLevelStage::AfterLevel if self.debug_enabled => {
                if let Some(physics_api) = api {
                    let (lines, mut quads) =
                        self.build_debug_geometry(physics_api, event.view.partial_tick);
                    quads.truncate(MAX_DEBUG_QUADS_PER_VIEW);
                    report.debug_lines = lines.len();
                    report.debug_quads = quads.len();
                    for chunk in lines.chunks(256) {
                        if event.draw.lines(chunk).is_err() {
                            break;
                        }
                    }
                    for chunk in quads.chunks(128) {
                        if event.draw.quads(chunk).is_err() {
                            break;
                        }
                    }
                }
            }
            _ => {}
        }

        report
    }
}

/// Validates that the embedded WGSL plume shader conforms strictly to Horizon's
/// `RenderPipeline` fragment shader contract.
pub fn validate_plume_shader_contract(source: &str) -> Result<(), String> {
    if !source.contains("fn pipeline_fragment(input: PipelineInput) -> vec4<f32>") {
        return Err("WGSL shader missing required `fn pipeline_fragment(input: PipelineInput) -> vec4<f32>` entry".to_owned());
    }
    for forbidden in ["@fragment", "@vertex", "@group(", "@binding(", "horizon_"] {
        if source.contains(forbidden) {
            return Err(format!(
                "WGSL shader contains disallowed pipeline token `{forbidden}`"
            ));
        }
    }
    Ok(())
}

#[inline]
fn rgba_to_argb(rgba: u32) -> u32 {
    let r = (rgba >> 24) & 0xFF;
    let g = (rgba >> 16) & 0xFF;
    let b = (rgba >> 8) & 0xFF;
    let a = rgba & 0xFF;
    (a << 24) | (r << 16) | (g << 8) | b
}

#[inline]
fn shade_argb(argb: u32, factor: f32) -> u32 {
    let a = (argb >> 24) & 0xFF;
    let r = ((((argb >> 16) & 0xFF) as f32) * factor).clamp(0.0, 255.0) as u32;
    let g = ((((argb >> 8) & 0xFF) as f32) * factor).clamp(0.0, 255.0) as u32;
    let b = (((argb & 0xFF) as f32) * factor).clamp(0.0, 255.0) as u32;
    (a << 24) | (r << 16) | (g << 8) | b
}

fn push_shaded_voxel_quads(
    out: &mut Vec<WorldQuad>,
    interp: &Transformd,
    center_local: Vec3d,
    base_argb: u32,
) {
    let min = center_local - Vec3d::splat(0.5);
    let max = center_local + Vec3d::splat(0.5);
    let p000 = interp
        .transform_point(Vec3d::new(min.x, min.y, min.z))
        .to_array();
    let p100 = interp
        .transform_point(Vec3d::new(max.x, min.y, min.z))
        .to_array();
    let p110 = interp
        .transform_point(Vec3d::new(max.x, max.y, min.z))
        .to_array();
    let p010 = interp
        .transform_point(Vec3d::new(min.x, max.y, min.z))
        .to_array();
    let p001 = interp
        .transform_point(Vec3d::new(min.x, min.y, max.z))
        .to_array();
    let p101 = interp
        .transform_point(Vec3d::new(max.x, min.y, max.z))
        .to_array();
    let p111 = interp
        .transform_point(Vec3d::new(max.x, max.y, max.z))
        .to_array();
    let p011 = interp
        .transform_point(Vec3d::new(min.x, max.y, max.z))
        .to_array();

    let faces: [([[f64; 3]; 4], f32); 6] = [
        ([p011, p111, p110, p010], 1.00), // +Y top
        ([p000, p100, p101, p001], 0.55), // -Y bottom
        ([p001, p101, p111, p011], 0.85), // +Z south
        ([p100, p000, p010, p110], 0.85), // -Z north
        ([p101, p100, p110, p111], 0.72), // +X east
        ([p000, p001, p011, p010], 0.72), // -X west
    ];
    for (corners, shade) in faces {
        out.push(WorldQuad::new(
            corners,
            shade_argb(base_argb, shade),
            WorldDepth::Tested,
        ));
    }
}

#[inline]
fn material_tint_argb(mat: MaterialId) -> u32 {
    match mat {
        MaterialId::WOOD => 0xFFD8_B480,
        MaterialId::IRON => 0xFFE6_EBF0,
        MaterialId::SLIME => 0xFF78_E070,
        MaterialId::ICE => 0xFFB8_E8FF,
        MaterialId::WOOL => 0xFFFF_FFFF,
        _ => 0xFFCC_CCCC,
    }
}

fn push_aabb_lines(out: &mut Vec<WorldLine>, aabb: &Aabbd, color: u32, width: f32) {
    if aabb.is_empty() || !aabb.min.is_finite() || !aabb.max.is_finite() {
        return;
    }
    let min = aabb.min;
    let max = aabb.max;
    let c = [
        [min.x, min.y, min.z],
        [max.x, min.y, min.z],
        [min.x, max.y, min.z],
        [max.x, max.y, min.z],
        [min.x, min.y, max.z],
        [max.x, min.y, max.z],
        [min.x, max.y, max.z],
        [max.x, max.y, max.z],
    ];
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
    for &(a, b) in &EDGES {
        out.push(WorldLine::new(c[a], c[b], color, width, WorldDepth::Tested));
    }
}

static GLOBAL_CLIENT_RENDER: OnceLock<Mutex<ClientRenderState>> = OnceLock::new();

/// Acquires the global [`ClientRenderState`] lock cleanly.
pub fn with_client_render<R>(f: impl FnOnce(&mut ClientRenderState) -> R) -> R {
    let lock = GLOBAL_CLIENT_RENDER.get_or_init(|| Mutex::new(ClientRenderState::new()));
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

/// Convenience helper invoked by the client payload receiver to ingest a pose packet.
pub fn apply_pose_packet(packet: &PoseSnapshotPacket) {
    with_client_render(|state| state.apply_pose_packet(packet));
}
