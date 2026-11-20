//! Comprehensive Horizon 26.3 `FakeHost` + `with_mods` integration test suite
//! for `mod-horizon_physics`.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use horizon_physics_core::{
    Aabbd, BodyKind, MaterialId, PhysicsConfig, Quatd, Transformd, Vec3d, VoxelMaterial,
    WorldVoxelProvider,
};
use mod_api::__private::built::LifecycleContext as BuiltLifecycleContext;
use mod_api::__private::game::{BackendMutRefOf, BackendRefOf};
use mod_api::prelude::*;
use mod_api::testing::{FakeHost, TestReader, with_mods};
use mod_api::{
    Authority, Backend, BackendMut, BackendMutRef, BackendRef, BlockKey, BlockPos, BlockState,
    BlockStateRecord, ClientHost, CollisionExclusionsError, CommandArgument, CommandArgumentValue,
    CommandInvocation, CommandSource, DimensionKey, DrawError, EntityRef, Hooks, Host, HostSide,
    IPayloadContext, IPayloadHandler, InstanceError, InstanceGroupRef, LevelDimension, LevelHost,
    LifecycleContext, LoopSide, ModEntry, ModelInstance, ModelRef, MoveCause, OwningLoop,
    PayloadAccepted, PayloadBytes, PayloadContextError, PayloadDirection, PayloadId,
    PayloadProtocol, PayloadSendError, PayloadToken, PayloadWork, PayloadWorkStatus,
    PermissionLevel, PipelineDrawError, PipelineQuad, Placement, PlayerRef, Registrar,
    RegistrationContext, RenderLevelStage, RenderPipeline, RenderPipelineRef, ServerHost,
    ServerTickPhase, Text, TickId, WorldDrawContext, WorldDrawUsage, WorldEpoch, WorldLine,
    WorldQuad, WorldRenderView,
};

use mod_horizon_physics::api::{
    BlockMassSpec, DistanceJointDesc, FixedWeldJointDesc, ForceContribution, ForceProvider,
    ForceQueryContext, HingeJointDesc, JointConstraint, JointMotor, JointRegistry, PhysicsApi,
    PhysicsRaycast, Propulsion, PropulsionOutput, RayQuery, RaycastFilter, RaycastHitTarget,
    SliderJointDesc, SubGridAssemblyDesc, SubGridHandle, SubGridRegistry, VoxelPlacement,
    with_physics_api, with_physics_api_read,
};
use mod_horizon_physics::client_render::{
    ClientRenderState, PLUME_PIPELINE_ID, PLUME_SHADER_SOURCE, validate_plume_shader_contract,
    with_client_render,
};
use mod_horizon_physics::entity_rider::{
    apply_client_compensation, apply_server_exclusions, with_rider_manager,
};
use mod_horizon_physics::sync::{
    DeltaSyncTracker, PoseSnapshotPacket, PoseSyncClientHandler, decode_subgrid_world_nbt,
    encode_subgrid_world_nbt, server_tick_sync_with,
};
use mod_horizon_physics::terrain_cache::{BlockMaterialTable, DEFAULT_BLOCKS_JSON};
use mod_horizon_physics::{HorizonPhysicsMod, reset_runtime_state};

/// Serializes integration tests sharing process-global mod singletons.
static TEST_GUARD: Mutex<()> = Mutex::new(());

fn with_horizon_physics<R>(f: impl FnOnce() -> R) -> R {
    let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    reset_runtime_state();
    let mods = vec![ModEntry::new(
        "horizon_physics",
        mod_horizon_physics::hooks(),
    )]
    .into();
    let out = with_mods(mods, f);
    reset_runtime_state();
    out
}

fn lifecycle_ctx(tick: u64) -> LifecycleContext {
    BuiltLifecycleContext {
        authority: Authority::IntegratedServer,
        epoch: WorldEpoch::new(1),
        tick: TickId::new(tick),
    }
    .into()
}

fn console_source() -> CommandSource {
    CommandSource::Console {
        permission: PermissionLevel::Admins,
    }
}

fn hphys_invocation(
    sub: &str,
    args: Vec<CommandArgument>,
    position: [f64; 3],
) -> CommandInvocation {
    CommandInvocation::new(
        vec!["hphys".to_owned(), sub.to_owned()],
        args,
        DimensionKey::OVERWORLD,
        position,
    )
}

fn text_to_plain(text: &Text) -> String {
    format!("{:?}", text.content)
}

fn feedback_lines(host: &FakeHost) -> Vec<String> {
    host.commands.feedback.iter().map(text_to_plain).collect()
}

// ---------------------------------------------------------------------------
// Test doubles for hooks that FakeHost leaves unimplemented.
// ---------------------------------------------------------------------------

struct BlockKeyCaptureMod {
    stone: Arc<Mutex<Option<BlockKey>>>,
    air: Arc<Mutex<Option<BlockKey>>>,
}

#[mod_api::__subscriptions]
impl Hooks for BlockKeyCaptureMod {
    fn register(&self, r: &mut dyn Registrar, _cx: &RegistrationContext) {
        *self.stone.lock().unwrap() = Some(r.block_key("minecraft:stone"));
        *self.air.lock().unwrap() = Some(r.block_key("minecraft:air"));
    }
}

struct StubLevelHost {
    states: HashMap<BlockState, &'static str>,
}

impl LevelHost for StubLevelHost {
    fn state_block(&self, state: BlockState) -> Option<&str> {
        self.states.get(&state).copied()
    }
}

#[derive(Default)]
struct RecordingWorldDraw {
    lines: Vec<WorldLine>,
    quads: Vec<WorldQuad>,
    pipeline_quads: Vec<PipelineQuad>,
    model_instances: usize,
}

impl WorldDrawContext for RecordingWorldDraw {
    fn draw_instance_group(&mut self, _group: InstanceGroupRef) -> Result<(), mod_api::GroupError> {
        Ok(())
    }

    fn lines(&mut self, lines: &[WorldLine]) -> Result<(), DrawError> {
        self.lines.extend_from_slice(lines);
        Ok(())
    }

    fn quads(&mut self, quads: &[WorldQuad]) -> Result<(), DrawError> {
        self.quads.extend_from_slice(quads);
        Ok(())
    }

    fn pipeline_quads(
        &mut self,
        _pipeline: RenderPipelineRef,
        quads: &[PipelineQuad],
    ) -> Result<(), PipelineDrawError> {
        self.pipeline_quads.extend_from_slice(quads);
        Ok(())
    }

    fn usage(&self) -> WorldDrawUsage {
        WorldDrawUsage::new(
            self.lines.len() as u32,
            16_384,
            (self.quads.len() + self.pipeline_quads.len()) as u32,
            4_096,
            self.model_instances as u32,
            16_384,
        )
    }

    fn draw_model_instances(
        &mut self,
        _model: ModelRef,
        instances: &[ModelInstance],
    ) -> Result<(), InstanceError> {
        self.model_instances += instances.len();
        Ok(())
    }
}

/// Recording double implementing server + client rider APIs (`set_collision_exclusions`,
/// `entity_rotation` / `entity_set_rotation`, and client `push_player`).
#[derive(Default)]
struct RecordingRiderHost {
    entity_rotations: BTreeMap<u32, [f32; 2]>,
    server_exclusions: Vec<(u32, Vec<[f64; 6]>)>,
    client_exclusions: Vec<Vec<[f64; 6]>>,
    client_kicks: Vec<[f32; 3]>,
}

impl Host for RecordingRiderHost {
    fn backend(&self) -> Option<BackendRef<'_>> {
        Some(BackendRef::of(self))
    }
}

impl EntitiesHost for RecordingRiderHost {
    fn entity(&self, id: u32) -> Option<EntityRef> {
        self.entity_rotations
            .contains_key(&id)
            .then(|| EntityRef::new(WorldEpoch::new(1), id))
    }
}

impl LevelsHost for RecordingRiderHost {
    fn is_loaded(&self, id: &str) -> bool {
        id == "horizon_physics"
    }
}

mod_api::impl_host_areas!(RecordingRiderHost, except(entities, levels));

impl ServerHost for RecordingRiderHost {
    fn backend_mut(&mut self) -> Option<BackendMutRef<'_>> {
        Some(BackendMutRef::of(self))
    }
}
mod_api::impl_server_host_areas!(RecordingRiderHost);

impl ClientHost for RecordingRiderHost {}
impl PlayersClientHost for RecordingRiderHost {
    fn push_player(&mut self, kick: [f32; 3]) {
        self.client_kicks.push(kick);
    }

    fn set_collision_exclusions(
        &mut self,
        boxes: &[[f64; 6]],
    ) -> Result<(), CollisionExclusionsError> {
        self.client_exclusions.push(boxes.to_vec());
        Ok(())
    }
}
mod_api::impl_client_host_areas!(RecordingRiderHost, except(players));

impl Backend for RecordingRiderHost {}
impl EntitiesBackend for RecordingRiderHost {
    fn entity_rotation(&self, entity: EntityRef) -> Option<[f32; 2]> {
        self.entity_rotations.get(&entity.id()).copied()
    }
}
mod_api::impl_backend_areas!(RecordingRiderHost, except(entities));

impl BackendMut for RecordingRiderHost {}
impl EntitiesBackendMut for RecordingRiderHost {
    fn entity_set_rotation(&mut self, entity: EntityRef, yaw: f32, pitch: f32) -> bool {
        self.entity_rotations.insert(entity.id(), [yaw, pitch]);
        true
    }

    fn entity_set_collision_exclusions(
        &mut self,
        entity: EntityRef,
        boxes: &[[f64; 6]],
    ) -> Result<(), CollisionExclusionsError> {
        self.server_exclusions.push((entity.id(), boxes.to_vec()));
        Ok(())
    }
}
mod_api::impl_backend_mut_areas!(RecordingRiderHost, except(entities));

struct DummyPayloadContext;
impl IPayloadContext for DummyPayloadContext {
    fn flow(&self) -> PayloadDirection {
        PayloadDirection::ToClient
    }

    fn protocol(&self) -> PayloadProtocol {
        PayloadProtocol::Play
    }

    fn player(&self) -> Result<PlayerRef, PayloadContextError> {
        Err(PayloadContextError::NoPlayer)
    }

    fn server_host(&mut self) -> Result<&mut dyn ServerHost, PayloadContextError> {
        Err(PayloadContextError::WrongSide)
    }

    fn client_host(&mut self) -> Result<&mut dyn ClientHost, PayloadContextError> {
        Err(PayloadContextError::WrongSide)
    }

    fn reply(
        &mut self,
        _id: &PayloadId,
        _payload: PayloadBytes,
    ) -> Result<PayloadAccepted, PayloadSendError> {
        Err(PayloadSendError::Disconnected)
    }

    fn check_payload_token(&self, _token: &PayloadToken) -> Result<(), PayloadSendError> {
        Ok(())
    }

    fn disconnect(&mut self, _reason: Text) {}

    fn enqueue_work(
        &mut self,
        _work: Arc<dyn PayloadWork>,
    ) -> Result<PayloadWorkStatus, PayloadContextError> {
        Err(PayloadContextError::Disconnected)
    }
}

// ---------------------------------------------------------------------------
// 1. Mod installation & server_tick physics driving
// ---------------------------------------------------------------------------

#[test]
fn test_01_mod_install_and_server_tick_gravity_step() {
    with_horizon_physics(|| {
        let mut host = FakeHost::new();

        // Spawn a 2x2x1 dynamic stone sub-grid high in the air.
        let handle = with_physics_api(|api| {
            api.spawn_named_subgrid(
                Vec3d::new(0.0, 40.0, 0.0),
                Quatd::IDENTITY,
                BodyKind::Dynamic,
                false,
                1.0,
                101,
                &[
                    ([0, 0, 0], "minecraft:stone".to_owned()),
                    ([1, 0, 0], "minecraft:stone".to_owned()),
                    ([0, 1, 0], "minecraft:stone".to_owned()),
                    ([1, 1, 0], "minecraft:stone".to_owned()),
                ],
            )
            .expect("spawn stone subgrid")
        });

        let start_y =
            with_physics_api_read(|api| api.subgrid_state(handle).unwrap().world_position[1]);

        // ServerTickStart and paused ServerTickEnd must NOT advance the simulation.
        mod_api::dispatch_server_tick(
            &mut host,
            &lifecycle_ctx(1),
            ServerTickPhase::ServerTickStart {
                world_running: true,
            },
        );
        mod_api::dispatch_server_tick(
            &mut host,
            &lifecycle_ctx(2),
            ServerTickPhase::ServerTickEnd {
                world_running: false,
            },
        );
        assert_eq!(with_physics_api_read(|api| api.world().step_index()), 0);

        // Drive 20 running server ticks (1.0 second at 20 TPS).
        for tick in 1..=20 {
            mod_api::dispatch_server_tick(
                &mut host,
                &lifecycle_ctx(tick),
                ServerTickPhase::ServerTickEnd {
                    world_running: true,
                },
            );
        }

        let (steps, end_state) = with_physics_api_read(|api| {
            (api.world().step_index(), api.subgrid_state(handle).unwrap())
        });
        assert_eq!(steps, 20);
        let drop_m = start_y - end_state.world_position[1];
        assert!(
            drop_m > 3.5 && drop_m < 6.5,
            "expected ~4.9m 1-second gravity drop, got {drop_m:.3}m"
        );
        assert!(end_state.linear_velocity[1] < -8.0);

        // Verify client render double buffer was synchronized on tick end.
        with_client_render(|render| {
            let body = render
                .body_state(handle.body_id())
                .expect("body synced to render");
            assert!(body.curr_transform.position.y < body.prev_transform.position.y);
        });
    });
}

// ---------------------------------------------------------------------------
// 2. Full /hphys subcommand suite
// ---------------------------------------------------------------------------

#[test]
fn test_02_hphys_full_command_suite() {
    with_horizon_physics(|| {
        let mut host = FakeHost::new();
        let src = console_source();

        // 2a. Register commands and verify all 9 subcommands are exposed.
        let mut commands = Vec::new();
        mod_api::dispatch_register_commands(&mut host, &mut commands);
        assert_eq!(commands.len(), 9, "expected 9 /hphys subcommands");
        let usages: Vec<&str> = commands.iter().map(|c| c.usage.as_str()).collect();
        for expected in [
            "hphys spawn",
            "hphys assemble",
            "hphys disassemble",
            "hphys impulse",
            "hphys joint",
            "hphys sleep",
            "hphys wake",
            "hphys stats",
            "hphys debug",
        ] {
            assert!(
                usages.iter().any(|u| u.starts_with(expected)),
                "missing registered command usage `{expected}`"
            );
        }

        // 2b. /hphys spawn (stone & iron)
        let mut ran = false;
        mod_api::dispatch_run_mod_command(
            &mut host,
            &src,
            &hphys_invocation(
                "spawn",
                vec![
                    CommandArgument::new("x".into(), CommandArgumentValue::Number(0.0)),
                    CommandArgument::new("y".into(), CommandArgumentValue::Number(15.0)),
                    CommandArgument::new("z".into(), CommandArgumentValue::Number(0.0)),
                    CommandArgument::new("size".into(), CommandArgumentValue::Integer(2)),
                    CommandArgument::new(
                        "material".into(),
                        CommandArgumentValue::String("stone".into()),
                    ),
                ],
                [0.0, 15.0, 0.0],
            ),
            &mut ran,
        );
        assert!(ran);
        assert!(
            feedback_lines(&host)
                .last()
                .unwrap()
                .contains("spawned subgrid #0")
        );

        ran = false;
        mod_api::dispatch_run_mod_command(
            &mut host,
            &src,
            &hphys_invocation(
                "spawn",
                vec![
                    CommandArgument::new("x".into(), CommandArgumentValue::Number(5.0)),
                    CommandArgument::new("y".into(), CommandArgumentValue::Number(15.0)),
                    CommandArgument::new("z".into(), CommandArgumentValue::Number(0.0)),
                    CommandArgument::new("size".into(), CommandArgumentValue::Integer(1)),
                    CommandArgument::new(
                        "material".into(),
                        CommandArgumentValue::String("iron".into()),
                    ),
                ],
                [5.0, 15.0, 0.0],
            ),
            &mut ran,
        );
        assert!(ran);
        assert_eq!(with_physics_api_read(|api| api.world().body_count()), 2);

        // 2c. /hphys assemble from TerrainCache cells
        with_physics_api(|api| {
            api.terrain_mut()
                .set_block([10, 5, 10], "minecraft:oak_planks");
            api.terrain_mut()
                .set_block([11, 5, 10], "minecraft:oak_planks");
            api.terrain_mut()
                .set_block([10, 5, 11], "minecraft:oak_planks");
        });
        ran = false;
        mod_api::dispatch_run_mod_command(
            &mut host,
            &src,
            &hphys_invocation(
                "assemble",
                vec![
                    CommandArgument::new("min".into(), CommandArgumentValue::BlockPos([10, 5, 10])),
                    CommandArgument::new("max".into(), CommandArgumentValue::BlockPos([11, 5, 11])),
                ],
                [10.0, 5.0, 10.0],
            ),
            &mut ran,
        );
        assert!(ran);
        assert!(
            feedback_lines(&host)
                .last()
                .unwrap()
                .contains("assembled subgrid #2")
        );
        assert_eq!(with_physics_api_read(|api| api.world().body_count()), 3);

        // 2d. /hphys impulse on body 0
        ran = false;
        mod_api::dispatch_run_mod_command(
            &mut host,
            &src,
            &hphys_invocation(
                "impulse",
                vec![
                    CommandArgument::new("body".into(), CommandArgumentValue::Integer(0)),
                    CommandArgument::new("ix".into(), CommandArgumentValue::Number(0.0)),
                    CommandArgument::new("iy".into(), CommandArgumentValue::Number(20_000.0)),
                    CommandArgument::new("iz".into(), CommandArgumentValue::Number(0.0)),
                ],
                [0.0, 0.0, 0.0],
            ),
            &mut ran,
        );
        assert!(ran);
        let vy_after_impulse = with_physics_api_read(|api| {
            let h0 = api.find_subgrid_by_index(0).unwrap();
            api.subgrid_state(h0).unwrap().linear_velocity[1]
        });
        assert!(
            vy_after_impulse > 0.5,
            "impulse must impart upward velocity"
        );

        // 2e. /hphys joint (weld + distance)
        for kind in ["weld", "distance"] {
            ran = false;
            mod_api::dispatch_run_mod_command(
                &mut host,
                &src,
                &hphys_invocation(
                    "joint",
                    vec![
                        CommandArgument::new(
                            "kind".into(),
                            CommandArgumentValue::String(kind.into()),
                        ),
                        CommandArgument::new("body_a".into(), CommandArgumentValue::Integer(0)),
                        CommandArgument::new("body_b".into(), CommandArgumentValue::Integer(1)),
                    ],
                    [0.0, 0.0, 0.0],
                ),
                &mut ran,
            );
            assert!(ran);
        }
        assert_eq!(with_physics_api_read(|api| api.world().joint_count()), 2);

        // 2f. /hphys sleep & wake
        ran = false;
        mod_api::dispatch_run_mod_command(
            &mut host,
            &src,
            &hphys_invocation(
                "sleep",
                vec![CommandArgument::new(
                    "body".into(),
                    CommandArgumentValue::Integer(0),
                )],
                [0.0, 0.0, 0.0],
            ),
            &mut ran,
        );
        assert!(ran);
        assert!(with_physics_api_read(|api| {
            let h0 = api.find_subgrid_by_index(0).unwrap();
            api.subgrid_state(h0).unwrap().sleeping
        }));

        ran = false;
        mod_api::dispatch_run_mod_command(
            &mut host,
            &src,
            &hphys_invocation(
                "wake",
                vec![CommandArgument::new(
                    "body".into(),
                    CommandArgumentValue::Integer(0),
                )],
                [0.0, 0.0, 0.0],
            ),
            &mut ran,
        );
        assert!(ran);
        assert!(!with_physics_api_read(|api| {
            let h0 = api.find_subgrid_by_index(0).unwrap();
            api.subgrid_state(h0).unwrap().sleeping
        }));

        // 2g. /hphys debug true -> false
        ran = false;
        mod_api::dispatch_run_mod_command(
            &mut host,
            &src,
            &hphys_invocation(
                "debug",
                vec![CommandArgument::new(
                    "enabled".into(),
                    CommandArgumentValue::Boolean(true),
                )],
                [0.0, 0.0, 0.0],
            ),
            &mut ran,
        );
        assert!(ran);
        assert!(with_physics_api_read(PhysicsApi::debug_overlay_enabled));
        assert!(with_client_render(|r| r.debug_enabled()));

        ran = false;
        mod_api::dispatch_run_mod_command(
            &mut host,
            &src,
            &hphys_invocation(
                "debug",
                vec![CommandArgument::new(
                    "enabled".into(),
                    CommandArgumentValue::Boolean(false),
                )],
                [0.0, 0.0, 0.0],
            ),
            &mut ran,
        );
        assert!(ran);
        assert!(!with_physics_api_read(PhysicsApi::debug_overlay_enabled));

        // 2h. /hphys stats
        ran = false;
        mod_api::dispatch_run_mod_command(
            &mut host,
            &src,
            &hphys_invocation("stats", vec![], [0.0, 0.0, 0.0]),
            &mut ran,
        );
        assert!(ran);
        let stats_msg = feedback_lines(&host).last().unwrap().clone();
        assert!(
            stats_msg.contains("bodies=3"),
            "unexpected stats: {stats_msg}"
        );
        assert!(
            stats_msg.contains("joints=2"),
            "unexpected stats: {stats_msg}"
        );

        // 2i. /hphys disassemble on body 2
        ran = false;
        mod_api::dispatch_run_mod_command(
            &mut host,
            &src,
            &hphys_invocation(
                "disassemble",
                vec![CommandArgument::new(
                    "body".into(),
                    CommandArgumentValue::Integer(2),
                )],
                [0.0, 0.0, 0.0],
            ),
            &mut ran,
        );
        assert!(ran);
        assert!(
            feedback_lines(&host)
                .last()
                .unwrap()
                .contains("disassembled subgrid #2")
        );
        assert_eq!(with_physics_api_read(|api| api.world().body_count()), 2);

        // 2j. Unrelated mod command must pass through unhandled.
        ran = false;
        let foreign = CommandInvocation::new(
            vec!["other_mod".into(), "ping".into()],
            vec![],
            DimensionKey::OVERWORLD,
            [0.0, 0.0, 0.0],
        );
        mod_api::dispatch_run_mod_command(&mut host, &src, &foreign, &mut ran);
        assert!(!ran);
    });
}

// ---------------------------------------------------------------------------
// 3. Third-party ForceProvider (aero wing) + Propulsion + dynamic voxels
// ---------------------------------------------------------------------------

struct AeroWingForceProvider {
    wing_area_m2: f64,
    cl_cruise: f64,
    cd_cruise: f64,
}

impl ForceProvider for AeroWingForceProvider {
    fn provider_id(&self) -> &'static str {
        "aero_addon:thin_airfoil_wing"
    }

    fn compute_force(
        &self,
        _grid: SubGridHandle,
        ctx: &ForceQueryContext,
    ) -> Option<ForceContribution> {
        let v = Vec3d::from_array(ctx.linear_velocity_world)
            - Vec3d::from_array(ctx.ambient_wind_world);
        let speed_sq = v.x * v.x + v.z * v.z;
        if speed_sq < 1e-4 {
            return None;
        }
        let rho = 1.225;
        let q = 0.5 * rho * speed_sq;
        let lift = q * self.wing_area_m2 * self.cl_cruise;
        let drag = q * self.wing_area_m2 * self.cd_cruise;
        let speed = speed_sq.sqrt();
        Some(ForceContribution::world_force_at_point(
            [-drag * (v.x / speed), lift, -drag * (v.z / speed)],
            ctx.world_position,
        ))
    }
}

struct VectorThrusterPropulsion {
    throttle: f64,
    pitch_rad: f64,
    yaw_rad: f64,
    max_thrust_n: f64,
}

impl Propulsion for VectorThrusterPropulsion {
    fn propulsion_id(&self) -> &'static str {
        "aero_addon:vector_jet"
    }

    fn set_throttle(&mut self, throttle: f64) {
        self.throttle = throttle.clamp(-1.0, 1.0);
    }

    fn throttle(&self) -> f64 {
        self.throttle
    }

    fn set_vector_angles(&mut self, pitch_rad: f64, yaw_rad: f64) {
        self.pitch_rad = pitch_rad.clamp(-0.5, 0.5);
        self.yaw_rad = yaw_rad.clamp(-0.5, 0.5);
    }

    fn evaluate_thrust(&self, _grid: SubGridHandle, _ctx: &ForceQueryContext) -> PropulsionOutput {
        let thrust = self.max_thrust_n * self.throttle;
        let fx = thrust * self.yaw_rad.sin();
        let fy = thrust * self.pitch_rad.sin();
        let fz = thrust * self.pitch_rad.cos() * self.yaw_rad.cos();
        PropulsionOutput {
            local_force: [fx, fy, fz],
            local_torque: [0.0, 0.0, 0.0],
            local_mount: [0.0, 0.0, -2.0],
            fuel_burn_rate: 1.5 * self.throttle.abs(),
        }
    }
}

#[test]
fn test_03_third_party_force_provider_propulsion_and_dynamic_voxels() {
    with_horizon_physics(|| {
        let mut host = FakeHost::new();

        let (aircraft, initial_mass) = with_physics_api(|api| {
            let handle = api
                .assemble_subgrid(SubGridAssemblyDesc {
                    world_origin: [0.0, 80.0, 0.0],
                    initial_rotation: [0.0, 0.0, 0.0, 1.0],
                    enable_ccd: true,
                    voxels: vec![
                        VoxelPlacement::from_spec([0, 0, 0], BlockMassSpec::wood()),
                        VoxelPlacement::from_spec([0, 0, 1], BlockMassSpec::wood()),
                        VoxelPlacement::from_spec([-1, 0, 0], BlockMassSpec::wood()),
                        VoxelPlacement::from_spec([1, 0, 0], BlockMassSpec::wood()),
                    ],
                })
                .expect("assemble aircraft");

            assert!(api.attach_force_provider(
                handle,
                Box::new(AeroWingForceProvider {
                    wing_area_m2: 90.0,
                    cl_cruise: 1.2,
                    cd_cruise: 0.04,
                }),
            ));
            assert!(api.attach_propulsion(
                handle,
                Box::new(VectorThrusterPropulsion {
                    throttle: 0.0,
                    pitch_rad: 0.0,
                    yaw_rad: 0.0,
                    max_thrust_n: 45_000.0,
                }),
            ));
            api.world_mut().set_body_velocity(
                handle.body_id(),
                Vec3d::new(0.0, 0.0, 18.0),
                Vec3d::ZERO,
            );
            assert!(api.set_propulsion_throttle(handle, "aero_addon:vector_jet", 1.0));
            assert!(api.set_propulsion_vector(handle, "aero_addon:vector_jet", 0.0, 0.0));
            let mass = api.subgrid_state(handle).unwrap().mass_kg;
            (handle, mass)
        });

        for tick in 1..=30 {
            mod_api::dispatch_server_tick(
                &mut host,
                &lifecycle_ctx(tick),
                ServerTickPhase::ServerTickEnd {
                    world_running: true,
                },
            );
        }

        with_physics_api(|api| {
            let state = api.subgrid_state(aircraft).unwrap();
            assert!(
                state.linear_velocity[2] > 28.0,
                "thruster must accelerate aircraft along +Z, got vz={:.2}",
                state.linear_velocity[2]
            );
            assert!(
                state.world_position[1] > 80.0,
                "aerodynamic wing lift must sustain/climb altitude against gravity, got y={:.2}",
                state.world_position[1]
            );
            assert!(api.last_fuel_burn_total() > 0.0);
            assert_eq!(api.active_plumes().len(), 1);
            assert!(api.active_plumes()[0].thrust_n > 40_000.0);

            // Dynamic voxel addition (heavy iron ballast at +X) updates mass & CoM immediately.
            api.set_voxel(
                aircraft,
                VoxelPlacement::from_spec([2, 0, 0], BlockMassSpec::iron()),
            )
            .expect("add iron ballast");
            let after_add = api.subgrid_state(aircraft).unwrap();
            assert!(
                after_add.mass_kg > initial_mass + 7_000.0,
                "adding iron ballast must raise total mass: {initial_mass} -> {}",
                after_add.mass_kg
            );
            assert_eq!(after_add.voxel_count, 5);

            // Remove the iron ballast and confirm mass returns.
            api.remove_voxel(aircraft, [2, 0, 0])
                .expect("remove iron ballast");
            let after_remove = api.subgrid_state(aircraft).unwrap();
            assert!((after_remove.mass_kg - initial_mass).abs() < 1e-3);
            assert_eq!(after_remove.voxel_count, 4);

            // Detach plugins cleanly.
            assert!(api.detach_force_provider(aircraft, "aero_addon:thin_airfoil_wing"));
            assert!(api.detach_propulsion(aircraft, "aero_addon:vector_jet"));
            assert_eq!(api.force_provider_count(), 0);
            assert_eq!(api.propulsion_count(), 0);
        });
    });
}

// ---------------------------------------------------------------------------
// 4. block_changed / break_block / entity_place / chunk_unload cache & wake
// ---------------------------------------------------------------------------

#[test]
fn test_04_block_changed_cache_invalidation_and_sleeping_island_wake() {
    let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    reset_runtime_state();

    let stone_key = Arc::new(Mutex::new(None));
    let air_key = Arc::new(Mutex::new(None));
    let probe = BlockKeyCaptureMod {
        stone: Arc::clone(&stone_key),
        air: Arc::clone(&air_key),
    };
    let mods = vec![
        ModEntry::new("horizon_physics", mod_horizon_physics::hooks()),
        ModEntry::new("test_probe", Box::new(probe)),
    ]
    .into();

    with_mods(mods, || {
        let mut host = FakeHost::new();
        let stone = BlockState::from(BlockStateRecord::new(
            stone_key.lock().unwrap().expect("registered stone key"),
            0,
        ));
        let air = BlockState::from(BlockStateRecord::new(
            air_key.lock().unwrap().expect("registered air key"),
            0,
        ));

        let mut states = HashMap::new();
        states.insert(stone, "minecraft:stone");
        states.insert(air, "minecraft:air");
        let level_host = StubLevelHost { states };

        // Populate chunk (0,0) with support floor under body, and unrelated chunk (5,5).
        let body = with_physics_api(|api| {
            api.terrain_mut().set_block([4, 64, 4], "minecraft:stone");
            api.terrain_mut().set_block([5, 64, 4], "minecraft:stone");
            api.terrain_mut()
                .set_block([82, 64, 82], "minecraft:iron_block");

            let h = api
                .spawn_named_subgrid(
                    Vec3d::new(4.5, 65.5, 4.5),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    1.0,
                    0,
                    &[([0, 0, 0], "minecraft:stone".to_owned())],
                )
                .unwrap();
            assert!(api.sleep_subgrid(h));
            assert!(api.subgrid_state(h).unwrap().sleeping);
            h
        });

        // Break supporting block via `dispatch_block_changed`.
        mod_api::dispatch_block_changed(
            &level_host,
            LevelDimension::Overworld,
            BlockPos::new(4, 64, 4),
            stone,
            air,
        );

        with_physics_api(|api| {
            // Sleeping body resting above [4,64,4] must be woken by wake_in_aabb.
            assert!(
                !api.subgrid_state(body).unwrap().sleeping,
                "block_changed beneath a sleeping body must wake it"
            );
            // Unrelated chunk (5,5) must remain cached untouched!
            assert!(
                api.terrain().chunk_revision(5, 5).is_some(),
                "unrelated chunk (5,5) must not be evicted on chunk (0,0) edit"
            );
            assert!(api.terrain().stats().block_invalidations >= 1);
            assert!(api.terrain_mut().voxel_collision(4, 64, 4).is_none());
        });

        // Also test break_block, entity_place, and chunk_unload hooks.
        let mut exp_drop = None;
        let mut canceled = false;
        mod_api::dispatch_break_block(&mut host, [5, 64, 4], stone, &mut exp_drop, &mut canceled);
        assert!(!canceled);

        let mut place_canceled = false;
        mod_api::dispatch_entity_place(
            &mut host,
            None,
            BlockPos::new(6, 64, 4),
            stone,
            air,
            Placement::Single,
            &mut place_canceled,
        );
        assert!(!place_canceled);

        mod_api::dispatch_chunk_unload(&mut host, "minecraft:overworld", [5, 5]);
        with_physics_api_read(|api| {
            assert!(
                api.terrain().chunk_revision(5, 5).is_none(),
                "chunk_unload must evict chunk (5,5)"
            );
            assert!(api.terrain().stats().chunk_evictions >= 1);
        });
    });
    reset_runtime_state();
}

// ---------------------------------------------------------------------------
// 5. entity_move deck carry, stair step-up, collision exclusions & push_player
// ---------------------------------------------------------------------------

#[test]
fn test_05_entity_move_deck_carry_step_up_exclusions_and_push_player() {
    with_horizon_physics(|| {
        let mut host = FakeHost::new();

        // Spawn a kinematic ship deck (5x1x5) with a 1-block stair step at x=+2, y=+1.
        let ship = with_physics_api(|api| {
            let mut blocks = Vec::new();
            for z in -2..=2 {
                for x in -2..=2 {
                    blocks.push(([x, 0, z], "minecraft:oak_planks".to_owned()));
                }
            }
            blocks.push(([2, 1, 0], "minecraft:oak_planks".to_owned()));
            api.spawn_named_subgrid(
                Vec3d::new(0.0, 10.0, 0.0),
                Quatd::IDENTITY,
                BodyKind::Kinematic,
                false,
                0.0,
                0,
                &blocks,
            )
            .unwrap()
        });

        // Step 1 tick at rest so entity establishes its initial foot anchor on the deck.
        let deck_top_y = with_physics_api_read(|api| {
            let s = api.world().body_snapshot(ship.body_id()).unwrap();
            s.transform.position.y - s.com_local.y + 1.0
        });
        let rider_id: u32 = 42;
        let standing_aabb = Aabbd::new(
            Vec3d::new(-0.3, deck_top_y + 0.01, -0.3),
            Vec3d::new(0.3, deck_top_y + 1.81, 0.3),
        );
        with_rider_manager(|rm| {
            rm.update_entity_pose(rider_id, standing_aabb, 0.0, 0.0, 0);
        });

        let reader = TestReader::default();
        let loop_ctx = OwningLoop::new(LoopSide::Server, &reader);
        let mut land_delta = [0.0, -0.08, 0.0];
        let mut move_canceled = false;
        mod_api::dispatch_entity_move(
            &loop_ctx,
            rider_id,
            "minecraft:player",
            MoveCause::Player,
            &mut land_delta,
            &mut move_canceled,
        );
        assert!(!move_canceled);
        assert_eq!(
            with_rider_manager(|rm| rm.anchor_of(rider_id))
                .expect("rider should anchor onto the ship deck on contact")
                .ground_body,
            ship.body_id()
        );

        // Now give the deck forward velocity + yaw angular velocity and advance 1 server tick.
        with_physics_api(|api| {
            api.world_mut().set_body_velocity(
                ship.body_id(),
                Vec3d::new(4.0, 0.0, 0.0),
                Vec3d::new(0.0, 0.8, 0.0),
            );
        });
        mod_api::dispatch_server_tick(
            &mut host,
            &lifecycle_ctx(1),
            ServerTickPhase::ServerTickEnd {
                world_running: true,
            },
        );

        // Dispatch `entity_move` while anchored on the moving/rotating deck: must carry rider!
        let mut delta = [0.0, -0.08, 0.0];
        mod_api::dispatch_entity_move(
            &loop_ctx,
            rider_id,
            "minecraft:player",
            MoveCause::Player,
            &mut delta,
            &mut move_canceled,
        );
        assert!(!move_canceled);
        assert!(
            delta[0] > 0.10,
            "moving deck (vx=4 m/s -> 0.2 m/tick) must carry rider in +X, got dx={:.4}",
            delta[0]
        );
        assert!(
            delta[1] >= -0.03,
            "solid deck floor must prevent sinking through deck, got dy={:.4}",
            delta[1]
        );

        let anchor = with_rider_manager(|rm| rm.anchor_of(rider_id))
            .expect("rider should remain anchored onto the ship deck");
        assert_eq!(anchor.ground_body, ship.body_id());
        assert!(
            anchor.last_carry_yaw_delta.abs() > 0.01,
            "rotating deck must produce non-zero carry_yaw_delta"
        );

        // Flush server + client actions through `RecordingRiderHost` via `level_tick_post`.
        let mut rider_host = RecordingRiderHost::default();
        rider_host.entity_rotations.insert(rider_id, [10.0, 0.0]);
        let level_ref = host.level.level();
        mod_api::dispatch_level_tick_post(HostSide::Server(&mut rider_host), level_ref);
        assert!(
            !rider_host.server_exclusions.is_empty(),
            "server must receive proxy collision exclusion boxes"
        );
        assert!(
            (rider_host.entity_rotations[&rider_id][0] - 10.0).abs() > 0.5,
            "server entity rotation must inherit deck yaw carry"
        );

        // Jump upward off the moving deck -> verify dismount velocity inheritance kick!
        let mut jump_delta = [0.0, 1.5, 0.0];
        mod_api::dispatch_entity_move(
            &loop_ctx,
            rider_id,
            "minecraft:player",
            MoveCause::Player,
            &mut jump_delta,
            &mut move_canceled,
        );
        mod_api::dispatch_level_tick_post(HostSide::Client(&mut rider_host), level_ref);
        assert!(
            !rider_host.client_exclusions.is_empty(),
            "client must receive collision exclusion boxes"
        );
        assert!(
            !rider_host.client_kicks.is_empty(),
            "dismounting from moving deck must emit push_player velocity inheritance kick"
        );
        assert!(rider_host.client_kicks[0][0] > 0.05);

        // Direct helper check for server/client exclusion truncation (max 64 boxes).
        let many_boxes = vec![[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; 80];
        let ent = EntityRef::new(WorldEpoch::new(1), rider_id);
        apply_server_exclusions(&mut rider_host, ent, &many_boxes).unwrap();
        assert_eq!(rider_host.server_exclusions.last().unwrap().1.len(), 64);
        apply_client_compensation(&mut rider_host, &many_boxes, Some([0.2, 0.1, 0.0])).unwrap();
        assert_eq!(rider_host.client_exclusions.last().unwrap().len(), 64);

        // Verify stair step-up (1.0m step <= 1.15m max step height) on stationary ship deck.
        with_physics_api(|api| {
            api.world_mut()
                .set_body_velocity(ship.body_id(), Vec3d::ZERO, Vec3d::ZERO);
            api.world_mut().set_body_transform(
                ship.body_id(),
                Transformd::from_position(Vec3d::new(0.0, 10.0, 0.0)),
            );
        });
        mod_api::dispatch_server_tick(
            &mut host,
            &lifecycle_ctx(2),
            ServerTickPhase::ServerTickEnd {
                world_running: true,
            },
        );
        let base_top = with_physics_api_read(|api| {
            let s = api.world().body_snapshot(ship.body_id()).unwrap();
            s.transform.position.y - s.com_local.y + 1.0
        });
        let before_step = Aabbd::new(
            Vec3d::new(1.1, base_top, -0.3),
            Vec3d::new(1.7, base_top + 1.8, 0.3),
        );
        let step_res = with_physics_api_read(|api| {
            with_rider_manager(|rm| {
                rm.resolve_entity_move(
                    api.world(),
                    99,
                    before_step,
                    Vec3d::new(0.6, 0.0, 0.0),
                    1.15,
                    2,
                )
            })
        });
        assert!(
            step_res.outcome.stepped_up_height > 0.5 || step_res.outcome.applied_delta.y > 0.5,
            "entity walking into a 1.0m deck step must step up smoothly: {:?}",
            step_res.outcome
        );
    });
}

// ---------------------------------------------------------------------------
// 6. AttachmentType NBT persistence roundtrip & server lifecycle hooks
// ---------------------------------------------------------------------------

#[test]
fn test_06_attachment_nbt_roundtrip_and_lifecycle_hooks() {
    with_horizon_physics(|| {
        let mut host = FakeHost::new();

        let (h_a, h_b) = with_physics_api(|api| {
            let rot = Quatd::from_axis_angle(Vec3d::Y, 0.45).normalize();
            let a = api
                .spawn_named_subgrid(
                    Vec3d::new(12.5, 70.25, -4.75),
                    rot,
                    BodyKind::Dynamic,
                    true,
                    1.0,
                    9001,
                    &[
                        ([0, 0, 0], "minecraft:oak_planks".to_owned()),
                        ([1, 0, 0], "minecraft:iron_block".to_owned()),
                    ],
                )
                .unwrap();
            let b = api
                .spawn_named_subgrid(
                    Vec3d::new(16.0, 70.25, -4.75),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    0.8,
                    9002,
                    &[([0, 0, 0], "minecraft:slime_block".to_owned())],
                )
                .unwrap();
            api.world_mut().set_body_velocity(
                a.body_id(),
                Vec3d::new(1.25, -0.5, 0.75),
                Vec3d::new(0.0, 0.3, 0.0),
            );

            api.add_joint(JointConstraint::Distance(DistanceJointDesc {
                body_a: a,
                body_b: Some(b),
                anchor_a: [0.5, 0.0, 0.0],
                anchor_b: [-0.5, 0.0, 0.0],
                rest_length: 2.5,
                min_length: Some(1.5),
                max_length: Some(3.5),
                compliance: 1e-4,
                damping: 0.2,
            }))
            .unwrap();
            api.add_joint(JointConstraint::FixedWeld(FixedWeldJointDesc {
                body_a: a,
                body_b: Some(b),
                anchor_a: [0.0, 0.0, 0.0],
                anchor_b: [0.0, 0.0, 0.0],
                relative_rotation: [0.0, 0.0, 0.0, 1.0],
                compliance: 0.0,
            }))
            .unwrap();
            api.add_joint(JointConstraint::Hinge(HingeJointDesc {
                body_a: a,
                body_b: b,
                anchor_a: [0.0, 1.0, 0.0],
                anchor_b: [0.0, 1.0, 0.0],
                axis_a: [0.0, 1.0, 0.0],
                axis_b: [0.0, 1.0, 0.0],
                min_angle_rad: -1.0,
                max_angle_rad: 1.0,
                compliance: 1e-5,
                motor: Some(JointMotor {
                    target_angle_rad: 0.5,
                    max_torque_nm: 500.0,
                    stiffness: 20.0,
                    damping: 5.0,
                }),
            }))
            .unwrap();
            api.add_joint(JointConstraint::Slider(SliderJointDesc {
                body_a: a,
                body_b: Some(b),
                anchor_a: [0.0, 0.0, 0.0],
                anchor_b: [0.0, 0.0, 0.0],
                axis_a: [1.0, 0.0, 0.0],
                axis_b: [1.0, 0.0, 0.0],
                limits: Some((-2.0, 2.0)),
                compliance: 0.0,
                motor_target_velocity: Some(0.5),
                motor_max_force: 250.0,
            }))
            .unwrap();
            (a, b)
        });

        let expected_a = with_physics_api_read(|api| api.subgrid_state(h_a).unwrap());
        let expected_b = with_physics_api_read(|api| api.subgrid_state(h_b).unwrap());

        // Direct Compound NBT codec roundtrip check.
        let persisted =
            with_physics_api_read(mod_horizon_physics::sync::PersistedSubGridWorld::capture);
        let nbt = encode_subgrid_world_nbt(&persisted).expect("encode NBT");
        let decoded = decode_subgrid_world_nbt(&nbt).expect("decode NBT");
        assert_eq!(persisted, decoded, "NBT roundtrip must be exact");

        // Persist via `world_saved` hook into FakeHost.attachments, wipe memory, restore via `server_started`.
        mod_api::dispatch_world_saved(&mut host, true);
        with_physics_api(PhysicsApi::reset);
        assert_eq!(with_physics_api_read(|api| api.world().body_count()), 0);

        mod_api::dispatch_server_started(&mut host, false);

        with_physics_api_read(|api| {
            assert_eq!(api.world().body_count(), 2);
            assert_eq!(api.world().joint_count(), 4);
            let snaps = api.world().snapshots();
            let ra = snaps
                .iter()
                .find(|s| s.user_data == 9001)
                .expect("restored body A");
            let rb = snaps
                .iter()
                .find(|s| s.user_data == 9002)
                .expect("restored body B");
            assert!((ra.transform.position.x - expected_a.world_position[0]).abs() < 1e-12);
            assert!((ra.transform.position.y - expected_a.world_position[1]).abs() < 1e-12);
            assert!((ra.linear_velocity.x - expected_a.linear_velocity[0]).abs() < 1e-12);
            assert!((rb.transform.position.x - expected_b.world_position[0]).abs() < 1e-12);
            let names_a = api.subgrid_block_names(SubGridHandle(ra.id)).unwrap();
            assert_eq!(
                names_a.get(&[0, 0, 0]).map(String::as_str),
                Some("minecraft:oak_planks")
            );
            assert_eq!(
                names_a.get(&[1, 0, 0]).map(String::as_str),
                Some("minecraft:iron_block")
            );
        });
    });
}

// ---------------------------------------------------------------------------
// 7. Compact binary PoseSnapshotPacket + DeltaSyncTracker + IPayloadHandler
// ---------------------------------------------------------------------------

#[test]
fn test_07_pose_snapshot_binary_codec_delta_and_payload_handler() {
    with_horizon_physics(|| {
        let mut tracker = DeltaSyncTracker::new();
        let (h1, h2) = with_physics_api(|api| {
            let a = api
                .spawn_named_subgrid(
                    Vec3d::new(100.25, 64.5, -200.75),
                    Quatd::from_axis_angle(Vec3d::Z, 0.3).normalize(),
                    BodyKind::Dynamic,
                    false,
                    1.0,
                    0,
                    &[([0, 0, 0], "minecraft:stone".to_owned())],
                )
                .unwrap();
            let b = api
                .spawn_named_subgrid(
                    Vec3d::new(110.0, 64.5, -200.75),
                    Quatd::IDENTITY,
                    BodyKind::Static,
                    false,
                    0.0,
                    0,
                    &[([0, 0, 0], "minecraft:iron_block".to_owned())],
                )
                .unwrap();
            (a, b)
        });

        // Full packet binary roundtrip.
        let full = with_physics_api_read(|api| tracker.build_full_packet(api));
        let wire = full.encode().expect("encode full pose packet");
        let decoded = PoseSnapshotPacket::decode(&wire).expect("decode full pose packet");
        assert_eq!(full, decoded);

        // Initial delta emits both bodies; immediate repeat with zero motion emits None.
        let d1 = with_physics_api_read(|api| tracker.build_delta_packet(api))
            .expect("initial delta must contain bodies");
        assert_eq!(d1.bodies.len(), 2);
        assert!(d1.is_delta);
        assert!(
            with_physics_api_read(|api| tracker.build_delta_packet(api)).is_none(),
            "stationary world must produce zero idle delta traffic"
        );

        // Move body 1 and disassemble body 2 -> delta contains 1 updated + 1 removed.
        with_physics_api(|api| {
            api.world_mut().set_body_transform(
                h1.body_id(),
                Transformd::from_position(Vec3d::new(101.5, 64.5, -200.75)),
            );
            let _ = api.disassemble_subgrid(h2);
        });
        let d2 = with_physics_api_read(|api| tracker.build_delta_packet(api))
            .expect("delta after move + remove");
        assert_eq!(d2.bodies.len(), 1);
        assert_eq!(d2.bodies[0].id, h1.body_id());
        assert_eq!(d2.removed, vec![h2.body_id()]);

        // Feed full + delta through `PoseSyncClientHandler` and verify double-buffered client state.
        let handler = PoseSyncClientHandler;
        let mut pctx = DummyPayloadContext;
        handler.handle(PayloadBytes(wire), &mut pctx).unwrap();
        let d2_wire = d2.encode().unwrap();
        handler.handle(PayloadBytes(d2_wire), &mut pctx).unwrap();

        with_client_render(|render| {
            assert_eq!(render.body_count(), 1);
            let b1 = render
                .body_state(h1.body_id())
                .expect("h1 tracked on client");
            assert!((b1.prev_transform.position.x - 100.25).abs() < 1e-12);
            assert!((b1.curr_transform.position.x - 101.5).abs() < 1e-12);
        });

        // Malformed / truncated packets must be rejected cleanly.
        assert!(PoseSnapshotPacket::decode(b"BAD!").is_err());
        assert!(PoseSnapshotPacket::decode(&[0u8; 4]).is_err());
    });
}

// ---------------------------------------------------------------------------
// 8. Client render interpolation, far-world affine, plume shader & debug draw
// ---------------------------------------------------------------------------

#[test]
fn test_08_client_render_interpolation_far_world_plume_and_debug_overlay() {
    with_horizon_physics(|| {
        let mut host = FakeHost::new();

        // 8a. Validate RenderPipeline registration & WGSL fragment shader contract.
        let mut pipelines: Vec<RenderPipeline> = Vec::new();
        ClientRenderState::register_pipelines(&mut pipelines);
        assert_eq!(pipelines.len(), 1);
        assert_eq!(pipelines[0].id.to_string(), PLUME_PIPELINE_ID);
        assert!(pipelines[0].optional);
        validate_plume_shader_contract(PLUME_SHADER_SOURCE)
            .expect("fx_plume.wgsl must satisfy Horizon RenderPipeline contract");

        // 8b. Spawn a multi-brick sub-grid at extreme far-world coordinates (~30M blocks)
        //     with a thruster attached to emit active plume quads.
        let far_origin = Vec3d::new(30_000_000.0, 64.0, -29_999_990.0);
        let far_ship = with_physics_api(|api| {
            let h = api
                .spawn_named_subgrid(
                    far_origin,
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    0.0,
                    0,
                    &[
                        ([0, 0, 0], "minecraft:stone".to_owned()),
                        ([16, 0, 0], "minecraft:stone".to_owned()), // second 16^3 brick!
                        ([0, 1, 0], "minecraft:iron_block".to_owned()),
                    ],
                )
                .unwrap();
            api.attach_propulsion(
                h,
                Box::new(VectorThrusterPropulsion {
                    throttle: 0.9,
                    pitch_rad: 0.0,
                    yaw_rad: 0.0,
                    max_thrust_n: 30_000.0,
                }),
            );
            api.set_debug_overlay_enabled(true);
            h
        });

        mod_api::dispatch_server_tick(
            &mut host,
            &lifecycle_ctx(1),
            ServerTickPhase::ServerTickEnd {
                world_running: true,
            },
        );

        let camera = [far_origin.x + 2.0, far_origin.y + 1.0, far_origin.z];

        with_client_render(|render| {
            // Double-buffered f64 interpolation check.
            let body = render.body_state(far_ship.body_id()).unwrap();
            let t0 = body.interpolated_transform(0.0);
            let t5 = body.interpolated_transform(0.5);
            let t1 = body.interpolated_transform(1.0);
            assert!((t5.position.z - 0.5 * (t0.position.z + t1.position.z)).abs() < 1e-9);

            // Multi-brick instanced batching + camera-relative f32 precision at 30M blocks.
            let batches = render.build_brick_instances(camera, 0.5);
            assert_eq!(
                batches.len(),
                3,
                "expected 3 (brick, block_id) batches across two 16^3 bricks"
            );
            let total_instances: usize = batches.iter().map(|b| b.instances.len()).sum();
            assert_eq!(total_instances, 3);
            for batch in &batches {
                for inst in &batch.instances {
                    // Row-major 3x4 affine translation column is indices [3], [7], [11].
                    let tx = inst.transform[3];
                    let ty = inst.transform[7];
                    let tz = inst.transform[11];
                    assert!(
                        tx.is_finite() && ty.is_finite() && tz.is_finite() && tx.abs() < 32.0,
                        "far-world camera-relative transform must be small and jitter-free: [{tx}, {ty}, {tz}]"
                    );
                }
            }

            // Plume quad generation for active propulsion.
            let plume_quads = render.build_plume_quads(0.5);
            assert!(
                plume_quads.len() >= 2,
                "active thruster must produce crossed-ribbon plume PipelineQuads"
            );
        });

        // 8c. Dispatch `render_level_stage(AfterLevel)` and verify debug WorldLines & WorldQuads.
        let view = WorldRenderView::new(camera, 0.5, 100);
        let mut draw = RecordingWorldDraw::default();
        mod_api::dispatch_render_level_stage(&host, RenderLevelStage::AfterLevel, &view, &mut draw);
        assert!(
            draw.lines.len() >= 27,
            "debug overlay should emit OBB (12) + BVH (12) + RGB CoM axes (3) lines, got {}",
            draw.lines.len()
        );
    });
}

// ---------------------------------------------------------------------------
// 9. Combined static world + dynamic sub-grid PhysicsRaycast
// ---------------------------------------------------------------------------

#[test]
fn test_09_combined_physics_raycast_world_and_subgrid() {
    with_horizon_physics(|| {
        let subgrid = with_physics_api(|api| {
            // Static wall at x = 12, y = 10, z = 0.
            api.terrain_mut().set_block([12, 10, 0], "minecraft:stone");

            // Dynamic sub-grid centered at (6.0, 10.5, 0.5).
            api.spawn_named_subgrid(
                Vec3d::new(6.0, 10.5, 0.5),
                Quatd::IDENTITY,
                BodyKind::Dynamic,
                false,
                0.0,
                0,
                &[([0, 0, 0], "minecraft:iron_block".to_owned())],
            )
            .unwrap()
        });

        with_physics_api_read(|api| {
            // Ray from x = 0 toward +X: hits nearer subgrid first under `all()`.
            let hit_all = api
                .cast_ray(&RayQuery {
                    origin: [0.0, 10.5, 0.5],
                    direction: [1.0, 0.0, 0.0],
                    max_distance: 30.0,
                    filter: RaycastFilter::all(),
                })
                .expect("should hit subgrid");
            match hit_all.target {
                RaycastHitTarget::SubGridVoxel { grid, .. } => assert_eq!(grid, subgrid),
                other => panic!("expected SubGridVoxel hit first, got {other:?}"),
            }
            assert!((hit_all.distance - 5.5).abs() < 0.2);

            // Excluding the subgrid must pass through and hit the static stone block at [12, 10, 0].
            let hit_world = api
                .cast_ray(&RayQuery {
                    origin: [0.0, 10.5, 0.5],
                    direction: [1.0, 0.0, 0.0],
                    max_distance: 30.0,
                    filter: RaycastFilter::exclude_subgrid(subgrid),
                })
                .expect("should hit static world block");
            assert_eq!(
                hit_world.target,
                RaycastHitTarget::WorldBlock {
                    block_pos: [12, 10, 0],
                    material: MaterialId::STONE,
                }
            );
            assert!((hit_world.distance - 12.0).abs() < 0.1);

            // world_only() also hits [12, 10, 0].
            let hit_world_only = api
                .cast_ray(&RayQuery {
                    origin: [0.0, 10.5, 0.5],
                    direction: [1.0, 0.0, 0.0],
                    max_distance: 30.0,
                    filter: RaycastFilter::world_only(),
                })
                .expect("world_only should hit wall");
            assert!(matches!(
                hit_world_only.target,
                RaycastHitTarget::WorldBlock { .. }
            ));

            // subgrids_only() aiming parallel to wall (y = 10.5, z = 5.0) misses cleanly.
            let miss = api.cast_ray(&RayQuery {
                origin: [0.0, 10.5, 5.0],
                direction: [1.0, 0.0, 0.0],
                max_distance: 30.0,
                filter: RaycastFilter::subgrids_only(),
            });
            assert!(miss.is_none());
        });
    });
}

// ---------------------------------------------------------------------------
// 10. Motorized hinge joint PD servo integration
// ---------------------------------------------------------------------------

#[test]
fn test_10_motorized_hinge_joint_and_slider_integration() {
    with_horizon_physics(|| {
        let mut host = FakeHost::new();

        let rotor = with_physics_api(|api| {
            let base = api
                .spawn_named_subgrid(
                    Vec3d::new(0.0, 20.0, 0.0),
                    Quatd::IDENTITY,
                    BodyKind::Static,
                    false,
                    0.0,
                    0,
                    &[([0, 0, 0], "minecraft:iron_block".to_owned())],
                )
                .unwrap();
            let arm = api
                .spawn_named_subgrid(
                    Vec3d::new(1.5, 20.0, 0.0),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    0.0,
                    0,
                    &[
                        ([0, 0, 0], "minecraft:oak_planks".to_owned()),
                        ([1, 0, 0], "minecraft:oak_planks".to_owned()),
                    ],
                )
                .unwrap();

            api.add_joint(JointConstraint::Hinge(HingeJointDesc {
                body_a: base,
                body_b: arm,
                anchor_a: [0.5, 0.0, 0.0],
                anchor_b: [-1.0, 0.0, 0.0],
                axis_a: [0.0, 1.0, 0.0],
                axis_b: [0.0, 1.0, 0.0],
                min_angle_rad: -std::f64::consts::PI,
                max_angle_rad: std::f64::consts::PI,
                compliance: 0.0,
                motor: Some(JointMotor {
                    target_angle_rad: 1.2,
                    max_torque_nm: 50_000.0,
                    stiffness: 15_000.0,
                    damping: 800.0,
                }),
            }))
            .expect("attach motorized hinge");
            arm
        });

        for tick in 1..=20 {
            mod_api::dispatch_server_tick(
                &mut host,
                &lifecycle_ctx(tick),
                ServerTickPhase::ServerTickEnd {
                    world_running: true,
                },
            );
        }

        let state = with_physics_api_read(|api| api.subgrid_state(rotor).unwrap());
        let rot = Quatd::new(
            state.world_rotation[0],
            state.world_rotation[1],
            state.world_rotation[2],
            state.world_rotation[3],
        );
        let turned_angle = 2.0 * rot.w.clamp(-1.0, 1.0).acos();
        assert!(
            turned_angle > 0.15 || state.angular_velocity[1].abs() > 0.2,
            "motorized hinge PD servo must drive rotor toward target angle: angle={turned_angle:.3}, wy={:.3}",
            state.angular_velocity[1]
        );
    });
}

// ---------------------------------------------------------------------------
// 11. Material data table loading & runtime custom block material override
// ---------------------------------------------------------------------------

#[test]
fn test_11_default_materials_json_and_custom_material_override() {
    with_horizon_physics(|| {
        let table = BlockMaterialTable::from_embedded_defaults();
        assert!(
            table.block_count() >= 25,
            "default_blocks.json must define at least 25 vanilla materials, got {}",
            table.block_count()
        );
        assert!(DEFAULT_BLOCKS_JSON.contains("minecraft:iron_block"));
        assert!(DEFAULT_BLOCKS_JSON.contains("minecraft:slime_block"));

        let (_, iron) = table.resolve_block("minecraft:iron_block");
        let (_, wood) = table.resolve_block("minecraft:oak_planks");
        let (_, ice) = table.resolve_block("minecraft:ice");
        let (_, slime) = table.resolve_block("minecraft:slime_block");
        assert!(iron.density > wood.density);
        assert!(ice.friction < 0.1);
        assert!(slime.restitution >= 0.8);
        assert!(table.resolve_fluid("minecraft:water").is_some());

        with_physics_api(|api| {
            let custom_mat = VoxelMaterial {
                density: 19_300.0,
                friction: 0.65,
                restitution: 0.05,
                linear_drag: 0.01,
                angular_drag: 0.01,
            };
            api.register_block_material("my_mod:tungsten_hull", custom_mat);
            let h = api
                .spawn_named_subgrid(
                    Vec3d::new(0.0, 10.0, 0.0),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    1.0,
                    0,
                    &[([0, 0, 0], "my_mod:tungsten_hull".to_owned())],
                )
                .unwrap();
            let mass = api.subgrid_state(h).unwrap().mass_kg;
            assert!((mass - 19_300.0).abs() < 1e-3);
        });
    });
}

// ---------------------------------------------------------------------------
// 12. Horizon 26.3 packaging & clean-room compliance self-check
// ---------------------------------------------------------------------------

#[test]
fn test_12_horizon_packaging_and_clean_room_self_check() {
    let mod_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Check dual license files exist.
    assert!(mod_dir.join("LICENSE-MIT").is_file());
    assert!(mod_dir.join("LICENSE-APACHE").is_file());
    assert!(
        !mod_dir.join("build.rs").exists(),
        "build.rs is disallowed in Horizon mods"
    );

    // Check horizon-mod.toml pins and allowed fields.
    let manifest =
        std::fs::read_to_string(mod_dir.join("horizon-mod.toml")).expect("read horizon-mod.toml");
    assert!(manifest.contains("id = \"horizon_physics\""));
    assert!(manifest.contains("api = \"0.2\""));
    assert!(manifest.contains("side = \"both\""));
    assert!(manifest.contains("capabilities = [\"registries\"]"));
    assert!(manifest.contains("game = \"26.3\""));
    assert!(manifest.contains("horizon = \"f00125f61f794aaa3d4e950d301f353b35b6ee3d\""));
    assert!(manifest.contains("mixins = \"e59da9094f0c2fdda6c627fe9eff22dec6a906f0\""));
    assert!(
        !manifest.contains("[sdk]"),
        "unrecognized [sdk] table must not appear"
    );

    // Check Cargo.toml constraints (no parent escape, no [features], MIT OR Apache-2.0).
    let cargo_toml = std::fs::read_to_string(mod_dir.join("Cargo.toml")).expect("read Cargo.toml");
    assert!(cargo_toml.contains("license = \"MIT OR Apache-2.0\""));
    assert!(
        !cargo_toml.contains(".."),
        "Cargo.toml must not escape the mod directory via `..`"
    );
    assert!(
        !cargo_toml.contains("[features]"),
        "Cargo.toml must not declare [features]"
    );

    // Check crate root uses deny(unsafe_code) instead of forbid(unsafe_code).
    let lib_rs = std::fs::read_to_string(mod_dir.join("src/lib.rs")).expect("read src/lib.rs");
    assert!(lib_rs.contains("#![deny(unsafe_code)]"));
    assert!(
        !lib_rs.contains("forbid(unsafe_code)"),
        "#[horizon_mod] requires deny(unsafe_code), never forbid(unsafe_code)"
    );

    // Check WGSL shaders exist at both canonical and Horizon pipeline asset paths.
    let plume_main =
        std::fs::read_to_string(mod_dir.join("assets/horizon_physics/shaders/fx_plume.wgsl"))
            .expect("read assets/horizon_physics/shaders/fx_plume.wgsl");
    let plume_pipe = std::fs::read_to_string(
        mod_dir.join("assets/horizon_physics/shaders/pipeline/fx_plume.wgsl"),
    )
    .expect("read assets/horizon_physics/shaders/pipeline/fx_plume.wgsl");
    assert_eq!(plume_main, plume_pipe);
    validate_plume_shader_contract(&plume_main).unwrap();

    // Ensure HorizonPhysicsMod implements Default + Hooks cleanly.
    let _mod_instance = HorizonPhysicsMod;
}

// ---------------------------------------------------------------------------
// 13. Red-team regressions: lock-free callbacks, CCD/gravity persistence,
//     GridSplit block-name propagation, standalone network brick sync, and
//     nearest-first collision exclusion prioritization
// ---------------------------------------------------------------------------

#[test]
fn test_13_redteam_regressions_lock_free_ccd_split_sync_and_exclusions() {
    use mod_horizon_physics::api::step_global_physics;
    use mod_horizon_physics::entity_rider::{EntityRiderManager, MAX_COLLISION_EXCLUSIONS};
    use mod_horizon_physics::sync::PersistedSubGridWorld;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // 13a. Reentrant read-lock inside ForceProvider callback during `step_global_physics`
    // must succeed without deadlock.
    with_horizon_physics(|| {
        struct RadarProbeProvider {
            hits: Arc<AtomicUsize>,
        }
        impl ForceProvider for RadarProbeProvider {
            fn provider_id(&self) -> &'static str {
                "audit:radar_probe"
            }
            fn compute_force(
                &self,
                grid: SubGridHandle,
                ctx: &ForceQueryContext,
            ) -> Option<ForceContribution> {
                let hit = with_physics_api_read(|api| {
                    api.cast_ray(&RayQuery {
                        origin: ctx.world_position,
                        direction: [0.0, -1.0, 0.0],
                        max_distance: 64.0,
                        filter: RaycastFilter::exclude_subgrid(grid),
                    })
                });
                if hit.is_some() {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                }
                None
            }
        }

        let hits = Arc::new(AtomicUsize::new(0));
        with_physics_api(|api| {
            api.terrain_mut().set_flat_floor(Some(60));
            let craft = api
                .spawn_named_subgrid(
                    Vec3d::new(0.0, 72.0, 0.0),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    true,
                    0.35,
                    42,
                    &[([0, 0, 0], "minecraft:iron_block".to_owned())],
                )
                .unwrap();
            assert!(api.attach_force_provider(
                craft,
                Box::new(RadarProbeProvider {
                    hits: Arc::clone(&hits),
                })
            ));
        });

        let metrics = step_global_physics(1.0 / 20.0);
        assert_eq!(metrics.total_bodies, 1);
        assert_eq!(hits.load(Ordering::Relaxed), 1);

        // 13b. Persistence preserves `enable_ccd` and `gravity_scale` faithfully.
        let persisted = with_physics_api_read(PersistedSubGridWorld::capture);
        assert_eq!(persisted.bodies.len(), 1);
        assert!(persisted.bodies[0].enable_ccd);
        assert!((persisted.bodies[0].gravity_scale - 0.35).abs() < 1e-12);

        let mut restored = PhysicsApi::default();
        let restored_ids = persisted.restore_into(&mut restored).unwrap();
        let snap = restored
            .world()
            .body_snapshot(restored_ids[0].body_id())
            .unwrap();
        assert!(snap.enable_ccd);
        assert!((snap.gravity_scale - 0.35).abs() < 1e-12);
    });

    // 13c. GridSplit preserves original block names across both surviving parent & child bodies.
    let mut api = PhysicsApi::default();
    let ship = api
        .spawn_named_subgrid(
            Vec3d::new(0.0, 80.0, 0.0),
            Quatd::IDENTITY,
            BodyKind::Dynamic,
            false,
            1.0,
            0,
            &[
                ([0, 0, 0], "minecraft:iron_block".to_owned()),
                ([1, 0, 0], "minecraft:stone".to_owned()),
                ([2, 0, 0], "minecraft:oak_planks".to_owned()),
            ],
        )
        .unwrap();
    api.remove_voxel(ship, [1, 0, 0]).unwrap();
    let active = api.active_subgrids();
    assert_eq!(
        active.len(),
        2,
        "splitting bridge voxel must create 2 bodies"
    );
    let mut seen_blocks: Vec<String> = Vec::new();
    for handle in &active {
        let names = api
            .subgrid_block_names(*handle)
            .expect("every split body must keep voxel_block_names");
        assert_eq!(names.len(), 1);
        seen_blocks.extend(names.values().cloned());
    }
    seen_blocks.sort();
    assert_eq!(
        seen_blocks,
        vec![
            "minecraft:iron_block".to_owned(),
            "minecraft:oak_planks".to_owned()
        ]
    );

    // 13d. Standalone remote ClientRenderState renders full voxel batches purely from
    // `DeltaSyncTracker` wire packet without shared `sync_from_api`, and stationary
    // repeat emits zero duplicate geometry traffic.
    let mut tracker = DeltaSyncTracker::new();
    let delta = tracker.build_delta_packet(&api).expect("initial delta");
    assert_eq!(delta.geometries.len(), 2);
    let wire = delta.encode().unwrap();
    let decoded = PoseSnapshotPacket::decode(&wire).unwrap();

    let mut remote = ClientRenderState::new();
    remote.apply_pose_packet(&decoded);
    let batches = remote.build_brick_instances([0.0, 80.0, 0.0], 1.0);
    assert_eq!(batches.len(), 2);
    let mut rendered_blocks: Vec<String> = batches.iter().map(|b| b.block_id.clone()).collect();
    rendered_blocks.sort();
    assert_eq!(rendered_blocks, seen_blocks);
    assert!(
        tracker.build_delta_packet(&api).is_none(),
        "stationary post-split world must emit zero delta traffic"
    );

    // 13e. Nearest-first collision exclusions under >64 overlapping bodies.
    let mut crowd = PhysicsApi::default();
    let entity_aabb = Aabbd::new(Vec3d::new(-0.3, 10.0, -0.3), Vec3d::new(0.3, 11.8, 0.3));
    for i in 0..90usize {
        let dist_x = 8.0 - (i as f64) * 0.085;
        crowd
            .spawn_named_subgrid(
                Vec3d::new(dist_x, 10.5, 0.0),
                Quatd::IDENTITY,
                BodyKind::Static,
                false,
                0.0,
                i as u64,
                &[([0, 0, 0], "minecraft:stone".to_owned())],
            )
            .unwrap();
    }
    let exclusions =
        EntityRiderManager::compute_collision_exclusions(crowd.world(), entity_aabb, 9.0);
    assert_eq!(exclusions.len(), MAX_COLLISION_EXCLUSIONS);
    // Closest body (i = 89 -> dist_x ~= 0.435) must be included near index 0!
    assert!(
        exclusions[0][0] < 1.0 && exclusions[0][3] > 0.0,
        "nearest subgrid must rank first in bounded collision exclusions: {:?}",
        exclusions[0]
    );
}

// ---------------------------------------------------------------------------
// 14. Round-2 heterogeneous review regressions (propulsion reentrancy,
//     lock-free terrain prefetch, WorldQuad fallback, chunked <1MiB sync,
//     unacked commit guard, and direct world_mut geometry_revision)
// ---------------------------------------------------------------------------

#[test]
fn test_14_codex_round2_p1_p2_regressions() {
    use horizon_physics_core::{VoxelCoord, VoxelEntry};
    use mod_horizon_physics::api::{PropulsionContext, step_global_physics_with_sampler};
    use mod_horizon_physics::sync::MAX_WIRE_PACKET_BYTES;
    use mod_horizon_physics::terrain_cache::HostTerrainSampler;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // 14a & 14b: Propulsion callback reentrantly calls `with_physics_api(set_propulsion_throttle)`
    // AND custom `HostTerrainSampler` reentrantly calls `with_physics_api_read` — zero deadlock!
    with_horizon_physics(|| {
        struct SelfModulatingJet {
            throttle: f64,
            requested_ramp: Arc<AtomicBool>,
        }
        impl Propulsion for SelfModulatingJet {
            fn propulsion_id(&self) -> &'static str {
                "audit:self_throttle_jet"
            }
            fn set_throttle(&mut self, throttle: f64) {
                self.throttle = throttle.clamp(0.0, 1.0);
            }
            fn throttle(&self) -> f64 {
                self.throttle
            }
            fn evaluate_thrust(
                &self,
                grid: SubGridHandle,
                _ctx: &PropulsionContext,
            ) -> PropulsionOutput {
                if !self.requested_ramp.swap(true, Ordering::SeqCst) {
                    with_physics_api(|api| {
                        assert!(api.set_propulsion_throttle(grid, "audit:self_throttle_jet", 0.95));
                    });
                }
                PropulsionOutput {
                    local_mount: [0.0, 0.0, 0.0],
                    local_force: [0.0, 0.0, 10_000.0 * self.throttle],
                    local_torque: [0.0; 3],
                    fuel_burn_rate: self.throttle,
                }
            }
        }

        struct ReentrantSampler {
            reads_ok: Arc<AtomicUsize>,
        }
        impl HostTerrainSampler for ReentrantSampler {
            fn sample_block(&self, _x: i32, y: i32, _z: i32) -> Option<(String, Option<Aabbd>)> {
                let cnt = with_physics_api_read(|api| api.world().body_count());
                if cnt >= 1 {
                    self.reads_ok.fetch_add(1, Ordering::Relaxed);
                }
                if y < 60 {
                    Some(("minecraft:stone".to_owned(), None))
                } else {
                    None
                }
            }
        }

        let requested_ramp = Arc::new(AtomicBool::new(false));
        let reads_ok = Arc::new(AtomicUsize::new(0));
        let craft = with_physics_api(|api| {
            let h = api
                .spawn_named_subgrid(
                    Vec3d::new(0.0, 62.0, 0.0),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    1.0,
                    0,
                    &[([0, 0, 0], "minecraft:iron_block".to_owned())],
                )
                .unwrap();
            assert!(api.attach_propulsion(
                h,
                Box::new(SelfModulatingJet {
                    throttle: 0.1,
                    requested_ramp: Arc::clone(&requested_ramp),
                })
            ));
            h
        });

        let sampler = ReentrantSampler {
            reads_ok: Arc::clone(&reads_ok),
        };
        let _ = step_global_physics_with_sampler(1.0 / 20.0, Some(&sampler));
        let burn_tick1 = with_physics_api_read(|api| api.last_fuel_burn_total());
        assert!(reads_ok.load(Ordering::Relaxed) > 0);
        assert!(requested_ramp.load(Ordering::SeqCst));

        // Step 2 must observe the queued deferred throttle (`0.95` vs initial `0.1`).
        let _ = step_global_physics_with_sampler(1.0 / 20.0, Some(&sampler));
        let burn_tick2 = with_physics_api_read(|api| api.last_fuel_burn_total());
        assert!(
            burn_tick2 > burn_tick1 * 5.0,
            "deferred throttle must take effect by step 2: burn1={burn_tick1}, burn2={burn_tick2}"
        );
        let state = with_physics_api_read(|api| api.subgrid_state(craft).unwrap());
        assert!(state.world_position[1] > 59.5);
    });

    // 14c: `standalone_models` registers vanilla blocks AND unknown unregistered mod blocks
    // fall back to shaded `WorldQuad` faces (never invisible).
    with_horizon_physics(|| {
        let mut models = Vec::new();
        ClientRenderState::register_standalone_models(&mut models);
        assert!(models.iter().any(|m| m == "minecraft:block/stone"));
        assert!(models.iter().any(|m| m == "minecraft:block/iron_block"));

        let host = FakeHost::new();
        with_physics_api(|api| {
            api.spawn_named_subgrid(
                Vec3d::new(0.0, 70.0, 0.0),
                Quatd::IDENTITY,
                BodyKind::Dynamic,
                false,
                0.0,
                0,
                &[([0, 0, 0], "other_mod:unregistered_thruster".to_owned())],
            )
            .unwrap();
            with_client_render(|r| r.sync_from_api(api));
        });

        let view = WorldRenderView::new([0.0, 70.0, 4.0], 1.0, 10);
        let mut draw = RecordingWorldDraw::default();
        mod_api::dispatch_render_level_stage(
            &host,
            RenderLevelStage::AfterOpaqueBlocks,
            &view,
            &mut draw,
        );
        assert_eq!(
            draw.quads.len(),
            6,
            "unregistered custom block must render 6 shaded fallback WorldQuads"
        );
        // Ensure FakeHost ref isn't unused if warning rules trigger.
        let _ = host.is_loaded("horizon_physics");
    });

    // 14d: Direct `world_mut()` edit bumps `geometry_revision`, unacked delta retains changes,
    // and oversized bodies safely chunk strictly below `MAX_WIRE_PACKET_BYTES` (< 1 MiB).
    let mut api = PhysicsApi::default();
    let long_block_name =
        "custom_long_namespace:reinforced_aerospace_composite_hull_block".to_owned();
    let mut blocks = Vec::with_capacity(12_000);
    for z in 0..30i32 {
        for y in 0..20i32 {
            for x in 0..20i32 {
                blocks.push(([x, y, z], long_block_name.clone()));
            }
        }
    }
    let giant = api
        .spawn_named_subgrid(
            Vec3d::new(0.0, 100.0, 0.0),
            Quatd::IDENTITY,
            BodyKind::Static,
            false,
            0.0,
            0,
            &blocks,
        )
        .unwrap();
    let rev_before = api.geometry_revision(giant.body_id());
    assert!(api.world_mut().set_body_voxel(
        giant.body_id(),
        VoxelCoord::new(21, 0, 0),
        Some(VoxelEntry::from_material(MaterialId::IRON)),
    ));
    assert!(
        api.geometry_revision(giant.body_id()) > rev_before,
        "direct world_mut().set_body_voxel must bump geometry_revision"
    );

    let mut tracker = DeltaSyncTracker::new();
    let unacked = tracker.prepare_delta_packets(&api).expect("delta ready");
    assert!(
        unacked.packets.len() >= 2,
        "giant hull must split into multiple <768KiB chunks"
    );
    for p in &unacked.packets {
        let encoded = p.encode().unwrap();
        assert!(
            encoded.len() <= MAX_WIRE_PACKET_BYTES,
            "chunk size {} exceeds MAX_WIRE_PACKET_BYTES",
            encoded.len()
        );
    }
    // Without commit_delta (e.g. send failed), preparing again still emits full delta!
    let retry = tracker
        .prepare_delta_packets(&api)
        .expect("unacked delta must retry");
    assert_eq!(retry.packets.len(), unacked.packets.len());
    tracker.commit_delta(retry);
    assert!(
        tracker.prepare_delta_packets(&api).is_none(),
        "committed tracker must emit zero idle delta"
    );

    // Feed all chunks of `build_chunked_full_packets` into fresh `ClientRenderState`
    let full_chunks = tracker.build_chunked_full_packets(&api);
    assert!(full_chunks.len() >= 2);
    let mut remote = ClientRenderState::new();
    for chunk in &full_chunks {
        let wire = chunk.encode().unwrap();
        assert!(wire.len() <= MAX_WIRE_PACKET_BYTES);
        let decoded = PoseSnapshotPacket::decode(&wire).unwrap();
        remote.apply_pose_packet(&decoded);
    }
    let batches = remote.build_brick_instances([0.0, 100.0, 0.0], 1.0);
    let total_instances: usize = batches.iter().map(|b| b.instances.len()).sum();
    assert_eq!(
        total_instances, 12_001,
        "remote client must reassemble all chunked voxels without loss"
    );

    // 14e: `set_propulsion_throttle` / `set_propulsion_vector` called inside `with_physics_api`
    // defers external trait callbacks until after the global write lock is released.
    struct ReentrantSetterPropulsion {
        throttle: f64,
        observed_read: Arc<AtomicUsize>,
    }
    impl Propulsion for ReentrantSetterPropulsion {
        fn propulsion_id(&self) -> &'static str {
            "audit:reentrant_setter"
        }

        fn set_throttle(&mut self, throttle: f64) {
            self.throttle = throttle.clamp(0.0, 1.0);
            let count = with_physics_api_read(|api| api.world().body_count());
            self.observed_read.store(count, Ordering::SeqCst);
        }

        fn throttle(&self) -> f64 {
            self.throttle
        }

        fn set_vector_angles(&mut self, _pitch_rad: f64, _yaw_rad: f64) {
            let count = with_physics_api_read(|api| api.world().body_count());
            self.observed_read.fetch_add(count, Ordering::SeqCst);
        }

        fn evaluate_thrust(&self, _: SubGridHandle, _: &PropulsionContext) -> PropulsionOutput {
            PropulsionOutput {
                local_mount: [0.0, 0.0, 0.0],
                local_force: [0.0, 200.0 * self.throttle, 0.0],
                local_torque: [0.0; 3],
                fuel_burn_rate: self.throttle,
            }
        }
    }

    with_horizon_physics(|| {
        let observed_read = Arc::new(AtomicUsize::new(0));
        let craft = with_physics_api(|api| {
            let h = api
                .spawn_named_subgrid(
                    Vec3d::new(0.0, 90.0, 0.0),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    1.0,
                    0,
                    &[([0, 0, 0], "minecraft:iron_block".to_owned())],
                )
                .unwrap();
            assert!(api.attach_propulsion(
                h,
                Box::new(ReentrantSetterPropulsion {
                    throttle: 0.0,
                    observed_read: Arc::clone(&observed_read),
                })
            ));
            h
        });
        assert!(with_physics_api(|api| {
            api.set_propulsion_throttle(craft, "audit:reentrant_setter", 0.75)
                && api.set_propulsion_vector(craft, "audit:reentrant_setter", 0.1, -0.1)
        }));
        assert_eq!(
            observed_read.load(Ordering::SeqCst),
            2,
            "deferred propulsion setters must run lock-free immediately upon write-lock release"
        );
    });

    // 14f: Multi-body uncached terrain spans strictly beyond 16,384 cells without re-entering write lock.
    struct ReentrantBatchSampler {
        calls: Arc<AtomicUsize>,
    }
    impl HostTerrainSampler for ReentrantBatchSampler {
        fn sample_block(&self, _: i32, y: i32, _: i32) -> Option<(String, Option<Aabbd>)> {
            let _ = with_physics_api_read(|api| api.world().body_count());
            self.calls.fetch_add(1, Ordering::Relaxed);
            if y < 60 {
                Some(("minecraft:stone".to_owned(), None))
            } else {
                None
            }
        }
    }

    with_horizon_physics(|| {
        let mut cube_blocks = Vec::with_capacity(27);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    cube_blocks.push(([dx, dy, dz], "minecraft:stone".to_owned()));
                }
            }
        }
        with_physics_api(|api| {
            for i in 0..64 {
                api.spawn_named_subgrid(
                    Vec3d::new(f64::from(i) * 80.0, 80.0, 0.0),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    1.0,
                    0,
                    &cube_blocks,
                )
                .unwrap();
            }
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let sampler = ReentrantBatchSampler {
            calls: Arc::clone(&calls),
        };
        let _ = step_global_physics_with_sampler(1.0 / 20.0, Some(&sampler));
        let total_calls = calls.load(Ordering::Relaxed);
        assert!(
            total_calls > 16_384,
            "expected >16,384 lock-free terrain samples across 64 3x3x3 bodies, got {total_calls}"
        );
    });

    // 14g: Late-joiner receiving a body created between deltas must receive removal if
    // that body despawns before the next delta tick.
    let mut offline_api = PhysicsApi::default();
    let base = offline_api
        .spawn_named_subgrid(
            Vec3d::new(0.0, 80.0, 0.0),
            Quatd::IDENTITY,
            BodyKind::Static,
            false,
            0.0,
            0,
            &[([0, 0, 0], "minecraft:stone".to_owned())],
        )
        .unwrap();
    let mut delta_tracker = DeltaSyncTracker::new();
    let init_pending = delta_tracker.prepare_delta_packets(&offline_api).unwrap();
    delta_tracker.commit_delta(init_pending);

    let added_offline = offline_api
        .spawn_named_subgrid(
            Vec3d::new(100.0, 80.0, 0.0),
            Quatd::IDENTITY,
            BodyKind::Static,
            false,
            0.0,
            0,
            &[([0, 0, 0], "minecraft:stone".to_owned())],
        )
        .unwrap();
    let mut late_client = ClientRenderState::new();
    for pkt in delta_tracker.build_chunked_full_packets(&offline_api) {
        late_client.apply_pose_packet(&pkt);
    }
    assert!(late_client.body_state(base.body_id()).is_some());
    assert!(late_client.body_state(added_offline.body_id()).is_some());

    assert!(
        offline_api
            .world_mut()
            .despawn_body(added_offline.body_id())
    );
    let cleanup_pending = delta_tracker
        .prepare_delta_packets(&offline_api)
        .expect("must emit removal for body announced via full baseline before despawn");
    assert!(
        cleanup_pending
            .packets
            .iter()
            .any(|p| p.removed.contains(&added_offline.body_id()))
    );
    for pkt in &cleanup_pending.packets {
        late_client.apply_pose_packet(pkt);
    }
    delta_tracker.commit_delta(cleanup_pending);
    assert!(
        late_client.body_state(added_offline.body_id()).is_none(),
        "late joiner must remove ghost body announced in chunked full baseline"
    );
    assert!(delta_tracker.prepare_delta_packets(&offline_api).is_none());

    // 14h: Substantially test write-lock `step_with_sampler` safety (`dt > 0` actually querying terrain),
    // zero air poisoning along a traversed path, hard-braking substep swept prefetch, unloaded `None`
    // non-caching until chunk load, and resync accumulation across `commit_delta` + partial broadcast failure.
    struct WallAt240And312Sampler {
        chunk_400_loaded: Arc<AtomicBool>,
    }
    impl HostTerrainSampler for WallAt240And312Sampler {
        fn sample_block(&self, x: i32, y: i32, z: i32) -> Option<(String, Option<Aabbd>)> {
            let _ = with_physics_api_read(|api| api.world().body_count());
            let in_240_or_312 = (240..=242).contains(&x) || (312..=314).contains(&x);
            let in_400 = self.chunk_400_loaded.load(Ordering::SeqCst) && (400..=402).contains(&x);
            if (in_240_or_312 || in_400) && (72..=92).contains(&y) && (-4..=4).contains(&z) {
                Some(("minecraft:stone".to_owned(), None))
            } else {
                None
            }
        }
    }

    struct HardBrakeForce;
    impl ForceProvider for HardBrakeForce {
        fn provider_id(&self) -> &'static str {
            "audit:hard_brake"
        }
        fn compute_force(
            &self,
            _grid: SubGridHandle,
            ctx: &ForceQueryContext,
        ) -> Option<ForceContribution> {
            // Opposes +X velocity strongly enough that end-of-tick speed reaches 0 while initial
            // substeps still travel +15..20 blocks past x=300.5 across x=312!
            Some(ForceContribution::world_force_at_point(
                [-ctx.mass_kg * 16_000.0, 0.0, 0.0],
                ctx.world_position,
            ))
        }
    }

    with_horizon_physics(|| {
        let wall_sampler = WallAt240And312Sampler {
            chunk_400_loaded: Arc::new(AtomicBool::new(false)),
        };
        let proj = with_physics_api(|api| {
            let h = api
                .spawn_named_subgrid(
                    Vec3d::new(238.5, 80.5, 0.5),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    0.0,
                    0,
                    &[([0, 0, 0], "minecraft:iron_block".to_owned())],
                )
                .unwrap();
            api.world_mut().set_body_velocity(
                h.body_id(),
                Vec3d::new(800.0, 0.0, 0.0),
                Vec3d::ZERO,
            );
            h
        });
        // Non-zero dt=0.05 under write lock: body moves right across [240, 80, 0], yet
        // `step_with_sampler` forces `None` sampler -> zero reentry panic and zero cache write!
        let proj2 = with_physics_api(|api| {
            let _ = api.step_with_sampler(1.0 / 20.0, &wall_sampler);
            assert!(
                !api.terrain().has_cached_cell(240, 80, 0),
                "step_with_sampler under write lock must ignore sampler and not populate cache"
            );
            let moved = api.subgrid_state(proj).unwrap();
            assert!(
                moved.world_position[0] > 245.0,
                "body actually traversed x=240 during unsampled write-lock step without poisoning air: {}",
                moved.world_position[0]
            );
            assert!(api.world_mut().despawn_body(proj.body_id()));
            let h2 = api
                .spawn_named_subgrid(
                    Vec3d::new(200.5, 80.5, 0.5),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    0.0,
                    0,
                    &[([0, 0, 0], "minecraft:iron_block".to_owned())],
                )
                .unwrap();
            api.world_mut().set_body_velocity(
                h2.body_id(),
                Vec3d::new(800.0, 0.0, 0.0),
                Vec3d::ZERO,
            );
            h2
        });
        assert!(with_physics_api_read(|api| !api
            .terrain()
            .has_cached_cell(240, 80, 0)));

        // Advance 3 ticks (0.15s * 800 m/s = 120 blocks uncapped travel, would reach x=320.5):
        // x=240..242 wall stops the projectile well below x=241.0.
        for _ in 0..3 {
            let _ = step_global_physics_with_sampler(1.0 / 20.0, Some(&wall_sampler));
        }
        let final_state = with_physics_api_read(|api| {
            assert!(
                api.terrain().has_cached_cell(240, 80, 0),
                "non-poisoned wall cell must be sampled and cached once sampler is active"
            );
            api.subgrid_state(proj2).unwrap()
        });
        assert!(
            final_state.world_position[0] < 241.0,
            "high-speed projectile must collide at x=240 over 3 ticks (vs 320.5 free): pos_x={}",
            final_state.world_position[0]
        );

        // Hard-braking bullet: 800 m/s at x=300.5 with -16,000 m/s^2 braking (v_end <= 0),
        // yet substep kinematics hit wall at x=312!
        let braker = with_physics_api(|api| {
            let h = api
                .spawn_named_subgrid(
                    Vec3d::new(300.5, 80.5, 0.5),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    0.0,
                    0,
                    &[([0, 0, 0], "minecraft:iron_block".to_owned())],
                )
                .unwrap();
            api.world_mut().set_body_velocity(
                h.body_id(),
                Vec3d::new(800.0, 0.0, 0.0),
                Vec3d::ZERO,
            );
            assert!(api.attach_force_provider(h, Box::new(HardBrakeForce)));
            h
        });
        let _ = step_global_physics_with_sampler(1.0 / 20.0, Some(&wall_sampler));
        let brake_state = with_physics_api_read(|api| {
            assert!(
                api.terrain().has_cached_cell(312, 80, 0),
                "hard-braking substep sweep must prefetch forward path around x=312"
            );
            api.subgrid_state(braker).unwrap()
        });
        assert!(
            brake_state.world_position[0] < 312.5,
            "hard-braking body must see wall at x=312 during early substeps: pos_x={}",
            brake_state.world_position[0]
        );

        // Unloaded chunk `None` must NOT be permanently cached as air until chunk loads.
        with_physics_api(|api| {
            api.spawn_named_subgrid(
                Vec3d::new(400.5, 80.5, 0.5),
                Quatd::IDENTITY,
                BodyKind::Dynamic,
                false,
                0.0,
                0,
                &[([0, 0, 0], "minecraft:iron_block".to_owned())],
            )
            .unwrap();
        });
        let _ = step_global_physics_with_sampler(1.0 / 20.0, Some(&wall_sampler));
        assert!(
            with_physics_api_read(|api| !api.terrain().has_cached_cell(400, 80, 0)),
            "unloaded chunk returning None must not poison cache as air"
        );
        wall_sampler.chunk_400_loaded.store(true, Ordering::SeqCst);
        let _ = step_global_physics_with_sampler(1.0 / 20.0, Some(&wall_sampler));
        assert!(
            with_physics_api_read(|api| api.terrain().has_cached_cell(400, 80, 0)),
            "once chunk loads and returns solid stone, cache must immediately capture it"
        );
    });

    // Substep=1 severe reverse braking: 800 m/s at x=350.5 with -32,000 m/s^2 acceleration
    // reverses single-substep velocity to -800 m/s (~ -40 blocks to x=310.5) past x=312 wall!
    struct ExtremeReverseBrakeForce;
    impl ForceProvider for ExtremeReverseBrakeForce {
        fn provider_id(&self) -> &'static str {
            "audit:extreme_reverse_brake"
        }
        fn compute_force(
            &self,
            _grid: SubGridHandle,
            ctx: &ForceQueryContext,
        ) -> Option<ForceContribution> {
            Some(ForceContribution::world_force_at_point(
                [-ctx.mass_kg * 32_000.0, 0.0, 0.0],
                ctx.world_position,
            ))
        }
    }

    with_horizon_physics(|| {
        with_physics_api(|api| {
            let cfg = PhysicsConfig {
                substeps: 1,
                gravity: Vec3d::ZERO,
                ..PhysicsConfig::default()
            };
            api.reset_with_config(cfg);
        });
        let wall_312 = WallAt240And312Sampler {
            chunk_400_loaded: Arc::new(AtomicBool::new(false)),
        };
        let rev_body = with_physics_api(|api| {
            let h = api
                .spawn_named_subgrid(
                    Vec3d::new(350.5, 80.5, 0.5),
                    Quatd::IDENTITY,
                    BodyKind::Dynamic,
                    false,
                    0.0,
                    0,
                    &[([0, 0, 0], "minecraft:iron_block".to_owned())],
                )
                .unwrap();
            api.world_mut().set_body_velocity(
                h.body_id(),
                Vec3d::new(800.0, 0.0, 0.0),
                Vec3d::ZERO,
            );
            assert!(api.attach_force_provider(h, Box::new(ExtremeReverseBrakeForce)));
            h
        });
        let _ = step_global_physics_with_sampler(1.0 / 20.0, Some(&wall_312));
        let rev_state = with_physics_api_read(|api| {
            assert!(
                api.terrain().has_cached_cell(312, 80, 0),
                "substeps=1 swept prefetch must honor actual config.substeps=1 backward travel"
            );
            api.subgrid_state(rev_body).unwrap()
        });
        assert!(
            rev_state.world_position[0] >= 312.5,
            "substeps=1 severe reverse brake must collide with wall at x=312 instead of tunneling to 310.5: pos_x={}",
            rev_state.world_position[0]
        );
    });

    // Real end-to-end `server_tick_sync_with` coverage across:
    // (a) multi-round commit_delta accumulation onto already-queued resync players;
    // (b) all-player multi-chunk broadcast failure (`delivered == 0`) where chunk 0 already
    //     reached clients before chunk 1 hit Backpressure, followed by despawn before next tick!
    let mut multi_api = PhysicsApi::default();
    let persistent_body = multi_api
        .spawn_named_subgrid(
            Vec3d::new(0.0, 80.0, 0.0),
            Quatd::IDENTITY,
            BodyKind::Static,
            false,
            0.0,
            0,
            &[([0, 0, 0], "minecraft:stone".to_owned())],
        )
        .unwrap();
    let ephemeral_1 = multi_api
        .spawn_named_subgrid(
            Vec3d::new(50.0, 80.0, 0.0),
            Quatd::IDENTITY,
            BodyKind::Static,
            false,
            0.0,
            0,
            &[([0, 0, 0], "minecraft:stone".to_owned())],
        )
        .unwrap();
    let ephemeral_2 = multi_api
        .spawn_named_subgrid(
            Vec3d::new(60.0, 80.0, 0.0),
            Quatd::IDENTITY,
            BodyKind::Static,
            false,
            0.0,
            0,
            &[([0, 0, 0], "minecraft:stone".to_owned())],
        )
        .unwrap();

    let mut fake_sync = FakeHost::new();
    let uuid_a: u128 = 0xDEAD_BEEF_CAFE_0001;
    let uuid_b: u128 = 0xDEAD_BEEF_CAFE_0002;
    let lvl_ref = fake_sync.level.level();
    fake_sync.players.join(uuid_a, 101, lvl_ref);
    fake_sync.players.join(uuid_b, 102, lvl_ref);
    let players = (&fake_sync as &dyn ServerHost).players().server_players();
    assert_eq!(players.len(), 2);

    let mut resync_tracker = DeltaSyncTracker::new();
    let mut clients: BTreeMap<u128, ClientRenderState> = [
        (uuid_a, ClientRenderState::new()),
        (uuid_b, ClientRenderState::new()),
    ]
    .into_iter()
    .collect();

    // Tick 1: Both clients receive baseline (persistent, ephemeral_1, ephemeral_2).
    server_tick_sync_with(&multi_api, &mut resync_tracker, &players, |p, wire| {
        let pkt = PoseSnapshotPacket::decode(wire).unwrap();
        clients.get_mut(&p.uuid()).unwrap().apply_pose_packet(&pkt);
        Ok::<_, PayloadSendError>(())
    });
    assert!(clients[&uuid_b].body_state(ephemeral_1.body_id()).is_some());
    assert!(clients[&uuid_b].body_state(ephemeral_2.body_id()).is_some());

    // Tick 2: Despawn ephemeral_1; client A succeeds while client B fails -> B queued!
    assert!(multi_api.world_mut().despawn_body(ephemeral_1.body_id()));
    server_tick_sync_with(&multi_api, &mut resync_tracker, &players, |p, wire| {
        if p.uuid() == uuid_b {
            Err(PayloadSendError::Backpressure)
        } else {
            let pkt = PoseSnapshotPacket::decode(wire).unwrap();
            clients.get_mut(&p.uuid()).unwrap().apply_pose_packet(&pkt);
            Ok(())
        }
    });
    assert_eq!(resync_tracker.pending_resync_player_count(), 1);
    assert!(clients[&uuid_a].body_state(ephemeral_1.body_id()).is_none());

    // Tick 3: While B is ALREADY queued and offline/still failing resync, despawn ephemeral_2;
    // healthy client A receives & commits delta -> B accumulates ephemeral_2 automatically!
    assert!(multi_api.world_mut().despawn_body(ephemeral_2.body_id()));
    server_tick_sync_with(&multi_api, &mut resync_tracker, &players, |p, wire| {
        if p.uuid() == uuid_b {
            Err(PayloadSendError::Backpressure)
        } else {
            let pkt = PoseSnapshotPacket::decode(wire).unwrap();
            clients.get_mut(&p.uuid()).unwrap().apply_pose_packet(&pkt);
            Ok(())
        }
    });
    assert_eq!(resync_tracker.pending_resync_player_count(), 1);

    // Tick 4: Recover client B via `server_tick_sync_with` -> B clears BOTH ephemeral_1 & ephemeral_2!
    server_tick_sync_with(&multi_api, &mut resync_tracker, &players, |p, wire| {
        let pkt = PoseSnapshotPacket::decode(wire).unwrap();
        clients.get_mut(&p.uuid()).unwrap().apply_pose_packet(&pkt);
        Ok::<_, PayloadSendError>(())
    });
    assert_eq!(resync_tracker.pending_resync_player_count(), 0);
    assert!(clients[&uuid_b].body_state(ephemeral_1.body_id()).is_none());
    assert!(clients[&uuid_b].body_state(ephemeral_2.body_id()).is_none());

    // Tick 5: Spawn an oversized 12,000-voxel body (`blocks` -> >= 2 chunks).
    // Allow chunk 0 through to BOTH players, then fail chunk 1 for ALL players (`delivered == 0`)!
    let giant_ephemeral = multi_api
        .spawn_named_subgrid(
            Vec3d::new(70.0, 80.0, 0.0),
            Quatd::IDENTITY,
            BodyKind::Static,
            false,
            0.0,
            0,
            &blocks,
        )
        .unwrap();
    let mut per_player_chunks: BTreeMap<u128, usize> = BTreeMap::new();
    server_tick_sync_with(&multi_api, &mut resync_tracker, &players, |p, wire| {
        let count = per_player_chunks.entry(p.uuid()).or_insert(0);
        *count += 1;
        if *count >= 2 {
            Err(PayloadSendError::Backpressure)
        } else {
            let pkt = PoseSnapshotPacket::decode(wire).unwrap();
            clients.get_mut(&p.uuid()).unwrap().apply_pose_packet(&pkt);
            Ok(())
        }
    });
    // Both clients received chunk 0 (spawning partial `giant_ephemeral`), yet `delivered == 0`!
    assert!(
        clients[&uuid_a]
            .body_state(giant_ephemeral.body_id())
            .is_some()
    );
    assert!(
        clients[&uuid_b]
            .body_state(giant_ephemeral.body_id())
            .is_some()
    );
    assert_eq!(
        resync_tracker.pending_resync_player_count(),
        2,
        "server_tick_sync_with must queue all failing players even when delivered == 0"
    );

    // Tick 6: Despawn `giant_ephemeral` BEFORE next sync, then restore transport and run `server_tick_sync_with`.
    assert!(
        multi_api
            .world_mut()
            .despawn_body(giant_ephemeral.body_id())
    );
    server_tick_sync_with(&multi_api, &mut resync_tracker, &players, |p, wire| {
        let pkt = PoseSnapshotPacket::decode(wire).unwrap();
        clients.get_mut(&p.uuid()).unwrap().apply_pose_packet(&pkt);
        Ok::<_, PayloadSendError>(())
    });
    assert_eq!(resync_tracker.pending_resync_player_count(), 0);
    assert!(
        clients[&uuid_a]
            .body_state(persistent_body.body_id())
            .is_some()
    );
    assert!(
        clients[&uuid_b]
            .body_state(persistent_body.body_id())
            .is_some()
    );
    assert!(
        clients[&uuid_a]
            .body_state(giant_ephemeral.body_id())
            .is_none(),
        "client A must not retain partially streamed body after zero-delivery recovery"
    );
    assert!(
        clients[&uuid_b]
            .body_state(giant_ephemeral.body_id())
            .is_none(),
        "client B must not retain partially streamed body after zero-delivery recovery"
    );
    assert!(resync_tracker.prepare_delta_packets(&multi_api).is_none());
}
