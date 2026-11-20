//! Horizon 26.3 native mod bridge for the clean-room `horizon_physics_core`
//! rigid-body and voxel sub-grid physics engine.
//!
//! Licensed under `MIT OR Apache-2.0`.

#![deny(unsafe_code)]

pub mod api;
pub mod client_render;
pub mod commands;
pub mod entity_rider;
pub mod sync;
pub mod terrain_cache;

pub use horizon_physics_core;

use std::sync::{Mutex, OnceLock};

use mod_api::{
    BlockChangedEvent, BreakBlockEvent, ChunkLoadEvent, ChunkUnloadEvent, ClientHost,
    ClientResourcesReloadedEvent, ClientSetupEvent, EntityMoveEvent, EntityPlaceEvent, Flow, Hooks,
    HostSide, LevelHost, LevelTickPostEvent, LevelTickPreEvent, LoopContext,
    ModelBakingCompletedEvent, PlayerLoggedInEvent, RegisterCommandsEvent,
    RegisterRenderPipelinesEvent, Registrar, RegistrationContext, RenderLevelStageEvent,
    RunModCommandEvent, ServerHost, ServerStartedEvent, ServerStoppingEvent, ServerTickEvent,
    ServerTickPhase, StandaloneModelsEvent, WorldSavedEvent,
};

use crate::api::{step_global_physics_with_sampler, with_physics_api, with_physics_api_read};
use crate::client_render::with_client_render;
use crate::entity_rider::with_rider_manager;
use crate::sync::DeltaSyncTracker;
use crate::terrain_cache::LiveLevelSampler;

/// Fixed Minecraft server tick interval (20 TPS = 50 ms).
pub const SERVER_TICK_DT: f64 = 1.0 / 20.0;

static GLOBAL_DELTA_TRACKER: OnceLock<Mutex<DeltaSyncTracker>> = OnceLock::new();

/// Accesses the global incremental pose-sync delta tracker.
pub fn with_delta_tracker<R>(f: impl FnOnce(&mut DeltaSyncTracker) -> R) -> R {
    let lock = GLOBAL_DELTA_TRACKER.get_or_init(|| Mutex::new(DeltaSyncTracker::new()));
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

/// Resets all mod-level global singletons (physics world, terrain cache,
/// rider anchors, client render buffers, and pose delta tracker).
pub fn reset_runtime_state() {
    with_physics_api(|api| api.reset());
    with_rider_manager(|rm| rm.clear());
    with_client_render(|r| r.reset());
    with_delta_tracker(|t| t.clear());
}

#[cfg_attr(target_arch = "wasm32", allow(unsafe_code))]
mod mod_entry {
    use super::*;

    /// Root Horizon 26.3 mod hooks implementation for `horizon_physics`.
    #[derive(Default)]
    pub struct HorizonPhysicsMod;

    #[mod_api::horizon_mod]
    impl Hooks for HorizonPhysicsMod {
        fn register(&self, r: &mut dyn Registrar, _cx: &RegistrationContext) {
            sync::register_sync_payloads(r);
        }

        fn server_started(&self, host: &mut dyn ServerHost, _event: &mut ServerStartedEvent) {
            with_delta_tracker(|t| t.clear());
            with_physics_api(|api| {
                let _ = sync::load_world_attachment(api, host);
            });
            with_physics_api_read(|api| {
                with_client_render(|r| r.sync_from_api(api));
            });
        }

        fn server_stopping(&self, host: &mut dyn ServerHost, _event: &mut ServerStoppingEvent) {
            with_physics_api_read(|api| {
                let _ = sync::save_world_attachment(api, host);
            });
        }

        fn world_saved(&self, host: &mut dyn ServerHost, _event: &mut WorldSavedEvent) {
            with_physics_api_read(|api| {
                let _ = sync::save_world_attachment(api, host);
            });
        }

        fn server_tick(&self, host: &mut dyn ServerHost, event: &mut ServerTickEvent<'_>) {
            let ServerTickPhase::ServerTickEnd {
                world_running: true,
            } = event.phase
            else {
                return;
            };

            let live_host = host.is_loaded("horizon_physics");
            if live_host && let Some(level) = host.level("minecraft:overworld") {
                let sampler = LiveLevelSampler::new(host, level);
                let _ = step_global_physics_with_sampler(SERVER_TICK_DT, Some(&sampler));
            } else {
                let _ = step_global_physics_with_sampler(SERVER_TICK_DT, None);
            }

            with_physics_api_read(|api| {
                with_client_render(|r| r.sync_from_api(api));
            });

            if live_host && !host.players().server_players().is_empty() {
                with_physics_api_read(|api| {
                    with_delta_tracker(|tracker| {
                        sync::server_tick_sync(api, tracker, host);
                    });
                });
            }
        }

        fn player_logged_in(&self, host: &mut dyn ServerHost, event: &mut PlayerLoggedInEvent<'_>) {
            if !host.is_loaded("horizon_physics") {
                return;
            }
            let Some(player) = host.players().player_by_name(event.name) else {
                return;
            };
            with_physics_api_read(|api| {
                with_delta_tracker(|tracker| {
                    sync::player_logged_in_sync(player, api, tracker, host);
                });
            });
        }

        fn level_tick_pre(&self, host: HostSide<'_>, event: &mut LevelTickPreEvent) {
            if let HostSide::Server(server) = host
                && server.is_loaded("horizon_physics")
            {
                with_physics_api_read(|api| {
                    let tick = api.server_tick_counter();
                    with_rider_manager(|rm| {
                        rm.snapshot_nearby_entities_from_level(
                            api.world(),
                            server,
                            event.level,
                            tick,
                        );
                    });
                });
            }
        }

        fn level_tick_post(&self, host: HostSide<'_>, _event: &mut LevelTickPostEvent) {
            match host {
                HostSide::Server(server) if server.is_loaded("horizon_physics") => {
                    let tick = with_physics_api_read(|api| api.server_tick_counter());
                    with_rider_manager(|rm| {
                        let _ = rm.flush_server_actions(server);
                        rm.prune_stale(tick, 40);
                    });
                }
                HostSide::Client(client) if client.is_loaded("horizon_physics") => {
                    with_rider_manager(|rm| {
                        let _ = rm.flush_client_actions(client);
                    });
                }
                _ => {}
            }
        }

        fn block_changed(&self, level: &dyn LevelHost, event: &mut BlockChangedEvent) {
            let new_block_name = level.state_block(event.new);
            with_physics_api(|api| {
                api.notify_block_changed([event.pos.x, event.pos.y, event.pos.z], new_block_name);
            });
        }

        fn break_block(&self, _host: &mut dyn ServerHost, event: &mut BreakBlockEvent<'_>) -> Flow {
            with_physics_api(|api| {
                api.notify_block_changed(event.pos, Some("minecraft:air"));
            });
            Flow::Continue
        }

        fn entity_place(&self, _host: &mut dyn ServerHost, event: &mut EntityPlaceEvent) -> Flow {
            with_physics_api(|api| {
                api.notify_block_changed([event.pos.x, event.pos.y, event.pos.z], None);
            });
            Flow::Continue
        }

        fn chunk_load(&self, _host: &mut dyn ServerHost, event: &mut ChunkLoadEvent<'_>) {
            with_physics_api(|api| {
                api.notify_chunk_unloaded(event.chunk[0], event.chunk[1]);
            });
        }

        fn chunk_unload(&self, _host: &mut dyn ServerHost, event: &mut ChunkUnloadEvent<'_>) {
            with_physics_api(|api| {
                api.notify_chunk_unloaded(event.chunk[0], event.chunk[1]);
            });
        }

        fn entity_move(&self, _context: &dyn LoopContext, event: &mut EntityMoveEvent<'_>) -> Flow {
            with_physics_api_read(|api| {
                let tick = api.server_tick_counter();
                with_rider_manager(|rm| rm.on_entity_move(api.world(), event, tick))
            })
        }

        fn register_commands(
            &self,
            _host: &mut dyn ServerHost,
            event: &mut RegisterCommandsEvent<'_>,
        ) {
            commands::register_hphys_commands(event.commands);
        }

        fn run_mod_command(
            &self,
            host: &mut dyn ServerHost,
            event: &mut RunModCommandEvent<'_>,
        ) -> Flow {
            commands::handle_hphys_command(host, event.source, event.command)
        }

        fn standalone_models(
            &self,
            _context: &RegistrationContext,
            event: &mut StandaloneModelsEvent<'_>,
        ) {
            client_render::ClientRenderState::register_standalone_models(event.models);
        }

        fn model_baking_completed(
            &self,
            host: &mut dyn ClientHost,
            _event: &mut ModelBakingCompletedEvent,
        ) {
            if host.is_loaded("horizon_physics") {
                with_client_render(|r| r.resolve_resources(host));
            }
        }

        fn register_render_pipelines(
            &self,
            _context: &RegistrationContext,
            event: &mut RegisterRenderPipelinesEvent<'_>,
        ) {
            client_render::ClientRenderState::register_pipelines(event.pipelines);
        }

        fn client_setup(&self, host: &mut dyn ClientHost, _event: &mut ClientSetupEvent) {
            if host.is_loaded("horizon_physics") {
                with_client_render(|r| r.resolve_resources(host));
            }
        }

        fn client_resources_reloaded(
            &self,
            host: &mut dyn ClientHost,
            _event: &mut ClientResourcesReloadedEvent,
        ) {
            if host.is_loaded("horizon_physics") {
                with_client_render(|r| r.resolve_resources(host));
            }
        }

        fn render_level_stage(&self, host: &dyn ClientHost, event: &mut RenderLevelStageEvent<'_>) {
            if matches!(event.stage, mod_api::RenderLevelStage::AfterLevel) {
                with_physics_api_read(|api| {
                    with_client_render(|r| {
                        let _ = r.render_stage(host, api, event);
                    });
                });
            } else {
                with_client_render(|r| {
                    let _ = r.render_stage_cached(host, event);
                });
            }
        }
    }
}

pub use mod_entry::{HorizonPhysicsMod, hooks};
