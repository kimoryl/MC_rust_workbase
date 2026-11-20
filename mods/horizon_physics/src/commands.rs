//! Registration and execution of the `/hphys` command suite (`spawn`, `assemble`,
//! `disassemble`, `impulse`, `joint`, `sleep`, `wake`, `stats`, `debug`).

use horizon_physics_core::{BodyKind, Quatd, Vec3d};
use mod_api::{
    Command, CommandArgumentValue, CommandInvocation, CommandSource, Flow, ServerHost, Text,
};

use crate::api::{
    DistanceJointDesc, FixedWeldJointDesc, HingeJointDesc, JointConstraint, JointRegistry,
    PhysicsApi, SliderJointDesc, SubGridRegistry, with_physics_api, with_physics_api_read,
};
use crate::client_render::with_client_render;
use crate::terrain_cache::CachedBlockCell;

/// Maximum total cells scanned by a single `/hphys assemble` bounding box.
const MAX_ASSEMBLE_SCAN_CELLS: i64 = 4096;

/// Registers all `/hphys` subcommand usages with Horizon's server command dispatcher.
pub fn register_hphys_commands(commands: &mut Vec<Command>) {
    commands.extend([
        Command::new("hphys spawn [x:number] [y:number] [z:number] [size:integer] [material:word]"),
        Command::new("hphys assemble <min:block_pos> <max:block_pos>"),
        Command::new("hphys disassemble <body:integer>"),
        Command::new("hphys impulse <body:integer> <ix:number> <iy:number> <iz:number>"),
        Command::new("hphys joint <kind:word> <body_a:integer> [body_b:integer]"),
        Command::new("hphys sleep [body:integer]"),
        Command::new("hphys wake [body:integer]"),
        Command::new("hphys stats"),
        Command::new("hphys debug [enabled:boolean]"),
    ]);
}

/// Dispatches a parsed `/hphys` command invocation.
///
/// Returns [`Flow::Cancel`] when this mod handled the command, or [`Flow::Continue`]
/// if the command belongs to another mod.
pub fn handle_hphys_command(
    host: &mut dyn ServerHost,
    _source: &CommandSource,
    cmd: &CommandInvocation,
) -> Flow {
    let Some(root) = cmd.literals.first().map(String::as_str) else {
        return Flow::Continue;
    };
    if root != "hphys" {
        return Flow::Continue;
    }

    let sub = cmd.literals.get(1).map(String::as_str).unwrap_or("stats");
    let reply = match sub {
        "spawn" => execute_spawn(cmd),
        "assemble" => execute_assemble(host, cmd),
        "disassemble" => execute_disassemble(cmd),
        "impulse" => execute_impulse(cmd),
        "joint" => execute_joint(cmd),
        "sleep" => execute_sleep(cmd),
        "wake" => execute_wake(cmd),
        "stats" => execute_stats(),
        "debug" => execute_debug(cmd),
        other => format!("[hphys] unknown subcommand `{other}`"),
    };

    let _ = host.commands().feedback(Text::literal(reply));
    Flow::Cancel
}

fn execute_spawn(cmd: &CommandInvocation) -> String {
    let x = arg_number(cmd, "x").unwrap_or(cmd.position[0]);
    let y = arg_number(cmd, "y").unwrap_or(cmd.position[1] + 2.0);
    let z = arg_number(cmd, "z").unwrap_or(cmd.position[2]);
    let size = arg_int(cmd, "size").unwrap_or(2).clamp(1, 8);
    let block_name = match arg_string(cmd, "material").unwrap_or("stone") {
        "wood" | "planks" => "minecraft:oak_planks",
        "iron" => "minecraft:iron_block",
        "slime" => "minecraft:slime_block",
        "ice" => "minecraft:ice",
        "wool" => "minecraft:white_wool",
        "stone" => "minecraft:stone",
        custom => custom,
    };

    let mut blocks = Vec::with_capacity((size * size * size) as usize);
    for bz in 0..size {
        for by in 0..size {
            for bx in 0..size {
                blocks.push(([bx, by, bz], block_name.to_owned()));
            }
        }
    }

    with_physics_api(|api| {
        match api.spawn_named_subgrid(
            Vec3d::new(x, y, z),
            Quatd::IDENTITY,
            BodyKind::Dynamic,
            false,
            1.0,
            0,
            &blocks,
        ) {
            Ok(handle) => {
                let mass = api.subgrid_state(handle).map_or(0.0, |s| s.mass_kg);
                format!(
                    "[hphys] spawned subgrid #{} (gen {}) at ({x:.2}, {y:.2}, {z:.2}), voxels={}, mass={mass:.1} kg",
                    handle.index(),
                    handle.generation(),
                    blocks.len()
                )
            }
            Err(err) => format!("[hphys] spawn failed: {err}"),
        }
    })
}

fn execute_assemble(host: &mut dyn ServerHost, cmd: &CommandInvocation) -> String {
    let min_raw = arg_block_pos(cmd, "min").unwrap_or([
        cmd.position[0].floor() as i32 - 1,
        cmd.position[1].floor() as i32,
        cmd.position[2].floor() as i32 - 1,
    ]);
    let max_raw = arg_block_pos(cmd, "max").unwrap_or([
        cmd.position[0].floor() as i32 + 1,
        cmd.position[1].floor() as i32 + 1,
        cmd.position[2].floor() as i32 + 1,
    ]);

    let min = [
        min_raw[0].min(max_raw[0]),
        min_raw[1].min(max_raw[1]),
        min_raw[2].min(max_raw[2]),
    ];
    let max = [
        min_raw[0].max(max_raw[0]),
        min_raw[1].max(max_raw[1]),
        min_raw[2].max(max_raw[2]),
    ];

    let span_x = i64::from(max[0]) - i64::from(min[0]) + 1;
    let span_y = i64::from(max[1]) - i64::from(min[1]) + 1;
    let span_z = i64::from(max[2]) - i64::from(min[2]) + 1;
    if span_x * span_y * span_z > MAX_ASSEMBLE_SCAN_CELLS {
        return format!(
            "[hphys] assemble volume ({} cells) exceeds limit of {MAX_ASSEMBLE_SCAN_CELLS}",
            span_x * span_y * span_z
        );
    }

    let level = host.level("minecraft:overworld");
    let resolve_block_names = host.is_loaded("horizon_physics");

    let mut sampled_blocks: Vec<([i32; 3], String)> = Vec::new();
    with_physics_api(|api| {
        for z in min[2]..=max[2] {
            for y in min[1]..=max[1] {
                for x in min[0]..=max[0] {
                    let local = [x - min[0], y - min[1], z - min[2]];
                    if let Some(lvl) = level
                        && let Some(block_key) = lvl.block(host, [x, y, z])
                    {
                        let name = if resolve_block_names {
                            block_key.id(host)
                        } else {
                            "minecraft:stone".to_owned()
                        };
                        if !api.terrain().materials().is_non_solid(&name) {
                            sampled_blocks.push((local, name));
                            continue;
                        }
                    }

                    if let CachedBlockCell::Solid { .. } = api.terrain().peek_cell(x, y, z) {
                        sampled_blocks.push((local, "minecraft:stone".to_owned()));
                    }
                }
            }
        }

        if sampled_blocks.is_empty() {
            return format!(
                "[hphys] assemble found no solid blocks in [{}, {}, {}]..[{}, {}, {}]",
                min[0], min[1], min[2], max[0], max[1], max[2]
            );
        }

        let center = Vec3d::new(
            0.5 * f64::from(min[0] + max[0] + 1),
            0.5 * f64::from(min[1] + max[1] + 1),
            0.5 * f64::from(min[2] + max[2] + 1),
        );

        match api.spawn_named_subgrid(
            center,
            Quatd::IDENTITY,
            BodyKind::Dynamic,
            false,
            1.0,
            0,
            &sampled_blocks,
        ) {
            Ok(handle) => {
                let mass = api.subgrid_state(handle).map_or(0.0, |s| s.mass_kg);
                format!(
                    "[hphys] assembled subgrid #{} (gen {}) with {} voxels ({mass:.1} kg)",
                    handle.index(),
                    handle.generation(),
                    sampled_blocks.len()
                )
            }
            Err(err) => format!("[hphys] assemble failed: {err}"),
        }
    })
}

fn execute_disassemble(cmd: &CommandInvocation) -> String {
    let Some(body_idx) = arg_int(cmd, "body").and_then(|v| u32::try_from(v).ok()) else {
        return "[hphys] disassemble requires a non-negative <body:integer>".to_owned();
    };
    with_physics_api(|api| {
        let Some(handle) = api.find_subgrid_by_index(body_idx) else {
            return format!("[hphys] subgrid #{body_idx} not found");
        };
        let Some(voxels) = api.disassemble_subgrid(handle) else {
            return format!("[hphys] failed to disassemble subgrid #{body_idx}");
        };
        let count = voxels.len();
        for v in &voxels {
            api.notify_block_changed(v.world_coord, Some(&v.block_id));
        }
        format!("[hphys] disassembled subgrid #{body_idx}, restored {count} blocks")
    })
}

fn execute_impulse(cmd: &CommandInvocation) -> String {
    let Some(body_idx) = arg_int(cmd, "body").and_then(|v| u32::try_from(v).ok()) else {
        return "[hphys] impulse requires a non-negative <body:integer>".to_owned();
    };
    let ix = arg_number(cmd, "ix").unwrap_or(0.0);
    let iy = arg_number(cmd, "iy").unwrap_or(0.0);
    let iz = arg_number(cmd, "iz").unwrap_or(0.0);
    if !ix.is_finite() || !iy.is_finite() || !iz.is_finite() {
        return "[hphys] impulse vector must be finite".to_owned();
    }

    with_physics_api(|api| {
        let Some(handle) = api.find_subgrid_by_index(body_idx) else {
            return format!("[hphys] subgrid #{body_idx} not found");
        };
        if !api.apply_impulse(handle, [ix, iy, iz], None) {
            return format!("[hphys] impulse rejected for subgrid #{body_idx}");
        }
        let vel = api
            .subgrid_state(handle)
            .map_or([0.0; 3], |s| s.linear_velocity);
        format!(
            "[hphys] applied impulse ({ix:.2}, {iy:.2}, {iz:.2}) to subgrid #{body_idx}, v=({:.2}, {:.2}, {:.2})",
            vel[0], vel[1], vel[2]
        )
    })
}

fn execute_joint(cmd: &CommandInvocation) -> String {
    let kind = arg_string(cmd, "kind").unwrap_or("weld");
    let Some(body_a_idx) = arg_int(cmd, "body_a").and_then(|v| u32::try_from(v).ok()) else {
        return "[hphys] joint requires <body_a:integer>".to_owned();
    };
    let body_b_idx = arg_int(cmd, "body_b").and_then(|v| u32::try_from(v).ok());

    with_physics_api(|api| {
        let Some(handle_a) = api.find_subgrid_by_index(body_a_idx) else {
            return format!("[hphys] subgrid #{body_a_idx} not found");
        };
        let handle_b = match body_b_idx {
            Some(idx) => match api.find_subgrid_by_index(idx) {
                Some(h) => Some(h),
                None => return format!("[hphys] subgrid #{idx} not found"),
            },
            None => None,
        };
        let pos_a = api
            .subgrid_state(handle_a)
            .map_or([0.0; 3], |s| s.world_position);
        let pos_b = handle_b
            .and_then(|hb| api.subgrid_state(hb))
            .map_or(pos_a, |s| s.world_position);

        let constraint = match kind.to_ascii_lowercase().as_str() {
            "hinge" => {
                let Some(hb) = handle_b else {
                    return "[hphys] hinge joint requires both body_a and body_b".to_owned();
                };
                JointConstraint::Hinge(HingeJointDesc {
                    body_a: handle_a,
                    body_b: hb,
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [0.0, 0.0, 0.0],
                    axis_a: [0.0, 1.0, 0.0],
                    axis_b: [0.0, 1.0, 0.0],
                    min_angle_rad: -std::f64::consts::FRAC_PI_2,
                    max_angle_rad: std::f64::consts::FRAC_PI_2,
                    compliance: 0.0,
                    motor: None,
                })
            }
            "distance" | "rope" | "spring" => {
                let dx = pos_b[0] - pos_a[0];
                let dy = pos_b[1] - pos_a[1];
                let dz = pos_b[2] - pos_a[2];
                let rest = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0);
                let anchor_b = if handle_b.is_some() { [0.0; 3] } else { pos_a };
                JointConstraint::Distance(DistanceJointDesc {
                    body_a: handle_a,
                    body_b: handle_b,
                    anchor_a: [0.0; 3],
                    anchor_b,
                    rest_length: rest,
                    min_length: None,
                    max_length: Some(rest),
                    compliance: 0.0,
                    damping: 0.05,
                })
            }
            "slider" | "prismatic" => {
                let anchor_b = if handle_b.is_some() { [0.0; 3] } else { pos_a };
                JointConstraint::Slider(SliderJointDesc {
                    body_a: handle_a,
                    body_b: handle_b,
                    anchor_a: [0.0; 3],
                    anchor_b,
                    axis_a: [1.0, 0.0, 0.0],
                    axis_b: [1.0, 0.0, 0.0],
                    limits: Some((-4.0, 4.0)),
                    compliance: 0.0,
                    motor_target_velocity: None,
                    motor_max_force: 0.0,
                })
            }
            "weld" | "fixed" | "fixed_weld" => {
                let anchor_b = if handle_b.is_some() { [0.0; 3] } else { pos_a };
                JointConstraint::FixedWeld(FixedWeldJointDesc {
                    body_a: handle_a,
                    body_b: handle_b,
                    anchor_a: [0.0; 3],
                    anchor_b,
                    relative_rotation: [0.0, 0.0, 0.0, 1.0],
                    compliance: 0.0,
                })
            }
            other => {
                return format!(
                    "[hphys] unsupported joint kind `{other}` (expected hinge|weld|distance|slider)"
                );
            }
        };

        match api.add_joint(constraint) {
            Some(jh) => format!(
                "[hphys] created {kind} joint #{} (gen {}) on body #{body_a_idx}",
                jh.joint_id().index,
                jh.joint_id().generation
            ),
            None => format!("[hphys] failed to create {kind} joint"),
        }
    })
}

fn execute_sleep(cmd: &CommandInvocation) -> String {
    with_physics_api(|api| {
        if let Some(idx) = arg_int(cmd, "body").and_then(|v| u32::try_from(v).ok()) {
            let Some(handle) = api.find_subgrid_by_index(idx) else {
                return format!("[hphys] subgrid #{idx} not found");
            };
            let ok = api.sleep_subgrid(handle);
            format!(
                "[hphys] sleep subgrid #{idx}: {}",
                if ok { "ok" } else { "skipped" }
            )
        } else {
            let mut count = 0usize;
            for h in api.active_subgrids() {
                if api.sleep_subgrid(h) {
                    count += 1;
                }
            }
            format!("[hphys] put {count} subgrids to sleep")
        }
    })
}

fn execute_wake(cmd: &CommandInvocation) -> String {
    with_physics_api(|api| {
        if let Some(idx) = arg_int(cmd, "body").and_then(|v| u32::try_from(v).ok()) {
            let Some(handle) = api.find_subgrid_by_index(idx) else {
                return format!("[hphys] subgrid #{idx} not found");
            };
            let ok = api.wake_subgrid(handle);
            format!(
                "[hphys] wake subgrid #{idx}: {}",
                if ok { "ok" } else { "skipped" }
            )
        } else {
            let mut count = 0usize;
            for h in api.active_subgrids() {
                if api.wake_subgrid(h) {
                    count += 1;
                }
            }
            format!("[hphys] woke {count} subgrids")
        }
    })
}

fn execute_stats() -> String {
    with_physics_api_read(format_stats)
}

/// Formats live engine telemetry into a concise single-line summary.
pub fn format_stats(api: &PhysicsApi) -> String {
    let m = api.last_metrics();
    let cache = api.terrain().stats();
    format!(
        "[hphys] step={} bodies={} (awake={}, sleeping={}) joints={} islands={} pairs={} manifolds={} contacts={} toi={} step_us={} cache(hits={}, misses={}, inv={}) providers={} propulsions={} debug={}",
        api.world().step_index(),
        api.world().body_count(),
        m.awake_bodies,
        m.sleeping_bodies,
        api.world().joint_count(),
        m.island_count,
        m.broadphase_pairs,
        m.contact_manifolds,
        m.contact_points,
        m.toi_events,
        m.step_time_us,
        cache.hits,
        cache.misses,
        cache.block_invalidations,
        api.force_provider_count(),
        api.propulsion_count(),
        if api.debug_overlay_enabled() {
            "on"
        } else {
            "off"
        }
    )
}

fn execute_debug(cmd: &CommandInvocation) -> String {
    let new_state = with_physics_api(|api| match arg_bool(cmd, "enabled") {
        Some(v) => {
            api.set_debug_overlay_enabled(v);
            v
        }
        None => api.toggle_debug_overlay(),
    });
    with_client_render(|render| render.set_debug_enabled(new_state));
    format!(
        "[hphys] debug overlay: {}",
        if new_state { "on" } else { "off" }
    )
}

fn arg_number(cmd: &CommandInvocation, name: &str) -> Option<f64> {
    match cmd.argument(name)? {
        CommandArgumentValue::Number(v) => Some(*v),
        CommandArgumentValue::Integer(v) => Some(f64::from(*v)),
        _ => None,
    }
}

fn arg_int(cmd: &CommandInvocation, name: &str) -> Option<i32> {
    match cmd.argument(name)? {
        CommandArgumentValue::Integer(v) => Some(*v),
        _ => None,
    }
}

fn arg_bool(cmd: &CommandInvocation, name: &str) -> Option<bool> {
    match cmd.argument(name)? {
        CommandArgumentValue::Boolean(v) => Some(*v),
        _ => None,
    }
}

fn arg_string<'a>(cmd: &'a CommandInvocation, name: &str) -> Option<&'a str> {
    match cmd.argument(name)? {
        CommandArgumentValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

fn arg_block_pos(cmd: &CommandInvocation, name: &str) -> Option<[i32; 3]> {
    match cmd.argument(name)? {
        CommandArgumentValue::BlockPos(p) => Some(*p),
        _ => None,
    }
}
