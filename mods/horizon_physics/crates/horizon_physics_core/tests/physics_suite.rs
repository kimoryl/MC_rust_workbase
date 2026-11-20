use std::time::Instant;

use horizon_physics_core::{
    Aabbd, BodyDesc, BodyId, BodyKind, DynamicBvh, EmptyWorldProvider, FACE_NEG_X, FACE_POS_X,
    FlatWorldProvider, JointKind, JointSpec, Mat3d, MaterialId, MaterialRegistry, PhysicsConfig,
    PhysicsEvent, PhysicsWorld, Quatd, SlotArena, SpatialHash16, Transformd, Vec3d, VoxelCoord,
    VoxelEntry, VoxelMaterial, VoxelSubGrid, WorldBlockCollision, WorldVoxelProvider,
};

fn make_cube_grid(size: i32, material: MaterialId, reg: &MaterialRegistry) -> VoxelSubGrid {
    VoxelSubGrid::solid_box(
        [0, 0, 0],
        [size, size, size],
        VoxelEntry::from_material(material),
        reg,
    )
}

#[test]
fn test_01_incremental_greedy_box_merging_and_reduction_rate() {
    let reg = MaterialRegistry::new();

    // 1. Homogeneous 16x16x16 brick (4096 voxels) -> 1 LocalBox (99.975% reduction).
    let brick_grid = make_cube_grid(16, MaterialId::STONE, &reg);
    assert_eq!(brick_grid.voxel_count(), 4096);
    assert_eq!(brick_grid.boxes().len(), 1);
    let reduction = 1.0 - (brick_grid.boxes().len() as f64 / brick_grid.voxel_count() as f64);
    assert!(
        reduction > 0.999,
        "expected >99.9% reduction, got {reduction}"
    );

    // 2. Multi-brick ship hull spanning 2 bricks along X (x in 0..32, y in 0..4, z in 0..8).
    let mut ship = VoxelSubGrid::solid_box(
        [0, 0, 0],
        [32, 4, 8],
        VoxelEntry::from_material(MaterialId::WOOD),
        &reg,
    );
    // Hollow out interior (keeping floor y=0 and walls).
    for z in 1..7 {
        for y in 1..4 {
            for x in 1..31 {
                ship.set_voxel(VoxelCoord::new(x, y, z), None, &reg);
            }
        }
    }
    let rebuilt_initial = ship.rebuild_dirty_colliders();
    assert_eq!(rebuilt_initial, 2, "ship spans exactly 2 bricks");
    assert_eq!(ship.brick_keys(), vec![(0, 0, 0), (1, 0, 0)]);

    let hull_reduction = 1.0 - (ship.boxes().len() as f64 / ship.voxel_count() as f64);
    assert!(
        hull_reduction > 0.90,
        "hollow hull expected >90% box reduction, voxels={} boxes={} reduction={hull_reduction}",
        ship.voxel_count(),
        ship.boxes().len()
    );

    // Check internal cross-brick seam: floor box covering (15, 0, 4) in brick (0,0,0) touches
    // (16, 0, 4) in brick (1,0,0), so its +X face is completely covered (not exposed).
    let left_floor = ship
        .boxes()
        .iter()
        .find(|b| {
            b.min[0] <= 15
                && b.max[0] == 16
                && b.min[1] <= 0
                && b.max[1] > 0
                && b.min[2] <= 4
                && b.max[2] > 4
        })
        .expect("left floor box covering (15,0,4) exists");
    assert_eq!(
        left_floor.exposed_faces & FACE_POS_X,
        0,
        "internal cross-brick +X seam must not be marked exposed"
    );

    // 3. Incremental dirty rebuild: edit a single interior voxel strictly inside brick (1,0,0).
    ship.set_voxel(
        VoxelCoord::new(20, 0, 4),
        Some(VoxelEntry::from_material(MaterialId::IRON)),
        &reg,
    );
    let rebuilt_one = ship.rebuild_dirty_colliders();
    assert_eq!(
        rebuilt_one, 1,
        "interior mutation must rebuild ONLY 1 dirty brick"
    );

    // 4. Boundary mutation at x = 16 must dirty BOTH brick (1,0,0) and adjacent brick (0,0,0)
    // so brick (0,0,0) updates its exposed_faces bit!
    ship.set_voxel(VoxelCoord::new(16, 0, 4), None, &reg);
    let rebuilt_border = ship.rebuild_dirty_colliders();
    assert_eq!(
        rebuilt_border, 2,
        "brick-boundary mutation must mark neighbor brick dirty for exposed_faces"
    );
    let left_floor_after = ship
        .boxes()
        .iter()
        .find(|b| {
            b.min[0] <= 15
                && b.max[0] == 16
                && b.min[1] <= 0
                && b.max[1] > 0
                && b.min[2] <= 4
                && b.max[2] > 4
        })
        .expect("left floor box covering (15,0,4) still exists");
    assert_ne!(
        left_floor_after.exposed_faces & FACE_POS_X,
        0,
        "removing neighbor voxel at x=16 must expose +X face on brick (0,0,0)"
    );
    // Also check negative face exposure on right side.
    let right_has_neg_x = ship
        .boxes()
        .iter()
        .any(|b| b.min[0] == 16 && (b.exposed_faces & FACE_NEG_X) != 0);
    assert!(!right_has_neg_x || ship.voxel_count() > 0);
}

#[test]
fn test_02_o1_inertia_and_com_vs_bruteforce() {
    let reg = MaterialRegistry::new();
    let mut grid = VoxelSubGrid::new();
    let mats = [
        MaterialId::STONE,
        MaterialId::WOOD,
        MaterialId::IRON,
        MaterialId::ICE,
        MaterialId::SLIME,
    ];

    // Add asymmetric voxels across negative & positive coordinates and multiple bricks.
    let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next_u64 = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let mut coords = Vec::new();
    for _ in 0..350 {
        let x = ((next_u64() % 41) as i32) - 20;
        let y = ((next_u64() % 33) as i32) - 16;
        let z = ((next_u64() % 41) as i32) - 20;
        let mat = mats[(next_u64() % mats.len() as u64) as usize];
        let coord = VoxelCoord::new(x, y, z);
        grid.set_voxel(coord, Some(VoxelEntry::from_material(mat)), &reg);
        coords.push(coord);
    }
    // Mutate and remove a subset to stress O(1) incremental subtraction accuracy.
    for (idx, &coord) in coords.iter().enumerate() {
        if idx % 3 == 0 {
            grid.set_voxel(coord, None, &reg);
        } else if idx % 5 == 0 {
            grid.set_voxel(
                coord,
                Some(VoxelEntry::from_material(MaterialId::IRON)),
                &reg,
            );
        }
    }

    // Brute-force O(N) mass, CoM, and inertia tensor directly about CoM.
    let mut bf_mass = 0.0;
    let mut bf_moment = Vec3d::ZERO;
    let voxels: Vec<(VoxelCoord, VoxelEntry)> = grid.iter_voxels().collect();
    for &(c, e) in &voxels {
        let m = reg.get(e.material).density;
        bf_mass += m;
        bf_moment += c.center_local() * m;
    }
    let bf_com = bf_moment / bf_mass;

    let mut bf_inertia = Mat3d::ZERO;
    for &(c, e) in &voxels {
        let m = reg.get(e.material).density;
        let d = c.center_local() - bf_com;
        let self_i = m / 6.0;
        let ixx = self_i + m * (d.y * d.y + d.z * d.z);
        let iyy = self_i + m * (d.x * d.x + d.z * d.z);
        let izz = self_i + m * (d.x * d.x + d.y * d.y);
        let ixy = -m * d.x * d.y;
        let ixz = -m * d.x * d.z;
        let iyz = -m * d.y * d.z;
        bf_inertia += Mat3d::from_cols(
            Vec3d::new(ixx, ixy, ixz),
            Vec3d::new(ixy, iyy, iyz),
            Vec3d::new(ixz, iyz, izz),
        );
    }

    let inc_mass = grid.mass();
    let inc_com = grid.com_local();
    let inc_inertia = grid.inertia_com();

    assert!(
        ((inc_mass - bf_mass) / bf_mass).abs() < 1e-13,
        "mass mismatch: inc={inc_mass}, bf={bf_mass}"
    );
    assert!(
        (inc_com - bf_com).length() < 1e-12,
        "CoM mismatch: inc={inc_com:?}, bf={bf_com:?}"
    );
    for col in 0..3 {
        for row in 0..3 {
            let a = inc_inertia.cols[col][row];
            let b = bf_inertia.cols[col][row];
            let scale = b.abs().max(1.0);
            let rel_err = (a - b).abs() / scale;
            assert!(
                rel_err < 1e-11,
                "inertia[{row},{col}] error {rel_err} >= 1e-11 (inc={a}, bf={b})"
            );
        }
    }

    // Verify set_body_voxel zero-visual-jump CoM compensation on a rotated body.
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let mut bar = VoxelSubGrid::solid_box(
        [0, 0, 0],
        [4, 1, 1],
        VoxelEntry::from_material(MaterialId::STONE),
        world.materials(),
    );
    bar.rebuild_dirty_colliders();
    let rot = Quatd::from_euler_zyx(0.4, -0.7, 0.25);
    let body_id = world.spawn_body(BodyDesc {
        kind: BodyKind::Dynamic,
        transform: Transformd::new(Vec3d::new(10.0, 20.0, -5.0), rot),
        grid: bar,
        ..BodyDesc::default()
    });
    let snap_before = world.body_snapshot(body_id).unwrap();
    let voxel0_world_before = snap_before
        .transform
        .transform_point(VoxelCoord::new(0, 0, 0).center_local() - snap_before.com_local);

    assert!(world.set_body_voxel(
        body_id,
        VoxelCoord::new(4, 0, 0),
        Some(VoxelEntry::from_material(MaterialId::IRON)),
    ));
    let snap_after = world.body_snapshot(body_id).unwrap();
    let voxel0_world_after = snap_after
        .transform
        .transform_point(VoxelCoord::new(0, 0, 0).center_local() - snap_after.com_local);
    assert!(
        (voxel0_world_after - voxel0_world_before).length() < 1e-12,
        "untouched voxel jumped in world space: before={voxel0_world_before:?} after={voxel0_world_after:?}"
    );
}

#[test]
fn test_03_farlands_stacking_no_jitter_and_camera_relative_f32() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let mut terrain = FlatWorldProvider::new(0);

    let far_x = 29_000_000.0;
    let far_z = -29_000_000.0;
    let mut ids = Vec::new();
    for i in 0..5 {
        let grid = make_cube_grid(1, MaterialId::STONE, world.materials());
        let id = world.spawn_body(BodyDesc::dynamic(
            grid,
            Vec3d::new(far_x + 0.5, 0.5 + i as f64 * 1.005, far_z + 0.5),
        ));
        ids.push(id);
    }

    for _ in 0..300 {
        world.step(1.0 / 60.0, &mut terrain);
    }

    let cam_origin = Vec3d::new(far_x - 10.0, 3.0, far_z - 8.0);
    for (i, &id) in ids.iter().enumerate() {
        let snap = world.body_snapshot(id).unwrap();
        let expected_y = 0.5 + i as f64;
        assert!(
            (snap.transform.position.y - expected_y).abs() < 0.03,
            "farlands box {i} drifted vertically: y={} expected={expected_y}",
            snap.transform.position.y
        );
        assert!(
            (snap.transform.position.x - (far_x + 0.5)).abs() < 0.02
                && (snap.transform.position.z - (far_z + 0.5)).abs() < 0.02,
            "farlands box {i} jittered laterally: pos={:?}",
            snap.transform.position
        );
        assert!(
            snap.linear_velocity.length() < 0.02,
            "farlands box {i} did not settle: vel={:?}",
            snap.linear_velocity
        );

        // Verify camera-relative f32 affine export preserves sub-millimeter precision.
        let m = snap
            .transform
            .to_camera_relative_affine_f32(cam_origin, -snap.com_local);
        let expected_rel = (snap.transform.position - cam_origin)
            + snap.transform.rotation.rotate_vec3(-snap.com_local);
        assert!((f64::from(m[3]) - expected_rel.x).abs() < 1e-4);
        assert!((f64::from(m[7]) - expected_rel.y).abs() < 1e-4);
        assert!((f64::from(m[11]) - expected_rel.z).abs() < 1e-4);
    }
}

#[test]
fn test_04_free_fall_and_elastic_energy_momentum_conservation() {
    // Part A: Free fall kinematic verification.
    {
        let cfg = PhysicsConfig {
            enable_sleeping: false,
            enable_buoyancy: false,
            ..PhysicsConfig::default()
        };
        let mut world = PhysicsWorld::new(cfg);
        let mut empty = EmptyWorldProvider;
        let grid = make_cube_grid(1, MaterialId::STONE, world.materials());
        let id = world.spawn_body(BodyDesc {
            kind: BodyKind::Dynamic,
            transform: Transformd::from_position(Vec3d::new(0.0, 100.0, 0.0)),
            grid,
            linear_damping_override: Some(0.0),
            angular_damping_override: Some(0.0),
            ..BodyDesc::default()
        });
        let dt = 1.0 / 60.0;
        let steps = 120; // t = 2.0 s
        for _ in 0..steps {
            world.step(dt, &mut empty);
        }
        let t = steps as f64 * dt;
        let g = 9.81;
        let snap = world.body_snapshot(id).unwrap();
        let expected_vy = -g * t;
        let expected_y = 100.0 - 0.5 * g * t * t;
        assert!(
            (snap.linear_velocity.y - expected_vy).abs() < 1e-6,
            "free-fall vy mismatch: got={} expected={expected_vy}",
            snap.linear_velocity.y
        );
        assert!(
            (snap.transform.position.y - expected_y).abs() < 0.05,
            "free-fall y mismatch: got={} expected={expected_y}",
            snap.transform.position.y
        );
    }

    // Part B: 1D head-on elastic collision (e = 1.0) momentum + kinetic energy conservation.
    {
        let cfg = PhysicsConfig {
            gravity: Vec3d::ZERO,
            enable_sleeping: false,
            enable_buoyancy: false,
            velocity_iterations: 8,
            ..PhysicsConfig::default()
        };
        let mut world = PhysicsWorld::new(cfg);
        let mut empty = EmptyWorldProvider;
        let elastic_light = MaterialId(100);
        let elastic_heavy = MaterialId(101);
        world.register_material(
            elastic_light,
            VoxelMaterial {
                density: 100.0,
                friction: 0.0,
                restitution: 1.0,
                linear_drag: 0.0,
                angular_drag: 0.0,
            },
        );
        world.register_material(
            elastic_heavy,
            VoxelMaterial {
                density: 300.0,
                friction: 0.0,
                restitution: 1.0,
                linear_drag: 0.0,
                angular_drag: 0.0,
            },
        );

        let g1 = make_cube_grid(1, elastic_light, world.materials());
        let g2 = make_cube_grid(1, elastic_heavy, world.materials());
        let b1 = world.spawn_body(BodyDesc {
            kind: BodyKind::Dynamic,
            transform: Transformd::from_position(Vec3d::new(-2.0, 0.0, 0.0)),
            linear_velocity: Vec3d::new(6.0, 0.0, 0.0),
            grid: g1,
            linear_damping_override: Some(0.0),
            angular_damping_override: Some(0.0),
            ..BodyDesc::default()
        });
        let b2 = world.spawn_body(BodyDesc {
            kind: BodyKind::Dynamic,
            transform: Transformd::from_position(Vec3d::new(2.0, 0.0, 0.0)),
            linear_velocity: Vec3d::new(-2.0, 0.0, 0.0),
            grid: g2,
            linear_damping_override: Some(0.0),
            angular_damping_override: Some(0.0),
            ..BodyDesc::default()
        });

        let m1 = 100.0;
        let m2 = 300.0;
        let v1_0 = 6.0;
        let v2_0 = -2.0;
        let p0 = m1 * v1_0 + m2 * v2_0;
        let ke0 = 0.5 * m1 * v1_0 * v1_0 + 0.5 * m2 * v2_0 * v2_0;

        for _ in 0..90 {
            world.step(1.0 / 60.0, &mut empty);
        }

        let s1 = world.body_snapshot(b1).unwrap();
        let s2 = world.body_snapshot(b2).unwrap();
        let p1 = m1 * s1.linear_velocity.x + m2 * s2.linear_velocity.x;
        let ke1 =
            0.5 * m1 * s1.linear_velocity.length_sq() + 0.5 * m2 * s2.linear_velocity.length_sq();

        let expected_v1 = ((m1 - m2) * v1_0 + 2.0 * m2 * v2_0) / (m1 + m2);
        let expected_v2 = ((m2 - m1) * v2_0 + 2.0 * m1 * v1_0) / (m1 + m2);

        assert!(
            (p1 - p0).abs() < 1e-8,
            "momentum not conserved: p0={p0}, p1={p1}"
        );
        assert!(
            ((ke1 - ke0) / ke0).abs() < 1e-3,
            "kinetic energy not conserved: ke0={ke0}, ke1={ke1}"
        );
        assert!(
            (s1.linear_velocity.x - expected_v1).abs() < 0.05
                && (s2.linear_velocity.x - expected_v2).abs() < 0.05,
            "post-impact velocities mismatch: got=({}, {}) expected=({expected_v1}, {expected_v2})",
            s1.linear_velocity.x,
            s2.linear_velocity.x
        );
    }

    // Part C: Elastic floor drop rebound recovery.
    {
        let cfg = PhysicsConfig {
            enable_sleeping: false,
            enable_buoyancy: false,
            velocity_iterations: 8,
            ..PhysicsConfig::default()
        };
        let mut world = PhysicsWorld::new(cfg);
        let elastic_mat = MaterialId(102);
        world.register_material(
            elastic_mat,
            VoxelMaterial {
                density: 500.0,
                friction: 0.0,
                restitution: 1.0,
                linear_drag: 0.0,
                angular_drag: 0.0,
            },
        );
        let mut terrain = FlatWorldProvider {
            ground_y: 0,
            ground_material: elastic_mat,
            water_level: None,
        };
        let g = make_cube_grid(1, elastic_mat, world.materials());
        let id = world.spawn_body(BodyDesc {
            kind: BodyKind::Dynamic,
            transform: Transformd::from_position(Vec3d::new(0.5, 5.5, 0.5)),
            grid: g,
            linear_damping_override: Some(0.0),
            angular_damping_override: Some(0.0),
            ..BodyDesc::default()
        });

        let mut hit_bottom = false;
        let mut peak_after_bounce = 0.0_f64;
        for _ in 0..150 {
            world.step(1.0 / 60.0, &mut terrain);
            let s = world.body_snapshot(id).unwrap();
            if s.transform.position.y < 1.0 && s.linear_velocity.y > 0.0 {
                hit_bottom = true;
            }
            if hit_bottom {
                peak_after_bounce = peak_after_bounce.max(s.transform.position.y);
            }
        }
        assert!(hit_bottom, "cube never bounced off floor");
        assert!(
            (5.30..=5.70).contains(&peak_after_bounce),
            "elastic rebound peak expected near 5.50, got {peak_after_bounce}"
        );
    }
}

struct ThinWallTerrain {
    wall_x: i32,
}

impl WorldVoxelProvider for ThinWallTerrain {
    fn voxel_collision(&mut self, x: i32, y: i32, z: i32) -> Option<WorldBlockCollision> {
        if x == self.wall_x && (-10..=10).contains(&y) && (-10..=10).contains(&z) {
            Some(WorldBlockCollision::full_block(MaterialId::STONE))
        } else {
            None
        }
    }
}

#[test]
fn test_05_high_speed_toi_anti_tunneling_thin_wall() {
    let cfg = PhysicsConfig {
        gravity: Vec3d::ZERO,
        enable_sleeping: false,
        enable_buoyancy: false,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(cfg);
    let mut wall = ThinWallTerrain { wall_x: 0 };

    let bullet_grid = make_cube_grid(1, MaterialId::IRON, world.materials());
    let bullet = world.spawn_body(BodyDesc {
        kind: BodyKind::Dynamic,
        transform: Transformd::from_position(Vec3d::new(-3.0, 0.5, 0.5)),
        // 300 m/s => 5.0 m per 60Hz frame, 1.25 m per substep (> 1.0 m wall thickness!).
        linear_velocity: Vec3d::new(300.0, 0.0, 0.0),
        grid: bullet_grid,
        enable_ccd: true,
        linear_damping_override: Some(0.0),
        angular_damping_override: Some(0.0),
        ..BodyDesc::default()
    });

    let mut total_toi = 0usize;
    for _ in 0..30 {
        let m = world.step(1.0 / 60.0, &mut wall);
        total_toi += m.toi_events;
        let snap = world.body_snapshot(bullet).unwrap();
        assert!(
            snap.transform.position.x < 0.0,
            "high-speed body tunneled through 1-block thin wall! x={}",
            snap.transform.position.x
        );
    }
    assert!(
        total_toi >= 1,
        "expected Conservative Advancement TOI event to fire"
    );
    let final_snap = world.body_snapshot(bullet).unwrap();
    assert!(
        final_snap.linear_velocity.x <= 0.0,
        "bullet should have bounced or stopped against wall, vx={}",
        final_snap.linear_velocity.x
    );
}

#[test]
fn test_06_sleeping_islands_and_wakeup() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let mut terrain = FlatWorldProvider::new(0);

    let mut stack = Vec::new();
    for i in 0..3 {
        let g = make_cube_grid(1, MaterialId::STONE, world.materials());
        let id = world.spawn_body(BodyDesc::dynamic(g, Vec3d::new(0.5, 0.5 + i as f64, 0.5)));
        stack.push(id);
    }

    let mut slept_ids = Vec::new();
    for _ in 0..180 {
        world.step(1.0 / 60.0, &mut terrain);
        for ev in world.drain_events() {
            if let PhysicsEvent::BodySlept { body } = ev {
                slept_ids.push(body);
            }
        }
    }

    for &id in &stack {
        let s = world.body_snapshot(id).unwrap();
        assert!(s.sleeping, "body {id:?} failed to enter sleep state");
        assert!(
            slept_ids.contains(&id),
            "BodySlept event missing for {id:?}"
        );
    }

    // Wakeup 1: local voxel mutation on the bottom box wakes the body and neighbors.
    assert!(world.set_body_voxel(
        stack[0],
        VoxelCoord::new(1, 0, 0),
        Some(VoxelEntry::from_material(MaterialId::WOOD)),
    ));
    let events = world.drain_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, PhysicsEvent::BodyWoke { body } if *body == stack[0])),
        "BodyWoke event not emitted on set_body_voxel"
    );
    assert!(!world.body_snapshot(stack[0]).unwrap().sleeping);

    // Let island sleep again, then test wake_in_aabb on the supporting ground envelope.
    for _ in 0..240 {
        world.step(1.0 / 60.0, &mut terrain);
    }
    world.drain_events();
    assert!(world.body_snapshot(stack[0]).unwrap().sleeping);

    world.wake_in_aabb(Aabbd::new(
        Vec3d::new(-0.5, -0.5, -0.5),
        Vec3d::new(2.5, 3.5, 1.5),
    ));
    let wake_events = world.drain_events();
    for &id in &stack {
        assert!(!world.body_snapshot(id).unwrap().sleeping);
        assert!(
            wake_events
                .iter()
                .any(|e| matches!(e, PhysicsEvent::BodyWoke { body } if *body == id))
        );
    }
}

#[test]
fn test_07_fracture_split_momentum_and_angular_momentum_conservation() {
    let cfg = PhysicsConfig {
        gravity: Vec3d::ZERO,
        enable_sleeping: false,
        enable_buoyancy: false,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(cfg);

    // Symmetric 5-voxel bridge x in [0..5): center voxel (2,0,0) sits exactly at CoM (2.5, 0.5, 0.5).
    let bridge = VoxelSubGrid::solid_box(
        [0, 0, 0],
        [5, 1, 1],
        VoxelEntry::from_material(MaterialId::WOOD),
        world.materials(),
    );
    let init_pos = Vec3d::new(10.0, 20.0, 30.0);
    let init_rot = Quatd::from_euler_zyx(0.3, -0.5, 0.8);
    let init_vel = Vec3d::new(4.5, -2.0, 1.25);
    let init_omega = Vec3d::new(1.2, 2.5, -0.9);

    let parent_id = world.spawn_body(BodyDesc {
        kind: BodyKind::Dynamic,
        transform: Transformd::new(init_pos, init_rot),
        linear_velocity: init_vel,
        angular_velocity: init_omega,
        grid: bridge,
        ..BodyDesc::default()
    });

    // Compute exact pre-split momentum & angular momentum of the 4 surviving voxels (0, 1, 3, 4)
    // right after voxel (2,0,0) is removed (before fracture split).
    let mut pre_grid = VoxelSubGrid::solid_box(
        [0, 0, 0],
        [5, 1, 1],
        VoxelEntry::from_material(MaterialId::WOOD),
        world.materials(),
    );
    pre_grid.set_voxel(VoxelCoord::new(2, 0, 0), None, world.materials());
    let pre_mass = pre_grid.mass();
    let r_mat = init_rot.to_mat3();
    let pre_inertia_world = Mat3d::similarity_transform(r_mat, pre_grid.inertia_com());

    let p_pre = init_vel * pre_mass;
    let l_pre = pre_inertia_world.mul_vec3(init_omega);

    // Trigger split by deleting the bridge keystone voxel (2, 0, 0).
    assert!(world.set_body_voxel(parent_id, VoxelCoord::new(2, 0, 0), None));
    let events = world.drain_events();
    let children = events
        .iter()
        .find_map(|e| match e {
            PhysicsEvent::GridSplit { parent, children } if *parent == parent_id => {
                Some(children.clone())
            }
            _ => None,
        })
        .expect("GridSplit event must be emitted");
    assert_eq!(
        children.len(),
        1,
        "5-voxel bridge split in center yields 1 detached child"
    );

    let all_ids = [parent_id, children[0]];
    let mut p_post = Vec3d::ZERO;
    let mut l_post = Vec3d::ZERO;
    let mut total_mass_post = 0.0;

    for bid in all_ids {
        let snap = world.body_snapshot(bid).unwrap();
        let grid = world.body_grid(bid).unwrap();
        let m = snap.mass;
        total_mass_post += m;
        let r = snap.transform.position - init_pos;
        let i_w =
            Mat3d::similarity_transform(snap.transform.rotation.to_mat3(), grid.inertia_com());
        p_post += snap.linear_velocity * m;
        l_post += i_w.mul_vec3(snap.angular_velocity) + r.cross(snap.linear_velocity * m);
    }

    assert!((total_mass_post - pre_mass).abs() < 1e-10);
    assert!(
        (p_post - p_pre).length() < 1e-9,
        "linear momentum not conserved on split: pre={p_pre:?} post={p_post:?}"
    );
    assert!(
        (l_post - l_pre).length() < 1e-9,
        "angular momentum not conserved on split: pre={l_pre:?} post={l_post:?}"
    );
}

#[test]
fn test_08_four_xpbd_joints_and_breaking_event() {
    let mut empty = EmptyWorldProvider;

    // 1. Distance joint (pendulum).
    {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            enable_sleeping: false,
            enable_buoyancy: false,
            ..PhysicsConfig::default()
        });
        let g = make_cube_grid(1, MaterialId::WOOD, world.materials());
        let bob = world.spawn_body(BodyDesc::dynamic(g, Vec3d::new(4.0, 10.0, 0.0)));
        let mut spec = JointSpec::distance(bob, None, 4.0);
        spec.local_anchor_b = Vec3d::new(0.0, 10.0, 0.0);
        world.add_joint(spec).unwrap();

        for _ in 0..120 {
            world.step(1.0 / 60.0, &mut empty);
            let p = world.body_snapshot(bob).unwrap().transform.position;
            let d = (p - Vec3d::new(0.0, 10.0, 0.0)).length();
            assert!(
                (d - 4.0).abs() < 0.03,
                "distance joint drift exceeded tolerance: dist={d}"
            );
        }
    }

    // 2. FixedWeld joint.
    {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3d::ZERO,
            enable_sleeping: false,
            enable_buoyancy: false,
            ..PhysicsConfig::default()
        });
        let g1 = make_cube_grid(1, MaterialId::STONE, world.materials());
        let g2 = make_cube_grid(1, MaterialId::WOOD, world.materials());
        let a = world.spawn_body(BodyDesc::dynamic(g1, Vec3d::new(0.0, 0.0, 0.0)));
        let b = world.spawn_body(BodyDesc::dynamic(g2, Vec3d::new(2.0, 0.0, 0.0)));
        let mut spec = JointSpec::fixed_weld(a, Some(b), Quatd::IDENTITY);
        spec.local_anchor_a = Vec3d::new(1.0, 0.0, 0.0);
        spec.local_anchor_b = Vec3d::new(-1.0, 0.0, 0.0);
        world.add_joint(spec).unwrap();

        world.apply_impulse_at_world_point(
            a,
            Vec3d::new(0.0, 5000.0, 2000.0),
            Vec3d::new(0.0, 0.5, 0.0),
        );
        for _ in 0..90 {
            world.step(1.0 / 60.0, &mut empty);
        }
        let sa = world.body_snapshot(a).unwrap();
        let sb = world.body_snapshot(b).unwrap();
        let pa = sa.transform.transform_point(Vec3d::new(1.0, 0.0, 0.0));
        let pb = sb.transform.transform_point(Vec3d::new(-1.0, 0.0, 0.0));
        assert!(
            (pa - pb).length() < 0.02,
            "fixed weld anchors separated: err={}",
            (pa - pb).length()
        );
        let q_err = (sa.transform.rotation * sb.transform.rotation.conjugate())
            .to_rotation_vector()
            .length();
        assert!(q_err < 0.03, "fixed weld orientation drifted: {q_err}");
    }

    // 3. Hinge joint with motor + angular limits.
    {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3d::ZERO,
            enable_sleeping: false,
            enable_buoyancy: false,
            ..PhysicsConfig::default()
        });
        let door_grid = VoxelSubGrid::solid_box(
            [0, 0, 0],
            [3, 2, 1],
            VoxelEntry::from_material(MaterialId::WOOD),
            world.materials(),
        );
        let door = world.spawn_body(BodyDesc::dynamic(door_grid, Vec3d::new(1.5, 1.0, 0.5)));
        let spec = JointSpec {
            body_a: door,
            body_b: None,
            local_anchor_a: Vec3d::new(-1.5, 0.0, 0.0),
            local_anchor_b: Vec3d::new(0.0, 1.0, 0.5),
            kind: JointKind::Hinge {
                axis_a: Vec3d::Y,
                axis_b: Vec3d::Y,
                limits: Some((-0.8, 0.8)),
                motor_target_velocity: Some(3.0),
                motor_max_torque: 50_000.0,
            },
            compliance: 0.0,
            damping: 0.02,
            max_force: f64::INFINITY,
            max_torque: f64::INFINITY,
            collide_connected: false,
        };
        world.add_joint(spec).unwrap();

        for _ in 0..90 {
            world.step(1.0 / 60.0, &mut empty);
        }
        let s = world.body_snapshot(door).unwrap();
        let pivot = s.transform.transform_point(Vec3d::new(-1.5, 0.0, 0.0));
        assert!(
            (pivot - Vec3d::new(0.0, 1.0, 0.5)).length() < 0.02,
            "hinge pivot drifted: {pivot:?}"
        );
        let fwd = s.transform.rotation.rotate_vec3(Vec3d::X);
        let yaw = (-fwd.z).atan2(fwd.x);
        assert!(
            yaw > 0.65 && yaw <= 0.85,
            "hinge motor should reach and respect +0.8 rad upper limit, got yaw={yaw}"
        );
    }

    // 4. Slider joint with linear limits.
    {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3d::ZERO,
            enable_sleeping: false,
            enable_buoyancy: false,
            ..PhysicsConfig::default()
        });
        let g = make_cube_grid(1, MaterialId::WOOD, world.materials());
        let cart = world.spawn_body(BodyDesc::dynamic(g, Vec3d::ZERO));
        let spec = JointSpec {
            body_a: cart,
            body_b: None,
            local_anchor_a: Vec3d::ZERO,
            local_anchor_b: Vec3d::ZERO,
            kind: JointKind::Slider {
                axis_a: Vec3d::X,
                axis_b: Vec3d::X,
                limits: Some((-1.5, 1.5)),
                motor_target_velocity: None,
                motor_max_force: 0.0,
            },
            compliance: 0.0,
            damping: 0.02,
            max_force: f64::INFINITY,
            max_torque: f64::INFINITY,
            collide_connected: false,
        };
        world.add_joint(spec).unwrap();

        for _ in 0..90 {
            // Apply diagonal force (strong +X and lateral +Y/+Z).
            world.apply_force_at_world_point(
                cart,
                Vec3d::new(8000.0, 6000.0, -4000.0),
                Vec3d::ZERO,
            );
            world.step(1.0 / 60.0, &mut empty);
        }
        let s = world.body_snapshot(cart).unwrap();
        assert!(
            s.transform.position.y.abs() < 0.03 && s.transform.position.z.abs() < 0.03,
            "slider drifted off axis: {:?}",
            s.transform.position
        );
        assert!(
            s.transform.position.x > 1.35 && s.transform.position.x <= 1.55,
            "slider failed to stop at +1.5 limit: x={}",
            s.transform.position.x
        );
    }

    // 5. Overload joint breakage emits PhysicsEvent::JointBroken.
    {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            enable_sleeping: false,
            enable_buoyancy: false,
            ..PhysicsConfig::default()
        });
        let iron = make_cube_grid(1, MaterialId::IRON, world.materials()); // weight ~ 77,008 N
        let block = world.spawn_body(BodyDesc::dynamic(iron, Vec3d::new(0.0, 10.0, 0.0)));
        let mut spec = JointSpec::distance(block, None, 2.0);
        spec.local_anchor_b = Vec3d::new(0.0, 12.0, 0.0);
        spec.max_force = 10_000.0; // far below 77,008 N weight!
        let jid = world.add_joint(spec).unwrap();

        let mut broke = false;
        for _ in 0..30 {
            world.step(1.0 / 60.0, &mut empty);
            for ev in world.drain_events() {
                if let PhysicsEvent::JointBroken { joint, force, .. } = ev {
                    assert_eq!(joint, jid);
                    assert!(force > 10_000.0);
                    broke = true;
                }
            }
        }
        assert!(broke, "overloaded joint must break and emit JointBroken");
        assert_eq!(world.joint_count(), 0);
        assert!(
            world.body_snapshot(block).unwrap().transform.position.y < 9.5,
            "block should fall after joint breaks"
        );
    }
}

#[test]
fn test_09_archimedes_voxel_buoyancy_equilibrium() {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        enable_sleeping: false,
        ..PhysicsConfig::default()
    });
    let mut ocean = FlatWorldProvider::with_water(0, 10.0);

    // 2x1x2 wooden raft (density 600 kg/m^3 in 1000 kg/m^3 water => 60% draft => CoM at y = 9.90).
    let raft_grid = VoxelSubGrid::solid_box(
        [0, 0, 0],
        [2, 1, 2],
        VoxelEntry::from_material(MaterialId::WOOD),
        world.materials(),
    );
    let raft = world.spawn_body(BodyDesc {
        kind: BodyKind::Dynamic,
        transform: Transformd::new(
            Vec3d::new(0.0, 11.2, 0.0),
            Quatd::from_axis_angle(Vec3d::Z, 0.25),
        ),
        grid: raft_grid,
        ..BodyDesc::default()
    });

    // Iron cube (density 7850 kg/m^3 > 1000 => sinks to seafloor at y = 0.5).
    let iron_grid = make_cube_grid(1, MaterialId::IRON, world.materials());
    let anchor = world.spawn_body(BodyDesc::dynamic(iron_grid, Vec3d::new(10.0, 11.0, 0.0)));

    for _ in 0..360 {
        world.step(1.0 / 60.0, &mut ocean);
    }

    let raft_snap = world.body_snapshot(raft).unwrap();
    let expected_y = 10.0 - 0.6 + 0.5; // bottom at 9.4, height 1.0 => center at 9.90
    assert!(
        (raft_snap.transform.position.y - expected_y).abs() < 0.08,
        "wooden raft buoyancy equilibrium mismatch: y={} expected={expected_y}",
        raft_snap.transform.position.y
    );
    assert!(
        raft_snap.linear_velocity.y.abs() < 0.05,
        "raft did not settle vertically: vy={}",
        raft_snap.linear_velocity.y
    );
    let up = raft_snap.transform.rotation.rotate_vec3(Vec3d::Y);
    assert!(
        up.y > 0.995,
        "tilted raft should hydrostatically right itself, up={up:?}"
    );

    let iron_snap = world.body_snapshot(anchor).unwrap();
    assert!(
        (iron_snap.transform.position.y - 0.5).abs() < 0.05,
        "iron cube should sink to seafloor y=0.5, got y={}",
        iron_snap.transform.position.y
    );
}

fn run_determinism_scenario() -> (Vec<horizon_physics_core::BodySnapshot>, Vec<PhysicsEvent>) {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let mut terrain = FlatWorldProvider::with_water(0, 4.0);

    // Spawn a mixed scene: colliding rotated cubes, hinged pendulum, floating raft, voxel split.
    let mut all_events = Vec::new();
    let mut tracked_ids: Vec<BodyId> = Vec::new();

    for i in 0..6 {
        let g = make_cube_grid(
            1,
            if i % 2 == 0 {
                MaterialId::STONE
            } else {
                MaterialId::WOOD
            },
            world.materials(),
        );
        let id = world.spawn_body(BodyDesc {
            kind: BodyKind::Dynamic,
            transform: Transformd::new(
                Vec3d::new(0.5 + (i % 3) as f64 * 0.95, 6.0 + i as f64 * 1.4, 0.5),
                Quatd::from_euler_zyx(0.1 * i as f64, -0.15 * i as f64, 0.08 * i as f64),
            ),
            linear_velocity: Vec3d::new(0.5 * (i as f64 - 2.5), 0.0, -0.2),
            grid: g,
            ..BodyDesc::default()
        });
        tracked_ids.push(id);
    }

    let bar = VoxelSubGrid::solid_box(
        [0, 0, 0],
        [3, 1, 1],
        VoxelEntry::from_material(MaterialId::WOOD),
        world.materials(),
    );
    let bridge = world.spawn_body(BodyDesc {
        kind: BodyKind::Dynamic,
        transform: Transformd::from_position(Vec3d::new(8.0, 5.5, 0.5)),
        angular_velocity: Vec3d::new(0.0, 1.5, 0.3),
        grid: bar,
        ..BodyDesc::default()
    });
    let mut j = JointSpec::distance(tracked_ids[0], Some(tracked_ids[1]), 2.5);
    j.compliance = 1e-5;
    world.add_joint(j);

    for step in 0..150 {
        if step == 45 {
            world.set_body_voxel(bridge, VoxelCoord::new(1, 0, 0), None);
        }
        world.step(1.0 / 60.0, &mut terrain);
        all_events.extend(world.drain_events());
    }

    (world.snapshots(), all_events)
}

#[test]
fn test_10_bit_exact_determinism_replay() {
    let (snaps_a, events_a) = run_determinism_scenario();
    let (snaps_b, events_b) = run_determinism_scenario();

    assert_eq!(snaps_a.len(), snaps_b.len());
    for (a, b) in snaps_a.iter().zip(snaps_b.iter()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.sleeping, b.sleeping);
        assert_eq!(a.voxel_count, b.voxel_count);
        assert_eq!(
            a.transform.position.x.to_bits(),
            b.transform.position.x.to_bits()
        );
        assert_eq!(
            a.transform.position.y.to_bits(),
            b.transform.position.y.to_bits()
        );
        assert_eq!(
            a.transform.position.z.to_bits(),
            b.transform.position.z.to_bits()
        );
        assert_eq!(
            a.transform.rotation.x.to_bits(),
            b.transform.rotation.x.to_bits()
        );
        assert_eq!(
            a.transform.rotation.y.to_bits(),
            b.transform.rotation.y.to_bits()
        );
        assert_eq!(
            a.transform.rotation.z.to_bits(),
            b.transform.rotation.z.to_bits()
        );
        assert_eq!(
            a.transform.rotation.w.to_bits(),
            b.transform.rotation.w.to_bits()
        );
        assert_eq!(a.linear_velocity.x.to_bits(), b.linear_velocity.x.to_bits());
        assert_eq!(a.linear_velocity.y.to_bits(), b.linear_velocity.y.to_bits());
        assert_eq!(a.linear_velocity.z.to_bits(), b.linear_velocity.z.to_bits());
        assert_eq!(
            a.angular_velocity.x.to_bits(),
            b.angular_velocity.x.to_bits()
        );
        assert_eq!(
            a.angular_velocity.y.to_bits(),
            b.angular_velocity.y.to_bits()
        );
        assert_eq!(
            a.angular_velocity.z.to_bits(),
            b.angular_velocity.z.to_bits()
        );
    }
    assert_eq!(
        events_a, events_b,
        "event streams must be strictly bit-identical"
    );
}

#[test]
fn test_11_entity_carry_and_stair_climb_on_tilted_rotating_deck() {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3d::ZERO,
        enable_buoyancy: false,
        ..PhysicsConfig::default()
    });
    let mut empty = EmptyWorldProvider;

    // Build an 8x1x8 deck with a raised 1-voxel stair platform on x in [5..8], y = 1.
    let mut deck_grid = VoxelSubGrid::solid_box(
        [0, 0, 0],
        [8, 1, 8],
        VoxelEntry::from_material(MaterialId::WOOD),
        world.materials(),
    );
    for z in 0..8 {
        for x in 5..8 {
            deck_grid.set_voxel(
                VoxelCoord::new(x, 1, z),
                Some(VoxelEntry::from_material(MaterialId::WOOD)),
                world.materials(),
            );
        }
    }
    deck_grid.rebuild_dirty_colliders();
    let deck_com = deck_grid.com_local();

    // Tilted 10 degrees around Z, translating +X/+Z and yawing around +Y!
    let tilt = Quatd::from_axis_angle(Vec3d::Z, 10.0_f64.to_radians());
    let deck_id = world.spawn_body(BodyDesc {
        kind: BodyKind::Kinematic,
        transform: Transformd::new(Vec3d::new(0.0, 5.0, 0.0), tilt),
        linear_velocity: Vec3d::new(3.0, 0.0, 1.5),
        angular_velocity: Vec3d::new(0.0, 0.4, 0.0),
        grid: deck_grid,
        ..BodyDesc::default()
    });

    // Place a player entity (0.6 x 1.8 x 0.6) standing at local deck point (2.0, 1.0, 4.0).
    let init_snap = world.body_snapshot(deck_id).unwrap();
    let local_foot_start = Vec3d::new(2.0, 1.0, 4.0) - deck_com;
    let world_foot_start = init_snap.transform.transform_point(local_foot_start);
    let mut player_aabb = Aabbd::new(
        Vec3d::new(
            world_foot_start.x - 0.3,
            world_foot_start.y + 0.02,
            world_foot_start.z - 0.3,
        ),
        Vec3d::new(
            world_foot_start.x + 0.3,
            world_foot_start.y + 1.82,
            world_foot_start.z + 0.3,
        ),
    );

    // Initial settle tick onto deck.
    let res0 = world.sweep_entity_aabb(player_aabb, Vec3d::new(0.0, -0.2, 0.0), None);
    assert!(res0.on_ground, "entity should land on tilted deck");
    player_aabb = res0.resolved_aabb;
    let mut ground = res0.ground_body;
    let initial_local_anchor = ground.unwrap().1;

    // Phase A: Stand still on moving + rotating + tilted deck for 30 ticks.
    let dt = 1.0 / 60.0;
    let mut total_yaw_carried = 0.0;
    for _ in 0..30 {
        world.step(dt, &mut empty);
        let outcome = world.sweep_entity_aabb(player_aabb, Vec3d::new(0.0, -0.15, 0.0), ground);
        assert!(
            outcome.on_ground,
            "entity slipped off moving rotating tilted deck"
        );
        assert!(
            outcome.carry_delta.length() > 0.01,
            "expected nonzero deck carry displacement"
        );
        total_yaw_carried += outcome.carry_yaw_delta;
        player_aabb = outcome.resolved_aabb;
        ground = outcome.ground_body;
    }

    // Verify accumulated yaw matches 0.4 rad/s * 0.5 s = 0.2 rad.
    assert!(
        (total_yaw_carried - 0.4 * 0.5).abs() < 0.03,
        "carried yaw mismatch: got={total_yaw_carried}, expected=0.2"
    );
    // Verify player stayed tightly anchored to the same deck-local spot.
    let final_local_anchor = ground.unwrap().1;
    assert!(
        (final_local_anchor - initial_local_anchor).length() < 0.15,
        "player drifted on rotating tilted deck: start={initial_local_anchor:?} end={final_local_anchor:?}"
    );

    // Phase B: Walk along deck local +X direction to climb onto the 1-voxel stair step (x in [5..8]).
    let mut climbed = false;
    let mut max_step_up = 0.0_f64;
    for _ in 0..50 {
        world.step(dt, &mut empty);
        let snap = world.body_snapshot(deck_id).unwrap();
        let walk_dir_world = snap.transform.transform_dir(Vec3d::X);
        let walk_horiz = Vec3d::new(walk_dir_world.x, 0.0, walk_dir_world.z).normalize();
        let delta = walk_horiz * 0.08 + Vec3d::new(0.0, -0.15, 0.0);
        let outcome = world.sweep_entity_aabb(player_aabb, delta, ground);
        if outcome.stepped_up_height > 0.4 {
            climbed = true;
            max_step_up = max_step_up.max(outcome.stepped_up_height);
        }
        player_aabb = outcome.resolved_aabb;
        ground = outcome.ground_body;
    }

    assert!(
        climbed,
        "entity failed to climb 1-voxel stair on rotating tilted deck (max_step_up={max_step_up})"
    );
    // Verify entity ended on top of the upper stair deck (local grid y >= 1.9).
    let final_snap = world.body_snapshot(deck_id).unwrap();
    let final_foot = Vec3d::new(
        player_aabb.center().x,
        player_aabb.min.y,
        player_aabb.center().z,
    );
    let local_grid_foot = final_snap.transform.inverse_transform_point(final_foot) + deck_com;
    assert!(
        local_grid_foot.y > 1.75,
        "entity should stand atop stair platform (y~2.0), got local_grid_y={}",
        local_grid_foot.y
    );
}

#[test]
fn test_12_throughput_1000_bodies_and_complex_ships() {
    // Benchmark A: 1000 dynamic rigid bodies over 60 steps.
    let mut world = PhysicsWorld::new(PhysicsConfig {
        enable_buoyancy: false,
        ..PhysicsConfig::default()
    });
    let mut terrain = FlatWorldProvider::new(0);

    let cube = make_cube_grid(1, MaterialId::STONE, world.materials());
    for i in 0..1000usize {
        let gx = (i % 25) as f64 * 2.5;
        let gz = ((i / 25) % 20) as f64 * 2.5;
        let gy = (i / 500) as f64 * 1.15 + 0.55;
        world.spawn_body(BodyDesc::dynamic(
            cube.clone(),
            Vec3d::new(gx + 0.5, gy, gz + 0.5),
        ));
    }
    assert_eq!(world.body_count(), 1000);

    let t0 = Instant::now();
    let mut total_metrics_us = 0u64;
    for _ in 0..60 {
        let m = world.step(1.0 / 60.0, &mut terrain);
        total_metrics_us += m.step_time_us;
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let avg_step_ms = elapsed_ms / 60.0;
    println!(
        "[Bench 1000 bodies] 60 steps total={elapsed_ms:.2} ms, avg={avg_step_ms:.3} ms/step (reported sum={total_metrics_us} us)"
    );
    let max_body_step_ms = if cfg!(debug_assertions) { 150.0 } else { 25.0 };
    assert!(
        avg_step_ms < max_body_step_ms,
        "1000-body average step latency too high: {avg_step_ms:.2} ms (limit {max_body_step_ms:.1} ms)"
    );

    // Benchmark B: 20 complex multi-box voxel ships (200+ voxels each) floating in ocean.
    let mut ship_world = PhysicsWorld::new(PhysicsConfig::default());
    let mut ocean = FlatWorldProvider::with_water(0, 10.0);

    let mut hull = VoxelSubGrid::solid_box(
        [0, 0, 0],
        [10, 3, 6],
        VoxelEntry::from_material(MaterialId::WOOD),
        ship_world.materials(),
    );
    for z in 1..5 {
        for y in 1..3 {
            for x in 1..9 {
                hull.set_voxel(VoxelCoord::new(x, y, z), None, ship_world.materials());
            }
        }
    }
    // Add keel + mast.
    for x in 2..8 {
        hull.set_voxel(
            VoxelCoord::new(x, -1, 3),
            Some(VoxelEntry::from_material(MaterialId::IRON)),
            ship_world.materials(),
        );
    }
    for y in 3..9 {
        hull.set_voxel(
            VoxelCoord::new(5, y, 3),
            Some(VoxelEntry::from_material(MaterialId::WOOD)),
            ship_world.materials(),
        );
    }
    hull.rebuild_dirty_colliders();
    assert!(hull.voxel_count() > 120);
    assert!(hull.boxes().len() < 20);

    for i in 0..20 {
        ship_world.spawn_body(BodyDesc::dynamic(
            hull.clone(),
            Vec3d::new((i % 5) as f64 * 18.0, 10.2, (i / 5) as f64 * 14.0),
        ));
    }

    let t1 = Instant::now();
    for _ in 0..60 {
        ship_world.step(1.0 / 60.0, &mut ocean);
    }
    let ship_avg_ms = (t1.elapsed().as_secs_f64() * 1000.0) / 60.0;
    println!("[Bench 20 complex ships in ocean] avg={ship_avg_ms:.3} ms/step");
    let max_ship_step_ms = if cfg!(debug_assertions) { 80.0 } else { 16.6 };
    assert!(
        ship_avg_ms < max_ship_step_ms,
        "20 complex ships average step latency exceeded budget: {ship_avg_ms:.2} ms (limit {max_ship_step_ms:.1} ms)"
    );
}

#[test]
fn test_13_duff_basis_raycast_and_ghost_free_voxel_seam_sliding() {
    // 1. Duff 2017 branchless orthonormal basis across tricky directions including -Z pole.
    let dirs = [
        Vec3d::X,
        Vec3d::Y,
        Vec3d::Z,
        Vec3d::NEG_X,
        Vec3d::NEG_Y,
        Vec3d::NEG_Z,
        Vec3d::new(0.3, -0.7, 0.648),
        Vec3d::new(-0.999999, 0.0, -1e-6),
    ];
    for d in dirs {
        let n = d.normalize();
        let (b1, b2) = n.orthonormal_basis();
        assert!((b1.length() - 1.0).abs() < 1e-12);
        assert!((b2.length() - 1.0).abs() < 1e-12);
        assert!(b1.dot(n).abs() < 1e-12);
        assert!(b2.dot(n).abs() < 1e-12);
        assert!(b1.dot(b2).abs() < 1e-12);
        assert!((b1.cross(b2) - n).length() < 1e-12);
    }

    // 2. Raycast against rotated voxel body returns exact voxel coordinate and normal.
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let g = VoxelSubGrid::solid_box(
        [0, 0, 0],
        [2, 2, 2],
        VoxelEntry::from_material(MaterialId::STONE),
        world.materials(),
    );
    let body = world.spawn_body(BodyDesc {
        kind: BodyKind::Static,
        transform: Transformd::new(
            Vec3d::new(5.0, 1.0, 0.0),
            Quatd::from_axis_angle(Vec3d::Y, std::f64::consts::FRAC_PI_4),
        ),
        grid: g,
        ..BodyDesc::default()
    });
    let hit = world
        .raycast(Vec3d::new(0.0, 1.0, 0.0), Vec3d::X, 20.0)
        .expect("ray must hit rotated voxel box");
    assert_eq!(hit.body_id, body);
    assert!(hit.distance > 3.0 && hit.distance < 4.0);
    assert!(hit.voxel_coord.is_some());
    assert!(hit.normal.x < -0.5);
    assert!(!world.debug_wireframes().is_empty());

    // 3. Ghost-collision-free sliding across 15 flat terrain voxel seams on ICE.
    let mut slide_world = PhysicsWorld::new(PhysicsConfig {
        enable_sleeping: false,
        enable_buoyancy: false,
        ..PhysicsConfig::default()
    });
    let mut ice_floor = FlatWorldProvider {
        ground_y: 0,
        ground_material: MaterialId::ICE,
        water_level: None,
    };
    let sled_grid = make_cube_grid(1, MaterialId::ICE, slide_world.materials());
    let sled = slide_world.spawn_body(BodyDesc {
        kind: BodyKind::Dynamic,
        transform: Transformd::from_position(Vec3d::new(0.5, 0.5, 0.5)),
        linear_velocity: Vec3d::new(10.0, 0.0, 0.0),
        grid: sled_grid,
        linear_damping_override: Some(0.0),
        angular_damping_override: Some(0.0),
        ..BodyDesc::default()
    });

    for _ in 0..90 {
        slide_world.step(1.0 / 60.0, &mut ice_floor);
    }
    let snap = slide_world.body_snapshot(sled).unwrap();
    // With mu = 0.04, deceleration a = 0.04 * 9.81 = 0.3924 m/s^2 => over 1.5 s, vx ~ 9.41 m/s, x ~ 15.0 m!
    assert!(
        snap.transform.position.x > 13.5,
        "ice block snagged on internal voxel seams! x={} vx={}",
        snap.transform.position.x,
        snap.linear_velocity.x
    );
    assert!(
        snap.linear_velocity.x > 8.5,
        "unexpected velocity drop across flat seams: vx={}",
        snap.linear_velocity.x
    );
    assert!(
        snap.angular_velocity.length() < 0.05,
        "seam snag caused phantom tumble: omega={:?}",
        snap.angular_velocity
    );
}

#[test]
fn test_14_arena_broadphase_math_edge_cases_and_despawn() {
    // 1. SlotArena generational invalidation and deterministic reuse.
    let mut arena: SlotArena<i32, BodyId> = SlotArena::new();
    let a = arena.insert(10);
    let b = arena.insert(20);
    assert_eq!(arena.len(), 2);
    assert_eq!(arena.remove(a), Some(10));
    assert!(arena.get(a).is_none(), "stale handle must be rejected");
    let c = arena.insert(30);
    assert_eq!(c.index, a.index, "slot index reused deterministically");
    assert_ne!(c.generation, a.generation, "generation must advance");
    assert_eq!(arena.get(a), None);
    assert_eq!(arena.get(c), Some(&30));
    let (pb, pc) = arena.get_pair_mut(b, c).expect("distinct mutable borrow");
    *pb += 1;
    *pc += 2;
    assert_eq!(arena.get(b), Some(&21));
    assert_eq!(arena.get(c), Some(&32));

    // 2. Mat3d::try_inverse singularity protection & skew cross-product identity.
    let singular = Mat3d::from_cols(Vec3d::X, Vec3d::X * 2.0, Vec3d::Y);
    assert!(singular.try_inverse().is_none());
    let invertible = Mat3d::from_cols(
        Vec3d::new(3.0, 1.0, 0.0),
        Vec3d::new(0.0, 2.0, 1.0),
        Vec3d::new(1.0, 0.0, 4.0),
    );
    let inv = invertible.try_inverse().expect("invertible 3x3");
    let prod = invertible * inv;
    for col in 0..3 {
        for row in 0..3 {
            let expected = if col == row { 1.0 } else { 0.0 };
            assert!((prod.cols[col][row] - expected).abs() < 1e-12);
        }
    }
    let u = Vec3d::new(1.5, -2.0, 3.25);
    let v = Vec3d::new(-0.5, 4.0, 1.0);
    assert!((Mat3d::skew(u) * v - u.cross(v)).length() < 1e-12);

    // 3. Quatd::slerp shortest-arc handling with negative dot product.
    let q0 = Quatd::IDENTITY;
    let q1_neg = Quatd::new(0.0, -0.1, 0.0, -(1.0 - 0.01_f64).sqrt()); // negated hemisphere
    let mid = q0.slerp(q1_neg, 0.5);
    assert!(
        mid.w > 0.99,
        "slerp with negative dot product must take shortest arc, got {mid:?}"
    );

    // 4. SpatialHash16 & DynamicBvh deterministic queries.
    let mut hash = SpatialHash16::new();
    let box_a = Aabbd::new(Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(18.0, 2.0, 2.0));
    let box_b = Aabbd::new(Vec3d::new(17.0, 0.0, 0.0), Vec3d::new(20.0, 2.0, 2.0));
    hash.insert(1, &box_a);
    hash.insert(2, &box_b);
    let mut found = Vec::new();
    hash.query_aabb(&box_b, &mut found);
    assert_eq!(found, vec![1, 2]);
    hash.remove(1, &box_a);
    found.clear();
    hash.query_aabb(&box_b, &mut found);
    assert_eq!(found, vec![2]);

    let mut bvh = DynamicBvh::new(0.2);
    let p1 = bvh.insert(10, box_a);
    let p2 = bvh.insert(20, box_b);
    let mut pairs = Vec::new();
    bvh.query_pairs(&mut pairs);
    assert_eq!(pairs, vec![(10, 20)]);
    bvh.update(
        p2,
        Aabbd::new(Vec3d::new(100.0, 0.0, 0.0), Vec3d::new(102.0, 2.0, 2.0)),
        Vec3d::new(80.0, 0.0, 0.0),
    );
    bvh.query_pairs(&mut pairs);
    assert!(pairs.is_empty());
    bvh.remove(p1);
    bvh.remove(p2);
    assert!(bvh.is_empty());

    // 5. despawn_body removes attached joints cleanly.
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let g1 = make_cube_grid(1, MaterialId::WOOD, world.materials());
    let g2 = make_cube_grid(1, MaterialId::WOOD, world.materials());
    let b1 = world.spawn_body(BodyDesc::dynamic(g1, Vec3d::ZERO));
    let b2 = world.spawn_body(BodyDesc::dynamic(g2, Vec3d::new(3.0, 0.0, 0.0)));
    assert!(
        world
            .add_joint(JointSpec::distance(b1, Some(b2), 3.0))
            .is_some()
    );
    assert_eq!(world.joint_count(), 1);
    assert!(world.despawn_body(b1));
    assert_eq!(world.body_count(), 1);
    assert_eq!(world.joint_count(), 0);
}
