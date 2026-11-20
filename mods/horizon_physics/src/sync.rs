//! World persistence via Horizon [`AttachmentType`] Compound NBT and compact
//! binary delta pose synchronization over Horizon custom [`PayloadId`] channels.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use horizon_physics_core::{
    BodyId, BodyKind, MaterialId, Quatd, Transformd, Vec3d, VoxelCoord, VoxelEntry,
};
use mod_api::{
    AttachmentCodec, AttachmentError, AttachmentType, Compound, Holder, IPayloadContext,
    IPayloadHandler, PayloadBytes, PayloadHandleError, PayloadId, PlayerRef, Registrar, ServerHost,
    Tag,
};

use crate::api::{
    DistanceJointDesc, FixedWeldJointDesc, HingeJointDesc, JointConstraint, JointMotor,
    JointRegistry, PhysicsApi, SliderJointDesc, SubGridHandle,
};

/// Horizon custom payload channel for compact rigid-body pose delta sync.
pub const POSE_SYNC_CHANNEL: &str = "horizon_physics:pose_sync";

/// Magic bytes prefixing every binary pose snapshot packet.
const PACKET_MAGIC: [u8; 4] = *b"HPHY";
const PACKET_VERSION: u16 = 2;
const ATTACHMENT_SCHEMA_VERSION: i32 = 1;
const MAX_PACKET_BODIES: usize = 4096;
const MAX_PACKET_REMOVALS: usize = 4096;
const MAX_PACKET_VOXELS_PER_BODY: usize = 65_535;
const MAX_BLOCK_ID_BYTES: usize = 256;
/// Safe payload budget strictly below Horizon's 1,048,576-byte custom payload cap.
pub const MAX_WIRE_PACKET_BYTES: usize = 768 * 1024;

/// Persisted voxel record inside a serialized sub-grid body.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistedVoxel {
    pub coord: [i32; 3],
    pub state_id: u32,
    pub material_id: u16,
    pub block_id: String,
}

/// Persisted rigid-body state saved in level NBT attachments.
#[derive(Clone, Debug, PartialEq)]
pub struct PersistedBody {
    pub kind: BodyKind,
    pub position: [f64; 3],
    pub rotation: [f64; 4],
    pub linear_velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub sleeping: bool,
    pub enable_ccd: bool,
    pub gravity_scale: f64,
    pub user_data: u64,
    pub voxels: Vec<PersistedVoxel>,
}

/// Persisted joint entry referencing bodies by their index inside [`PersistedSubGridWorld::bodies`].
#[derive(Clone, Debug, PartialEq)]
pub struct PersistedJoint {
    pub body_a_idx: usize,
    pub body_b_idx: Option<usize>,
    pub kind: PersistedJointKind,
    pub anchor_a: [f64; 3],
    pub anchor_b: [f64; 3],
    pub compliance: f64,
    pub damping: f64,
}

/// Joint-specific parameters persisted to NBT.
#[derive(Clone, Debug, PartialEq)]
pub enum PersistedJointKind {
    Distance {
        rest_length: f64,
        min_length: Option<f64>,
        max_length: Option<f64>,
    },
    FixedWeld {
        relative_rotation: [f64; 4],
    },
    Hinge {
        axis_a: [f64; 3],
        axis_b: [f64; 3],
        min_angle_rad: f64,
        max_angle_rad: f64,
        motor: Option<JointMotor>,
    },
    Slider {
        axis_a: [f64; 3],
        axis_b: [f64; 3],
        limits: Option<(f64, f64)>,
        motor_target_velocity: Option<f64>,
        motor_max_force: f64,
    },
}

/// Complete serialized snapshot of all physics sub-grids and joints in a level.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PersistedSubGridWorld {
    pub bodies: Vec<PersistedBody>,
    pub joints: Vec<PersistedJoint>,
}

impl PersistedSubGridWorld {
    /// Captures all active sub-grids and joints from [`PhysicsApi`].
    pub fn capture(api: &PhysicsApi) -> Self {
        let snapshots = api.world().snapshots();
        let mut handle_to_idx: BTreeMap<SubGridHandle, usize> = BTreeMap::new();
        let mut bodies = Vec::with_capacity(snapshots.len());

        for snap in snapshots {
            let handle = SubGridHandle(snap.id);
            let idx = bodies.len();
            handle_to_idx.insert(handle, idx);

            let block_names = api.subgrid_block_names(handle);
            let mut voxels = Vec::new();
            if let Some(grid) = api.world().body_grid(snap.id) {
                for (vc, entry) in grid.iter_voxels() {
                    let coord = [vc.x, vc.y, vc.z];
                    let block_id = block_names
                        .and_then(|m| m.get(&coord))
                        .cloned()
                        .unwrap_or_else(|| "minecraft:stone".to_owned());
                    voxels.push(PersistedVoxel {
                        coord,
                        state_id: entry.state_id,
                        material_id: entry.material.0,
                        block_id,
                    });
                }
            }

            bodies.push(PersistedBody {
                kind: snap.kind,
                position: snap.transform.position.to_array(),
                rotation: [
                    snap.transform.rotation.x,
                    snap.transform.rotation.y,
                    snap.transform.rotation.z,
                    snap.transform.rotation.w,
                ],
                linear_velocity: snap.linear_velocity.to_array(),
                angular_velocity: snap.angular_velocity.to_array(),
                sleeping: snap.sleeping,
                enable_ccd: snap.enable_ccd,
                gravity_scale: snap.gravity_scale,
                user_data: snap.user_data,
                voxels,
            });
        }

        let mut joints = Vec::new();
        for constraint in api.joints().values() {
            match *constraint {
                JointConstraint::Hinge(h) => {
                    let (Some(&ia), Some(&ib)) =
                        (handle_to_idx.get(&h.body_a), handle_to_idx.get(&h.body_b))
                    else {
                        continue;
                    };
                    joints.push(PersistedJoint {
                        body_a_idx: ia,
                        body_b_idx: Some(ib),
                        kind: PersistedJointKind::Hinge {
                            axis_a: h.axis_a,
                            axis_b: h.axis_b,
                            min_angle_rad: h.min_angle_rad,
                            max_angle_rad: h.max_angle_rad,
                            motor: h.motor,
                        },
                        anchor_a: h.anchor_a,
                        anchor_b: h.anchor_b,
                        compliance: h.compliance,
                        damping: 0.02,
                    });
                }
                JointConstraint::Slider(s) => {
                    let Some(&ia) = handle_to_idx.get(&s.body_a) else {
                        continue;
                    };
                    let ib = s.body_b.and_then(|b| handle_to_idx.get(&b).copied());
                    if s.body_b.is_some() && ib.is_none() {
                        continue;
                    }
                    joints.push(PersistedJoint {
                        body_a_idx: ia,
                        body_b_idx: ib,
                        kind: PersistedJointKind::Slider {
                            axis_a: s.axis_a,
                            axis_b: s.axis_b,
                            limits: s.limits,
                            motor_target_velocity: s.motor_target_velocity,
                            motor_max_force: s.motor_max_force,
                        },
                        anchor_a: s.anchor_a,
                        anchor_b: s.anchor_b,
                        compliance: s.compliance,
                        damping: 0.02,
                    });
                }
                JointConstraint::Distance(d) => {
                    let Some(&ia) = handle_to_idx.get(&d.body_a) else {
                        continue;
                    };
                    let ib = d.body_b.and_then(|b| handle_to_idx.get(&b).copied());
                    if d.body_b.is_some() && ib.is_none() {
                        continue;
                    }
                    joints.push(PersistedJoint {
                        body_a_idx: ia,
                        body_b_idx: ib,
                        kind: PersistedJointKind::Distance {
                            rest_length: d.rest_length,
                            min_length: d.min_length,
                            max_length: d.max_length,
                        },
                        anchor_a: d.anchor_a,
                        anchor_b: d.anchor_b,
                        compliance: d.compliance,
                        damping: d.damping,
                    });
                }
                JointConstraint::FixedWeld(w) => {
                    let Some(&ia) = handle_to_idx.get(&w.body_a) else {
                        continue;
                    };
                    let ib = w.body_b.and_then(|b| handle_to_idx.get(&b).copied());
                    if w.body_b.is_some() && ib.is_none() {
                        continue;
                    }
                    joints.push(PersistedJoint {
                        body_a_idx: ia,
                        body_b_idx: ib,
                        kind: PersistedJointKind::FixedWeld {
                            relative_rotation: w.relative_rotation,
                        },
                        anchor_a: w.anchor_a,
                        anchor_b: w.anchor_b,
                        compliance: w.compliance,
                        damping: 0.05,
                    });
                }
            }
        }

        Self { bodies, joints }
    }

    /// Restores this persisted snapshot into `api`, returning newly spawned body handles.
    pub fn restore_into(&self, api: &mut PhysicsApi) -> Result<Vec<SubGridHandle>, String> {
        let mut spawned = Vec::with_capacity(self.bodies.len());
        for body in &self.bodies {
            let blocks: Vec<([i32; 3], String)> = body
                .voxels
                .iter()
                .map(|v| (v.coord, v.block_id.clone()))
                .collect();
            let handle = api.spawn_named_subgrid(
                Vec3d::from_array(body.position),
                Quatd::new(
                    body.rotation[0],
                    body.rotation[1],
                    body.rotation[2],
                    body.rotation[3],
                )
                .normalize(),
                body.kind,
                body.enable_ccd,
                body.gravity_scale,
                body.user_data,
                &blocks,
            )?;

            // Restore exact persisted voxel state_id/material metadata.
            for v in &body.voxels {
                api.world_mut().set_body_voxel(
                    handle.body_id(),
                    VoxelCoord::new(v.coord[0], v.coord[1], v.coord[2]),
                    Some(VoxelEntry::new(v.state_id, MaterialId(v.material_id))),
                );
            }
            // Re-apply exact transform because voxel edits can shift CoM relative to origin.
            api.world_mut().set_body_transform(
                handle.body_id(),
                Transformd::new(
                    Vec3d::from_array(body.position),
                    Quatd::new(
                        body.rotation[0],
                        body.rotation[1],
                        body.rotation[2],
                        body.rotation[3],
                    )
                    .normalize(),
                ),
            );
            api.world_mut().set_body_velocity(
                handle.body_id(),
                Vec3d::from_array(body.linear_velocity),
                Vec3d::from_array(body.angular_velocity),
            );
            if body.sleeping {
                api.world_mut().sleep_body(handle.body_id());
            }
            spawned.push(handle);
        }

        for j in &self.joints {
            let Some(&body_a) = spawned.get(j.body_a_idx) else {
                continue;
            };
            let body_b = j.body_b_idx.and_then(|idx| spawned.get(idx).copied());
            let constraint = match &j.kind {
                PersistedJointKind::Distance {
                    rest_length,
                    min_length,
                    max_length,
                } => JointConstraint::Distance(DistanceJointDesc {
                    body_a,
                    body_b,
                    anchor_a: j.anchor_a,
                    anchor_b: j.anchor_b,
                    rest_length: *rest_length,
                    min_length: *min_length,
                    max_length: *max_length,
                    compliance: j.compliance,
                    damping: j.damping,
                }),
                PersistedJointKind::FixedWeld { relative_rotation } => {
                    JointConstraint::FixedWeld(FixedWeldJointDesc {
                        body_a,
                        body_b,
                        anchor_a: j.anchor_a,
                        anchor_b: j.anchor_b,
                        relative_rotation: *relative_rotation,
                        compliance: j.compliance,
                    })
                }
                PersistedJointKind::Hinge {
                    axis_a,
                    axis_b,
                    min_angle_rad,
                    max_angle_rad,
                    motor,
                } => {
                    let Some(hb) = body_b else {
                        continue;
                    };
                    JointConstraint::Hinge(HingeJointDesc {
                        body_a,
                        body_b: hb,
                        anchor_a: j.anchor_a,
                        anchor_b: j.anchor_b,
                        axis_a: *axis_a,
                        axis_b: *axis_b,
                        min_angle_rad: *min_angle_rad,
                        max_angle_rad: *max_angle_rad,
                        compliance: j.compliance,
                        motor: *motor,
                    })
                }
                PersistedJointKind::Slider {
                    axis_a,
                    axis_b,
                    limits,
                    motor_target_velocity,
                    motor_max_force,
                } => JointConstraint::Slider(SliderJointDesc {
                    body_a,
                    body_b,
                    anchor_a: j.anchor_a,
                    anchor_b: j.anchor_b,
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                    limits: *limits,
                    compliance: j.compliance,
                    motor_target_velocity: *motor_target_velocity,
                    motor_max_force: *motor_max_force,
                }),
            };
            let _ = api.add_joint(constraint);
        }

        Ok(spawned)
    }
}

/// Level-attached persistent storage for all voxel sub-grids and joints.
pub static SUBGRID_WORLD_ATTACHMENT: AttachmentType<PersistedSubGridWorld> = AttachmentType {
    namespace: "horizon_physics",
    name: "subgrid_world",
    default: PersistedSubGridWorld::default,
    codec: Some(AttachmentCodec {
        encode: encode_subgrid_world_nbt,
        decode: decode_subgrid_world_nbt,
    }),
    copy_on_death: false,
    sync: true,
};

/// Saves current physics sub-grids and joints into the level's [`SUBGRID_WORLD_ATTACHMENT`].
pub fn save_world_attachment(
    api: &PhysicsApi,
    host: &mut dyn ServerHost,
) -> Result<(), AttachmentError> {
    let snapshot = PersistedSubGridWorld::capture(api);
    SUBGRID_WORLD_ATTACHMENT.set(host.attachments(), Holder::Level, snapshot)
}

/// Loads persisted sub-grids and joints from the level's [`SUBGRID_WORLD_ATTACHMENT`].
/// Returns `Ok(true)` if saved data existed and was restored.
pub fn load_world_attachment(
    api: &mut PhysicsApi,
    host: &mut dyn ServerHost,
) -> Result<bool, AttachmentError> {
    let Some(saved) = SUBGRID_WORLD_ATTACHMENT.get_existing(host.attachments(), Holder::Level)?
    else {
        return Ok(false);
    };
    if saved.bodies.is_empty() {
        return Ok(false);
    }
    saved
        .restore_into(api)
        .map_err(|_| AttachmentError::Decode)?;
    Ok(true)
}

/// Encodes [`PersistedSubGridWorld`] into a Horizon Compound NBT tree.
pub fn encode_subgrid_world_nbt(
    world: &PersistedSubGridWorld,
) -> Result<Compound, AttachmentError> {
    let mut root = Compound::new();
    root.insert("schema".into(), Tag::Int(ATTACHMENT_SCHEMA_VERSION));

    let mut body_tags = Vec::with_capacity(world.bodies.len());
    for body in &world.bodies {
        let mut bc = Compound::new();
        let kind_byte: i8 = match body.kind {
            BodyKind::Dynamic => 0,
            BodyKind::Static => 1,
            BodyKind::Kinematic => 2,
        };
        bc.insert("kind".into(), Tag::Byte(kind_byte));
        insert_vec3_nbt(&mut bc, "pos", body.position)?;
        insert_quat_nbt(&mut bc, "rot", body.rotation)?;
        insert_vec3_nbt(&mut bc, "lin_vel", body.linear_velocity)?;
        insert_vec3_nbt(&mut bc, "ang_vel", body.angular_velocity)?;
        bc.insert("sleeping".into(), Tag::Byte(i8::from(body.sleeping)));
        bc.insert("ccd".into(), Tag::Byte(i8::from(body.enable_ccd)));
        bc.insert("grav".into(), Tag::Double(body.gravity_scale.to_bits()));
        bc.insert("user_data".into(), Tag::Long(body.user_data as i64));

        let mut voxel_tags = Vec::with_capacity(body.voxels.len());
        for v in &body.voxels {
            let mut vc = Compound::new();
            vc.insert("x".into(), Tag::Int(v.coord[0]));
            vc.insert("y".into(), Tag::Int(v.coord[1]));
            vc.insert("z".into(), Tag::Int(v.coord[2]));
            vc.insert("state".into(), Tag::Int(v.state_id as i32));
            vc.insert("mat".into(), Tag::Short(v.material_id as i16));
            vc.insert("block".into(), Tag::String(v.block_id.as_str().into()));
            voxel_tags.push(Tag::Compound(vc));
        }
        bc.insert("voxels".into(), Tag::List(voxel_tags));
        body_tags.push(Tag::Compound(bc));
    }
    root.insert("bodies".into(), Tag::List(body_tags));

    let mut joint_tags = Vec::with_capacity(world.joints.len());
    for joint in &world.joints {
        let mut jc = Compound::new();
        jc.insert(
            "a".into(),
            Tag::Int(i32::try_from(joint.body_a_idx).map_err(|_| AttachmentError::Encode)?),
        );
        let b_idx = match joint.body_b_idx {
            Some(i) => i32::try_from(i).map_err(|_| AttachmentError::Encode)?,
            None => -1,
        };
        jc.insert("b".into(), Tag::Int(b_idx));
        insert_vec3_nbt(&mut jc, "anchor_a", joint.anchor_a)?;
        insert_vec3_nbt(&mut jc, "anchor_b", joint.anchor_b)?;
        jc.insert("compliance".into(), Tag::Double(joint.compliance.to_bits()));
        jc.insert("damping".into(), Tag::Double(joint.damping.to_bits()));

        match &joint.kind {
            PersistedJointKind::Distance {
                rest_length,
                min_length,
                max_length,
            } => {
                jc.insert("kind".into(), Tag::String("distance".into()));
                jc.insert("rest".into(), Tag::Double(rest_length.to_bits()));
                if let Some(min_l) = min_length {
                    jc.insert("min_l".into(), Tag::Double(min_l.to_bits()));
                }
                if let Some(max_l) = max_length {
                    jc.insert("max_l".into(), Tag::Double(max_l.to_bits()));
                }
            }
            PersistedJointKind::FixedWeld { relative_rotation } => {
                jc.insert("kind".into(), Tag::String("fixed_weld".into()));
                insert_quat_nbt(&mut jc, "rel_rot", *relative_rotation)?;
            }
            PersistedJointKind::Hinge {
                axis_a,
                axis_b,
                min_angle_rad,
                max_angle_rad,
                motor,
            } => {
                jc.insert("kind".into(), Tag::String("hinge".into()));
                insert_vec3_nbt(&mut jc, "axis_a", *axis_a)?;
                insert_vec3_nbt(&mut jc, "axis_b", *axis_b)?;
                jc.insert("min_ang".into(), Tag::Double(min_angle_rad.to_bits()));
                jc.insert("max_ang".into(), Tag::Double(max_angle_rad.to_bits()));
                if let Some(m) = motor {
                    let mut mc = Compound::new();
                    mc.insert("target".into(), Tag::Double(m.target_angle_rad.to_bits()));
                    mc.insert("max_t".into(), Tag::Double(m.max_torque_nm.to_bits()));
                    mc.insert("stiff".into(), Tag::Double(m.stiffness.to_bits()));
                    mc.insert("damp".into(), Tag::Double(m.damping.to_bits()));
                    jc.insert("motor".into(), Tag::Compound(mc));
                }
            }
            PersistedJointKind::Slider {
                axis_a,
                axis_b,
                limits,
                motor_target_velocity,
                motor_max_force,
            } => {
                jc.insert("kind".into(), Tag::String("slider".into()));
                insert_vec3_nbt(&mut jc, "axis_a", *axis_a)?;
                insert_vec3_nbt(&mut jc, "axis_b", *axis_b)?;
                if let Some((lo, hi)) = limits {
                    jc.insert("lim_lo".into(), Tag::Double(lo.to_bits()));
                    jc.insert("lim_hi".into(), Tag::Double(hi.to_bits()));
                }
                if let Some(tv) = motor_target_velocity {
                    jc.insert("motor_vel".into(), Tag::Double(tv.to_bits()));
                }
                jc.insert("motor_max_f".into(), Tag::Double(motor_max_force.to_bits()));
            }
        }
        joint_tags.push(Tag::Compound(jc));
    }
    root.insert("joints".into(), Tag::List(joint_tags));
    Ok(root)
}

/// Decodes [`PersistedSubGridWorld`] from a Horizon Compound NBT tree.
pub fn decode_subgrid_world_nbt(root: &Compound) -> Result<PersistedSubGridWorld, AttachmentError> {
    let schema = read_i32(root, "schema")?;
    if schema != ATTACHMENT_SCHEMA_VERSION {
        return Err(AttachmentError::Decode);
    }

    let mut bodies = Vec::new();
    if let Some(Tag::List(body_list)) = root.get(&"bodies".into()) {
        for item in body_list {
            let Tag::Compound(bc) = item else {
                return Err(AttachmentError::Decode);
            };
            let kind = match read_i8(bc, "kind")? {
                0 => BodyKind::Dynamic,
                1 => BodyKind::Static,
                2 => BodyKind::Kinematic,
                _ => return Err(AttachmentError::Decode),
            };
            let position = read_vec3_nbt(bc, "pos")?;
            let rotation = read_quat_nbt(bc, "rot")?;
            let linear_velocity = read_vec3_nbt(bc, "lin_vel")?;
            let angular_velocity = read_vec3_nbt(bc, "ang_vel")?;
            let sleeping = read_i8(bc, "sleeping")? != 0;
            let enable_ccd = read_i8(bc, "ccd")? != 0;
            let gravity_scale = read_f64(bc, "grav")?;
            let user_data = read_i64(bc, "user_data")? as u64;

            let mut voxels = Vec::new();
            if let Some(Tag::List(vlist)) = bc.get(&"voxels".into()) {
                for vitem in vlist {
                    let Tag::Compound(vc) = vitem else {
                        return Err(AttachmentError::Decode);
                    };
                    let coord = [read_i32(vc, "x")?, read_i32(vc, "y")?, read_i32(vc, "z")?];
                    let state_id = read_i32(vc, "state")? as u32;
                    let material_id = read_i16(vc, "mat")? as u16;
                    let block_id = read_string(vc, "block")?;
                    voxels.push(PersistedVoxel {
                        coord,
                        state_id,
                        material_id,
                        block_id,
                    });
                }
            }

            bodies.push(PersistedBody {
                kind,
                position,
                rotation,
                linear_velocity,
                angular_velocity,
                sleeping,
                enable_ccd,
                gravity_scale,
                user_data,
                voxels,
            });
        }
    }

    let mut joints = Vec::new();
    if let Some(Tag::List(joint_list)) = root.get(&"joints".into()) {
        for item in joint_list {
            let Tag::Compound(jc) = item else {
                return Err(AttachmentError::Decode);
            };
            let body_a_idx =
                usize::try_from(read_i32(jc, "a")?).map_err(|_| AttachmentError::Decode)?;
            let raw_b = read_i32(jc, "b")?;
            let body_b_idx = if raw_b < 0 {
                None
            } else {
                Some(usize::try_from(raw_b).map_err(|_| AttachmentError::Decode)?)
            };
            let anchor_a = read_vec3_nbt(jc, "anchor_a")?;
            let anchor_b = read_vec3_nbt(jc, "anchor_b")?;
            let compliance = read_f64(jc, "compliance")?;
            let damping = read_f64(jc, "damping")?;
            let kind_name = read_string(jc, "kind")?;

            let kind = match kind_name.as_str() {
                "distance" => PersistedJointKind::Distance {
                    rest_length: read_f64(jc, "rest")?,
                    min_length: read_opt_f64(jc, "min_l")?,
                    max_length: read_opt_f64(jc, "max_l")?,
                },
                "fixed_weld" => PersistedJointKind::FixedWeld {
                    relative_rotation: read_quat_nbt(jc, "rel_rot")?,
                },
                "hinge" => {
                    let motor = match jc.get(&"motor".into()) {
                        Some(Tag::Compound(mc)) => Some(JointMotor {
                            target_angle_rad: read_f64(mc, "target")?,
                            max_torque_nm: read_f64(mc, "max_t")?,
                            stiffness: read_f64(mc, "stiff")?,
                            damping: read_f64(mc, "damp")?,
                        }),
                        None => None,
                        _ => return Err(AttachmentError::Decode),
                    };
                    PersistedJointKind::Hinge {
                        axis_a: read_vec3_nbt(jc, "axis_a")?,
                        axis_b: read_vec3_nbt(jc, "axis_b")?,
                        min_angle_rad: read_f64(jc, "min_ang")?,
                        max_angle_rad: read_f64(jc, "max_ang")?,
                        motor,
                    }
                }
                "slider" => {
                    let limits = match (read_opt_f64(jc, "lim_lo")?, read_opt_f64(jc, "lim_hi")?) {
                        (Some(lo), Some(hi)) => Some((lo, hi)),
                        (None, None) => None,
                        _ => return Err(AttachmentError::Decode),
                    };
                    PersistedJointKind::Slider {
                        axis_a: read_vec3_nbt(jc, "axis_a")?,
                        axis_b: read_vec3_nbt(jc, "axis_b")?,
                        limits,
                        motor_target_velocity: read_opt_f64(jc, "motor_vel")?,
                        motor_max_force: read_f64(jc, "motor_max_f")?,
                    }
                }
                _ => return Err(AttachmentError::Decode),
            };

            joints.push(PersistedJoint {
                body_a_idx,
                body_b_idx,
                kind,
                anchor_a,
                anchor_b,
                compliance,
                damping,
            });
        }
    }

    Ok(PersistedSubGridWorld { bodies, joints })
}

fn insert_vec3_nbt(c: &mut Compound, prefix: &str, v: [f64; 3]) -> Result<(), AttachmentError> {
    if !v.iter().all(|x| x.is_finite()) {
        return Err(AttachmentError::Encode);
    }
    c.insert(
        format!("{prefix}_x").as_str().into(),
        Tag::Double(v[0].to_bits()),
    );
    c.insert(
        format!("{prefix}_y").as_str().into(),
        Tag::Double(v[1].to_bits()),
    );
    c.insert(
        format!("{prefix}_z").as_str().into(),
        Tag::Double(v[2].to_bits()),
    );
    Ok(())
}

fn insert_quat_nbt(c: &mut Compound, prefix: &str, q: [f64; 4]) -> Result<(), AttachmentError> {
    if !q.iter().all(|x| x.is_finite()) {
        return Err(AttachmentError::Encode);
    }
    c.insert(
        format!("{prefix}_x").as_str().into(),
        Tag::Double(q[0].to_bits()),
    );
    c.insert(
        format!("{prefix}_y").as_str().into(),
        Tag::Double(q[1].to_bits()),
    );
    c.insert(
        format!("{prefix}_z").as_str().into(),
        Tag::Double(q[2].to_bits()),
    );
    c.insert(
        format!("{prefix}_w").as_str().into(),
        Tag::Double(q[3].to_bits()),
    );
    Ok(())
}

fn read_vec3_nbt(c: &Compound, prefix: &str) -> Result<[f64; 3], AttachmentError> {
    Ok([
        read_f64(c, &format!("{prefix}_x"))?,
        read_f64(c, &format!("{prefix}_y"))?,
        read_f64(c, &format!("{prefix}_z"))?,
    ])
}

fn read_quat_nbt(c: &Compound, prefix: &str) -> Result<[f64; 4], AttachmentError> {
    Ok([
        read_f64(c, &format!("{prefix}_x"))?,
        read_f64(c, &format!("{prefix}_y"))?,
        read_f64(c, &format!("{prefix}_z"))?,
        read_f64(c, &format!("{prefix}_w"))?,
    ])
}

fn read_i8(c: &Compound, key: &str) -> Result<i8, AttachmentError> {
    match c.get(&key.into()) {
        Some(&Tag::Byte(v)) => Ok(v),
        _ => Err(AttachmentError::Decode),
    }
}

fn read_i16(c: &Compound, key: &str) -> Result<i16, AttachmentError> {
    match c.get(&key.into()) {
        Some(&Tag::Short(v)) => Ok(v),
        _ => Err(AttachmentError::Decode),
    }
}

fn read_i32(c: &Compound, key: &str) -> Result<i32, AttachmentError> {
    match c.get(&key.into()) {
        Some(&Tag::Int(v)) => Ok(v),
        _ => Err(AttachmentError::Decode),
    }
}

fn read_i64(c: &Compound, key: &str) -> Result<i64, AttachmentError> {
    match c.get(&key.into()) {
        Some(&Tag::Long(v)) => Ok(v),
        _ => Err(AttachmentError::Decode),
    }
}

fn read_f64(c: &Compound, key: &str) -> Result<f64, AttachmentError> {
    match c.get(&key.into()) {
        Some(&Tag::Double(bits)) => {
            let v = f64::from_bits(bits);
            if v.is_finite() {
                Ok(v)
            } else {
                Err(AttachmentError::Decode)
            }
        }
        _ => Err(AttachmentError::Decode),
    }
}

fn read_opt_f64(c: &Compound, key: &str) -> Result<Option<f64>, AttachmentError> {
    match c.get(&key.into()) {
        None => Ok(None),
        Some(&Tag::Double(bits)) => {
            let v = f64::from_bits(bits);
            if v.is_finite() {
                Ok(Some(v))
            } else {
                Err(AttachmentError::Decode)
            }
        }
        _ => Err(AttachmentError::Decode),
    }
}

fn read_string(c: &Compound, key: &str) -> Result<String, AttachmentError> {
    match c.get(&key.into()) {
        Some(Tag::String(js)) => Ok(js.to_string_lossy()),
        _ => Err(AttachmentError::Decode),
    }
}

// ---------------------------------------------------------------------------
// Compact binary pose snapshot wire codec & delta synchronization tracker
// ---------------------------------------------------------------------------

/// Minimal per-body kinematic pose transmitted over the custom payload channel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BodyPoseRecord {
    pub id: BodyId,
    pub transform: Transformd,
    pub lin_vel: Vec3d,
    pub ang_vel: Vec3d,
    pub sleeping: bool,
}

impl BodyPoseRecord {
    pub fn roughly_equal(&self, other: &Self) -> bool {
        self.id == other.id
            && self.sleeping == other.sleeping
            && (self.transform.position - other.transform.position).length_sq() < 1e-10
            && self.transform.rotation.dot(other.transform.rotation).abs() > 1.0 - 1e-10
            && (self.lin_vel - other.lin_vel).length_sq() < 1e-10
            && (self.ang_vel - other.ang_vel).length_sq() < 1e-10
    }
}

/// Compact synced voxel entry carried only when a sub-grid spawns or mutates geometry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyncedVoxelRecord {
    pub coord: VoxelCoord,
    pub material: MaterialId,
    pub block_id: String,
}

/// Complete voxel geometry snapshot for a newly spawned or modified sub-grid.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyGeometryRecord {
    pub id: BodyId,
    pub geometry_revision: u64,
    pub com_local: Vec3d,
    pub replace_geometry: bool,
    pub voxels: Vec<SyncedVoxelRecord>,
}

/// Compact binary pose synchronization packet (`BodyId`, `Transformd`, `lin_vel`, `ang_vel`,
/// `sleeping`, plus optional on-demand sub-grid geometry changes).
#[derive(Clone, Debug, PartialEq)]
pub struct PoseSnapshotPacket {
    pub server_tick: u64,
    pub is_delta: bool,
    pub removed: Vec<BodyId>,
    pub bodies: Vec<BodyPoseRecord>,
    pub geometries: Vec<BodyGeometryRecord>,
}

impl PoseSnapshotPacket {
    /// Encodes the packet into a compact little-endian binary payload.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        if self.bodies.len() > MAX_PACKET_BODIES
            || self.removed.len() > MAX_PACKET_REMOVALS
            || self.geometries.len() > MAX_PACKET_BODIES
        {
            return Err("pose snapshot packet exceeds body/removal capacity".to_owned());
        }
        let capacity = 22
            + self.removed.len() * 8
            + self.bodies.len() * 113
            + self
                .geometries
                .iter()
                .map(|g| 44 + g.voxels.len() * 28)
                .sum::<usize>();
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(&PACKET_MAGIC);
        out.extend_from_slice(&PACKET_VERSION.to_le_bytes());
        let flags: u16 = if self.is_delta { 1 } else { 0 };
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&self.server_tick.to_le_bytes());
        out.extend_from_slice(&(self.removed.len() as u16).to_le_bytes());
        out.extend_from_slice(&(self.bodies.len() as u16).to_le_bytes());

        for id in &self.removed {
            out.extend_from_slice(&id.index.to_le_bytes());
            out.extend_from_slice(&id.generation.to_le_bytes());
        }

        for b in &self.bodies {
            if !b.transform.is_finite() || !b.lin_vel.is_finite() || !b.ang_vel.is_finite() {
                return Err("non-finite kinematics in PoseSnapshotPacket".to_owned());
            }
            out.extend_from_slice(&b.id.index.to_le_bytes());
            out.extend_from_slice(&b.id.generation.to_le_bytes());
            push_vec3_le(&mut out, b.transform.position);
            push_f64_le(&mut out, b.transform.rotation.x);
            push_f64_le(&mut out, b.transform.rotation.y);
            push_f64_le(&mut out, b.transform.rotation.z);
            push_f64_le(&mut out, b.transform.rotation.w);
            push_vec3_le(&mut out, b.lin_vel);
            push_vec3_le(&mut out, b.ang_vel);
            out.push(u8::from(b.sleeping));
        }

        out.extend_from_slice(&(self.geometries.len() as u16).to_le_bytes());
        for g in &self.geometries {
            if !g.com_local.is_finite() {
                return Err("non-finite com_local in BodyGeometryRecord".to_owned());
            }
            if g.voxels.len() > MAX_PACKET_VOXELS_PER_BODY {
                return Err("body voxel geometry exceeds packet limit".to_owned());
            }
            out.extend_from_slice(&g.id.index.to_le_bytes());
            out.extend_from_slice(&g.id.generation.to_le_bytes());
            out.extend_from_slice(&g.geometry_revision.to_le_bytes());
            push_vec3_le(&mut out, g.com_local);
            out.push(u8::from(g.replace_geometry));
            out.extend_from_slice(&(g.voxels.len() as u32).to_le_bytes());
            for v in &g.voxels {
                out.extend_from_slice(&v.coord.x.to_le_bytes());
                out.extend_from_slice(&v.coord.y.to_le_bytes());
                out.extend_from_slice(&v.coord.z.to_le_bytes());
                out.extend_from_slice(&v.material.0.to_le_bytes());
                let name_bytes = v.block_id.as_bytes();
                if name_bytes.len() > MAX_BLOCK_ID_BYTES {
                    return Err("block_id exceeds wire limit".to_owned());
                }
                out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
                out.extend_from_slice(name_bytes);
            }
        }
        Ok(out)
    }

    /// Decodes and validates a binary pose snapshot packet (supports protocol v1 and v2).
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 20 {
            return Err("pose snapshot header truncated".to_owned());
        }
        if bytes[0..4] != PACKET_MAGIC {
            return Err("pose snapshot magic mismatch".to_owned());
        }
        let mut cursor = 4usize;
        let version = read_u16_le(bytes, &mut cursor)?;
        if version != 1 && version != PACKET_VERSION {
            return Err(format!("unsupported pose snapshot version {version}"));
        }
        let flags = read_u16_le(bytes, &mut cursor)?;
        let is_delta = (flags & 1) != 0;
        let server_tick = read_u64_le(bytes, &mut cursor)?;
        let removed_count = usize::from(read_u16_le(bytes, &mut cursor)?);
        let body_count = usize::from(read_u16_le(bytes, &mut cursor)?);
        if removed_count > MAX_PACKET_REMOVALS || body_count > MAX_PACKET_BODIES {
            return Err("pose snapshot counts exceed protocol limits".to_owned());
        }

        let base_len = 20 + removed_count * 8 + body_count * 113;
        if version == 1 && bytes.len() != base_len {
            return Err(format!(
                "pose snapshot length mismatch: expected {base_len}, got {}",
                bytes.len()
            ));
        }
        if version >= 2 && bytes.len() < base_len + 2 {
            return Err("pose snapshot v2 geometry footer truncated".to_owned());
        }

        let mut removed = Vec::with_capacity(removed_count);
        for _ in 0..removed_count {
            let index = read_u32_le(bytes, &mut cursor)?;
            let generation = read_u32_le(bytes, &mut cursor)?;
            removed.push(BodyId::new(index, generation));
        }

        let mut bodies = Vec::with_capacity(body_count);
        for _ in 0..body_count {
            let index = read_u32_le(bytes, &mut cursor)?;
            let generation = read_u32_le(bytes, &mut cursor)?;
            let position = read_vec3_le(bytes, &mut cursor)?;
            let qx = read_f64_le(bytes, &mut cursor)?;
            let qy = read_f64_le(bytes, &mut cursor)?;
            let qz = read_f64_le(bytes, &mut cursor)?;
            let qw = read_f64_le(bytes, &mut cursor)?;
            let lin_vel = read_vec3_le(bytes, &mut cursor)?;
            let ang_vel = read_vec3_le(bytes, &mut cursor)?;
            let sleeping = bytes[cursor] != 0;
            cursor += 1;
            let rotation = Quatd::new(qx, qy, qz, qw);
            if !position.is_finite()
                || !rotation.is_finite()
                || !lin_vel.is_finite()
                || !ang_vel.is_finite()
            {
                return Err("decoded non-finite pose value".to_owned());
            }
            bodies.push(BodyPoseRecord {
                id: BodyId::new(index, generation),
                transform: Transformd::new(position, rotation),
                lin_vel,
                ang_vel,
                sleeping,
            });
        }

        let mut geometries = Vec::new();
        if version >= 2 {
            let geom_count = usize::from(read_u16_le(bytes, &mut cursor)?);
            if geom_count > MAX_PACKET_BODIES {
                return Err("pose snapshot geometry count exceeds protocol limits".to_owned());
            }
            geometries.reserve(geom_count);
            for _ in 0..geom_count {
                let index = read_u32_le(bytes, &mut cursor)?;
                let generation = read_u32_le(bytes, &mut cursor)?;
                let geometry_revision = read_u64_le(bytes, &mut cursor)?;
                let com_local = read_vec3_le(bytes, &mut cursor)?;
                if !com_local.is_finite() {
                    return Err("decoded non-finite com_local".to_owned());
                }
                let replace_byte = *bytes
                    .get(cursor)
                    .ok_or_else(|| "unexpected EOF reading replace_geometry flag".to_owned())?;
                let replace_geometry = replace_byte != 0;
                cursor += 1;
                let voxel_count = usize::try_from(read_u32_le(bytes, &mut cursor)?)
                    .map_err(|_| "voxel count overflow".to_owned())?;
                if voxel_count > MAX_PACKET_VOXELS_PER_BODY {
                    return Err("decoded voxel count exceeds protocol limit".to_owned());
                }
                let mut voxels = Vec::with_capacity(voxel_count.min(4096));
                for _ in 0..voxel_count {
                    let vx = i32::from_le_bytes(
                        bytes
                            .get(cursor..cursor + 4)
                            .ok_or_else(|| "unexpected EOF reading voxel x".to_owned())?
                            .try_into()
                            .map_err(|_| "slice conversion failed".to_owned())?,
                    );
                    cursor += 4;
                    let vy = i32::from_le_bytes(
                        bytes
                            .get(cursor..cursor + 4)
                            .ok_or_else(|| "unexpected EOF reading voxel y".to_owned())?
                            .try_into()
                            .map_err(|_| "slice conversion failed".to_owned())?,
                    );
                    cursor += 4;
                    let vz = i32::from_le_bytes(
                        bytes
                            .get(cursor..cursor + 4)
                            .ok_or_else(|| "unexpected EOF reading voxel z".to_owned())?
                            .try_into()
                            .map_err(|_| "slice conversion failed".to_owned())?,
                    );
                    cursor += 4;
                    let mat_raw = read_u16_le(bytes, &mut cursor)?;
                    let name_len = usize::from(read_u16_le(bytes, &mut cursor)?);
                    if name_len > MAX_BLOCK_ID_BYTES {
                        return Err("decoded block_id length exceeds protocol limit".to_owned());
                    }
                    let name_slice = bytes
                        .get(cursor..cursor + name_len)
                        .ok_or_else(|| "unexpected EOF reading block_id".to_owned())?;
                    cursor += name_len;
                    let block_id = std::str::from_utf8(name_slice)
                        .map_err(|_| "invalid UTF-8 block_id".to_owned())?
                        .to_owned();
                    voxels.push(SyncedVoxelRecord {
                        coord: VoxelCoord::new(vx, vy, vz),
                        material: MaterialId(mat_raw),
                        block_id,
                    });
                }
                geometries.push(BodyGeometryRecord {
                    id: BodyId::new(index, generation),
                    geometry_revision,
                    com_local,
                    replace_geometry,
                    voxels,
                });
            }
            if cursor != bytes.len() {
                return Err(format!(
                    "trailing bytes in pose snapshot v2 packet: {} leftover",
                    bytes.len() - cursor
                ));
            }
        }

        Ok(Self {
            server_tick,
            is_delta,
            removed,
            bodies,
            geometries,
        })
    }
}

fn push_f64_le(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_bits().to_le_bytes());
}

fn push_vec3_le(buf: &mut Vec<u8>, v: Vec3d) {
    push_f64_le(buf, v.x);
    push_f64_le(buf, v.y);
    push_f64_le(buf, v.z);
}

fn read_u16_le(bytes: &[u8], cursor: &mut usize) -> Result<u16, String> {
    let slice: [u8; 2] = bytes
        .get(*cursor..*cursor + 2)
        .ok_or_else(|| "unexpected EOF reading u16".to_owned())?
        .try_into()
        .map_err(|_| "slice conversion failed".to_owned())?;
    *cursor += 2;
    Ok(u16::from_le_bytes(slice))
}

fn read_u32_le(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let slice: [u8; 4] = bytes
        .get(*cursor..*cursor + 4)
        .ok_or_else(|| "unexpected EOF reading u32".to_owned())?
        .try_into()
        .map_err(|_| "slice conversion failed".to_owned())?;
    *cursor += 4;
    Ok(u32::from_le_bytes(slice))
}

fn read_u64_le(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
    let slice: [u8; 8] = bytes
        .get(*cursor..*cursor + 8)
        .ok_or_else(|| "unexpected EOF reading u64".to_owned())?
        .try_into()
        .map_err(|_| "slice conversion failed".to_owned())?;
    *cursor += 8;
    Ok(u64::from_le_bytes(slice))
}

fn read_f64_le(bytes: &[u8], cursor: &mut usize) -> Result<f64, String> {
    Ok(f64::from_bits(read_u64_le(bytes, cursor)?))
}

fn read_vec3_le(bytes: &[u8], cursor: &mut usize) -> Result<Vec3d, String> {
    Ok(Vec3d::new(
        read_f64_le(bytes, cursor)?,
        read_f64_le(bytes, cursor)?,
        read_f64_le(bytes, cursor)?,
    ))
}

fn capture_body_geometry(
    api: &PhysicsApi,
    id: BodyId,
    com_local: Vec3d,
    geometry_revision: u64,
) -> BodyGeometryRecord {
    let handle = SubGridHandle(id);
    let block_names = api.subgrid_block_names(handle);
    let mut voxels = Vec::new();
    if let Some(grid) = api.world().body_grid(id) {
        voxels.reserve(grid.voxel_count());
        for (vc, entry) in grid.iter_voxels() {
            let block_id = block_names
                .and_then(|m| m.get(&[vc.x, vc.y, vc.z]))
                .cloned()
                .unwrap_or_else(|| "minecraft:stone".to_owned());
            voxels.push(SyncedVoxelRecord {
                coord: vc,
                material: entry.material,
                block_id,
            });
        }
    }
    BodyGeometryRecord {
        id,
        geometry_revision,
        com_local,
        replace_geometry: true,
        voxels,
    }
}

fn estimated_voxel_wire_bytes(v: &SyncedVoxelRecord) -> usize {
    16 + v.block_id.len().min(MAX_BLOCK_ID_BYTES)
}

fn chunk_pose_packets(
    server_tick: u64,
    is_delta: bool,
    removed: Vec<BodyId>,
    bodies: Vec<BodyPoseRecord>,
    geometries: Vec<BodyGeometryRecord>,
) -> Vec<PoseSnapshotPacket> {
    let mut packets: Vec<PoseSnapshotPacket> = Vec::new();
    let mut current = PoseSnapshotPacket {
        server_tick,
        is_delta,
        removed: Vec::new(),
        bodies: Vec::new(),
        geometries: Vec::new(),
    };
    let mut approx_bytes = 22usize;

    let flush_current = |packets_out: &mut Vec<PoseSnapshotPacket>,
                         cur: &mut PoseSnapshotPacket,
                         bytes_used: &mut usize| {
        if cur.removed.is_empty() && cur.bodies.is_empty() && cur.geometries.is_empty() {
            return;
        }
        let finished = std::mem::replace(
            cur,
            PoseSnapshotPacket {
                server_tick,
                is_delta: true, // continuation chunks always merge incrementally
                removed: Vec::new(),
                bodies: Vec::new(),
                geometries: Vec::new(),
            },
        );
        packets_out.push(finished);
        *bytes_used = 22;
    };

    for id in removed {
        if current.removed.len() >= MAX_PACKET_REMOVALS || approx_bytes + 8 > MAX_WIRE_PACKET_BYTES
        {
            flush_current(&mut packets, &mut current, &mut approx_bytes);
        }
        current.removed.push(id);
        approx_bytes += 8;
    }

    for body in bodies {
        if current.bodies.len() >= MAX_PACKET_BODIES || approx_bytes + 113 > MAX_WIRE_PACKET_BYTES {
            flush_current(&mut packets, &mut current, &mut approx_bytes);
        }
        current.bodies.push(body);
        approx_bytes += 113;
    }

    for geom in geometries {
        if geom.voxels.is_empty() {
            if current.geometries.len() >= MAX_PACKET_BODIES
                || approx_bytes + 45 > MAX_WIRE_PACKET_BYTES
            {
                flush_current(&mut packets, &mut current, &mut approx_bytes);
            }
            current.geometries.push(geom);
            approx_bytes += 45;
            continue;
        }

        let mut cursor = 0usize;
        let total = geom.voxels.len();
        let mut first_slice = geom.replace_geometry;
        while cursor < total {
            if current.geometries.len() >= MAX_PACKET_BODIES
                || approx_bytes + 45 + estimated_voxel_wire_bytes(&geom.voxels[cursor])
                    > MAX_WIRE_PACKET_BYTES
            {
                flush_current(&mut packets, &mut current, &mut approx_bytes);
            }
            let mut slice_end = cursor;
            let mut slice_bytes = 45usize;
            while slice_end < total {
                let vb = estimated_voxel_wire_bytes(&geom.voxels[slice_end]);
                if approx_bytes + slice_bytes + vb > MAX_WIRE_PACKET_BYTES && slice_end > cursor {
                    break;
                }
                if (slice_end - cursor) >= MAX_PACKET_VOXELS_PER_BODY {
                    break;
                }
                slice_bytes += vb;
                slice_end += 1;
            }
            current.geometries.push(BodyGeometryRecord {
                id: geom.id,
                geometry_revision: geom.geometry_revision,
                com_local: geom.com_local,
                replace_geometry: first_slice,
                voxels: geom.voxels[cursor..slice_end].to_vec(),
            });
            approx_bytes += slice_bytes;
            first_slice = false;
            cursor = slice_end;
        }
    }

    flush_current(&mut packets, &mut current, &mut approx_bytes);
    if packets.is_empty() && !is_delta {
        packets.push(PoseSnapshotPacket {
            server_tick,
            is_delta: false,
            removed: Vec::new(),
            bodies: Vec::new(),
            geometries: Vec::new(),
        });
    }
    packets
}

/// Staged delta sync payload awaiting delivery acknowledgement before committing tracker state.
#[derive(Clone, Debug)]
pub struct PendingDeltaSync {
    pub packets: Vec<PoseSnapshotPacket>,
    next_sent: BTreeMap<BodyId, BodyPoseRecord>,
    next_geometry_revs: BTreeMap<BodyId, u64>,
}

/// Stateful delta encoder emitting compact pose updates only for moved, newly
/// spawned, state-changed, geometry-mutated, or despawned bodies, plus per-player
/// resync state so any transient per-player delivery failure never drops removals.
#[derive(Default)]
pub struct DeltaSyncTracker {
    last_sent: BTreeMap<BodyId, BodyPoseRecord>,
    sent_geometry_revs: BTreeMap<BodyId, u64>,
    announced_full_bodies: Mutex<BTreeSet<BodyId>>,
    pending_resync_players: BTreeMap<u128, BTreeSet<BodyId>>,
}

impl Clone for DeltaSyncTracker {
    fn clone(&self) -> Self {
        let announced = self
            .announced_full_bodies
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        Self {
            last_sent: self.last_sent.clone(),
            sent_geometry_revs: self.sent_geometry_revs.clone(),
            announced_full_bodies: Mutex::new(announced),
            pending_resync_players: self.pending_resync_players.clone(),
        }
    }
}

impl std::fmt::Debug for DeltaSyncTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaSyncTracker")
            .field("last_sent_len", &self.last_sent.len())
            .field("sent_geometry_revs_len", &self.sent_geometry_revs.len())
            .field("pending_resync_players", &self.pending_resync_players.len())
            .finish()
    }
}

impl DeltaSyncTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.last_sent.clear();
        self.sent_geometry_revs.clear();
        self.pending_resync_players.clear();
        if let Ok(mut announced) = self.announced_full_bodies.lock() {
            announced.clear();
        }
    }

    pub fn mark_player_needs_resync(&mut self, player_uuid: u128, extra_removals: &[BodyId]) {
        let entry = self.pending_resync_players.entry(player_uuid).or_default();
        entry.extend(extra_removals.iter().copied());
    }

    pub fn ack_player_resync(&mut self, player_uuid: u128) {
        self.pending_resync_players.remove(&player_uuid);
    }

    pub fn retain_online_resync_players(&mut self, online_uuids: &[u128]) {
        self.pending_resync_players
            .retain(|uuid, _| online_uuids.contains(uuid));
    }

    #[inline]
    pub fn needs_resync(&self, player_uuid: u128) -> bool {
        self.pending_resync_players.contains_key(&player_uuid)
    }

    #[inline]
    pub fn pending_resync_player_count(&self) -> usize {
        self.pending_resync_players.len()
    }

    /// Records every body ID referenced by an outbound packet slice so even if a multi-chunk
    /// broadcast partially fails for ALL online players (`delivered == 0`) and that body is
    /// later despawned before the next commit, subsequent deltas and resyncs still clear it.
    pub fn record_announced_packets(&self, packets: &[PoseSnapshotPacket]) {
        if let Ok(mut announced) = self.announced_full_bodies.lock() {
            for pkt in packets {
                for b in &pkt.bodies {
                    announced.insert(b.id);
                }
                for g in &pkt.geometries {
                    announced.insert(g.id);
                }
            }
        }
    }

    /// Builds a chunked full baseline snapshot for `player_uuid`, including any accumulated
    /// removals that occurred while that player's channel failed prior broadcasts.
    pub fn build_player_resync_packets(
        &self,
        player_uuid: u128,
        api: &PhysicsApi,
    ) -> Vec<PoseSnapshotPacket> {
        let full = self.build_full_packet(api);
        let live_ids: BTreeSet<BodyId> = full.bodies.iter().map(|b| b.id).collect();
        let removals: Vec<BodyId> = self
            .pending_resync_players
            .get(&player_uuid)
            .map(|set| {
                set.iter()
                    .copied()
                    .filter(|id| !live_ids.contains(id))
                    .collect()
            })
            .unwrap_or_default();
        chunk_pose_packets(
            full.server_tick,
            false,
            removals,
            full.bodies,
            full.geometries,
        )
    }

    /// Produces a full baseline snapshot for a newly connected client (including
    /// voxel geometry so standalone remote clients render immediately).
    pub fn build_full_packet(&self, api: &PhysicsApi) -> PoseSnapshotPacket {
        let snapshots = api.world().snapshots();
        let mut bodies = Vec::with_capacity(snapshots.len());
        let mut geometries = Vec::with_capacity(snapshots.len());
        let mut announced_ids = Vec::with_capacity(snapshots.len());
        for s in snapshots {
            announced_ids.push(s.id);
            bodies.push(BodyPoseRecord {
                id: s.id,
                transform: s.transform,
                lin_vel: s.linear_velocity,
                ang_vel: s.angular_velocity,
                sleeping: s.sleeping,
            });
            let rev = api.geometry_revision(s.id).max(1);
            geometries.push(capture_body_geometry(api, s.id, s.com_local, rev));
        }
        if let Ok(mut announced) = self.announced_full_bodies.lock() {
            announced.extend(announced_ids);
        }
        PoseSnapshotPacket {
            server_tick: api.server_tick_counter(),
            is_delta: false,
            removed: Vec::new(),
            bodies,
            geometries,
        }
    }

    /// Produces size-bounded (`<= MAX_WIRE_PACKET_BYTES`) full baseline packets for newly
    /// connected players so large sub-grid worlds stay under Horizon's 1 MiB payload cap.
    pub fn build_chunked_full_packets(&self, api: &PhysicsApi) -> Vec<PoseSnapshotPacket> {
        let full = self.build_full_packet(api);
        chunk_pose_packets(
            full.server_tick,
            false,
            full.removed,
            full.bodies,
            full.geometries,
        )
    }

    /// Computes chunked incremental delta packets WITHOUT mutating tracker state until
    /// [`Self::commit_delta`] is called after network queueing succeeds.
    pub fn prepare_delta_packets(&self, api: &PhysicsApi) -> Option<PendingDeltaSync> {
        let snapshots = api.world().snapshots();
        let mut current: BTreeMap<BodyId, BodyPoseRecord> = BTreeMap::new();
        let mut current_geoms: BTreeMap<BodyId, (u64, Vec3d)> = BTreeMap::new();

        for s in snapshots {
            current.insert(
                s.id,
                BodyPoseRecord {
                    id: s.id,
                    transform: s.transform,
                    lin_vel: s.linear_velocity,
                    ang_vel: s.angular_velocity,
                    sleeping: s.sleeping,
                },
            );
            current_geoms.insert(s.id, (api.geometry_revision(s.id).max(1), s.com_local));
        }

        let mut removed_set: BTreeSet<BodyId> = self
            .last_sent
            .keys()
            .copied()
            .filter(|id| !current.contains_key(id))
            .collect();
        if let Ok(announced) = self.announced_full_bodies.lock() {
            for &id in announced.iter() {
                if !current.contains_key(&id) {
                    removed_set.insert(id);
                }
            }
        }
        let removed: Vec<BodyId> = removed_set.into_iter().collect();

        let mut next_geometry_revs = self.sent_geometry_revs.clone();
        for id in &removed {
            next_geometry_revs.remove(id);
        }

        let mut changed = Vec::new();
        for (&id, &rec) in &current {
            let needs_send = match self.last_sent.get(&id) {
                None => true,
                Some(prev) => !prev.roughly_equal(&rec),
            };
            if needs_send {
                changed.push(rec);
            }
        }

        let mut geometries = Vec::new();
        for (&id, &(rev, com_local)) in &current_geoms {
            if self.sent_geometry_revs.get(&id).copied() != Some(rev) {
                geometries.push(capture_body_geometry(api, id, com_local, rev));
                next_geometry_revs.insert(id, rev);
            }
        }

        if removed.is_empty() && changed.is_empty() && geometries.is_empty() {
            return None;
        }

        let packets = chunk_pose_packets(
            api.server_tick_counter(),
            true,
            removed,
            changed,
            geometries,
        );
        Some(PendingDeltaSync {
            packets,
            next_sent: current,
            next_geometry_revs,
        })
    }

    /// Commits tracker state once at least one packet batch has been successfully sent.
    /// Accumulates any committed removals onto players still queued for resync.
    pub fn commit_delta(&mut self, pending: PendingDeltaSync) {
        let removed_in_batch: Vec<BodyId> = pending
            .packets
            .iter()
            .flat_map(|p| p.removed.iter().copied())
            .collect();
        if !removed_in_batch.is_empty() {
            for set in self.pending_resync_players.values_mut() {
                set.extend(removed_in_batch.iter().copied());
            }
        }
        self.last_sent = pending.next_sent;
        self.sent_geometry_revs = pending.next_geometry_revs;
        if let Ok(mut announced) = self.announced_full_bodies.lock() {
            announced.retain(|id| self.last_sent.contains_key(id));
        }
    }

    /// Convenience single-step delta builder for in-process testing (commits immediately).
    pub fn build_delta_packet(&mut self, api: &PhysicsApi) -> Option<PoseSnapshotPacket> {
        let mut pending = self.prepare_delta_packets(api)?;
        let packet = pending.packets.drain(..).next()?;
        self.commit_delta(pending);
        Some(packet)
    }
}

/// Client-side [`IPayloadHandler`] feeding decoded pose packets into client interpolation state.
pub struct PoseSyncClientHandler;

impl IPayloadHandler for PoseSyncClientHandler {
    fn handle(
        &self,
        payload: PayloadBytes,
        _context: &mut dyn IPayloadContext,
    ) -> Result<(), PayloadHandleError> {
        let packet = PoseSnapshotPacket::decode(&payload.0).map_err(PayloadHandleError::Failed)?;
        crate::client_render::apply_pose_packet(&packet);
        Ok(())
    }
}

/// Registers the `horizon_physics:pose_sync` clientbound play payload channel.
pub fn register_sync_payloads(registrar: &mut dyn Registrar) {
    let mut payloads = registrar.payload_handlers();
    payloads.registrar("1").optional().play_to_client(
        PayloadId(POSE_SYNC_CHANNEL.to_owned()),
        Some(Arc::new(PoseSyncClientHandler)),
    );
}

/// Per-player delivery summary from broadcasting pose/geometry packets.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BroadcastReport {
    pub delivered: usize,
    pub total_players: usize,
    pub failed_player_uuids: Vec<u128>,
}

/// Broadcasts bounded packets to a specified slice of target players, reporting exact
/// delivery counts and any per-player failures that require a baseline resync.
fn encode_bounded_packets(packets: &[PoseSnapshotPacket]) -> Option<Vec<Vec<u8>>> {
    let mut encoded = Vec::with_capacity(packets.len());
    for packet in packets {
        let bytes = packet.encode().ok()?;
        if bytes.len() > MAX_WIRE_PACKET_BYTES {
            return None;
        }
        encoded.push(bytes);
    }
    Some(encoded)
}

fn broadcast_encoded_with<R, E, F>(
    encoded: &[Vec<u8>],
    players: &[PlayerRef],
    mut send_fn: F,
) -> BroadcastReport
where
    F: FnMut(PlayerRef, &[u8]) -> Result<R, E>,
{
    let mut report = BroadcastReport {
        delivered: 0,
        total_players: players.len(),
        failed_player_uuids: Vec::new(),
    };
    for &player in players {
        let mut all_ok = true;
        for wire in encoded {
            if send_fn(player, wire).is_err() {
                all_ok = false;
                break;
            }
        }
        if all_ok {
            report.delivered += 1;
        } else {
            report.failed_player_uuids.push(player.uuid());
        }
    }
    report
}

/// Broadcasts bounded packets using a custom per-frame sender closure.
pub fn broadcast_pose_packets_with<R, E, F>(
    packets: &[PoseSnapshotPacket],
    players: &[PlayerRef],
    send_fn: F,
) -> BroadcastReport
where
    F: FnMut(PlayerRef, &[u8]) -> Result<R, E>,
{
    if packets.is_empty() || players.is_empty() {
        return BroadcastReport {
            delivered: 0,
            total_players: players.len(),
            failed_player_uuids: Vec::new(),
        };
    }
    let Some(encoded) = encode_bounded_packets(packets) else {
        return BroadcastReport {
            delivered: 0,
            total_players: players.len(),
            failed_player_uuids: players.iter().map(|p| p.uuid()).collect(),
        };
    };
    broadcast_encoded_with(&encoded, players, send_fn)
}

/// Broadcasts bounded packets to a specified slice of target players, reporting exact
/// delivery counts and any per-player failures that require a baseline resync.
pub fn broadcast_pose_packets_to(
    packets: &[PoseSnapshotPacket],
    players: &[PlayerRef],
    host: &mut dyn ServerHost,
) -> BroadcastReport {
    let channel = PayloadId(POSE_SYNC_CHANNEL.to_owned());
    broadcast_pose_packets_with(packets, players, |player, wire| {
        player.send_payload(host, &channel, PayloadBytes(wire.to_vec()))
    })
}

/// Broadcasts a sequence of bounded pose/geometry packets to all online players.
/// Returns the number of players for whom ALL packets queued successfully.
pub fn broadcast_pose_packets(packets: &[PoseSnapshotPacket], host: &mut dyn ServerHost) -> usize {
    let players = host.players().server_players();
    broadcast_pose_packets_to(packets, &players, host).delivered
}

/// Broadcasts a single pose snapshot packet (chunking safely if needed).
pub fn broadcast_pose_packet(packet: &PoseSnapshotPacket, host: &mut dyn ServerHost) -> usize {
    let chunks = chunk_pose_packets(
        packet.server_tick,
        packet.is_delta,
        packet.removed.clone(),
        packet.bodies.clone(),
        packet.geometries.clone(),
    );
    broadcast_pose_packets(&chunks, host)
}

/// Sends a full chunked baseline snapshot using a custom byte-delivery callback.
pub fn send_full_baseline_with<R, E, F>(
    player: PlayerRef,
    packets: &[PoseSnapshotPacket],
    mut send_fn: F,
) -> bool
where
    F: FnMut(PlayerRef, &[u8]) -> Result<R, E>,
{
    let Some(encoded) = encode_bounded_packets(packets) else {
        return false;
    };
    for wire in &encoded {
        if send_fn(player, wire).is_err() {
            return false;
        }
    }
    true
}

/// Sends a full chunked baseline snapshot to a newly joined player.
pub fn send_full_baseline_to_player(
    player: PlayerRef,
    packets: &[PoseSnapshotPacket],
    host: &mut dyn ServerHost,
) -> bool {
    let channel = PayloadId(POSE_SYNC_CHANNEL.to_owned());
    send_full_baseline_with(player, packets, |p, wire| {
        p.send_payload(host, &channel, PayloadBytes(wire.to_vec()))
    })
}

/// Synchronizes a newly logged-in player with a chunked full baseline; marks them for
/// automatic retry in [`server_tick_sync`] if any baseline chunk fails to enqueue.
pub fn player_logged_in_sync(
    player: PlayerRef,
    api: &PhysicsApi,
    tracker: &mut DeltaSyncTracker,
    host: &mut dyn ServerHost,
) -> bool {
    let full_packets = tracker.build_chunked_full_packets(api);
    if full_packets.is_empty() {
        return true;
    }
    if send_full_baseline_to_player(player, &full_packets, host) {
        tracker.ack_player_resync(player.uuid());
        true
    } else {
        tracker.mark_player_needs_resync(player.uuid(), &[]);
        false
    }
}

/// Executes per-tick server sync state machine against a caller-supplied delivery closure:
/// 1. Prunes disconnected players and flushes pending full resyncs (with missed removals).
/// 2. Broadcasts delta packets to healthy players; registers partially sent outbound body IDs
///    when any player fails (even if `delivered == 0`) and commits once `delivered > 0`.
pub fn server_tick_sync_with<R, E, F>(
    api: &PhysicsApi,
    tracker: &mut DeltaSyncTracker,
    players: &[PlayerRef],
    mut send_fn: F,
) where
    F: FnMut(PlayerRef, &[u8]) -> Result<R, E>,
{
    if players.is_empty() {
        tracker.retain_online_resync_players(&[]);
        return;
    }
    let online_uuids: Vec<u128> = players.iter().map(|p| p.uuid()).collect();
    tracker.retain_online_resync_players(&online_uuids);

    for &player in players {
        if tracker.needs_resync(player.uuid()) {
            let resync_packets = tracker.build_player_resync_packets(player.uuid(), api);
            if send_full_baseline_with(player, &resync_packets, &mut send_fn) {
                tracker.ack_player_resync(player.uuid());
            }
        }
    }

    let healthy_players: Vec<PlayerRef> = players
        .iter()
        .copied()
        .filter(|p| !tracker.needs_resync(p.uuid()))
        .collect();
    if healthy_players.is_empty() {
        return;
    }

    if let Some(pending) = tracker.prepare_delta_packets(api) {
        let removed_in_batch: Vec<BodyId> = pending
            .packets
            .iter()
            .flat_map(|p| p.removed.iter().copied())
            .collect();
        let report = broadcast_pose_packets_with(&pending.packets, &healthy_players, &mut send_fn);
        if !report.failed_player_uuids.is_empty() {
            // Even when `delivered == 0`, partial chunk writes on packet[0..k] may have spawned
            // new bodies on failing clients; register those IDs and queue full resyncs for all.
            tracker.record_announced_packets(&pending.packets);
            for failed_uuid in &report.failed_player_uuids {
                tracker.mark_player_needs_resync(*failed_uuid, &removed_in_batch);
            }
        }
        if report.delivered > 0 {
            tracker.commit_delta(pending);
        }
    }
}

/// Executes per-tick server sync over [`ServerHost`] payload channels.
pub fn server_tick_sync(
    api: &PhysicsApi,
    tracker: &mut DeltaSyncTracker,
    host: &mut dyn ServerHost,
) {
    let players = host.players().server_players();
    let channel = PayloadId(POSE_SYNC_CHANNEL.to_owned());
    // Synchronize existing/partial tracker state safely:
    server_tick_sync_with(api, tracker, &players, |player, wire| {
        player.send_payload(host, &channel, PayloadBytes(wire.to_vec()))
    });
}
