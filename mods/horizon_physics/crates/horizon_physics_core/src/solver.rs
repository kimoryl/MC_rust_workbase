//! Hybrid TGS (Temporal Gauss-Seidel) soft-contact + XPBD joint substep solver
//! with Structure-of-Arrays (`SolverSoA`) hot data layout.

use std::f64::consts::PI;

use crate::arena::{BodyId, JointId};
use crate::math::{Mat3d, Quatd, Vec3d};
use crate::narrowphase::ContactManifold;

/// Hot-path Structure-of-Arrays layout for active rigid-body kinematics and inertia.
#[derive(Clone, Debug, Default)]
pub struct SolverSoA {
    pub body_ids: Vec<BodyId>,
    pub positions: Vec<Vec3d>,
    pub rotations: Vec<Quatd>,
    pub linear_velocities: Vec<Vec3d>,
    pub angular_velocities: Vec<Vec3d>,
    pub inv_masses: Vec<f64>,
    pub inv_inertias_local: Vec<Mat3d>,
    pub inv_inertias_world: Vec<Mat3d>,
    pub linear_damping: Vec<f64>,
    pub angular_damping: Vec<f64>,
    pub is_dynamic: Vec<bool>,
    pub sleeping: Vec<bool>,
    pub awake_dense_indices: Vec<usize>,
}

impl SolverSoA {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.body_ids.clear();
        self.positions.clear();
        self.rotations.clear();
        self.linear_velocities.clear();
        self.angular_velocities.clear();
        self.inv_masses.clear();
        self.inv_inertias_local.clear();
        self.inv_inertias_world.clear();
        self.linear_damping.clear();
        self.angular_damping.clear();
        self.is_dynamic.clear();
        self.sleeping.clear();
        self.awake_dense_indices.clear();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.body_ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.body_ids.is_empty()
    }

    #[inline]
    pub fn update_world_inertia(&mut self, idx: usize) {
        if self.inv_masses[idx] > 0.0 {
            let r = self.rotations[idx].to_mat3();
            self.inv_inertias_world[idx] =
                Mat3d::similarity_transform(r, self.inv_inertias_local[idx]).symmetrize();
        } else {
            self.inv_inertias_world[idx] = Mat3d::ZERO;
        }
    }
}

/// Soft-contact harmonic-oscillator coefficients derived from `contact_hertz`
/// and `contact_damping_ratio` for substep duration `h`.
#[derive(Copy, Clone, Debug)]
pub struct SoftContactParams {
    pub bias_rate: f64,
    pub mass_scale: f64,
    pub impulse_scale: f64,
}

impl SoftContactParams {
    #[inline]
    pub fn from_hertz(hertz: f64, damping_ratio: f64, h: f64) -> Self {
        let hz = hertz.max(1.0);
        let zeta = damping_ratio.max(0.0);
        let omega = 2.0 * PI * hz;
        let a1 = 2.0 * zeta + h * omega;
        let a2 = h * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Self {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }
}

/// Supported mechanical joint constraint kinds.
#[derive(Clone, Debug, PartialEq)]
pub enum JointKind {
    /// Bilateral or bounded distance constraint between two anchor points.
    Distance {
        rest_length: f64,
        min_length: Option<f64>,
        max_length: Option<f64>,
    },
    /// Rigid 6-DOF weld locking position and relative orientation.
    FixedWeld { relative_rotation: Quatd },
    /// Revolute hinge locking 3 positional and 2 angular DOFs, with optional limits & motor.
    Hinge {
        axis_a: Vec3d,
        axis_b: Vec3d,
        limits: Option<(f64, f64)>,
        motor_target_velocity: Option<f64>,
        motor_max_torque: f64,
    },
    /// Prismatic slider locking 3 angular and 2 lateral DOFs, with optional limits & motor.
    Slider {
        axis_a: Vec3d,
        axis_b: Vec3d,
        limits: Option<(f64, f64)>,
        motor_target_velocity: Option<f64>,
        motor_max_force: f64,
    },
}

/// Public specification for creating a joint via `PhysicsWorld::add_joint`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointSpec {
    pub body_a: BodyId,
    /// `None` anchors the joint to the static world at `local_anchor_b`.
    pub body_b: Option<BodyId>,
    /// Anchor offset in body A's local frame (relative to A's CoM).
    pub local_anchor_a: Vec3d,
    /// Anchor offset in body B's local frame (or world point if `body_b` is `None`).
    pub local_anchor_b: Vec3d,
    pub kind: JointKind,
    /// XPBD compliance `alpha` (`0.0` = rigid).
    pub compliance: f64,
    /// Joint velocity damping factor in `[0.0, 1.0]`.
    pub damping: f64,
    /// Maximum allowable constraint force (N) before breaking (`f64::INFINITY` = unbreakable).
    pub max_force: f64,
    /// Maximum allowable constraint torque (N*m) before breaking (`f64::INFINITY` = unbreakable).
    pub max_torque: f64,
    /// Whether connected bodies still collide with each other.
    pub collide_connected: bool,
}

impl JointSpec {
    pub fn distance(body_a: BodyId, body_b: Option<BodyId>, rest_length: f64) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3d::ZERO,
            local_anchor_b: Vec3d::ZERO,
            kind: JointKind::Distance {
                rest_length,
                min_length: None,
                max_length: None,
            },
            compliance: 0.0,
            damping: 0.05,
            max_force: f64::INFINITY,
            max_torque: f64::INFINITY,
            collide_connected: false,
        }
    }

    pub fn fixed_weld(body_a: BodyId, body_b: Option<BodyId>, relative_rotation: Quatd) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3d::ZERO,
            local_anchor_b: Vec3d::ZERO,
            kind: JointKind::FixedWeld { relative_rotation },
            compliance: 0.0,
            damping: 0.05,
            max_force: f64::INFINITY,
            max_torque: f64::INFINITY,
            collide_connected: false,
        }
    }

    pub fn hinge(body_a: BodyId, body_b: Option<BodyId>, axis_a: Vec3d, axis_b: Vec3d) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3d::ZERO,
            local_anchor_b: Vec3d::ZERO,
            kind: JointKind::Hinge {
                axis_a: axis_a.normalize_or(Vec3d::Y),
                axis_b: axis_b.normalize_or(Vec3d::Y),
                limits: None,
                motor_target_velocity: None,
                motor_max_torque: 0.0,
            },
            compliance: 0.0,
            damping: 0.02,
            max_force: f64::INFINITY,
            max_torque: f64::INFINITY,
            collide_connected: false,
        }
    }

    pub fn slider(body_a: BodyId, body_b: Option<BodyId>, axis_a: Vec3d, axis_b: Vec3d) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3d::ZERO,
            local_anchor_b: Vec3d::ZERO,
            kind: JointKind::Slider {
                axis_a: axis_a.normalize_or(Vec3d::X),
                axis_b: axis_b.normalize_or(Vec3d::X),
                limits: None,
                motor_target_velocity: None,
                motor_max_force: 0.0,
            },
            compliance: 0.0,
            damping: 0.02,
            max_force: f64::INFINITY,
            max_torque: f64::INFINITY,
            collide_connected: false,
        }
    }
}

/// Internal runtime record for an active joint during a substep.
#[derive(Clone, Debug)]
pub(crate) struct ActiveJoint {
    pub id: JointId,
    pub spec: JointSpec,
    pub idx_a: usize,
    pub idx_b: Option<usize>,
    /// Reference perpendicular vectors (for hinge angle measurement).
    pub ref_a: Vec3d,
    pub ref_b: Vec3d,
    /// Target relative rotation for sliders (`q_a^-1 * q_b` at spawn).
    pub slider_rel_rot: Quatd,
    /// Accumulated XPBD positional impulse vector (`N * s^2`) in this substep.
    pub xpbd_pos_lambda: Vec3d,
    /// Accumulated XPBD angular impulse vector (`N * m * s^2`) in this substep.
    pub xpbd_ang_lambda: Vec3d,
    /// Accumulated velocity-level linear impulse vector (`N * s`) in this substep.
    pub vel_pos_impulse: Vec3d,
    /// Accumulated velocity-level angular impulse vector (`N * m * s`) in this substep.
    pub vel_ang_impulse: Vec3d,
    /// Accumulated scalar XPBD multipliers per substep for compliance scaling.
    pub scalar_lambda_pos: [f64; 3],
    pub scalar_lambda_ang: [f64; 3],
    pub motor_impulse: f64,
}

/// Reported when a joint exceeds its breaking force or torque threshold.
#[derive(Copy, Clone, Debug)]
pub(crate) struct BrokenJointInfo {
    pub id: JointId,
    pub body_a: BodyId,
    pub body_b: Option<BodyId>,
    pub force: f64,
    pub torque: f64,
}

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct PreparedContactPoint {
    r_a: Vec3d,
    r_b: Vec3d,
    penetration: f64,
    normal_mass: f64,
    tangent_mass: [f64; 2],
    restitution_vel: f64,
    is_bouncing: bool,
    normal_impulse: f64,
    tangent_impulse: [f64; 2],
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedManifold {
    pub manifold_index: usize,
    pub idx_a: usize,
    pub idx_b: Option<usize>,
    pub normal: Vec3d,
    pub t1: Vec3d,
    pub t2: Vec3d,
    pub friction: f64,
    pub points: [PreparedContactPoint; 4],
    pub point_count: usize,
    pub twist_mass: f64,
    pub twist_radius: f64,
    pub twist_impulse: f64,
}

/// Solver tuning parameters passed per substep.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SubstepSolverParams {
    pub dt: f64,
    pub velocity_iterations: u32,
    pub contact_slop: f64,
    pub max_depenetration_velocity: f64,
    pub warm_start_factor: f64,
    pub restitution_threshold: f64,
    pub soft: SoftContactParams,
}

#[inline]
fn effective_mass_linear(
    inv_m_a: f64,
    inv_i_a: &Mat3d,
    r_a: Vec3d,
    inv_m_b: f64,
    inv_i_b: &Mat3d,
    r_b: Vec3d,
    dir: Vec3d,
) -> f64 {
    let ra_x_n = r_a.cross(dir);
    let rb_x_n = r_b.cross(dir);
    let k = inv_m_a
        + inv_m_b
        + ra_x_n.dot(inv_i_a.mul_vec3(ra_x_n))
        + rb_x_n.dot(inv_i_b.mul_vec3(rb_x_n));
    if k > 1e-18 { 1.0 / k } else { 0.0 }
}

#[inline]
fn apply_vel_impulse(
    soa: &mut SolverSoA,
    idx_a: usize,
    idx_b: Option<usize>,
    r_a: Vec3d,
    r_b: Vec3d,
    impulse: Vec3d,
) {
    let ima = soa.inv_masses[idx_a];
    if ima > 0.0 {
        soa.linear_velocities[idx_a] += impulse * ima;
        soa.angular_velocities[idx_a] += soa.inv_inertias_world[idx_a].mul_vec3(r_a.cross(impulse));
    }
    if let Some(ib) = idx_b {
        let imb = soa.inv_masses[ib];
        if imb > 0.0 {
            soa.linear_velocities[ib] -= impulse * imb;
            soa.angular_velocities[ib] -= soa.inv_inertias_world[ib].mul_vec3(r_b.cross(impulse));
        }
    }
}

#[inline]
fn apply_ang_vel_impulse(
    soa: &mut SolverSoA,
    idx_a: usize,
    idx_b: Option<usize>,
    ang_impulse: Vec3d,
) {
    if soa.inv_masses[idx_a] > 0.0 {
        soa.angular_velocities[idx_a] += soa.inv_inertias_world[idx_a].mul_vec3(ang_impulse);
    }
    if let Some(ib) = idx_b
        && soa.inv_masses[ib] > 0.0
    {
        soa.angular_velocities[ib] -= soa.inv_inertias_world[ib].mul_vec3(ang_impulse);
    }
}

/// Builds solver-ready contact constraints and applies warm-starting impulses.
pub(crate) fn prepare_contacts(
    soa: &mut SolverSoA,
    manifolds: &[ContactManifold],
    body_to_dense: &dyn Fn(BodyId) -> Option<usize>,
    params: &SubstepSolverParams,
    out: &mut Vec<PreparedManifold>,
) {
    out.clear();
    out.reserve(manifolds.len());

    for (m_idx, m) in manifolds.iter().enumerate() {
        if m.point_count == 0 {
            continue;
        }
        let Some(idx_a) = body_to_dense(m.body_a) else {
            continue;
        };
        let idx_b = match m.body_b {
            Some(bid) => match body_to_dense(bid) {
                Some(ib) => Some(ib),
                None => continue,
            },
            None => None,
        };

        let ima = soa.inv_masses[idx_a];
        let imb = idx_b.map_or(0.0, |ib| soa.inv_masses[ib]);
        if ima == 0.0 && imb == 0.0 {
            continue;
        }

        let inv_i_a = soa.inv_inertias_world[idx_a];
        let inv_i_b = idx_b.map_or(Mat3d::ZERO, |ib| soa.inv_inertias_world[ib]);
        let pos_a = soa.positions[idx_a];
        let pos_b = idx_b.map_or(Vec3d::ZERO, |ib| soa.positions[ib]);
        let va = soa.linear_velocities[idx_a];
        let wa = soa.angular_velocities[idx_a];
        let vb = idx_b.map_or(Vec3d::ZERO, |ib| soa.linear_velocities[ib]);
        let wb = idx_b.map_or(Vec3d::ZERO, |ib| soa.angular_velocities[ib]);

        let (t1, t2) = m.normal.orthonormal_basis();
        let mut pts = [PreparedContactPoint::default(); 4];
        let mut centroid = Vec3d::ZERO;

        for (i, cp) in m.points.iter().enumerate().take(m.point_count) {
            let r_a = cp.world_point_a - pos_a;
            let r_b = if idx_b.is_some() {
                cp.world_point_b - pos_b
            } else {
                Vec3d::ZERO
            };
            centroid += cp.world_point_a;

            let v_rel = (va + wa.cross(r_a)) - (vb + wb.cross(r_b));
            let vn0 = v_rel.dot(m.normal);
            let bounce_thresh = if m.restitution > 0.5 {
                params.restitution_threshold * 0.2
            } else {
                params.restitution_threshold
            };
            let is_bouncing = m.restitution > 1e-4 && vn0 < -bounce_thresh;
            let restitution_vel = if is_bouncing {
                -m.restitution * vn0
            } else {
                0.0
            };

            let ws = if is_bouncing {
                0.0
            } else {
                params.warm_start_factor
            };

            let normal_mass =
                effective_mass_linear(ima, &inv_i_a, r_a, imb, &inv_i_b, r_b, m.normal);
            let tm1 = effective_mass_linear(ima, &inv_i_a, r_a, imb, &inv_i_b, r_b, t1);
            let tm2 = effective_mass_linear(ima, &inv_i_a, r_a, imb, &inv_i_b, r_b, t2);

            pts[i] = PreparedContactPoint {
                r_a,
                r_b,
                penetration: cp.penetration,
                normal_mass,
                tangent_mass: [tm1, tm2],
                restitution_vel,
                is_bouncing,
                normal_impulse: (cp.normal_impulse * ws).max(0.0),
                tangent_impulse: [cp.tangent_impulse[0] * ws, cp.tangent_impulse[1] * ws],
            };
        }

        centroid /= m.point_count as f64;
        let mut avg_r = 0.0;
        for cp in m.points.iter().take(m.point_count) {
            avg_r += (cp.world_point_a - centroid).length();
        }
        let twist_radius = (avg_r / m.point_count as f64).max(0.2);
        let k_twist =
            m.normal.dot(inv_i_a.mul_vec3(m.normal)) + m.normal.dot(inv_i_b.mul_vec3(m.normal));
        let twist_mass = if k_twist > 1e-18 { 1.0 / k_twist } else { 0.0 };

        // Apply warm-starting impulses before iterations.
        for p in pts.iter().take(m.point_count) {
            let j =
                m.normal * p.normal_impulse + t1 * p.tangent_impulse[0] + t2 * p.tangent_impulse[1];
            if j.length_sq() > 0.0 {
                apply_vel_impulse(soa, idx_a, idx_b, p.r_a, p.r_b, j);
            }
        }

        out.push(PreparedManifold {
            manifold_index: m_idx,
            idx_a,
            idx_b,
            normal: m.normal,
            t1,
            t2,
            friction: m.friction,
            points: pts,
            point_count: m.point_count,
            twist_mass,
            twist_radius,
            twist_impulse: 0.0,
        });
    }
}

/// Runs TGS soft-contact + Coulomb cone friction + twist friction + joint velocity solving.
pub(crate) fn solve_velocities(
    soa: &mut SolverSoA,
    contacts: &mut [PreparedManifold],
    joints: &mut [ActiveJoint],
    params: &SubstepSolverParams,
    use_bias: bool,
) {
    let iters = if use_bias {
        params.velocity_iterations.max(1)
    } else {
        1
    };

    for _ in 0..iters {
        // 1. Joint velocity constraints (motors + damping + rigid velocity locking).
        for joint in joints.iter_mut() {
            solve_joint_velocity(soa, joint, params.dt);
        }

        // 2. Contact constraints (normal + coupled 2D Coulomb cone + twist friction).
        for cm in contacts.iter_mut() {
            let idx_a = cm.idx_a;
            let idx_b = cm.idx_b;
            let n = cm.normal;
            let t1 = cm.t1;
            let t2 = cm.t2;

            let mut total_jn = 0.0;

            for i in 0..cm.point_count {
                let p = &mut cm.points[i];
                if !use_bias && p.is_bouncing {
                    continue;
                }
                let va = soa.linear_velocities[idx_a];
                let wa = soa.angular_velocities[idx_a];
                let (vb, wb) = match idx_b {
                    Some(ib) => (soa.linear_velocities[ib], soa.angular_velocities[ib]),
                    None => (Vec3d::ZERO, Vec3d::ZERO),
                };

                let v_rel = (va + wa.cross(p.r_a)) - (vb + wb.cross(p.r_b));
                let vn = v_rel.dot(n);

                // Normal impulse: hard elastic bounce vs soft harmonic-oscillator resting contact.
                let delta_jn = if p.is_bouncing {
                    p.normal_mass * (p.restitution_vel - vn)
                } else if use_bias {
                    let pen_err = (p.penetration - params.contact_slop).max(0.0);
                    let bias =
                        (params.soft.bias_rate * pen_err).min(params.max_depenetration_velocity);
                    p.normal_mass * params.soft.mass_scale * (-vn + bias)
                        - params.soft.impulse_scale * p.normal_impulse
                } else {
                    p.normal_mass * (-vn)
                };

                let old_jn = p.normal_impulse;
                let new_jn = (old_jn + delta_jn).max(0.0);
                let applied_jn = new_jn - old_jn;
                p.normal_impulse = new_jn;
                total_jn += new_jn;

                if applied_jn.abs() > 0.0 {
                    apply_vel_impulse(soa, idx_a, idx_b, p.r_a, p.r_b, n * applied_jn);
                }

                // Re-read velocities after normal impulse for accurate Coulomb friction cone.
                if cm.friction > 0.0 && new_jn > 0.0 {
                    let va = soa.linear_velocities[idx_a];
                    let wa = soa.angular_velocities[idx_a];
                    let (vb, wb) = match idx_b {
                        Some(ib) => (soa.linear_velocities[ib], soa.angular_velocities[ib]),
                        None => (Vec3d::ZERO, Vec3d::ZERO),
                    };
                    let v_rel = (va + wa.cross(p.r_a)) - (vb + wb.cross(p.r_b));
                    let dj1 = -v_rel.dot(t1) * p.tangent_mass[0];
                    let dj2 = -v_rel.dot(t2) * p.tangent_mass[1];

                    let old_t1 = p.tangent_impulse[0];
                    let old_t2 = p.tangent_impulse[1];
                    let mut cand_t1 = old_t1 + dj1;
                    let mut cand_t2 = old_t2 + dj2;

                    let max_f = cm.friction * new_jn;
                    let mag_sq = cand_t1 * cand_t1 + cand_t2 * cand_t2;
                    if mag_sq > max_f * max_f {
                        let scale = max_f / mag_sq.sqrt();
                        cand_t1 *= scale;
                        cand_t2 *= scale;
                    }

                    let app_t1 = cand_t1 - old_t1;
                    let app_t2 = cand_t2 - old_t2;
                    p.tangent_impulse[0] = cand_t1;
                    p.tangent_impulse[1] = cand_t2;

                    let jt = t1 * app_t1 + t2 * app_t2;
                    if jt.length_sq() > 0.0 {
                        apply_vel_impulse(soa, idx_a, idx_b, p.r_a, p.r_b, jt);
                    }
                }
            }

            // Twist (torsional) friction around the contact normal.
            if cm.friction > 0.0 && cm.twist_mass > 0.0 && total_jn > 0.0 {
                let wa = soa.angular_velocities[idx_a];
                let wb = idx_b.map_or(Vec3d::ZERO, |ib| soa.angular_velocities[ib]);
                let w_rel_n = (wa - wb).dot(n);
                let delta_tw = -w_rel_n * cm.twist_mass;
                let max_tw = cm.friction * cm.twist_radius * total_jn;
                let old_tw = cm.twist_impulse;
                let new_tw = (old_tw + delta_tw).clamp(-max_tw, max_tw);
                let app_tw = new_tw - old_tw;
                cm.twist_impulse = new_tw;
                if app_tw.abs() > 0.0 {
                    apply_ang_vel_impulse(soa, idx_a, idx_b, n * app_tw);
                }
            }
        }
    }
}

/// Writes solved impulses back into `ContactManifold` for warm-starting and collision events.
pub(crate) fn write_back_contacts(
    prepared: &[PreparedManifold],
    manifolds: &mut [ContactManifold],
) {
    for pm in prepared {
        let m = &mut manifolds[pm.manifold_index];
        for i in 0..pm.point_count {
            m.points[i].normal_impulse = pm.points[i].normal_impulse;
            m.points[i].tangent_impulse = pm.points[i].tangent_impulse;
        }
    }
}

fn solve_joint_velocity(soa: &mut SolverSoA, joint: &mut ActiveJoint, h: f64) {
    let idx_a = joint.idx_a;
    let idx_b = joint.idx_b;
    let ima = soa.inv_masses[idx_a];
    let imb = idx_b.map_or(0.0, |ib| soa.inv_masses[ib]);
    if ima == 0.0 && imb == 0.0 {
        return;
    }

    let inv_i_a = soa.inv_inertias_world[idx_a];
    let inv_i_b = idx_b.map_or(Mat3d::ZERO, |ib| soa.inv_inertias_world[ib]);
    let qa = soa.rotations[idx_a];
    let qb = idx_b.map_or(Quatd::IDENTITY, |ib| soa.rotations[ib]);
    let r_a = qa.rotate_vec3(joint.spec.local_anchor_a);
    let r_b = if idx_b.is_some() {
        qb.rotate_vec3(joint.spec.local_anchor_b)
    } else {
        Vec3d::ZERO
    };

    let damp = joint.spec.damping.clamp(0.0, 1.0);

    match &joint.spec.kind {
        JointKind::Distance {
            min_length,
            max_length,
            ..
        } => {
            let pa = soa.positions[idx_a] + r_a;
            let pb = match idx_b {
                Some(ib) => soa.positions[ib] + r_b,
                None => joint.spec.local_anchor_b,
            };
            let Some(dir) = (pa - pb).try_normalize() else {
                return;
            };
            if min_length.is_none() && max_length.is_none() && damp > 0.0 {
                let va = soa.linear_velocities[idx_a] + soa.angular_velocities[idx_a].cross(r_a);
                let vb = idx_b.map_or(Vec3d::ZERO, |ib| {
                    soa.linear_velocities[ib] + soa.angular_velocities[ib].cross(r_b)
                });
                let v_rel = (va - vb).dot(dir);
                let m_eff = effective_mass_linear(ima, &inv_i_a, r_a, imb, &inv_i_b, r_b, dir);
                let j = -v_rel * m_eff * damp;
                let imp = dir * j;
                apply_vel_impulse(soa, idx_a, idx_b, r_a, r_b, imp);
                joint.vel_pos_impulse += imp;
            }
        }
        JointKind::FixedWeld { .. } => {
            if damp > 0.0 {
                let va = soa.linear_velocities[idx_a] + soa.angular_velocities[idx_a].cross(r_a);
                let vb = idx_b.map_or(Vec3d::ZERO, |ib| {
                    soa.linear_velocities[ib] + soa.angular_velocities[ib].cross(r_b)
                });
                let v_rel = va - vb;
                for axis in [Vec3d::X, Vec3d::Y, Vec3d::Z] {
                    let m_eff = effective_mass_linear(ima, &inv_i_a, r_a, imb, &inv_i_b, r_b, axis);
                    let imp = axis * (-v_rel.dot(axis) * m_eff * damp);
                    apply_vel_impulse(soa, idx_a, idx_b, r_a, r_b, imp);
                    joint.vel_pos_impulse += imp;
                }
                let wa = soa.angular_velocities[idx_a];
                let wb = idx_b.map_or(Vec3d::ZERO, |ib| soa.angular_velocities[ib]);
                let w_rel = wa - wb;
                for axis in [Vec3d::X, Vec3d::Y, Vec3d::Z] {
                    let k = axis.dot(inv_i_a.mul_vec3(axis)) + axis.dot(inv_i_b.mul_vec3(axis));
                    if k > 1e-18 {
                        let ang_imp = axis * (-w_rel.dot(axis) / k * damp);
                        apply_ang_vel_impulse(soa, idx_a, idx_b, ang_imp);
                        joint.vel_ang_impulse += ang_imp;
                    }
                }
            }
        }
        JointKind::Hinge {
            axis_a,
            motor_target_velocity,
            motor_max_torque,
            ..
        } => {
            let world_axis = qa.rotate_vec3(*axis_a).normalize_or(Vec3d::Y);
            if let Some(target_w) = *motor_target_velocity
                && *motor_max_torque > 0.0
            {
                let wa = soa.angular_velocities[idx_a];
                let wb = idx_b.map_or(Vec3d::ZERO, |ib| soa.angular_velocities[ib]);
                let w_rel = (wa - wb).dot(world_axis);
                let k = world_axis.dot(inv_i_a.mul_vec3(world_axis))
                    + world_axis.dot(inv_i_b.mul_vec3(world_axis));
                if k > 1e-18 {
                    let dj = (target_w - w_rel) / k;
                    let max_j = motor_max_torque * h;
                    let old_j = joint.motor_impulse;
                    let new_j = (old_j + dj).clamp(-max_j, max_j);
                    let applied = new_j - old_j;
                    joint.motor_impulse = new_j;
                    let ang_imp = world_axis * applied;
                    apply_ang_vel_impulse(soa, idx_a, idx_b, ang_imp);
                    joint.vel_ang_impulse += ang_imp;
                }
            } else if damp > 0.0 {
                // Mild hinge damping on perpendicular rotational axes only (do not fight free spin).
                let (u, v) = world_axis.orthonormal_basis();
                let wa = soa.angular_velocities[idx_a];
                let wb = idx_b.map_or(Vec3d::ZERO, |ib| soa.angular_velocities[ib]);
                let w_rel = wa - wb;
                for axis in [u, v] {
                    let k = axis.dot(inv_i_a.mul_vec3(axis)) + axis.dot(inv_i_b.mul_vec3(axis));
                    if k > 1e-18 {
                        let ang_imp = axis * (-w_rel.dot(axis) / k * damp);
                        apply_ang_vel_impulse(soa, idx_a, idx_b, ang_imp);
                        joint.vel_ang_impulse += ang_imp;
                    }
                }
            }
        }
        JointKind::Slider {
            axis_a,
            motor_target_velocity,
            motor_max_force,
            ..
        } => {
            let world_axis = qa.rotate_vec3(*axis_a).normalize_or(Vec3d::X);
            if let Some(target_v) = *motor_target_velocity
                && *motor_max_force > 0.0
            {
                let va = soa.linear_velocities[idx_a] + soa.angular_velocities[idx_a].cross(r_a);
                let vb = idx_b.map_or(Vec3d::ZERO, |ib| {
                    soa.linear_velocities[ib] + soa.angular_velocities[ib].cross(r_b)
                });
                let v_rel = (va - vb).dot(world_axis);
                let m_eff =
                    effective_mass_linear(ima, &inv_i_a, r_a, imb, &inv_i_b, r_b, world_axis);
                if m_eff > 0.0 {
                    let dj = (target_v - v_rel) * m_eff;
                    let max_j = motor_max_force * h;
                    let old_j = joint.motor_impulse;
                    let new_j = (old_j + dj).clamp(-max_j, max_j);
                    let applied = new_j - old_j;
                    joint.motor_impulse = new_j;
                    let imp = world_axis * applied;
                    apply_vel_impulse(soa, idx_a, idx_b, r_a, r_b, imp);
                    joint.vel_pos_impulse += imp;
                }
            }
        }
    }
}

/// Positional XPBD joint projection with exact compliance and velocity synchronization.
///
/// Returns any joints whose measured force or torque exceeded `max_force` / `max_torque`.
pub(crate) fn solve_joints_xpbd_positions(
    soa: &mut SolverSoA,
    joints: &mut [ActiveJoint],
    h: f64,
    iterations: u32,
) -> Vec<BrokenJointInfo> {
    if joints.is_empty() || h <= 0.0 {
        return Vec::new();
    }
    let inv_h = 1.0 / h;
    let inv_h2 = inv_h * inv_h;

    for _ in 0..iterations.max(1) {
        for joint in joints.iter_mut() {
            let idx_a = joint.idx_a;
            let idx_b = joint.idx_b;
            let ima = soa.inv_masses[idx_a];
            let imb = idx_b.map_or(0.0, |ib| soa.inv_masses[ib]);
            if ima == 0.0 && imb == 0.0 {
                continue;
            }
            let alpha_tilde = joint.spec.compliance.max(0.0) * inv_h2;

            match joint.spec.kind.clone() {
                JointKind::Distance {
                    rest_length,
                    min_length,
                    max_length,
                } => {
                    let (pa, pb, r_a, r_b) = world_anchors(soa, joint);
                    let diff = pa - pb;
                    let dist = diff.length();
                    if dist < 1e-12 {
                        continue;
                    }
                    let dir = diff / dist;
                    let err = match (min_length, max_length) {
                        (None, None) => dist - rest_length,
                        (min_l, max_l) => {
                            if let Some(mn) = min_l
                                && dist < mn
                            {
                                dist - mn
                            } else if let Some(mx) = max_l
                                && dist > mx
                            {
                                dist - mx
                            } else {
                                0.0
                            }
                        }
                    };
                    if err.abs() > 1e-12 {
                        let dlam = apply_xpbd_positional_1d(
                            soa,
                            idx_a,
                            idx_b,
                            r_a,
                            r_b,
                            dir,
                            err,
                            alpha_tilde,
                            &mut joint.scalar_lambda_pos[0],
                            inv_h,
                        );
                        joint.xpbd_pos_lambda += dir * dlam;
                    }
                }
                JointKind::FixedWeld { relative_rotation } => {
                    // 3 positional DOFs.
                    for (axis_i, axis) in [Vec3d::X, Vec3d::Y, Vec3d::Z].into_iter().enumerate() {
                        let (pa, pb, r_a, r_b) = world_anchors(soa, joint);
                        let err = (pa - pb).dot(axis);
                        if err.abs() > 1e-12 {
                            let dlam = apply_xpbd_positional_1d(
                                soa,
                                idx_a,
                                idx_b,
                                r_a,
                                r_b,
                                axis,
                                err,
                                alpha_tilde,
                                &mut joint.scalar_lambda_pos[axis_i],
                                inv_h,
                            );
                            joint.xpbd_pos_lambda += axis * dlam;
                        }
                    }

                    // 3 rotational DOFs: target q_a = q_b * relative_rotation^-1 (so q_b = q_a * relative_rotation).
                    let qa = soa.rotations[idx_a];
                    let qb = idx_b.map_or(Quatd::IDENTITY, |ib| soa.rotations[ib]);
                    let q_target_a = (qb * relative_rotation.conjugate()).normalize();
                    let q_err = (qa * q_target_a.conjugate()).normalize();
                    let rot_err = q_err.to_rotation_vector();
                    let err_len = rot_err.length();
                    if err_len > 1e-12 {
                        let axis = rot_err / err_len;
                        let dlam = apply_xpbd_angular_1d(
                            soa,
                            idx_a,
                            idx_b,
                            axis,
                            err_len,
                            alpha_tilde,
                            &mut joint.scalar_lambda_ang[0],
                            inv_h,
                        );
                        joint.xpbd_ang_lambda += axis * dlam;
                    }
                }
                JointKind::Hinge {
                    axis_a,
                    axis_b,
                    limits,
                    ..
                } => {
                    // 1. Lock 3 positional DOFs at the hinge pivot.
                    for (axis_i, axis) in [Vec3d::X, Vec3d::Y, Vec3d::Z].into_iter().enumerate() {
                        let (pa, pb, r_a, r_b) = world_anchors(soa, joint);
                        let err = (pa - pb).dot(axis);
                        if err.abs() > 1e-12 {
                            let dlam = apply_xpbd_positional_1d(
                                soa,
                                idx_a,
                                idx_b,
                                r_a,
                                r_b,
                                axis,
                                err,
                                alpha_tilde,
                                &mut joint.scalar_lambda_pos[axis_i],
                                inv_h,
                            );
                            joint.xpbd_pos_lambda += axis * dlam;
                        }
                    }

                    // 2. Align hinge axes wa (on A) and wb (on B): locks 2 angular DOFs.
                    let qa = soa.rotations[idx_a];
                    let qb = idx_b.map_or(Quatd::IDENTITY, |ib| soa.rotations[ib]);
                    let wa = qa.rotate_vec3(axis_a).normalize_or(Vec3d::Y);
                    let wb = qb.rotate_vec3(axis_b).normalize_or(Vec3d::Y);
                    // Rotation taking wb -> wa has vector approximately wb x wa,
                    // so error of A relative to B is `wb.cross(wa)`.
                    let align_err = wb.cross(wa);
                    let align_len = align_err.length();
                    if align_len > 1e-12 {
                        let corr_axis = align_err / align_len;
                        let angle_err = align_len.min(1.0).asin();
                        let dlam = apply_xpbd_angular_1d(
                            soa,
                            idx_a,
                            idx_b,
                            corr_axis,
                            angle_err,
                            alpha_tilde,
                            &mut joint.scalar_lambda_ang[0],
                            inv_h,
                        );
                        joint.xpbd_ang_lambda += corr_axis * dlam;
                    }

                    // 3. Optional angular limits around the hinge axis.
                    if let Some((min_ang, max_ang)) = limits {
                        let qa = soa.rotations[idx_a];
                        let qb = idx_b.map_or(Quatd::IDENTITY, |ib| soa.rotations[ib]);
                        let wa = qa.rotate_vec3(axis_a).normalize_or(Vec3d::Y);
                        let ra = qa.rotate_vec3(joint.ref_a);
                        let rb = qb.rotate_vec3(joint.ref_b);
                        let rb_proj = (rb - wa * rb.dot(wa)).normalize_or(ra);
                        let ra_proj = (ra - wa * ra.dot(wa)).normalize_or(rb_proj);
                        let angle = rb_proj.cross(ra_proj).dot(wa).atan2(rb_proj.dot(ra_proj));
                        let limit_err = if angle > max_ang {
                            angle - max_ang
                        } else if angle < min_ang {
                            angle - min_ang
                        } else {
                            0.0
                        };
                        if limit_err.abs() > 1e-12 {
                            let dlam = apply_xpbd_angular_1d(
                                soa,
                                idx_a,
                                idx_b,
                                wa,
                                limit_err,
                                alpha_tilde,
                                &mut joint.scalar_lambda_ang[1],
                                inv_h,
                            );
                            joint.xpbd_ang_lambda += wa * dlam;
                        }
                    }
                }
                JointKind::Slider { axis_a, limits, .. } => {
                    // 1. Lock all 3 rotational DOFs to initial relative rotation.
                    let qa = soa.rotations[idx_a];
                    let qb = idx_b.map_or(Quatd::IDENTITY, |ib| soa.rotations[ib]);
                    let q_target_a = (qb * joint.slider_rel_rot.conjugate()).normalize();
                    let q_err = (qa * q_target_a.conjugate()).normalize();
                    let rot_err = q_err.to_rotation_vector();
                    let err_len = rot_err.length();
                    if err_len > 1e-12 {
                        let axis = rot_err / err_len;
                        let dlam = apply_xpbd_angular_1d(
                            soa,
                            idx_a,
                            idx_b,
                            axis,
                            err_len,
                            alpha_tilde,
                            &mut joint.scalar_lambda_ang[0],
                            inv_h,
                        );
                        joint.xpbd_ang_lambda += axis * dlam;
                    }

                    // 2. Lock 2 lateral positional DOFs perpendicular to slider axis.
                    let world_axis = soa.rotations[idx_a]
                        .rotate_vec3(axis_a)
                        .normalize_or(Vec3d::X);
                    let (u, v) = world_axis.orthonormal_basis();
                    for (axis_i, lat) in [u, v].into_iter().enumerate() {
                        let (pa, pb, r_a, r_b) = world_anchors(soa, joint);
                        let err = (pa - pb).dot(lat);
                        if err.abs() > 1e-12 {
                            let dlam = apply_xpbd_positional_1d(
                                soa,
                                idx_a,
                                idx_b,
                                r_a,
                                r_b,
                                lat,
                                err,
                                alpha_tilde,
                                &mut joint.scalar_lambda_pos[axis_i],
                                inv_h,
                            );
                            joint.xpbd_pos_lambda += lat * dlam;
                        }
                    }

                    // 3. Optional linear limits along slider axis.
                    if let Some((min_d, max_d)) = limits {
                        let (pa, pb, r_a, r_b) = world_anchors(soa, joint);
                        let slide = (pa - pb).dot(world_axis);
                        let limit_err = if slide > max_d {
                            slide - max_d
                        } else if slide < min_d {
                            slide - min_d
                        } else {
                            0.0
                        };
                        if limit_err.abs() > 1e-12 {
                            let dlam = apply_xpbd_positional_1d(
                                soa,
                                idx_a,
                                idx_b,
                                r_a,
                                r_b,
                                world_axis,
                                limit_err,
                                alpha_tilde,
                                &mut joint.scalar_lambda_pos[2],
                                inv_h,
                            );
                            joint.xpbd_pos_lambda += world_axis * dlam;
                        }
                    }
                }
            }
        }
    }

    // Evaluate breaking force and torque thresholds after positional projection.
    let mut broken = Vec::new();
    for joint in joints.iter() {
        let total_force = (joint.xpbd_pos_lambda * inv_h2 + joint.vel_pos_impulse * inv_h).length();
        let total_torque =
            (joint.xpbd_ang_lambda * inv_h2 + joint.vel_ang_impulse * inv_h).length();
        if total_force > joint.spec.max_force || total_torque > joint.spec.max_torque {
            broken.push(BrokenJointInfo {
                id: joint.id,
                body_a: joint.spec.body_a,
                body_b: joint.spec.body_b,
                force: total_force,
                torque: total_torque,
            });
        }
    }
    broken
}

#[inline]
fn world_anchors(soa: &SolverSoA, joint: &ActiveJoint) -> (Vec3d, Vec3d, Vec3d, Vec3d) {
    let idx_a = joint.idx_a;
    let r_a = soa.rotations[idx_a].rotate_vec3(joint.spec.local_anchor_a);
    let pa = soa.positions[idx_a] + r_a;
    let (pb, r_b) = match joint.idx_b {
        Some(ib) => {
            let rb = soa.rotations[ib].rotate_vec3(joint.spec.local_anchor_b);
            (soa.positions[ib] + rb, rb)
        }
        None => (joint.spec.local_anchor_b, Vec3d::ZERO),
    };
    (pa, pb, r_a, r_b)
}

#[allow(clippy::too_many_arguments)]
fn apply_xpbd_positional_1d(
    soa: &mut SolverSoA,
    idx_a: usize,
    idx_b: Option<usize>,
    r_a: Vec3d,
    r_b: Vec3d,
    dir: Vec3d,
    err: f64,
    alpha_tilde: f64,
    lambda_acc: &mut f64,
    inv_h: f64,
) -> f64 {
    let ima = soa.inv_masses[idx_a];
    let imb = idx_b.map_or(0.0, |ib| soa.inv_masses[ib]);
    let inv_i_a = soa.inv_inertias_world[idx_a];
    let inv_i_b = idx_b.map_or(Mat3d::ZERO, |ib| soa.inv_inertias_world[ib]);

    let ra_x_n = r_a.cross(dir);
    let rb_x_n = r_b.cross(dir);
    let w_a = ima + ra_x_n.dot(inv_i_a.mul_vec3(ra_x_n));
    let w_b = imb + rb_x_n.dot(inv_i_b.mul_vec3(rb_x_n));
    let denom = w_a + w_b + alpha_tilde;
    if denom <= 1e-18 {
        return 0.0;
    }

    let dlam = (-err - alpha_tilde * (*lambda_acc)) / denom;
    *lambda_acc += dlam;
    let p = dir * dlam;

    if ima > 0.0 {
        let dx = p * ima;
        let dtheta = inv_i_a.mul_vec3(r_a.cross(p));
        soa.positions[idx_a] += dx;
        soa.rotations[idx_a] = soa.rotations[idx_a].integrate_angular_velocity(dtheta, 1.0);
        soa.linear_velocities[idx_a] += dx * inv_h;
        soa.angular_velocities[idx_a] += dtheta * inv_h;
        soa.update_world_inertia(idx_a);
    }
    if let Some(ib) = idx_b
        && imb > 0.0
    {
        let dx = -p * imb;
        let dtheta = -inv_i_b.mul_vec3(r_b.cross(p));
        soa.positions[ib] += dx;
        soa.rotations[ib] = soa.rotations[ib].integrate_angular_velocity(dtheta, 1.0);
        soa.linear_velocities[ib] += dx * inv_h;
        soa.angular_velocities[ib] += dtheta * inv_h;
        soa.update_world_inertia(ib);
    }
    dlam
}

#[allow(clippy::too_many_arguments)]
fn apply_xpbd_angular_1d(
    soa: &mut SolverSoA,
    idx_a: usize,
    idx_b: Option<usize>,
    axis: Vec3d,
    err: f64,
    alpha_tilde: f64,
    lambda_acc: &mut f64,
    inv_h: f64,
) -> f64 {
    let ima = soa.inv_masses[idx_a];
    let imb = idx_b.map_or(0.0, |ib| soa.inv_masses[ib]);
    let inv_i_a = soa.inv_inertias_world[idx_a];
    let inv_i_b = idx_b.map_or(Mat3d::ZERO, |ib| soa.inv_inertias_world[ib]);

    let w_a = if ima > 0.0 {
        axis.dot(inv_i_a.mul_vec3(axis))
    } else {
        0.0
    };
    let w_b = if imb > 0.0 {
        axis.dot(inv_i_b.mul_vec3(axis))
    } else {
        0.0
    };
    let denom = w_a + w_b + alpha_tilde;
    if denom <= 1e-18 {
        return 0.0;
    }

    let dlam = (-err - alpha_tilde * (*lambda_acc)) / denom;
    *lambda_acc += dlam;
    let tau = axis * dlam;

    if ima > 0.0 {
        let dtheta = inv_i_a.mul_vec3(tau);
        soa.rotations[idx_a] = soa.rotations[idx_a].integrate_angular_velocity(dtheta, 1.0);
        soa.angular_velocities[idx_a] += dtheta * inv_h;
        soa.update_world_inertia(idx_a);
    }
    if let Some(ib) = idx_b
        && imb > 0.0
    {
        let dtheta = -inv_i_b.mul_vec3(tau);
        soa.rotations[ib] = soa.rotations[ib].integrate_angular_velocity(dtheta, 1.0);
        soa.angular_velocities[ib] += dtheta * inv_h;
        soa.update_world_inertia(ib);
    }
    dlam
}
