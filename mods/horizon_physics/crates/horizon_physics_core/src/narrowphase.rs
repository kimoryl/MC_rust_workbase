//! Pair-anchor local-frame SAT narrowphase (`OBB-OBB` and specialized `OBB-AABB`),
//! exposed-face internal-edge filtering, Sutherland-Hodgman face clipping, and
//! deterministic 4-point contact manifold reduction.
//!
//! # Normal Convention
//! Every [`ContactManifold::normal`] is a unit vector pointing from **B toward A**
//! (the direction that pushes body A out of body B / terrain).

use crate::arena::BodyId;
use crate::math::{Aabbd, Obbd, Vec3d};
use crate::voxel::{FACE_NEG_X, FACE_NEG_Y, FACE_NEG_Z, FACE_POS_X, FACE_POS_Y, FACE_POS_Z};

/// Persistent topological feature key used to warm-start impulses across steps.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FeatureKey {
    pub shape_a: u32,
    pub shape_b: u32,
    pub feature_id: u32,
}

impl FeatureKey {
    #[inline]
    pub const fn new(shape_a: u32, shape_b: u32, feature_id: u32) -> Self {
        Self {
            shape_a,
            shape_b,
            feature_id,
        }
    }
}

/// Single contact point in a [`ContactManifold`].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ContactPoint {
    pub world_point_a: Vec3d,
    pub world_point_b: Vec3d,
    pub local_point_a: Vec3d,
    pub local_point_b: Vec3d,
    /// Positive depth when penetrating.
    pub penetration: f64,
    pub feature: FeatureKey,
    /// Cached normal impulse for warm-starting.
    pub normal_impulse: f64,
    /// Cached orthogonal tangent impulses `(t1, t2)` for warm-starting.
    pub tangent_impulse: [f64; 2],
}

/// Deterministic up-to-4-point contact manifold between body A and body B (or terrain).
#[derive(Clone, Debug)]
pub struct ContactManifold {
    pub body_a: BodyId,
    /// `None` indicates static world terrain.
    pub body_b: Option<BodyId>,
    /// Unit contact normal pointing from B -> A.
    pub normal: Vec3d,
    pub friction: f64,
    pub restitution: f64,
    pub points: [ContactPoint; 4],
    pub point_count: usize,
    pub is_terrain: bool,
}

impl ContactManifold {
    #[inline]
    pub fn empty(body_a: BodyId, body_b: Option<BodyId>, normal: Vec3d) -> Self {
        Self {
            body_a,
            body_b,
            normal,
            friction: 0.5,
            restitution: 0.0,
            points: [ContactPoint::default(); 4],
            point_count: 0,
            is_terrain: body_b.is_none(),
        }
    }

    #[inline]
    pub fn active_points(&self) -> &[ContactPoint] {
        &self.points[..self.point_count]
    }

    #[inline]
    pub fn active_points_mut(&mut self) -> &mut [ContactPoint] {
        &mut self.points[..self.point_count]
    }
}

/// Raw SAT collision output in world space before attaching body IDs and materials.
#[derive(Clone, Debug)]
pub struct BoxContactResult {
    /// Unit normal pointing from B -> A.
    pub normal: Vec3d,
    pub points: [ContactPoint; 4],
    pub point_count: usize,
}

#[derive(Copy, Clone, Debug)]
struct LocalBoxFrame {
    center: Vec3d,
    half_extents: Vec3d,
    axes: [Vec3d; 3],
    exposed_faces: u8,
}

#[derive(Copy, Clone, Debug)]
enum SatAxisKind {
    FaceA(usize),
    FaceB(usize),
    EdgeEdge(usize, usize),
}

#[derive(Copy, Clone, Debug)]
struct SatBest {
    overlap: f64,
    normal: Vec3d, // B -> A
    kind: SatAxisKind,
}

#[inline]
const fn face_bit(axis: usize, positive: bool) -> u8 {
    match (axis, positive) {
        (0, false) => FACE_NEG_X,
        (0, true) => FACE_POS_X,
        (1, false) => FACE_NEG_Y,
        (1, true) => FACE_POS_Y,
        (2, false) => FACE_NEG_Z,
        (2, true) => FACE_POS_Z,
        _ => 0,
    }
}

#[inline]
const fn is_face_exposed(mask: u8, axis: usize, positive: bool) -> bool {
    (mask & face_bit(axis, positive)) != 0
}

/// Checks if a normal aligns predominantly with an UNEXPOSED (internal seam) face
/// of a box. `outward_dir` is the direction pointing OUT of that box toward the other.
#[inline]
fn aligns_with_buried_face(box_frame: &LocalBoxFrame, outward_dir: Vec3d) -> bool {
    let mut best_abs = 0.0;
    let mut best_axis = 0usize;
    let mut best_pos = true;
    for (i, &axis) in box_frame.axes.iter().enumerate() {
        let d = outward_dir.dot(axis);
        if d.abs() > best_abs {
            best_abs = d.abs();
            best_axis = i;
            best_pos = d >= 0.0;
        }
    }
    best_abs > 0.72 && !is_face_exposed(box_frame.exposed_faces, best_axis, best_pos)
}

/// Full 15-axis SAT `OBB-OBB` collision test executed in pair-anchor local coordinates.
pub fn collide_obb_obb(
    obb_a: &Obbd,
    exposed_a: u8,
    shape_a: u32,
    obb_b: &Obbd,
    exposed_b: u8,
    shape_b: u32,
) -> Option<BoxContactResult> {
    let anchor = (obb_a.center + obb_b.center) * 0.5;
    let frame_a = LocalBoxFrame {
        center: obb_a.center - anchor,
        half_extents: obb_a.half_extents,
        axes: obb_a.axes(),
        exposed_faces: exposed_a,
    };
    let frame_b = LocalBoxFrame {
        center: obb_b.center - anchor,
        half_extents: obb_b.half_extents,
        axes: obb_b.axes(),
        exposed_faces: exposed_b,
    };

    let t = frame_a.center - frame_b.center; // B -> A

    // Precompute rotation matrix R[i][j] = A_i . B_j and abs(R) + eps.
    let mut r = [[0.0_f64; 3]; 3];
    let mut abs_r = [[0.0_f64; 3]; 3];
    const EPS: f64 = 1e-9;
    for i in 0..3 {
        for j in 0..3 {
            let dot = frame_a.axes[i].dot(frame_b.axes[j]);
            r[i][j] = dot;
            abs_r[i][j] = dot.abs() + EPS;
        }
    }

    let ea = frame_a.half_extents;
    let eb = frame_b.half_extents;

    let mut best_exposed: Option<SatBest> = None;
    let mut best_any: Option<SatBest> = None;

    let update_candidate =
        |slot: &mut Option<SatBest>, overlap: f64, normal: Vec3d, kind: SatAxisKind, bias: f64| {
            let better = match slot {
                None => true,
                Some(prev) => overlap < prev.overlap * bias - 1e-6,
            };
            if better {
                *slot = Some(SatBest {
                    overlap,
                    normal,
                    kind,
                });
            }
        };

    // 1. Face axes of B (0..3): preferred slightly on ties for stable terrain/support normals.
    for j in 0..3 {
        let axis = frame_b.axes[j];
        let proj = t.dot(axis);
        let ra = ea.x * abs_r[0][j] + ea.y * abs_r[1][j] + ea.z * abs_r[2][j];
        let rb = eb[j];
        let overlap = ra + rb - proj.abs();
        if overlap < 0.0 {
            return None;
        }
        let sign = if proj >= 0.0 { 1.0 } else { -1.0 };
        let normal = axis * sign; // B -> A
        let exposed = is_face_exposed(exposed_b, j, sign > 0.0)
            && !aligns_with_buried_face(&frame_a, -normal);
        update_candidate(&mut best_any, overlap, normal, SatAxisKind::FaceB(j), 1.0);
        if exposed {
            update_candidate(
                &mut best_exposed,
                overlap,
                normal,
                SatAxisKind::FaceB(j),
                1.0,
            );
        }
    }

    // 2. Face axes of A (0..3).
    for i in 0..3 {
        let axis = frame_a.axes[i];
        let proj = t.dot(axis);
        let ra = ea[i];
        let rb = eb.x * abs_r[i][0] + eb.y * abs_r[i][1] + eb.z * abs_r[i][2];
        let overlap = ra + rb - proj.abs();
        if overlap < 0.0 {
            return None;
        }
        let sign = if proj >= 0.0 { 1.0 } else { -1.0 };
        let normal = axis * sign; // B -> A
        // Outward contacting face on A points along -normal => sign < 0 means +axis face of A.
        let exposed =
            is_face_exposed(exposed_a, i, sign < 0.0) && !aligns_with_buried_face(&frame_b, normal);
        update_candidate(&mut best_any, overlap, normal, SatAxisKind::FaceA(i), 0.999);
        if exposed {
            update_candidate(
                &mut best_exposed,
                overlap,
                normal,
                SatAxisKind::FaceA(i),
                0.999,
            );
        }
    }

    // 3. 9 Edge-Edge cross axes A_i x B_j.
    let face_edge_bias = 0.95;
    for i in 0..3 {
        for j in 0..3 {
            let cross = frame_a.axes[i].cross(frame_b.axes[j]);
            let len_sq = cross.length_sq();
            if len_sq < 1e-10 {
                continue;
            }
            let inv_len = 1.0 / len_sq.sqrt();
            let axis = cross * inv_len;
            let proj = t.dot(axis);

            let ra = ea.x * axis.dot(frame_a.axes[0]).abs()
                + ea.y * axis.dot(frame_a.axes[1]).abs()
                + ea.z * axis.dot(frame_a.axes[2]).abs();
            let rb = eb.x * axis.dot(frame_b.axes[0]).abs()
                + eb.y * axis.dot(frame_b.axes[1]).abs()
                + eb.z * axis.dot(frame_b.axes[2]).abs();
            let overlap = ra + rb - proj.abs();
            if overlap < 0.0 {
                return None;
            }

            let sign = if proj >= 0.0 { 1.0 } else { -1.0 };
            let normal = axis * sign;
            update_candidate(
                &mut best_any,
                overlap,
                normal,
                SatAxisKind::EdgeEdge(i, j),
                face_edge_bias,
            );
            if !aligns_with_buried_face(&frame_b, normal)
                && !aligns_with_buried_face(&frame_a, -normal)
            {
                update_candidate(
                    &mut best_exposed,
                    overlap,
                    normal,
                    SatAxisKind::EdgeEdge(i, j),
                    face_edge_bias,
                );
            }
        }
    }

    // If an exposed axis exists, always prefer it over buried internal-seam faces.
    let winner = best_exposed.or(best_any)?;
    build_manifold_from_sat(anchor, &frame_a, shape_a, &frame_b, shape_b, winner)
}

/// Specialized SAT `OBB-AABB` collision test in pair-anchor local coordinates.
///
/// Exploits B's world-aligned canonical axes `[X, Y, Z]` to avoid redundant matrix
/// products while preserving exact exposed-face filtering and 4-point clipping.
pub fn collide_obb_aabb(
    obb_a: &Obbd,
    exposed_a: u8,
    shape_a: u32,
    aabb_b: &Aabbd,
    exposed_b: u8,
    shape_b: u32,
) -> Option<BoxContactResult> {
    let b_center = aabb_b.center();
    let b_he = aabb_b.half_extents();
    let anchor = (obb_a.center + b_center) * 0.5;

    let frame_a = LocalBoxFrame {
        center: obb_a.center - anchor,
        half_extents: obb_a.half_extents,
        axes: obb_a.axes(),
        exposed_faces: exposed_a,
    };
    let frame_b = LocalBoxFrame {
        center: b_center - anchor,
        half_extents: b_he,
        axes: [Vec3d::X, Vec3d::Y, Vec3d::Z],
        exposed_faces: exposed_b,
    };

    let t = frame_a.center - frame_b.center; // B -> A
    let ea = frame_a.half_extents;
    let eb = frame_b.half_extents;
    const EPS: f64 = 1e-9;

    let abs_a = [
        frame_a.axes[0].abs() + Vec3d::splat(EPS),
        frame_a.axes[1].abs() + Vec3d::splat(EPS),
        frame_a.axes[2].abs() + Vec3d::splat(EPS),
    ];

    let mut best_exposed: Option<SatBest> = None;
    let mut best_any: Option<SatBest> = None;

    let update_candidate =
        |slot: &mut Option<SatBest>, overlap: f64, normal: Vec3d, kind: SatAxisKind, bias: f64| {
            let better = match slot {
                None => true,
                Some(prev) => overlap < prev.overlap * bias - 1e-6,
            };
            if better {
                *slot = Some(SatBest {
                    overlap,
                    normal,
                    kind,
                });
            }
        };

    // 1. Canonical face axes of AABB B (X, Y, Z).
    for j in 0..3 {
        let proj = t[j];
        let ra = ea.x * abs_a[0][j] + ea.y * abs_a[1][j] + ea.z * abs_a[2][j];
        let rb = eb[j];
        let overlap = ra + rb - proj.abs();
        if overlap < 0.0 {
            return None;
        }
        let sign = if proj >= 0.0 { 1.0 } else { -1.0 };
        let mut normal = Vec3d::ZERO;
        normal[j] = sign;
        let exposed = is_face_exposed(exposed_b, j, sign > 0.0)
            && !aligns_with_buried_face(&frame_a, -normal);
        update_candidate(&mut best_any, overlap, normal, SatAxisKind::FaceB(j), 1.0);
        if exposed {
            update_candidate(
                &mut best_exposed,
                overlap,
                normal,
                SatAxisKind::FaceB(j),
                1.0,
            );
        }
    }

    // 2. Face axes of OBB A.
    for i in 0..3 {
        let axis = frame_a.axes[i];
        let proj = t.dot(axis);
        let ra = ea[i];
        let rb = eb.x * abs_a[i].x + eb.y * abs_a[i].y + eb.z * abs_a[i].z;
        let overlap = ra + rb - proj.abs();
        if overlap < 0.0 {
            return None;
        }
        let sign = if proj >= 0.0 { 1.0 } else { -1.0 };
        let normal = axis * sign;
        let exposed =
            is_face_exposed(exposed_a, i, sign < 0.0) && !aligns_with_buried_face(&frame_b, normal);
        update_candidate(&mut best_any, overlap, normal, SatAxisKind::FaceA(i), 0.999);
        if exposed {
            update_candidate(
                &mut best_exposed,
                overlap,
                normal,
                SatAxisKind::FaceA(i),
                0.999,
            );
        }
    }

    // 3. 9 Edge-Edge cross axes A_i x E_j.
    let face_edge_bias = 0.95;
    for i in 0..3 {
        let ai = frame_a.axes[i];
        let crosses = [
            Vec3d::new(0.0, ai.z, -ai.y), // ai x X
            Vec3d::new(-ai.z, 0.0, ai.x), // ai x Y
            Vec3d::new(ai.y, -ai.x, 0.0), // ai x Z
        ];
        for (j, cross) in crosses.into_iter().enumerate() {
            let len_sq = cross.length_sq();
            if len_sq < 1e-10 {
                continue;
            }
            let axis = cross / len_sq.sqrt();
            let proj = t.dot(axis);
            let ra = ea.x * axis.dot(frame_a.axes[0]).abs()
                + ea.y * axis.dot(frame_a.axes[1]).abs()
                + ea.z * axis.dot(frame_a.axes[2]).abs();
            let rb = eb.x * axis.x.abs() + eb.y * axis.y.abs() + eb.z * axis.z.abs();
            let overlap = ra + rb - proj.abs();
            if overlap < 0.0 {
                return None;
            }

            let sign = if proj >= 0.0 { 1.0 } else { -1.0 };
            let normal = axis * sign;
            update_candidate(
                &mut best_any,
                overlap,
                normal,
                SatAxisKind::EdgeEdge(i, j),
                face_edge_bias,
            );
            if !aligns_with_buried_face(&frame_b, normal)
                && !aligns_with_buried_face(&frame_a, -normal)
            {
                update_candidate(
                    &mut best_exposed,
                    overlap,
                    normal,
                    SatAxisKind::EdgeEdge(i, j),
                    face_edge_bias,
                );
            }
        }
    }

    let winner = best_exposed.or(best_any)?;
    build_manifold_from_sat(anchor, &frame_a, shape_a, &frame_b, shape_b, winner)
}

#[derive(Copy, Clone, Debug)]
struct ClipVertex {
    pos: Vec3d,
    feature_id: u32,
}

fn build_manifold_from_sat(
    anchor: Vec3d,
    frame_a: &LocalBoxFrame,
    shape_a: u32,
    frame_b: &LocalBoxFrame,
    shape_b: u32,
    winner: SatBest,
) -> Option<BoxContactResult> {
    match winner.kind {
        SatAxisKind::FaceB(ref_axis) => {
            // Reference face on B with outward normal = winner.normal (B -> A).
            let ref_n = winner.normal;
            let ref_pos_sign = ref_n.dot(frame_b.axes[ref_axis]) >= 0.0;
            let clipped = clip_incident_against_reference(
                frame_b,
                ref_axis,
                ref_pos_sign,
                ref_n,
                frame_a,
                1, // tag: B is ref
            );
            finalize_face_contacts(
                anchor,
                winner.normal,
                shape_a,
                shape_b,
                true, // reference is B, incident vertices lie on A
                &clipped,
            )
        }
        SatAxisKind::FaceA(ref_axis) => {
            // Reference face on A with outward normal = -winner.normal (A -> B).
            let ref_n = -winner.normal;
            let ref_pos_sign = ref_n.dot(frame_a.axes[ref_axis]) >= 0.0;
            let clipped = clip_incident_against_reference(
                frame_a,
                ref_axis,
                ref_pos_sign,
                ref_n,
                frame_b,
                0, // tag: A is ref
            );
            finalize_face_contacts(
                anchor,
                winner.normal,
                shape_a,
                shape_b,
                false, // reference is A, incident vertices lie on B
                &clipped,
            )
        }
        SatAxisKind::EdgeEdge(edge_a, edge_b) => {
            let (a0, a1) = support_edge(frame_a, edge_a, -winner.normal);
            let (b0, b1) = support_edge(frame_b, edge_b, winner.normal);
            let (pa, pb) = closest_points_segments(a0, a1, b0, b1);

            let feature_id = 0x8000_0000u32 | ((edge_a as u32) << 8) | (edge_b as u32);
            let pt = ContactPoint {
                world_point_a: anchor + pa,
                world_point_b: anchor + pb,
                local_point_a: Vec3d::ZERO,
                local_point_b: Vec3d::ZERO,
                penetration: winner.overlap.max(0.0),
                feature: FeatureKey::new(shape_a, shape_b, feature_id),
                normal_impulse: 0.0,
                tangent_impulse: [0.0, 0.0],
            };
            let mut points = [ContactPoint::default(); 4];
            points[0] = pt;
            Some(BoxContactResult {
                normal: winner.normal,
                points,
                point_count: 1,
            })
        }
    }
}

/// Clips the most anti-parallel face of `inc` against the 4 side planes of `ref_box`'s
/// reference face, returning `(clipped_vertex_on_inc, penetration_depth)` pairs.
fn clip_incident_against_reference(
    ref_box: &LocalBoxFrame,
    ref_axis: usize,
    ref_pos_sign: bool,
    ref_normal: Vec3d,
    inc_box: &LocalBoxFrame,
    ref_tag: u32,
) -> Vec<(ClipVertex, f64)> {
    // Tangent axes of reference face.
    let u_axis = (ref_axis + 1) % 3;
    let v_axis = (ref_axis + 2) % 3;
    let u_dir = ref_box.axes[u_axis];
    let v_dir = ref_box.axes[v_axis];
    let hu = ref_box.half_extents[u_axis];
    let hv = ref_box.half_extents[v_axis];
    let h_ref = ref_box.half_extents[ref_axis];
    let ref_face_center = ref_box.center + ref_normal * h_ref;

    // Find incident face on inc_box most anti-parallel to ref_normal.
    let mut best_dot = f64::INFINITY;
    let mut inc_axis = 0usize;
    let mut inc_pos = true;
    for (i, &axis) in inc_box.axes.iter().enumerate() {
        let d = axis.dot(ref_normal);
        if d < best_dot {
            best_dot = d;
            inc_axis = i;
            inc_pos = true;
        }
        if -d < best_dot {
            best_dot = -d;
            inc_axis = i;
            inc_pos = false;
        }
    }

    let inc_u_axis = (inc_axis + 1) % 3;
    let inc_v_axis = (inc_axis + 2) % 3;
    let inc_n_sign = if inc_pos { 1.0 } else { -1.0 };
    let inc_center =
        inc_box.center + inc_box.axes[inc_axis] * (inc_n_sign * inc_box.half_extents[inc_axis]);
    let du = inc_box.axes[inc_u_axis] * inc_box.half_extents[inc_u_axis];
    let dv = inc_box.axes[inc_v_axis] * inc_box.half_extents[inc_v_axis];

    let face_code = (ref_tag << 24)
        | ((ref_axis as u32) << 20)
        | (u32::from(ref_pos_sign) << 19)
        | ((inc_axis as u32) << 16)
        | (u32::from(inc_pos) << 15);

    let mut poly: Vec<ClipVertex> = vec![
        ClipVertex {
            pos: inc_center - du - dv,
            feature_id: face_code,
        },
        ClipVertex {
            pos: inc_center + du - dv,
            feature_id: face_code | 1,
        },
        ClipVertex {
            pos: inc_center + du + dv,
            feature_id: face_code | 2,
        },
        ClipVertex {
            pos: inc_center - du + dv,
            feature_id: face_code | 3,
        },
    ];

    // Clip against the 4 side planes of the reference face using Sutherland-Hodgman.
    let side_planes = [
        (u_dir, ref_face_center.dot(u_dir) + hu, 1u32),
        (-u_dir, -ref_face_center.dot(u_dir) + hu, 2u32),
        (v_dir, ref_face_center.dot(v_dir) + hv, 3u32),
        (-v_dir, -ref_face_center.dot(v_dir) + hv, 4u32),
    ];

    let mut scratch: Vec<ClipVertex> = Vec::with_capacity(8);
    for (plane_n, plane_d, plane_id) in side_planes {
        clip_polygon_against_plane(&poly, plane_n, plane_d, plane_id, &mut scratch);
        std::mem::swap(&mut poly, &mut scratch);
        if poly.is_empty() {
            return Vec::new();
        }
    }

    // Keep vertices on or behind the reference face plane.
    let ref_plane_d = ref_face_center.dot(ref_normal);
    let mut out = Vec::with_capacity(poly.len());
    for v in poly {
        let signed_dist = v.pos.dot(ref_normal) - ref_plane_d;
        let penetration = -signed_dist;
        if penetration >= -1e-5 {
            out.push((v, penetration.max(0.0)));
        }
    }
    out
}

/// Clips polygon to half-space `p . plane_n <= plane_d`.
fn clip_polygon_against_plane(
    input: &[ClipVertex],
    plane_n: Vec3d,
    plane_d: f64,
    plane_id: u32,
    output: &mut Vec<ClipVertex>,
) {
    output.clear();
    let n = input.len();
    if n == 0 {
        return;
    }

    let mut prev = input[n - 1];
    let mut prev_dist = prev.pos.dot(plane_n) - plane_d;

    for &curr in input {
        let curr_dist = curr.pos.dot(plane_n) - plane_d;
        let prev_in = prev_dist <= 1e-9;
        let curr_in = curr_dist <= 1e-9;

        if prev_in && curr_in {
            output.push(curr);
        } else if prev_in && !curr_in {
            let denom = prev_dist - curr_dist;
            let t = if denom.abs() > 1e-15 {
                (prev_dist / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            output.push(ClipVertex {
                pos: prev.pos.lerp(curr.pos, t),
                feature_id: (prev.feature_id & 0xFFFF_F000)
                    | (plane_id << 8)
                    | ((curr.feature_id & 0xF) << 4)
                    | (prev.feature_id & 0xF),
            });
        } else if !prev_in && curr_in {
            let denom = prev_dist - curr_dist;
            let t = if denom.abs() > 1e-15 {
                (prev_dist / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            output.push(ClipVertex {
                pos: prev.pos.lerp(curr.pos, t),
                feature_id: (curr.feature_id & 0xFFFF_F000)
                    | (plane_id << 8)
                    | ((prev.feature_id & 0xF) << 4)
                    | (curr.feature_id & 0xF),
            });
            output.push(curr);
        }

        prev = curr;
        prev_dist = curr_dist;
    }
}

fn finalize_face_contacts(
    anchor: Vec3d,
    normal_b_to_a: Vec3d,
    shape_a: u32,
    shape_b: u32,
    ref_is_b: bool,
    clipped: &[(ClipVertex, f64)],
) -> Option<BoxContactResult> {
    if clipped.is_empty() {
        return None;
    }

    let mut candidates: Vec<ContactPoint> = Vec::with_capacity(clipped.len());
    for &(cv, pen) in clipped {
        let (world_a, world_b) = if ref_is_b {
            // Incident point lies on A; project along +normal onto B's reference face.
            let wa = anchor + cv.pos;
            let wb = wa + normal_b_to_a * pen;
            (wa, wb)
        } else {
            // Incident point lies on B; project along -normal onto A's reference face.
            let wb = anchor + cv.pos;
            let wa = wb - normal_b_to_a * pen;
            (wa, wb)
        };
        candidates.push(ContactPoint {
            world_point_a: world_a,
            world_point_b: world_b,
            local_point_a: Vec3d::ZERO,
            local_point_b: Vec3d::ZERO,
            penetration: pen,
            feature: FeatureKey::new(shape_a, shape_b, cv.feature_id),
            normal_impulse: 0.0,
            tangent_impulse: [0.0, 0.0],
        });
    }

    let (points, point_count) = reduce_contacts_to_4(&candidates, normal_b_to_a);
    if point_count == 0 {
        return None;
    }
    Some(BoxContactResult {
        normal: normal_b_to_a,
        points,
        point_count,
    })
}

/// Deterministically reduces an arbitrary set of coplanar/near-coplanar contact
/// points down to at most 4 points preserving deepest penetration and maximal
/// support quadrilateral area.
pub fn reduce_contacts_to_4(
    candidates: &[ContactPoint],
    normal: Vec3d,
) -> ([ContactPoint; 4], usize) {
    let mut out = [ContactPoint::default(); 4];
    if candidates.is_empty() {
        return (out, 0);
    }

    // Deduplicate nearly coincident points first, retaining deepest penetration.
    let mut unique: Vec<ContactPoint> = Vec::with_capacity(candidates.len());
    for &c in candidates {
        let mut merged = false;
        for u in &mut unique {
            if (u.world_point_a - c.world_point_a).length_sq() < 1e-6 {
                if c.penetration > u.penetration
                    || ((c.penetration - u.penetration).abs() <= 1e-12 && c.feature < u.feature)
                {
                    *u = c;
                }
                merged = true;
                break;
            }
        }
        if !merged {
            unique.push(c);
        }
    }

    if unique.len() <= 4 {
        unique.sort_by_key(|p| p.feature);
        for (i, p) in unique.iter().enumerate() {
            out[i] = *p;
        }
        return (out, unique.len());
    }

    // Center coordinates around unique[0] so area computation stays accurate at farlands.
    let origin = unique[0].world_point_a;
    let pts: Vec<Vec3d> = unique.iter().map(|c| c.world_point_a - origin).collect();

    // 1. Deepest penetration point.
    let mut i0 = 0usize;
    for i in 1..unique.len() {
        if unique[i].penetration > unique[i0].penetration
            || ((unique[i].penetration - unique[i0].penetration).abs() <= 1e-12
                && unique[i].feature < unique[i0].feature)
        {
            i0 = i;
        }
    }

    // 2. Point farthest from i0.
    let mut i1 = i0;
    let mut max_dist_sq = -1.0;
    for i in 0..unique.len() {
        if i == i0 {
            continue;
        }
        let d2 = (pts[i] - pts[i0]).length_sq();
        if d2 > max_dist_sq + 1e-12
            || ((d2 - max_dist_sq).abs() <= 1e-12 && unique[i].feature < unique[i1].feature)
        {
            max_dist_sq = d2;
            i1 = i;
        }
    }

    // 3. Point maximizing triangle area (i0, i1, i2).
    let mut i2 = i0;
    let mut max_area = -1.0;
    let e01 = pts[i1] - pts[i0];
    for i in 0..unique.len() {
        if i == i0 || i == i1 {
            continue;
        }
        let area = e01.cross(pts[i] - pts[i0]).dot(normal).abs();
        if area > max_area + 1e-12
            || ((area - max_area).abs() <= 1e-12 && unique[i].feature < unique[i2].feature)
        {
            max_area = area;
            i2 = i;
        }
    }

    // Orient triangle (a, b, c) positively wrt normal so exterior edges have negative signed area.
    let (a, mut b, mut c) = (i0, i1, i2);
    if (pts[b] - pts[a]).cross(pts[c] - pts[a]).dot(normal) < 0.0 {
        std::mem::swap(&mut b, &mut c);
    }

    // 4. Point maximizing exterior quad area beyond edges (a,b), (b,c), (c,a).
    let mut i3 = i0;
    let mut best_ext_area = -1.0;
    for i in 0..unique.len() {
        if i == a || i == b || i == c {
            continue;
        }
        let s_ab = -(pts[b] - pts[a]).cross(pts[i] - pts[a]).dot(normal);
        let s_bc = -(pts[c] - pts[b]).cross(pts[i] - pts[b]).dot(normal);
        let s_ca = -(pts[a] - pts[c]).cross(pts[i] - pts[c]).dot(normal);
        let ext = s_ab.max(s_bc).max(s_ca);
        if ext > best_ext_area + 1e-12
            || ((ext - best_ext_area).abs() <= 1e-12 && unique[i].feature < unique[i3].feature)
        {
            best_ext_area = ext;
            i3 = i;
        }
    }

    let mut chosen = [unique[i0], unique[i1], unique[i2], unique[i3]];
    chosen.sort_by_key(|p| p.feature);
    out = chosen;
    (out, 4)
}

fn support_edge(box_frame: &LocalBoxFrame, edge_axis: usize, dir: Vec3d) -> (Vec3d, Vec3d) {
    let mut mid = box_frame.center;
    for k in 0..3 {
        if k == edge_axis {
            continue;
        }
        let s = if dir.dot(box_frame.axes[k]) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        mid += box_frame.axes[k] * (s * box_frame.half_extents[k]);
    }
    let half = box_frame.axes[edge_axis] * box_frame.half_extents[edge_axis];
    (mid - half, mid + half)
}

/// Computes the closest points `(pa, pb)` between 3D segments `[a0, a1]` and `[b0, b1]`.
pub fn closest_points_segments(a0: Vec3d, a1: Vec3d, b0: Vec3d, b1: Vec3d) -> (Vec3d, Vec3d) {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let (s, t) = if a <= 1e-14 && e <= 1e-14 {
        (0.0, 0.0)
    } else if a <= 1e-14 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e <= 1e-14 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let mut s = if denom.abs() > 1e-14 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };

    (a0 + d1 * s, b0 + d2 * t)
}
