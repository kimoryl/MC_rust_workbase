//! Continuous Collision Detection (CCD) via adaptive Conservative Advancement
//! (up to 12 iterations) backed by exact 3D GJK distance between oriented boxes.

use crate::math::{Obbd, Quatd, Vec3d};

/// Maximum Conservative Advancement iterations per swept shape pair.
pub const MAX_CA_ITERATIONS: usize = 12;

/// Distance query result between two convex boxes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GjkDistanceResult {
    /// Euclidean separation distance (`0.0` when overlapping).
    pub distance: f64,
    /// Closest point on shape A in world space.
    pub point_a: Vec3d,
    /// Closest point on shape B in world space.
    pub point_b: Vec3d,
    /// Unit separating normal pointing from B -> A.
    pub normal_b_to_a: Vec3d,
}

/// Time-of-impact result from Conservative Advancement.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ToiHit {
    /// Normalized fraction of `dt` in `[0.0, 1.0]`.
    pub fraction: f64,
    /// Absolute impact time in `[0.0, dt]`.
    pub time: f64,
    /// Unit contact normal pointing from B -> A.
    pub normal: Vec3d,
    /// Contact point on A at impact time.
    pub point_a: Vec3d,
    /// Contact point on B at impact time.
    pub point_b: Vec3d,
}

/// Kinematic sweep specification for a single OBB attached to a rigid body.
#[derive(Copy, Clone, Debug)]
pub struct SweptObb {
    /// Body CoM position at `t = 0`.
    pub body_pos: Vec3d,
    /// Body orientation at `t = 0`.
    pub body_rot: Quatd,
    /// Linear velocity of body CoM over `[0, dt]`.
    pub linear_vel: Vec3d,
    /// Angular velocity of body over `[0, dt]`.
    pub angular_vel: Vec3d,
    /// Box center offset relative to body CoM in body-local coordinates.
    pub local_offset: Vec3d,
    /// Box half-extents in body-local coordinates.
    pub half_extents: Vec3d,
}

impl SweptObb {
    #[inline]
    pub fn obb_at(&self, t: f64) -> Obbd {
        let rot = if self.angular_vel.length_sq() > 1e-24 && t != 0.0 {
            self.body_rot
                .integrate_angular_velocity(self.angular_vel, t)
        } else {
            self.body_rot
        };
        let pos = self.body_pos + self.linear_vel * t + rot.rotate_vec3(self.local_offset);
        Obbd::new(pos, self.half_extents, rot)
    }

    #[inline]
    pub fn max_angular_radius(&self) -> f64 {
        self.local_offset.length() + self.half_extents.length()
    }
}

/// Computes the Time of Impact (TOI) between two swept OBBs over `[0, dt]` using
/// adaptive Conservative Advancement (up to [`MAX_CA_ITERATIONS`] iterations).
///
/// Returns `None` if the boxes do not collide within `dt` or are already
/// penetrating at `t = 0` (leaving initial overlaps to discrete SAT).
pub fn conservative_advancement_obb_pair(
    a: &SweptObb,
    b: &SweptObb,
    dt: f64,
    tolerance: f64,
) -> Option<ToiHit> {
    if dt <= 0.0 {
        return None;
    }
    let tol = tolerance.max(1e-5);
    let r_a = a.max_angular_radius();
    let r_b = b.max_angular_radius();
    let omega_bound = a.angular_vel.length() * r_a + b.angular_vel.length() * r_b;

    let mut t = 0.0_f64;

    for iter in 0..MAX_CA_ITERATIONS {
        let obb_a = a.obb_at(t);
        let obb_b = b.obb_at(t);
        let dist_res = gjk_obb_distance(&obb_a, &obb_b);

        if iter == 0 && dist_res.distance <= 1e-9 {
            // Already overlapping at t = 0: let discrete SAT manifold handle it.
            return None;
        }

        if dist_res.distance <= tol {
            return Some(ToiHit {
                fraction: (t / dt).clamp(0.0, 1.0),
                time: t.clamp(0.0, dt),
                normal: dist_res.normal_b_to_a,
                point_a: dist_res.point_a,
                point_b: dist_res.point_b,
            });
        }

        // Conservative upper bound on closing rate along separating normal (B -> A).
        let lin_closing = -(a.linear_vel - b.linear_vel).dot(dist_res.normal_b_to_a);
        let v_bound = lin_closing + omega_bound;
        if v_bound <= 1e-9 {
            return None;
        }

        let delta_t = ((dist_res.distance - 0.5 * tol).max(0.0)) / v_bound;
        t += delta_t.max(1e-9 * dt);
        if t > dt {
            return None;
        }
    }

    // Final check after MAX_CA_ITERATIONS.
    let obb_a = a.obb_at(t);
    let obb_b = b.obb_at(t);
    let dist_res = gjk_obb_distance(&obb_a, &obb_b);
    if dist_res.distance <= tol * 4.0 {
        Some(ToiHit {
            fraction: (t / dt).clamp(0.0, 1.0),
            time: t.clamp(0.0, dt),
            normal: dist_res.normal_b_to_a,
            point_a: dist_res.point_a,
            point_b: dist_res.point_b,
        })
    } else {
        None
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct SupportVertex {
    w: Vec3d,  // sa - sb in anchor frame
    sa: Vec3d, // point on A in anchor frame
    sb: Vec3d, // point on B in anchor frame
}

/// Exact 3D GJK Euclidean distance between two OBBs in pair-anchor local coordinates.
pub fn gjk_obb_distance(obb_a: &Obbd, obb_b: &Obbd) -> GjkDistanceResult {
    let anchor = (obb_a.center + obb_b.center) * 0.5;
    let local_a = Obbd::new(obb_a.center - anchor, obb_a.half_extents, obb_a.orientation);
    let local_b = Obbd::new(obb_b.center - anchor, obb_b.half_extents, obb_b.orientation);

    let support = |dir: Vec3d| -> SupportVertex {
        let sa = local_a.support(dir);
        let sb = local_b.support(-dir);
        SupportVertex { w: sa - sb, sa, sb }
    };

    let init_dir = (local_a.center - local_b.center).normalize_or(Vec3d::X);
    let mut simplex = [SupportVertex::default(); 4];
    let mut count = 1usize;
    simplex[0] = support(-init_dir);

    let mut weights = [1.0, 0.0, 0.0, 0.0];
    let mut v = simplex[0].w;

    for _ in 0..24 {
        let v_sq = v.length_sq();
        if v_sq <= 1e-18 {
            return GjkDistanceResult {
                distance: 0.0,
                point_a: obb_a.center,
                point_b: obb_b.center,
                normal_b_to_a: init_dir,
            };
        }

        let new_pt = support(-v);
        // Check if already in simplex or makes negligible progress toward origin.
        let progress = v_sq - v.dot(new_pt.w);
        if progress <= 1e-11 * v_sq.max(1.0) {
            break;
        }
        let mut duplicate = false;
        for s in simplex.iter().take(count) {
            if (s.w - new_pt.w).length_sq() <= 1e-18 {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            break;
        }

        simplex[count] = new_pt;
        count += 1;

        let (contains_origin, new_count, new_simplex, new_weights, new_v) =
            reduce_simplex(&simplex[..count]);
        count = new_count;
        simplex = new_simplex;
        weights = new_weights;
        v = new_v;

        if contains_origin {
            return GjkDistanceResult {
                distance: 0.0,
                point_a: obb_a.center,
                point_b: obb_b.center,
                normal_b_to_a: init_dir,
            };
        }
    }

    let mut pa = Vec3d::ZERO;
    let mut pb = Vec3d::ZERO;
    for i in 0..count {
        pa += simplex[i].sa * weights[i];
        pb += simplex[i].sb * weights[i];
    }
    let diff = pa - pb;
    let dist = diff.length();
    let normal = if dist > 1e-12 { diff / dist } else { init_dir };

    GjkDistanceResult {
        distance: dist,
        point_a: anchor + pa,
        point_b: anchor + pb,
        normal_b_to_a: normal,
    }
}

fn reduce_simplex(pts: &[SupportVertex]) -> (bool, usize, [SupportVertex; 4], [f64; 4], Vec3d) {
    let mut out = [SupportVertex::default(); 4];
    let mut w = [0.0; 4];

    match pts.len() {
        1 => {
            out[0] = pts[0];
            w[0] = 1.0;
            (false, 1, out, w, pts[0].w)
        }
        2 => {
            let (cnt, idxs, bw, v) = closest_segment(pts[0].w, pts[1].w);
            for i in 0..cnt {
                out[i] = pts[idxs[i]];
                w[i] = bw[i];
            }
            (false, cnt, out, w, v)
        }
        3 => {
            let (cnt, idxs, bw, v) = closest_triangle(pts[0].w, pts[1].w, pts[2].w);
            for i in 0..cnt {
                out[i] = pts[idxs[i]];
                w[i] = bw[i];
            }
            (false, cnt, out, w, v)
        }
        4 => closest_tetrahedron(pts),
        _ => unreachable!(),
    }
}

#[inline]
fn closest_segment(a: Vec3d, b: Vec3d) -> (usize, [usize; 2], [f64; 2], Vec3d) {
    let ab = b - a;
    let ab_sq = ab.length_sq();
    if ab_sq <= 1e-24 {
        return (1, [0, 0], [1.0, 0.0], a);
    }
    let t = (-a.dot(ab)) / ab_sq;
    if t <= 0.0 {
        (1, [0, 0], [1.0, 0.0], a)
    } else if t >= 1.0 {
        (1, [1, 0], [1.0, 0.0], b)
    } else {
        (2, [0, 1], [1.0 - t, t], a + ab * t)
    }
}

/// Closest point to origin on triangle `(a, b, c)` (Ericson, Real-Time Collision Detection §5.1.5).
fn closest_triangle(a: Vec3d, b: Vec3d, c: Vec3d) -> (usize, [usize; 3], [f64; 3], Vec3d) {
    let ab = b - a;
    let ac = c - a;
    let ap = -a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (1, [0, 0, 0], [1.0, 0.0, 0.0], a);
    }

    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (1, [1, 0, 0], [1.0, 0.0, 0.0], b);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (2, [0, 1, 0], [1.0 - v, v, 0.0], a + ab * v);
    }

    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (1, [2, 0, 0], [1.0, 0.0, 0.0], c);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (2, [0, 2, 0], [1.0 - w, w, 0.0], a + ac * w);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (2, [1, 2, 0], [1.0 - w, w, 0.0], b + (c - b) * w);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let u = 1.0 - v - w;
    (3, [0, 1, 2], [u, v, w], a * u + b * v + c * w)
}

/// Closest point to origin on tetrahedron `(pts[0..4])` (Ericson §5.1.6).
fn closest_tetrahedron(
    pts: &[SupportVertex],
) -> (bool, usize, [SupportVertex; 4], [f64; 4], Vec3d) {
    let a = pts[0].w;
    let b = pts[1].w;
    let c = pts[2].w;
    let d = pts[3].w;

    let faces: [([usize; 3], usize); 4] = [
        ([0, 1, 2], 3),
        ([0, 2, 3], 1),
        ([0, 3, 1], 2),
        ([1, 3, 2], 0),
    ];

    let mut best_dist_sq = f64::INFINITY;
    let mut best_cnt = 0usize;
    let mut best_out = [SupportVertex::default(); 4];
    let mut best_w = [0.0; 4];
    let mut best_v = Vec3d::ZERO;
    let mut outside_any = false;

    let verts = [a, b, c, d];
    for &(face, opp) in &faces {
        let p0 = verts[face[0]];
        let p1 = verts[face[1]];
        let p2 = verts[face[2]];
        let p_opp = verts[opp];

        let n = (p1 - p0).cross(p2 - p0);
        let sign_origin = (-p0).dot(n);
        let sign_opp = (p_opp - p0).dot(n);

        // Origin is strictly on the opposite side of this face from `p_opp` (outside).
        if sign_origin * sign_opp < 0.0 {
            outside_any = true;
            let (cnt, local_idxs, bw, v) = closest_triangle(p0, p1, p2);
            let d2 = v.length_sq();
            if d2 < best_dist_sq {
                best_dist_sq = d2;
                best_cnt = cnt;
                best_v = v;
                best_w = [0.0; 4];
                for i in 0..cnt {
                    best_out[i] = pts[face[local_idxs[i]]];
                    best_w[i] = bw[i];
                }
            }
        }
    }

    if !outside_any {
        // Origin lies inside or on the boundary of the tetrahedron!
        let mut all = [SupportVertex::default(); 4];
        all.copy_from_slice(&pts[..4]);
        return (true, 4, all, [0.25; 4], Vec3d::ZERO);
    }

    (false, best_cnt, best_out, best_w, best_v)
}
