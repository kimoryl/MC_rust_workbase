//! Double-precision 3D math primitives, rigid transforms, and bounding volumes.
//!
//! All simulation coordinates use `f64` to retain sub-millimeter precision at
//! far-world coordinates (e.g., `|x| >= 2.9e7`), converting to camera-relative
//! `f32` only at the rendering boundary.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Double-precision 3D vector.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Vec3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3d {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    pub const NEG_X: Self = Self {
        x: -1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const NEG_Y: Self = Self {
        x: 0.0,
        y: -1.0,
        z: 0.0,
    };
    pub const NEG_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    };
    pub const MIN: Self = Self {
        x: f64::MIN,
        y: f64::MIN,
        z: f64::MIN,
    };
    pub const MAX: Self = Self {
        x: f64::MAX,
        y: f64::MAX,
        z: f64::MAX,
    };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline]
    pub const fn from_array(a: [f64; 3]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
        }
    }

    #[inline]
    pub const fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    #[inline]
    pub const fn as_f32_array(self) -> [f32; 3] {
        [self.x as f32, self.y as f32, self.z as f32]
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    #[inline]
    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn distance_sq(self, rhs: Self) -> f64 {
        (self - rhs).length_sq()
    }

    #[inline]
    pub fn distance(self, rhs: Self) -> f64 {
        (self - rhs).length()
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let len_sq = self.length_sq();
        if len_sq > 1e-30 && len_sq.is_finite() {
            Some(self / len_sq.sqrt())
        } else {
            None
        }
    }

    #[inline]
    pub fn normalize(self) -> Self {
        self.try_normalize().unwrap_or(Self::ZERO)
    }

    #[inline]
    pub fn normalize_or(self, fallback: Self) -> Self {
        self.try_normalize().unwrap_or(fallback)
    }

    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        Self {
            x: self.x.min(rhs.x),
            y: self.y.min(rhs.y),
            z: self.z.min(rhs.z),
        }
    }

    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        Self {
            x: self.x.max(rhs.x),
            y: self.y.max(rhs.y),
            z: self.z.max(rhs.z),
        }
    }

    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
            z: self.z.clamp(min.z, max.z),
        }
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    #[inline]
    pub fn component_mul(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }

    #[inline]
    pub fn component_div(self, rhs: Self) -> Self {
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
    }

    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        self + (rhs - self) * t
    }

    #[inline]
    pub fn min_element(self) -> f64 {
        self.x.min(self.y).min(self.z)
    }

    #[inline]
    pub fn max_element(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Constructs a right-handed orthonormal tangent pair `(b1, b2)` from a
    /// direction vector using the branchless method of Duff et al. (JCGT 2017).
    #[inline]
    pub fn orthonormal_basis(self) -> (Self, Self) {
        let n = self.normalize_or(Self::Y);
        let sign = 1.0_f64.copysign(n.z);
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;
        let b1 = Self::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        let b2 = Self::new(b, sign + n.y * n.y * a, -n.y);
        (b1, b2)
    }
}

impl Add for Vec3d {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3d {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vec3d {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3d {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<f64> for Vec3d {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3d> for f64 {
    type Output = Vec3d;
    #[inline]
    fn mul(self, rhs: Vec3d) -> Vec3d {
        Vec3d::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}

impl MulAssign<f64> for Vec3d {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Div<f64> for Vec3d {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        let inv = 1.0 / rhs;
        Self::new(self.x * inv, self.y * inv, self.z * inv)
    }
}

impl DivAssign<f64> for Vec3d {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        let inv = 1.0 / rhs;
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

impl Neg for Vec3d {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Index<usize> for Vec3d {
    type Output = f64;
    #[inline]
    fn index(&self, idx: usize) -> &f64 {
        match idx {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3d index out of bounds: {idx}"),
        }
    }
}

impl IndexMut<usize> for Vec3d {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut f64 {
        match idx {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3d index out of bounds: {idx}"),
        }
    }
}

/// Double-precision unit quaternion `(x, y, z, w)` representing a 3D rotation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quatd {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Default for Quatd {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quatd {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn from_axis_angle(axis: Vec3d, angle: f64) -> Self {
        let Some(unit) = axis.try_normalize() else {
            return Self::IDENTITY;
        };
        let half = 0.5 * angle;
        let (s, c) = half.sin_cos();
        Self {
            x: unit.x * s,
            y: unit.y * s,
            z: unit.z * s,
            w: c,
        }
    }

    /// Constructs a quaternion from intrinsic Z-Y-X Euler angles
    /// (`q = q_z(yaw_z) * q_y(pitch_y) * q_x(roll_x)`).
    #[inline]
    pub fn from_euler_zyx(yaw_z: f64, pitch_y: f64, roll_x: f64) -> Self {
        let (sz, cz) = (0.5 * yaw_z).sin_cos();
        let (sy, cy) = (0.5 * pitch_y).sin_cos();
        let (sx, cx) = (0.5 * roll_x).sin_cos();
        Self {
            x: sx * cy * cz - cx * sy * sz,
            y: cx * sy * cz + sx * cy * sz,
            z: cx * cy * sz - sx * sy * cz,
            w: cx * cy * cz + sx * sy * sz,
        }
        .normalize()
    }

    /// Constructs the shortest-arc rotation mapping unit direction `from` to `to`.
    #[inline]
    pub fn from_rotation_arc(from: Vec3d, to: Vec3d) -> Self {
        let Some(a) = from.try_normalize() else {
            return Self::IDENTITY;
        };
        let Some(b) = to.try_normalize() else {
            return Self::IDENTITY;
        };
        let d = a.dot(b);
        if d >= 1.0 - 1e-12 {
            return Self::IDENTITY;
        }
        if d <= -1.0 + 1e-12 {
            let (ortho, _) = a.orthonormal_basis();
            return Self::new(ortho.x, ortho.y, ortho.z, 0.0);
        }
        let c = a.cross(b);
        Self::new(c.x, c.y, c.z, 1.0 + d).normalize()
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    #[inline]
    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len_sq = self.length_sq();
        if len_sq > 1e-30 && len_sq.is_finite() {
            let inv = 1.0 / len_sq.sqrt();
            Self {
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
                w: self.w * inv,
            }
        } else {
            Self::IDENTITY
        }
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let len_sq = self.length_sq();
        if len_sq > 1e-30 && len_sq.is_finite() {
            let inv = 1.0 / len_sq;
            Self {
                x: -self.x * inv,
                y: -self.y * inv,
                z: -self.z * inv,
                w: self.w * inv,
            }
        } else {
            Self::IDENTITY
        }
    }

    #[inline]
    pub fn mul_quat(self, rhs: Self) -> Self {
        Self {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
    }

    #[inline]
    pub fn rotate_vec3(self, v: Vec3d) -> Vec3d {
        let qv = Vec3d::new(self.x, self.y, self.z);
        let t = qv.cross(v) * 2.0;
        v + t * self.w + qv.cross(t)
    }

    #[inline]
    pub fn inverse_rotate_vec3(self, v: Vec3d) -> Vec3d {
        self.conjugate().rotate_vec3(v)
    }

    #[inline]
    pub fn to_mat3(self) -> Mat3d {
        let x2 = self.x + self.x;
        let y2 = self.y + self.y;
        let z2 = self.z + self.z;

        let xx = self.x * x2;
        let xy = self.x * y2;
        let xz = self.x * z2;
        let yy = self.y * y2;
        let yz = self.y * z2;
        let zz = self.z * z2;
        let wx = self.w * x2;
        let wy = self.w * y2;
        let wz = self.w * z2;

        Mat3d {
            cols: [
                Vec3d::new(1.0 - (yy + zz), xy + wz, xz - wy),
                Vec3d::new(xy - wz, 1.0 - (xx + zz), yz + wx),
                Vec3d::new(xz + wy, yz - wx, 1.0 - (xx + yy)),
            ],
        }
    }

    /// Exact exponential-map orientation integration for world-space angular
    /// velocity `omega` over time step `dt`: `q' = exp(0.5 * omega * dt) * q`.
    #[inline]
    pub fn integrate_angular_velocity(self, omega: Vec3d, dt: f64) -> Self {
        let dtheta = omega * dt;
        let theta_sq = dtheta.length_sq();
        if theta_sq <= 1e-32 {
            return self.normalize();
        }
        let dq = if theta_sq < 1e-12 {
            let s = 0.5 - theta_sq / 48.0;
            let c = 1.0 - theta_sq / 8.0;
            Self::new(dtheta.x * s, dtheta.y * s, dtheta.z * s, c)
        } else {
            let theta = theta_sq.sqrt();
            let half = 0.5 * theta;
            let s = half.sin() / theta;
            let c = half.cos();
            Self::new(dtheta.x * s, dtheta.y * s, dtheta.z * s, c)
        };
        dq.mul_quat(self).normalize()
    }

    /// Shortest-arc spherical linear interpolation between `self` and `rhs`.
    #[inline]
    pub fn slerp(self, rhs: Self, t: f64) -> Self {
        let mut dot = self.dot(rhs);
        let mut target = rhs;
        if dot < 0.0 {
            dot = -dot;
            target = Self::new(-rhs.x, -rhs.y, -rhs.z, -rhs.w);
        }
        dot = dot.clamp(0.0, 1.0);
        if dot > 0.999_999_5 {
            return Self::new(
                self.x + (target.x - self.x) * t,
                self.y + (target.y - self.y) * t,
                self.z + (target.z - self.z) * t,
                self.w + (target.w - self.w) * t,
            )
            .normalize();
        }
        let theta = dot.acos();
        let inv_sin = 1.0 / theta.sin();
        let w0 = ((1.0 - t) * theta).sin() * inv_sin;
        let w1 = (t * theta).sin() * inv_sin;
        Self::new(
            self.x * w0 + target.x * w1,
            self.y * w0 + target.y * w1,
            self.z * w0 + target.z * w1,
            self.w * w0 + target.w * w1,
        )
        .normalize()
    }

    /// Converts the shortest-arc rotation into an axis-angle rotation vector `theta * axis`.
    #[inline]
    pub fn to_rotation_vector(self) -> Vec3d {
        let q = if self.w < 0.0 {
            Self::new(-self.x, -self.y, -self.z, -self.w)
        } else {
            self
        };
        let v = Vec3d::new(q.x, q.y, q.z);
        let v_len = v.length();
        if v_len < 1e-12 {
            v * 2.0
        } else {
            let angle = 2.0 * v_len.atan2(q.w);
            v * (angle / v_len)
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

impl Mul<Quatd> for Quatd {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_quat(rhs)
    }
}

/// Column-major double-precision 3x3 matrix.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Mat3d {
    pub cols: [Vec3d; 3],
}

impl Mat3d {
    pub const ZERO: Self = Self {
        cols: [Vec3d::ZERO, Vec3d::ZERO, Vec3d::ZERO],
    };
    pub const IDENTITY: Self = Self {
        cols: [Vec3d::X, Vec3d::Y, Vec3d::Z],
    };

    #[inline]
    pub const fn from_cols(c0: Vec3d, c1: Vec3d, c2: Vec3d) -> Self {
        Self { cols: [c0, c1, c2] }
    }

    #[inline]
    pub const fn from_rows(r0: Vec3d, r1: Vec3d, r2: Vec3d) -> Self {
        Self {
            cols: [
                Vec3d::new(r0.x, r1.x, r2.x),
                Vec3d::new(r0.y, r1.y, r2.y),
                Vec3d::new(r0.z, r1.z, r2.z),
            ],
        }
    }

    #[inline]
    pub const fn from_diagonal(diag: Vec3d) -> Self {
        Self {
            cols: [
                Vec3d::new(diag.x, 0.0, 0.0),
                Vec3d::new(0.0, diag.y, 0.0),
                Vec3d::new(0.0, 0.0, diag.z),
            ],
        }
    }

    /// Skew-symmetric cross-product matrix `[v]_x` such that `Mat3d::skew(v) * w == v.cross(w)`.
    #[inline]
    pub const fn skew(v: Vec3d) -> Self {
        Self {
            cols: [
                Vec3d::new(0.0, v.z, -v.y),
                Vec3d::new(-v.z, 0.0, v.x),
                Vec3d::new(v.y, -v.x, 0.0),
            ],
        }
    }

    #[inline]
    pub const fn col(&self, i: usize) -> Vec3d {
        self.cols[i]
    }

    #[inline]
    pub fn row(&self, i: usize) -> Vec3d {
        Vec3d::new(self.cols[0][i], self.cols[1][i], self.cols[2][i])
    }

    #[inline]
    pub const fn transpose(self) -> Self {
        Self::from_rows(self.cols[0], self.cols[1], self.cols[2])
    }

    #[inline]
    pub fn mul_vec3(self, v: Vec3d) -> Vec3d {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z
    }

    #[inline]
    pub fn mul_mat3(self, rhs: Self) -> Self {
        Self {
            cols: [
                self.mul_vec3(rhs.cols[0]),
                self.mul_vec3(rhs.cols[1]),
                self.mul_vec3(rhs.cols[2]),
            ],
        }
    }

    #[inline]
    pub fn determinant(self) -> f64 {
        self.cols[0].dot(self.cols[1].cross(self.cols[2]))
    }

    /// Robust 3x3 matrix inversion with scale-aware singularity protection.
    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        let r0 = self.cols[1].cross(self.cols[2]);
        let r1 = self.cols[2].cross(self.cols[0]);
        let r2 = self.cols[0].cross(self.cols[1]);
        let det = self.cols[0].dot(r0);
        let norm_sq =
            self.cols[0].length_sq() + self.cols[1].length_sq() + self.cols[2].length_sq();
        let eps = (norm_sq * norm_sq.sqrt() * 1e-15).max(1e-30);
        if !det.is_finite() || det.abs() <= eps {
            return None;
        }
        let inv_det = 1.0 / det;
        let inv = Self::from_rows(r0 * inv_det, r1 * inv_det, r2 * inv_det);
        if inv.is_finite() { Some(inv) } else { None }
    }

    /// Similarity transform `R * local_tensor * R^T` used to rotate inertia tensors into world space.
    #[inline]
    pub fn similarity_transform(rotation: Self, local_tensor: Self) -> Self {
        rotation
            .mul_mat3(local_tensor)
            .mul_mat3(rotation.transpose())
    }

    /// Symmetrizes a matrix as `0.5 * (M + M^T)` to remove floating-point drift on symmetric tensors.
    #[inline]
    pub fn symmetrize(self) -> Self {
        (self + self.transpose()) * 0.5
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self {
            cols: [self.cols[0].abs(), self.cols[1].abs(), self.cols[2].abs()],
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.cols[0].is_finite() && self.cols[1].is_finite() && self.cols[2].is_finite()
    }
}

impl Add for Mat3d {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            cols: [
                self.cols[0] + rhs.cols[0],
                self.cols[1] + rhs.cols[1],
                self.cols[2] + rhs.cols[2],
            ],
        }
    }
}

impl AddAssign for Mat3d {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.cols[0] += rhs.cols[0];
        self.cols[1] += rhs.cols[1];
        self.cols[2] += rhs.cols[2];
    }
}

impl Sub for Mat3d {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            cols: [
                self.cols[0] - rhs.cols[0],
                self.cols[1] - rhs.cols[1],
                self.cols[2] - rhs.cols[2],
            ],
        }
    }
}

impl SubAssign for Mat3d {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.cols[0] -= rhs.cols[0];
        self.cols[1] -= rhs.cols[1];
        self.cols[2] -= rhs.cols[2];
    }
}

impl Mul<f64> for Mat3d {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            cols: [self.cols[0] * rhs, self.cols[1] * rhs, self.cols[2] * rhs],
        }
    }
}

impl Mul<Vec3d> for Mat3d {
    type Output = Vec3d;
    #[inline]
    fn mul(self, rhs: Vec3d) -> Vec3d {
        self.mul_vec3(rhs)
    }
}

impl Mul<Mat3d> for Mat3d {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_mat3(rhs)
    }
}

/// Rigid-body double-precision Euclidean transform (translation + rotation).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Transformd {
    pub position: Vec3d,
    pub rotation: Quatd,
}

impl Transformd {
    pub const IDENTITY: Self = Self {
        position: Vec3d::ZERO,
        rotation: Quatd::IDENTITY,
    };

    #[inline]
    pub const fn new(position: Vec3d, rotation: Quatd) -> Self {
        Self { position, rotation }
    }

    #[inline]
    pub const fn from_position(position: Vec3d) -> Self {
        Self {
            position,
            rotation: Quatd::IDENTITY,
        }
    }

    #[inline]
    pub fn transform_point(&self, p: Vec3d) -> Vec3d {
        self.position + self.rotation.rotate_vec3(p)
    }

    #[inline]
    pub fn inverse_transform_point(&self, p: Vec3d) -> Vec3d {
        self.rotation.inverse_rotate_vec3(p - self.position)
    }

    #[inline]
    pub fn transform_dir(&self, d: Vec3d) -> Vec3d {
        self.rotation.rotate_vec3(d)
    }

    #[inline]
    pub fn inverse_transform_dir(&self, d: Vec3d) -> Vec3d {
        self.rotation.inverse_rotate_vec3(d)
    }

    #[inline]
    pub fn mul_transform(&self, child: &Self) -> Self {
        Self {
            position: self.transform_point(child.position),
            rotation: (self.rotation * child.rotation).normalize(),
        }
    }

    #[inline]
    pub fn inverse(&self) -> Self {
        let inv_rot = self.rotation.conjugate();
        Self {
            position: inv_rot.rotate_vec3(-self.position),
            rotation: inv_rot,
        }
    }

    #[inline]
    pub fn interpolate(&self, other: &Self, alpha: f64) -> Self {
        Self {
            position: self.position.lerp(other.position, alpha),
            rotation: self.rotation.slerp(other.rotation, alpha),
        }
    }

    /// Exports a camera-relative row-major 3x4 `f32` affine matrix
    /// `[r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]`.
    ///
    /// Translation subtraction is performed in `f64` prior to casting to `f32`
    /// so far-world coordinates (`|x| > 1e7`) suffer zero floating-point jitter.
    #[inline]
    pub fn to_camera_relative_affine_f32(
        &self,
        origin_world: Vec3d,
        local_offset: Vec3d,
    ) -> [f32; 12] {
        let rot = self.rotation.to_mat3();
        let rel = (self.position - origin_world) + self.rotation.rotate_vec3(local_offset);
        [
            rot.cols[0].x as f32,
            rot.cols[1].x as f32,
            rot.cols[2].x as f32,
            rel.x as f32,
            rot.cols[0].y as f32,
            rot.cols[1].y as f32,
            rot.cols[2].y as f32,
            rel.y as f32,
            rot.cols[0].z as f32,
            rot.cols[1].z as f32,
            rot.cols[2].z as f32,
            rel.z as f32,
        ]
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.position.is_finite() && self.rotation.is_finite()
    }
}

/// Double-precision axis-aligned bounding box.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Aabbd {
    pub min: Vec3d,
    pub max: Vec3d,
}

impl Default for Aabbd {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Aabbd {
    pub const EMPTY: Self = Self {
        min: Vec3d::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        max: Vec3d::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
    };

    #[inline]
    pub const fn new(min: Vec3d, max: Vec3d) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn from_center_half_extents(center: Vec3d, half_extents: Vec3d) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    #[inline]
    pub fn center(&self) -> Vec3d {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn extents(&self) -> Vec3d {
        self.max - self.min
    }

    #[inline]
    pub fn half_extents(&self) -> Vec3d {
        (self.max - self.min) * 0.5
    }

    #[inline]
    pub fn surface_area(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let d = self.extents();
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    #[inline]
    pub fn volume(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let d = self.extents();
        d.x * d.y * d.z
    }

    #[inline]
    pub fn expand_margin(self, margin: f64) -> Self {
        let m = Vec3d::splat(margin);
        Self {
            min: self.min - m,
            max: self.max + m,
        }
    }

    #[inline]
    pub fn expand_point(self, p: Vec3d) -> Self {
        Self {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    #[inline]
    pub fn translate(self, delta: Vec3d) -> Self {
        Self {
            min: self.min + delta,
            max: self.max + delta,
        }
    }

    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    #[inline]
    pub fn contains_aabb(&self, other: &Self) -> bool {
        self.min.x <= other.min.x
            && self.min.y <= other.min.y
            && self.min.z <= other.min.z
            && self.max.x >= other.max.x
            && self.max.y >= other.max.y
            && self.max.z >= other.max.z
    }

    #[inline]
    pub fn contains_point(&self, p: Vec3d) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Slab-method ray-AABB intersection returning `(distance, outward_normal)`.
    #[inline]
    pub fn ray_intersect(&self, origin: Vec3d, dir: Vec3d, max_dist: f64) -> Option<(f64, Vec3d)> {
        ray_intersect_aabb(origin, dir, max_dist, self)
    }
}

/// Double-precision oriented bounding box.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Obbd {
    pub center: Vec3d,
    pub half_extents: Vec3d,
    pub orientation: Quatd,
}

impl Obbd {
    #[inline]
    pub const fn new(center: Vec3d, half_extents: Vec3d, orientation: Quatd) -> Self {
        Self {
            center,
            half_extents,
            orientation,
        }
    }

    #[inline]
    pub fn from_aabb(aabb: &Aabbd) -> Self {
        Self {
            center: aabb.center(),
            half_extents: aabb.half_extents(),
            orientation: Quatd::IDENTITY,
        }
    }

    #[inline]
    pub fn axes(&self) -> [Vec3d; 3] {
        self.orientation.to_mat3().cols
    }

    #[inline]
    pub fn to_aabb(&self) -> Aabbd {
        let abs_rot = self.orientation.to_mat3().abs();
        let world_he = abs_rot.mul_vec3(self.half_extents);
        Aabbd::from_center_half_extents(self.center, world_he)
    }

    /// Extreme vertex of this OBB along world direction `dir`.
    #[inline]
    pub fn support(&self, dir: Vec3d) -> Vec3d {
        let local_dir = self.orientation.inverse_rotate_vec3(dir);
        let local_pt = Vec3d::new(
            self.half_extents.x.copysign(local_dir.x),
            self.half_extents.y.copysign(local_dir.y),
            self.half_extents.z.copysign(local_dir.z),
        );
        self.center + self.orientation.rotate_vec3(local_pt)
    }

    /// Ray-OBB intersection returning `(distance, outward_normal)` in world space.
    #[inline]
    pub fn ray_intersect(&self, origin: Vec3d, dir: Vec3d, max_dist: f64) -> Option<(f64, Vec3d)> {
        ray_intersect_obb(origin, dir, max_dist, self)
    }
}

/// Intersects a ray `origin + t * dir` (`0 <= t <= max_dist`) against an AABB.
/// Returns `Some((t, normal))` for the nearest entry hit (or `t = 0` if starting inside).
pub fn ray_intersect_aabb(
    origin: Vec3d,
    dir: Vec3d,
    max_dist: f64,
    aabb: &Aabbd,
) -> Option<(f64, Vec3d)> {
    if aabb.is_empty() || max_dist < 0.0 || !dir.is_finite() || !origin.is_finite() {
        return None;
    }

    let mut t_min = 0.0_f64;
    let mut t_max = max_dist;
    let mut hit_axis = usize::MAX;
    let mut hit_sign = 0.0_f64;

    for axis in 0..3 {
        let o = origin[axis];
        let d = dir[axis];
        let bmin = aabb.min[axis];
        let bmax = aabb.max[axis];

        if d.abs() < 1e-15 {
            if o < bmin || o > bmax {
                return None;
            }
        } else {
            let inv_d = 1.0 / d;
            let mut t1 = (bmin - o) * inv_d;
            let mut t2 = (bmax - o) * inv_d;
            let mut sign = -1.0;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
                sign = 1.0;
            }
            if t1 > t_min {
                t_min = t1;
                hit_axis = axis;
                hit_sign = sign;
            }
            if t2 < t_max {
                t_max = t2;
            }
            if t_min > t_max {
                return None;
            }
        }
    }

    let normal = if hit_axis < 3 {
        let mut n = Vec3d::ZERO;
        n[hit_axis] = hit_sign;
        n
    } else {
        // Ray started inside the AABB: return outward normal toward closest face.
        let c = aabb.center();
        let he = aabb.half_extents();
        let rel = origin - c;
        let dx = he.x - rel.x.abs();
        let dy = he.y - rel.y.abs();
        let dz = he.z - rel.z.abs();
        if dx <= dy && dx <= dz {
            Vec3d::new(1.0_f64.copysign(rel.x), 0.0, 0.0)
        } else if dy <= dz {
            Vec3d::new(0.0, 1.0_f64.copysign(rel.y), 0.0)
        } else {
            Vec3d::new(0.0, 0.0, 1.0_f64.copysign(rel.z))
        }
    };

    Some((t_min, normal))
}

/// Intersects a ray `origin + t * dir` (`0 <= t <= max_dist`) against an OBB.
/// Returns `Some((t, world_normal))`.
pub fn ray_intersect_obb(
    origin: Vec3d,
    dir: Vec3d,
    max_dist: f64,
    obb: &Obbd,
) -> Option<(f64, Vec3d)> {
    let local_origin = obb.orientation.inverse_rotate_vec3(origin - obb.center);
    let local_dir = obb.orientation.inverse_rotate_vec3(dir);
    let local_aabb = Aabbd::new(-obb.half_extents, obb.half_extents);
    let (t, local_normal) = ray_intersect_aabb(local_origin, local_dir, max_dist, &local_aabb)?;
    let world_normal = obb.orientation.rotate_vec3(local_normal).normalize();
    Some((t, world_normal))
}
