# 第三方 Mod 接入实战指南（`horizon_physics` API）

本指南面向基于 **Horizon 26.3** 开发**航空学、船舶、铁道、装甲载具与机械动力 Mod** 的开发者，介绍如何通过 `horizon_physics` 暴露的稳定扩展接口接入物理引擎，而无需自行维护碰撞检测、惯量计算或渲染同步。

---

## 1. 在 `horizon-mod.toml` 中声明依赖

首先使用 `cargo horizon-mod new mods/my_aero_mod --template native` 创建你的原生 Mod，并在 `mods/my_aero_mod/horizon-mod.toml` 中声明对 `horizon_physics` 的依赖以及与工作区一致的 SDK Commit Pin：

```toml
id = "my_aero_mod"
name = "My Aero Mod"
version = "0.1.0"
description = "基于 horizon_physics 的固定翼航空与螺旋桨推进 Mod"
authors = ["YourName"]
api = "0.2"
side = "both"
capabilities = ["registries"]

[native]
game = "26.3"
horizon = "f00125f61f794aaa3d4e950d301f353b35b6ee3d"
mixins = "e59da9094f0c2fdda6c627fe9eff22dec6a906f0"

[dependencies]
horizon_physics = ">=0.1.0, <0.2.0"
```

> **合规提醒**：
> - 第三方 Mod 的 `Cargo.toml` 中严禁出现指向 Mod 根目录之外的 `..` 路径逃逸，也请避免添加自定义 `build.rs` 与 `#![forbid(unsafe_code)]`（Horizon 原生桥接宏需要允许必要的内部 `unsafe`）。
> - `horizon_physics` 采用 `MIT OR Apache-2.0` 双许可，你的扩展 Mod 可以自由选择任意兼容许可证发布。

---

## 2. 核心对外接口全景

| 接口 / 注册表 | 职责 | 典型应用场景 |
| --- | --- | --- |
| `SubGridRegistry` | 组装、注册、查询与注销体素子网格，动态增删方块 | 船坞组装船体、机库拼装飞机、战损块拆除 |
| `ForceProvider` | 每个物理子步注入连续分布外力与力矩 | 机翼升力与诱导阻力、浮力、风场、磁悬浮 |
| `Propulsion` | 受油门/控制信号驱动的推进器抽象 | 螺旋桨、涡扇发动机、火箭矢量喷口、侧推器 |
| `JointConstraint` | 连接两个刚体/子网格的机械自由度约束（XPBD/TGS） | 铰链副翼/方向舵、起落架弹簧减震、万向节炮塔 |
| `PhysicsRaycast` | 跨静态世界与动态子网格的双精度射线检测 | 无线电高度表、火控测距、玩家交互点击船体按钮 |

---

## 3. 使用 `SubGridRegistry` 组装船体与机体

当玩家激活“组装核心”方块时，你的 Mod 只需扫描连通的载具方块集合，提交至 `SubGridRegistry` 即可在 `O(N)` 初始化后交给引擎接管，后续单个方块的破坏或增建将以 `O(1)` 平行轴定理实时更新惯量：

```rust
use mod_horizon_physics::api::{
    BlockMassSpec, SubGridAssemblyDesc, SubGridHandle, SubGridRegistry, VoxelPlacement,
};

/// 将世界坐标中的一组连通方块组装为一个动态物理子网格
pub fn assemble_aircraft_fuselage(
    registry: &mut dyn SubGridRegistry,
    origin_world: [f64; 3],
    blocks: &[([i32; 3], BlockMassSpec)],
) -> Result<SubGridHandle, String> {
    let placements: Vec<VoxelPlacement> = blocks
        .iter()
        .map(|&(local_coord, spec)| VoxelPlacement {
            local_coord,
            mass_kg: spec.mass_kg,
            friction: spec.friction,
            restitution: spec.restitution,
        })
        .collect();

    let desc = SubGridAssemblyDesc {
        world_origin: origin_world,
        initial_rotation: [0.0, 0.0, 0.0, 1.0], // Quatd (x, y, z, w)
        voxels: placements,
        enable_ccd: true, // 高速固定翼建议开启 TOI 保守推进防穿模
    };

    registry.assemble_subgrid(desc)
}

/// 战斗损伤或飞行中抛弃副油箱：O(1) 增量更新质心与惯量张量
pub fn jettison_drop_tank(
    registry: &mut dyn SubGridRegistry,
    grid: SubGridHandle,
    tank_voxels: &[[i32; 3]],
) {
    for &coord in tank_voxels {
        let _ = registry.remove_voxel(grid, coord);
    }
}
```

---

## 4. 实现 `ForceProvider`：气动机翼升阻力模型

`ForceProvider` 在每个物理子步（Sub-Step）都会收到当前子网格的双精度位姿、局部线/角速度与环境采样信息，返回施加在指定局部挂载点（气动压心）上的力与力矩。

下面给出一个简化的薄翼型升阻力计算实现（升力 $L = \frac{1}{2}\rho v^2 S C_L(\alpha)$，阻力 $D = \frac{1}{2}\rho v^2 S C_D(\alpha)$）：

```rust
use mod_horizon_physics::api::{ForceContribution, ForceProvider, ForceQueryContext, SubGridHandle};

pub struct AirfoilWingProvider {
    /// 机翼气动压心在子网格局部坐标系中的位置 (m)
    pub local_cp: [f64; 3],
    /// 局部翼面法向（通常为 +Y 向上）
    pub local_normal: [f64; 3],
    /// 翼面积 S (m^2)
    pub wing_area_m2: f64,
    /// 展弦比 AR（用于估算诱导阻力）
    pub aspect_ratio: f64,
}

impl ForceProvider for AirfoilWingProvider {
    fn provider_id(&self) -> &'static str {
        "my_aero_mod:airfoil_wing"
    }

    fn compute_force(
        &self,
        _grid: SubGridHandle,
        ctx: &ForceQueryContext,
    ) -> Option<ForceContribution> {
        // 1. 计算气动压心处的局部相对气流速度（机体速度 + 旋转切向速度 - 环境风速）
        let v_local = ctx.local_point_velocity(self.local_cp);
        let speed_sq = v_local[0] * v_local[0] + v_local[1] * v_local[1] + v_local[2] * v_local[2];
        if speed_sq < 1e-4 {
            return None;
        }
        let speed = speed_sq.sqrt();

        // 2. 迎角 alpha 近似：airflow = -v_local
        let dir = [-v_local[0] / speed, -v_local[1] / speed, -v_local[2] / speed];
        let sin_alpha = -(dir[0] * self.local_normal[0]
            + dir[1] * self.local_normal[1]
            + dir[2] * self.local_normal[2])
            .clamp(-1.0, 1.0);
        let alpha = sin_alpha.asin();

        // 3. 失速钳位后的升力系数 C_L 与阻力系数 C_D
        let cl = if alpha.abs() < 0.30 {
            (2.0 * std::f64::consts::PI * alpha).clamp(-1.45, 1.45)
        } else {
            // 失速后平滑衰减（sin(2*alpha) 近似）
            1.1 * (2.0 * alpha).sin()
        };
        let cd0 = 0.025;
        let cd = cd0 + (cl * cl) / (std::f64::consts::PI * self.aspect_ratio * 0.85);

        // 4. 随海拔高度衰减的大气密度 rho (kg/m^3)
        let altitude_m = (ctx.world_position[1] - 64.0).max(0.0);
        let rho = 1.225 * (-altitude_m / 8500.0).exp();
        let dynamic_pressure = 0.5 * rho * speed_sq * self.wing_area_m2;

        // 升力方向垂直于气流并与翼面法向共面；阻力与速度反向
        let lift_mag = dynamic_pressure * cl;
        let drag_mag = dynamic_pressure * cd;

        let local_force = [
            self.local_normal[0] * lift_mag + dir[0] * drag_mag,
            self.local_normal[1] * lift_mag + dir[1] * drag_mag,
            self.local_normal[2] * lift_mag + dir[2] * drag_mag,
        ];

        Some(ForceContribution::local_force_at_point(
            local_force,
            self.local_cp,
        ))
    }
}
```

---

## 5. 实现 `Propulsion`：螺旋桨与矢量喷口推力

与被动响应速度的 `ForceProvider` 不同，`Propulsion` 额外封装了**油门指令（Throttle）**、**矢量偏转角**、**反扭矩（Torque Reaction）**以及**燃料/能源状态反馈**，便于统一对接座舱仪表盘与自动驾驶仪：

```rust
use mod_horizon_physics::api::{Propulsion, PropulsionContext, PropulsionOutput, SubGridHandle};

pub struct TurbopropEngine {
    pub local_mount: [f64; 3],
    /// 局部推力方向（例如 [0.0, 0.0, -1.0] 朝机头前方）
    pub local_thrust_axis: [f64; 3],
    pub max_static_thrust_n: f64,
    pub max_airspeed_ms: f64,
    pub throttle: f64, // [0.0, 1.0]
}

impl Propulsion for TurbopropEngine {
    fn propulsion_id(&self) -> &'static str {
        "my_aero_mod:turboprop"
    }

    fn set_throttle(&mut self, throttle: f64) {
        self.throttle = throttle.clamp(0.0, 1.0);
    }

    fn evaluate_thrust(
        &self,
        _grid: SubGridHandle,
        ctx: &PropulsionContext,
    ) -> PropulsionOutput {
        // 螺旋桨推力随前进空速线性衰减，随空气密度衰减
        let v_forward = ctx.local_velocity_along(self.local_thrust_axis).max(0.0);
        let speed_factor = (1.0 - v_forward / self.max_airspeed_ms).clamp(0.0, 1.0);
        let altitude_m = (ctx.world_position[1] - 64.0).max(0.0);
        let air_density_ratio = (-altitude_m / 8500.0).exp();

        let thrust_n = self.max_static_thrust_n * self.throttle * speed_factor * air_density_ratio;
        let local_force = [
            self.local_thrust_axis[0] * thrust_n,
            self.local_thrust_axis[1] * thrust_n,
            self.local_thrust_axis[2] * thrust_n,
        ];

        // 螺旋桨滚转反扭矩（绕推力轴）
        let prop_reaction_torque = thrust_n * 0.04;
        let local_torque = [
            self.local_thrust_axis[0] * prop_reaction_torque,
            self.local_thrust_axis[1] * prop_reaction_torque,
            self.local_thrust_axis[2] * prop_reaction_torque,
        ];

        PropulsionOutput {
            local_mount: self.local_mount,
            local_force,
            local_torque,
            fuel_burn_rate: self.throttle * 0.12,
        }
    }
}
```

---

## 6. 实现 `JointConstraint`：铰链舵面与起落架悬挂

通过向物理世界注册 `JointConstraint`，可以将独立子网格（例如可动方向舵、炮塔、车轮悬挂总成）与主机体刚性或弹性连接。底层 XPBD 位置约束求解器会按柔度系数 $\alpha$ 自动保证高刚度与无漂移：

```rust
use mod_horizon_physics::api::{
    HingeJointDesc, JointConstraint, JointHandle, JointMotor, JointRegistry, SubGridHandle,
};

/// 将方向舵子网格通过带限位与伺服电机的铰链连接到主垂尾上
pub fn attach_rudder_hinge(
    joints: &mut dyn JointRegistry,
    fuselage: SubGridHandle,
    rudder: SubGridHandle,
) -> Option<JointHandle> {
    let hinge = HingeJointDesc {
        body_a: fuselage,
        body_b: rudder,
        anchor_a: [0.0, 2.0, 8.5],     // 机身局部铰接点
        anchor_b: [0.0, 0.0, -0.5],    // 舵面局部铰接点
        axis_a: [0.0, 1.0, 0.0],       // 绕 Y 轴偏航
        axis_b: [0.0, 1.0, 0.0],
        min_angle_rad: -0.45,          // -25.8°
        max_angle_rad: 0.45,           // +25.8°
        compliance: 0.0,               // 0.0 = 绝对刚性铰链（>0 则为弹簧柔度）
        motor: Some(JointMotor {
            target_angle_rad: 0.0,
            max_torque_nm: 8_000.0,
            stiffness: 1_200.0,
            damping: 180.0,
        }),
    };

    joints.add_joint(JointConstraint::Hinge(hinge))
}
```

---

## 7. 使用 `PhysicsRaycast`：跨世界与子网格高精度射线检测

`PhysicsRaycast` 会同时查询静态地形方块与所有活动子网格的局部 SAH BVH，并返回统一的双精度命中坐标、法线及所属 `SubGridHandle`（若命中子网格则附带精确局部体素坐标）：

```rust
use mod_horizon_physics::api::{PhysicsRaycast, RayQuery, RaycastFilter, RaycastHitTarget};

/// 机载雷达高度表：向下发射 256m 射线，自动忽略本机子网格
pub fn sample_radar_altitude(
    raycaster: &dyn PhysicsRaycast,
    aircraft_grid: mod_horizon_physics::api::SubGridHandle,
    sensor_world_pos: [f64; 3],
) -> Option<f64> {
    let query = RayQuery {
        origin: sensor_world_pos,
        direction: [0.0, -1.0, 0.0],
        max_distance: 256.0,
        filter: RaycastFilter::exclude_subgrid(aircraft_grid),
    };

    let hit = raycaster.cast_ray(&query)?;
    match hit.target {
        RaycastHitTarget::WorldBlock { .. } => Some(hit.distance),
        RaycastHitTarget::SubGridVoxel { .. } => Some(hit.distance), // 例如降落到航母甲板上！
    }
}
```

---

## 8. 离线集成测试最佳实践（`FakeHost`）

在开发第三方航空/载具 Mod 时，推荐利用 Horizon SDK 的 `mod_api::testing::{FakeHost, with_mods}` 与 `mod_mod_horizon_physics::api::PhysicsApi`（对应 `[package] name = "mod-horizon_physics"` 的 Rust 导入名）编写毫秒级纯 Rust 自动化测试，无需启动客户端即可验证升力平衡、巡航速度、铰链限位与防穿模：

```rust
use horizon_physics_core::EmptyWorldProvider;
use mod_mod_horizon_physics::api::{
    PhysicsApi, SubGridRegistry, VoxelPlacement,
};

#[test]
fn test_aircraft_cruising_lift_balances_gravity() {
    let mut api = PhysicsApi::default();
    let grid = assemble_aircraft_fuselage(&mut api, [0.0, 120.0, 0.0], &sample_plane_blocks())
        .expect("assembly should succeed");

    api.attach_force_provider(grid, Box::new(AirfoilWingProvider {
        local_cp: [0.0, 0.2, 0.0],
        local_normal: [0.0, 1.0, 0.0],
        wing_area_m2: 18.0,
        aspect_ratio: 7.5,
    }));
    api.attach_propulsion(grid, Box::new(TurbopropEngine {
        local_mount: [0.0, 0.0, -4.0],
        local_thrust_axis: [0.0, 0.0, -1.0],
        max_static_thrust_n: 22_000.0,
        max_airspeed_ms: 90.0,
        throttle: 0.75,
    }));

    // 模拟 5 秒（100 服务端 tick @ 20 TPS）
    let world_provider = EmptyWorldProvider;
    for _ in 0..100 {
        api.step_with_provider(1.0 / 20.0, &world_provider);
    }

    let state = api.subgrid_state(grid).expect("subgrid must remain active");
    assert!(state.world_position[1] > 100.0, "飞机应维持巡航高度而非坠落");
}
```

> 若需测试 Hook 调度、命令分发、RenderPipeline 与实体骑乘补偿等完整宿主链路，可参考 [`tests/fake_host_integration.rs`](../mods/horizon_physics/tests/fake_host_integration.rs) 中的 `mod_api::testing::{FakeHost, with_mods}` 夹具写法。

运行验证：

```powershell
# 若需加载本地 MSVC/Horizon 编译环境，请先配置工具链或导入上层工作区脚本
cargo test
```
