# MC_rust_workbase

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](./LICENSE-MIT)
[![Target: Horizon 26.3](https://img.shields.io/badge/Horizon-Minecraft%2026.3%20Rust-orange.svg)](./pack/horizon-modpack.toml)
[![Clean-Room Verified](https://img.shields.io/badge/compliance-Clean--Room%20Gate-brightgreen.svg)](./docs/CLEAN_ROOM_COMPLIANCE.md)

基于 **Horizon**（Minecraft: Java Edition 26.3 原生 Rust 实现）构建的开源双许可整合包与子网格刚体物理引擎工程。

本仓库同时提供：
- **`horizon_physics_core`**：100% 纯 Rust、零外部 crates.io 依赖的 Clean-Room 刚体与方块子网格（Sub-Grid）物理内核；
- **`horizon_physics`**：面向 Horizon 26.3 SDK 的原生桥接 Mod，提供子网格组装、实体随动、合批渲染与第三方载具/航空学 Mod 扩展接口；
- **`workbase_core`**：内容 Mod 脚手架（方块/物品/配方/战利品）；
- **`mc_rust_workbase` (`.hpack`)**：整合包构建、红队合规门禁与开发调试全套工具链。

---

## 为什么选择 100% Clean-Room 开源替代？

在 Minecraft 物理与载具生态中，部分既有闭源或限制性源码可用（Source-Available）方案采用了如 **PolyForm Shield 1.0.0** 等带有竞争性限制的许可证。这类条款不属于 OSI 认可的标准开源许可证，会给社区二次开发、衍生航空/机械 Mod 生态、商业服务器与开源整合包分发带来严重的合规传染与法律不确定性。

`MC_rust_workbase` 与 `horizon_physics` 采用 **100% Clean-Room（洁净室）独立工程实现**：
- **纯公开学术文献推导**：所有碰撞检测、惯量张量、求解器与时间步推进算法均基于公开教科书与 GDC/SIGGRAPH 经典论文独立编写；
- **标准 OSI 双许可**：全仓采用与 Rust 生态完全一致的 **`MIT OR Apache-2.0`** 双许可，允许任意社区 Mod、开源整合包与商业项目安全依赖；
- **自动化红队合规门禁**：根目录内置零依赖集成测试 [tests/redteam_compliance.rs](./tests/redteam_compliance.rs)，在 CI 与本地提交前强制拦截许可证缺失、违规依赖、路径逃逸、非法 `#![forbid(unsafe_code)]` 与 SDK Commit 漂移。

详细法律与工程隔离规范请参阅：[Clean-Room 开源合规白皮书](./docs/CLEAN_ROOM_COMPLIANCE.md)。

---

## 核心架构特性

| 模块层级 | 核心设计 | 工程收益 |
| --- | --- | --- |
| **混合精度空间坐标** | `f64` 世界位置 + `Quatd` 单位四元数 + Pair-Anchor 相对局部化 | 彻底消除高坐标（$10^6+$ 格）下的浮点抖动与旋转漂移 |
| **内存与数据布局** | 刚体状态 SoA（Structure-of-Arrays）紧凑数组 + 稳定代际句柄 | 提升 CPU 缓存命中率，便于 SIMD 向量化与无锁只读快照 |
| **稀疏体素碰撞体** | $16^3$ 稀疏砖块（Sparse Brick）+ 增量贪心盒合并（Greedy Meshing） | 船体/飞艇破坏或增建方块时局部 $O(k)$ 重建，大幅减少凸包数量 |
| **惯量动态维护** | 平行轴定理（Parallel-Axis Theorem）$O(1)$ 增量更新质心与惯量张量 | 船体实时增删数百方块无需全量重扫 |
| **两级加速结构** | 粗相 $16\text{m}$ 空间哈希网格 + 细相局部 SAH BVH | 稳定支撑数百子网格与数千合并盒碰撞查询 |
| **接触生成与求解** | SAT 分离轴 + Sutherland-Hodgman 多边形流形裁切 + TGS / XPBD 求解器 | 稳定堆叠、低穿透、同时支持刚性接触与柔性关节约束 |
| **性能与防隧穿** | 基于接触图的休眠岛屿（Sleeping Islands）+ TOI 保守推进（Conservative Advancement） | 静止船只零 CPU 开销，高速炮弹/战机不穿模 |
| **Horizon 引擎桥接** | Sub-Grid 玩家/实体随动变换 + Horizon 合批矩阵渲染 + 第三方 Trait 注册表 | 无缝兼容 Horizon 26.3 原生客户端/服务端与 `FakeHost` 离线测试 |

完整技术细节与数学推导见：[系统架构设计文档](./docs/ARCHITECTURE.md)。

---

## 目录结构

```text
MC_rust_workbase/
├── Cargo.toml                           # 根级零依赖合规门禁包（避免离网强绑 .horizon/sdk）
├── LICENSE-MIT                          # 标准 MIT 许可证全文
├── LICENSE-APACHE                       # 标准 Apache-2.0 许可证全文
├── README.md                            # 项目全景说明与开发手册
├── docs/
│   ├── ARCHITECTURE.md                  # 物理引擎架构、数学公式与 Horizon 桥接详述
│   ├── THIRD_PARTY_MOD_API.md           # 航空学/载具/机械 Mod 实战接入指南
│   └── CLEAN_ROOM_COMPLIANCE.md         # Clean-Room 开源合规白皮书与文献溯源
├── tests/
│   └── redteam_compliance.rs            # 自动化静态合规红队测试（支持 STRICT_FULL_AUDIT=1）
├── pack/
│   ├── horizon-modpack.toml             # .hpack 整合包清单（format = 1, game = "26.3"）
│   └── overrides/                       # 允许覆写的客户端配置文件与资源包（保留 .gitkeep）
├── mods/
│   ├── workbase_core/                   # 核心内容 Mod 脚手架（.hmod）
│   └── horizon_physics/                 # 原生物理桥接 Mod（SDK 固定 Pin）
│       ├── Cargo.toml
│       ├── horizon-mod.toml
│       └── crates/
│           └── horizon_physics_core/    # 零外部依赖纯 Rust 物理内核
├── scripts/
│   ├── build_pack.ps1                   # 全链路构建：内核 Release 测试 + Mod 校验 + .hpack 打包
│   ├── build-pack.ps1                   # 兼容旧版脚本：快速构建 Mod 与整合包
│   ├── verify_redteam.ps1               # 一键红队门禁：合规 + 内核单测/基准 + FakeHost + check
│   └── run-dev.ps1                      # 加载 MSVC 2022 并用本地 horizon-client 调试 Mod
├── port_work/                           # Java 整合包导入与分析工作区（Git 忽略大产物）
└── dist/                                # 构建产物输出目录（*.hmod / *.hpack）
```

> **为什么根 `Cargo.toml` 不把 `mods/horizon_physics` 放进 `[workspace].members`？**
> `mods/horizon_physics` 依赖由 `cargo horizon-mod` 拉取的 `.horizon/sdk/`；在离网 CI 或首次拉取代码尚未初始化 SDK 时，将其硬编码进根工作区会导致 `cargo metadata` 直接报错。根包保持零依赖独立运行合规审计，由 `scripts/build_pack.ps1` 与 `scripts/verify_redteam.ps1` 分阶段编排内核、Mod 与整合包。

---

## 快速开始：一键编译、测试与打包

### 0. 环境准备（Windows + MSVC 2022 / 可选 Linux 离线内核开发）

> - **纯物理内核与合规门禁**：`horizon_physics_core` 与根合规测试套件零外部依赖，可在任意 Windows / Linux Rust 环境中直接编译与测试，无需 Horizon SDK。
> - **完整 Mod 打包与联调**：需配置 MSVC 2022 工具链与 `cargo horizon-mod` / Horizon 26.3 本地 SDK 缓存（`.horizon/sdk/`）。

```powershell
# 可选：若上层工作区提供环境脚本，或通过 $env:HORIZON_DEV_ENV 指定 MSVC/Horizon 环境
if (Test-Path "..\enter-dev-env.ps1") { . "..\enter-dev-env.ps1" }
```

### 1. 运行根级静态合规测试

```powershell
# 常规模式（对当前已存在模块执行合规扫描）
cargo test --test redteam_compliance

# 严格全仓模式（强制要求内核、桥接 Mod、文档、脚本全部就绪）
$env:STRICT_FULL_AUDIT = "1"
cargo test --test redteam_compliance
```

### 2. 一键全链路红队门禁验收

```powershell
# 依次执行：STRICT_FULL_AUDIT 合规 -> 内核单测与 Release 基准 -> FakeHost 集成测试 -> horizon-mod check
.\scripts\verify_redteam.ps1

# 可选：导出结构化 JSON 门禁报告供 CI 存档
.\scripts\verify_redteam.ps1 -JsonReport .\dist\redteam-report.json
```

### 3. 一键构建 Mod 与打包整合包（`.hpack`）

```powershell
# 构建 horizon_physics_core (--release)、校验 workbase_core 与 horizon_physics 并生成 .hpack
.\scripts\build_pack.ps1

# 构建完成后自动将真实 .hmod SHA-256 回填至 pack/horizon-modpack.toml 并重新打包校验
.\scripts\build_pack.ps1 -UpdateManifest
```

### 4. 本地客户端快速调试

```powershell
.\scripts\run-dev.ps1 -ModName workbase_core
.\scripts\run-dev.ps1 -ModName horizon_physics
```

### 5. 新建子 Mod 与 Java 整合包迁移工具

```powershell
# 新建内容 / 原生 / WebAssembly Mod
cargo horizon-mod new mods/<mod_id> --template content
cargo horizon-mod new mods/<mod_id> --template native
cargo horizon-mod new mods/<mod_id> --template wasm

# 导入分析 Java 整合包（.mrpack / CurseForge / Prism）
horizon-port-pack import "<整合包路径>" .\port_work\target_pack
horizon-port-pack inventory .\port_work\target_pack
horizon-port-pack report .\port_work\target_pack
```

---

## 第三方 Mod 接入

如果你正在开发基于 Horizon 26.3 的**航空学、船舶、铁道或机械动力 Mod**，请阅读：
- [第三方 Mod 接入实战指南（`ForceProvider` / `Propulsion` / `JointConstraint` / `PhysicsRaycast` / `SubGridRegistry`）](./docs/THIRD_PARTY_MOD_API.md)

---

## 开源许可与合规说明

本项目采用 **MIT License** 或 **Apache License, Version 2.0** 双许可，用户与贡献者可任选其一使用：
- [LICENSE-MIT](./LICENSE-MIT)
- [LICENSE-APACHE](./LICENSE-APACHE)

向本仓库提交 Pull Request 前，请确保：
1. 所有贡献代码均为原创或符合 `MIT OR Apache-2.0` 兼容许可，**严禁**引入任何受限制性源码可用许可证（如 PolyForm Shield）约束的代码或派生片段；
2. 已通过 `.\scripts\verify_redteam.ps1` 全部门禁检查且零警告退出。
