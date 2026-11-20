# Clean-Room 开源合规边界白皮书

本文档明确界定 `MC_rust_workbase`（含 `horizon_physics_core`、`horizon_physics`、`workbase_core` 及工具链脚本）的知识产权隔离红线、许可证策略、独立学术参考文献与自动化红队审计机制，以确保整个仓库 **100% 符合 OSI 标准开源双许可（`MIT OR Apache-2.0`）**，不存在任何受限许可证传染或著作权污染风险。

---

## 1. 为何必须规避 PolyForm Shield 1.0.0 等限制性许可证？

在社区体素物理引擎与沙盒载具模组领域，存在若干采用 **PolyForm Shield License 1.0.0**（如 Sable 生态相关项目）或其他 Source-Available（源码可用但非开源）协议发布的项目。本项目**严格禁止**复制、移植、改写或依赖任何受 PolyForm Shield 1.0.0 约束的源码与文本，原因如下：

1. **非 OSI 认证的开源许可证**：PolyForm Shield 1.0.0 含有“非竞争条款（Non-Compete）”，限制用户将软件用于与许可方存在竞争关系的产品或服务，违反了开源定义（OSD 第 6 条：不得歧视特定使用领域）。
2. **与主流开源生态不兼容**：PolyForm Shield 无法与 MIT、Apache-2.0、BSD 或 GPL/LGPL 自由组合分发；一旦源码层面发生拷贝、直译（例如将 Java/Kotlin 实现逐行转译为 Rust）或直接依赖，将导致整个下游整合包、服务端与第三方衍生 Mod 面临严重的违约与侵权风险。
3. **生态长期演进风险**：航空学、船舶、铁道与机械动力等第三方 Mod 开发者需要清晰、不可撤销的版权与专利授权。仅采用行业公认的 `MIT OR Apache-2.0` 双许可，才能为整个 Horizon 26.3 中文/国际社区提供零顾虑的基础设施底座。

> **合规红线声明**：本文件（`docs/CLEAN_ROOM_COMPLIANCE.md`）与 `README.md` 中的上述提及仅用于声明“**严禁使用、严禁参考、严禁依赖**”的合规边界。仓库内所有 `.rs` 源码与 `.toml` 清单文件均严禁出现任何 PolyForm、Sable 或相关作者标识与依赖项，并由自动化红队测试强制校验。

---

## 2. Clean-Room（洁净室）隔离工程方法论

本仓库物理引擎的研发遵循经典的 **Clean-Room Design（洁净室设计）** 原则：

1. **只定义通用领域需求，不阅读受限实现**：
   - 功能目标仅抽象为计算机图形学与刚体力学中的通用问题（例如：“在 $16^3$ 稀疏体素网格上执行贪心盒合并”、“在双精度坐标下执行 OBB 碰撞检测与 TGS 约束求解”）；
   - 编码与测试人员**不读取、不反编译、不逐行对照**任何 PolyForm Shield 许可仓库的源代码、数据结构命名或私有协议。
2. **基于第一性原理与公开学术文献独立推导**：
   - 所有公式（四元数微分、平行轴定理惯量张量、SAH 划分代价、SAT 15 轴判定、Sutherland-Hodgman 裁剪、TGS/XPBD 拉格朗日乘子、TOI 保守推进）均直接来源于公开教科书、IEEE/ACM 论文与 GDC 公开演讲资料，使用惯用 Rust 2024 风格与 SoA 布局从零实现。
3. **零外部依赖核心 Crate（`horizon_physics_core`）**：
   - `mods/horizon_physics/crates/horizon_physics_core` 强制要求**零 `[dependencies]`**、**零 `[features]`**、**无 `build.rs`**，从供应链源头切断间接引入非兼容许可证传递依赖的可能。

---

## 3. 独立公开数学与算法参考文献清单

`horizon_physics_core` 与 `horizon_physics` 所采用的算法全部源自以下公开学术与工程经典文献：

| 引擎子模块 | 独立公开参考文献 |
| --- | --- |
| **刚体动力学与平行轴惯量** | Brian Mirtich, *"Fast and Accurate Computation of Polyhedral Mass Properties"*, Journal of Graphics Tools, 1(2), 1996；David Baraff, *"An Introduction to Physically Based Modeling: Rigid Body Simulation"*, SIGGRAPH Course Notes, 1997。 |
| **实时碰撞检测 / SAT / OBB / 空间哈希** | Christer Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2004（第 4 章包围体、第 5 章基本几何测试、第 7 章空间划分）。 |
| **多边形接触流形裁切** | Ivan E. Sutherland & Gary W. Hodgman, *"Reentrant Polygon Clipping"*, Communications of the ACM, 17(1):32–42, 1974；Dirk Gregorius, *"The Separating Axis Test between Convex Polyhedra"*, GDC Physics Tutorial, 2013。 |
| **SAH BVH 加速结构** | Ingo Wald, *"On Fast Construction of SAH-based Bounding Volume Hierarchies"*, IEEE Symposium on Interactive Ray Tracing, 2007。 |
| **TGS 脉冲约束求解与热启动** | Erin Catto, *"Iterative Dynamics with Temporal Coherence"*, GDC, 2005；Erin Catto, *"Modeling and Solving Constraints"*, GDC, 2009；Erin Catto, *"Solver2D / Soft Constraints & Sub-Stepping (TGS)"*, Box2D Technical Notes, 2024。 |
| **XPBD 位置级柔度约束** | Miles Macklin, Matthias Müller & Nuttapong Chentanez, *"XPBD: Position-Based Simulation of Compliant Constrained Dynamics"*, MIG / ACM SIGGRAPH, 2016；Matthias Müller et al., *"Detailed Rigid Body Simulation with Extended Position Based Dynamics"*, Eurographics / SCA, 2020。 |
| **连续碰撞检测（TOI 保守推进）** | Brian Mirtich, *"Impulse-Based Dynamic Simulation of Rigid Body Systems"*, PhD Thesis, UC Berkeley, 1996；Erwin Coumans, *"Continuous Collision Detection and Physics"*, GDC, 2005。 |
| **体素贪心合并（Greedy Meshing）** | 通用三维游程与二维最大矩形贪心合并算法（标准计算机图形学体素网格压缩技术）。 |

---

## 4. 双许可策略与贡献者条款

本项目遵循 Rust 官方生态标准双许可模式：**`MIT OR Apache-2.0`**。

- **MIT License**（[LICENSE-MIT](../LICENSE-MIT)）：极简宽松条款，便于学术研究、轻量工具与各类项目直接复用。
- **Apache License 2.0**（[LICENSE-APACHE](../LICENSE-APACHE)）：提供明确的**专利授权条款（Section 3）**与**贡献提交条款（Section 5）**，保护企业和社区用户免受潜在专利诉讼。

### 贡献者声明（Inbound = Outbound）

依据 Apache License 2.0 第 5 条，除非明确书面声明，否则你向本仓库提交的任何 Pull Request 或补丁均默认按 **`MIT OR Apache-2.0`** 双许可授权给项目与全体下游用户，且你确认提交内容完全原创或拥有合法的兼容开源再分发权。

---

## 5. 自动化红队静态合规审计机制

为避免人工审查疏漏，仓库根目录内置了零依赖 Rust 红队审计测试套件 [tests/redteam_compliance.rs](../tests/redteam_compliance.rs) 与自动化门禁脚本 [scripts/verify_redteam.ps1](../scripts/verify_redteam.ps1)。

### 5.1 自动化检查项清单

| 检查编号 | 审计目标 | 强制约束条件 |
| --- | --- | --- |
| **C-01** | `LICENSE-MIT` & `LICENSE-APACHE` & 根 `Cargo.toml` | 必须存在且包含完整官方许可证文本；根包声明 `license = "MIT OR Apache-2.0"` |
| **C-02** | 全仓所有 `.rs` 与 `.toml` 文件 | 严禁出现受限许可证/竞品标识关键字；严禁出现 `#![forbid(unsafe_code)]` |
| **C-03** | `mods/horizon_physics/crates/horizon_physics_core/Cargo.toml` | 零 `[dependencies]`、零 `[features]`、无 `build.rs`、无 `.workspace = true`、无 `..` 路径逃逸、`license = "MIT OR Apache-2.0"` |
| **C-04** | `mods/horizon_physics/Cargo.toml` & `horizon-mod.toml` | 无 `..` 路径逃逸出 Mod 根目录、无 `build.rs`、无自定义 `[features]`、SDK pins 精确锁定 `horizon = "f00125f61f794aaa3d4e950d301f353b35b6ee3d"` 与 `mixins = "e59da9094f0c2fdda6c627fe9eff22dec6a906f0"` |
| **C-05** | `pack/horizon-modpack.toml` | 强制校验 `format = 1`、`game = "26.3"`、必需元数据非空以及全部 `[[mods]]` 条目的 64 位小写 SHA-256 合法性 |
| **C-06** | `STRICT_FULL_AUDIT=1` 严格模式 | 要求全仓核心源码、Mod 清单、文档与脚本全部就绪，防止缺失模块被静默跳过 |

### 5.2 本地与 CI 执行命令

```powershell
# 1. 仅运行根级合规审计（任意环境即可执行，无需 Horizon SDK）
if (Test-Path "..\enter-dev-env.ps1") { . "..\enter-dev-env.ps1" }
cargo test --test redteam_compliance

# 2. 运行全链路红队门禁（自动开启 STRICT_FULL_AUDIT=1）
.\scripts\verify_redteam.ps1
```
