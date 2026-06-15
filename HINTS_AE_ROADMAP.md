# 把 HINTS 应用于波动/Helmholtz 正演并用声发射(AE)实验验证 —— 详细书面方案

> 状态：方案稿 v1（2026-06-14）。本文件只规划、不改动现有求解器代码。
> 目标：把 `HINTS_numpy` 的"DeepONet + 松弛法谱互补"思想，从标量 Poisson/Helmholtz
> 扩展到**频域弹性波(AE)正演**，并用自有声发射实验数据验证。

---

## 0. 一页纸结论（TL;DR）

- **落点**：以**频域弹性 Helmholtz（Navier 方程）**为 HINTS 的加速对象；时域弹性
  FDTD（已存在于 `wave_solvers/elastic_fdtd.py`）作为"真值"做交叉验证。
- **为什么是频域**：HINTS 加速的是 `A u = f` 的迭代解。频域正演天然就是解线性系统；
  时域显式 FDTD 每步不解线性系统，HINTS 无处接入（除非改隐式时间步进，见 §7 备选）。
- **创新点（三合一，互相支撑）**：
  1. **HINTS 首次用于矢量/弹性算子**（标量→耦合 P-S 矢量系统）。
  2. **复数、高波数、吸收边界(Sommerfeld/PML)的 HINTS-CSLP**（原 HINTS-Helmholtz 仅低波数实数）。
  3. **首个用真实 AE 实验验证的 HINTS**（铅芯折断校准源 → 真实裂纹源）。
- **AE 工作流**：震源频谱 → 每个 ω 用 HINTS 解一次弹性 Helmholtz → IFFT 合成传感器
  时域波形 → 与 FDTD 合成解 + 实测 AE 波形比对（到时/频散/幅值）。
- **可靠性风险**：参考 "Are DL-Based Hybrid PDE Solvers Reliable?"(2026)，把训练范式与
  DeepONet/松弛更新策略作为受控变量研究。

---

## 1. 创新定位（与已有工作的差距）

| 已有工作 | 物理 | 范式 | 与本方案的差距 |
| --- | --- | --- | --- |
| HINTS (Zhang et al., NMI 2024) | 标量 Poisson/Helmholtz | DeepONet+松弛，迭代到机器精度 | **标量**；低波数实数 Helmholtz |
| HINTS 几何可迁移 (Comput. Mech. 2023) | 标量 | 同上+几何迁移 | 仍标量 |
| 注意力混合解 (2024)、谱分析混合解 (2024) | 标量 | 改进网络/谱分析 | 仍标量 |
| Helmholtz U-Net 预条件子 (Azulay & Treister, SISC) | 标量声学 | CNN 预条件子+MG | 非 DeepONet；非弹性；无 AE |
| 弹性 Helmholtz 神经算子 (Zou et al., GJI 2024) | **弹性** | **纯代理模型** | 不到机器精度；不是混合迭代解 |
| NOWS 神经算子热启动 (2025) | 一般 | 热启动迭代解 | 非 HINTS 谱互补；无 AE |
| 引导波/SAW SciML 综述 (2025) | 弹性/导波 | PINN/算子综述 | 无 HINTS；无混合求解器 |

**未被占用的交集 = 本方案：**
> 复数高波数 + 吸收边界 + **弹性(矢量)** Helmholtz 的 **HINTS** 混合求解器，
> 并以**真实声发射实验**验证。

---

## 2. 数学与方法

### 2.1 控制方程（频域弹性 / Navier-Helmholtz）
位移场 `u(x) ∈ C^d`（2D 取 `d=2`，分量 `ux, uz`），时谐 `e^{-iωt}`：

```
∇·σ(u) + ρ ω² u = -f,     σ = λ (∇·u) I + μ (∇u + ∇uᵀ)
```

离散后得复数线性系统 `A(ω) u = f`。`A` 随频率 ω 与材料 (λ, μ, ρ) 变化、**强不定**
（高波数时特征值跨虚实、Jacobi 发散）—— 正是 HINTS 的用武之地。

### 2.2 HINTS 的接入方式（两条，都做）
对照 `HINTS_numpy/iterative_solver.py`：

- **(A) 独立混合迭代** `Numerical_DeepONet_Hybrid`（`iterative_solver.py:243`）：
  每 `NUMERICAL_TO_DON_RATIO` 步松弛/CSLP 平滑，插一步 DeepONet 残差校正。
  最贴近原 HINTS，便于复现谱互补曲线。
- **(B) DeepONet 作为 Krylov 预条件子**（对照 `HINTS_petsc`、`wave_solvers/helmholtz.py`
  的 GMRES/BiCGSTAB + CSLP）：把"CSLP 内解 + DeepONet 低频校正"组合成一个
  `LinearOperator` 作为 GMRES 的 M。这是大规模可扩展路线，命名 **HINTS-CSLP**。

### 2.3 DeepONet 的改造要点
现有 `deeponet.py` 是实值、标量输出。弹性频域需要：
1. **复数支持**：实部/虚部双通道，或复权重。最省事：把 `δu` 的 re/im 当两个输出通道。
2. **矢量输出**：分支/主干网络输出 `(δux, δuz)`，共 4 个实通道（re/im × 2 分量）。
3. **分支输入**：材料场 `(vp, vs, ρ)` 或 `(λ, μ, ρ)` + 频率 ω（或无量纲波数 k=ω/c）。
   ω 作为分支输入是**多频摊销**的关键（§4 Stage 3）。
4. **物理单位与缩放**：AE 是 mm/μs/MHz 量级，需无量纲化（特征长度=板厚或波长，
   特征时间=1/中心频率），避免数值病态。

### 2.4 谱互补诊断（必须复现）
沿用 `iterative_solver.py:271` 的 `update_metrics` 思路：对误差做特征模态分解，
画"模态误差 vs 迭代"。要证明：**松弛压高频、DeepONet 压低频、合起来一致收敛**。
弹性情形需分别看 P 模与 S 模 —— 这本身是论文里的新图。

---

## 3. 代码桥接（现状 → 目标）

现状两套独立组装，需统一：

- `HINTS_numpy/utils.py` 自带实数 Helmholtz 组装（HINTS 在其上迭代）。
- `wave_solvers/helmholtz.py` 有复数 Sommerfeld 组装 + CSLP（但没接 HINTS）。
- `wave_solvers/elastic_fdtd.py` 是时域弹性真值。

**目标桥接层**（新增，不破坏现有）：
1. `wave_solvers/elastic_helmholtz.py`（新）：组装复数弹性 Helmholtz `A(ω)`、震源 `f`、
   吸收边界（先 Sommerfeld，后 PML）。
2. `wave_solvers/hints_bridge.py`（新）：把 `A, f` 喂给改造后的 HINTS 迭代器；
   提供 `solve_hints(A, f, deeponet, ratio, ...)` 与 `make_hints_preconditioner(...)`。
3. 复用 `wave_solvers/sources.py` 的 moment tensor / 铅芯折断作为 `f` 的空间+频谱部分。
4. 频域→时域合成：新增 `freq_to_time(...)`（对一组 ω 解，乘源谱，IFFT）。

---

## 4. 分阶段计划（含交付物与验证指标）

> 每阶段都有**可量化验证**，避免"看起来能跑"。

### Stage 0 — 打通基线（0.5 周）
- 跑通 `wave_solvers/tests/test_wave_solvers.py`（9 个测试）、`helmholtz_cslp_vs_jacobi.py`、
  `ae_pencil_break_plate.py`。
- 跑通 `HINTS_numpy` 的 1D Helmholtz 例子（`configs_1D_Helmholtz_HINTS_Jacobi.py`）。
- **验证**：测试全绿；CSLP-GMRES 收敛、Jacobi 在高 k 发散（复现失效模式）；
  铅芯折断例子的 P 波到时与理论 `offset/vp` 吻合。
- **交付**：环境就绪报告 + 基线收敛/波形图。

### Stage 1 — 标量复数 2D Helmholtz 的 HINTS（1.5 周）
- 把 HINTS 从实数 1D 扩到**复数 2D + Sommerfeld**（先标量声学，铺路）。
- DeepONet 加复数(re/im 双通道)支持；分支输入异质 `k(x,y)` + ω。
- 训练数据：板状/块状 `k` 场（含缺陷扰动），用 `solve_direct` 生成真解。
- **验证**：HINTS-CSLP 的 GMRES 迭代数 vs 纯 CSLP 显著下降；解与直接解 L2 误差到 ~1e-8；
  谱互补图成立。
- **交付**：`elastic_helmholtz.py`(标量分支) + `hints_bridge.py` + 训练脚本 + 收敛对比图。

### Stage 2 — 弹性(矢量) Helmholtz 的 HINTS【核心创新】（3 周）
- 组装 2D P-SV 频域 Navier-Helmholtz；DeepONet 输出矢量 `(δux,δuz)` re/im。
- 震源用 moment tensor / double-couple（对接 `sources.py`）。
- **验证**：
  - 频域 HINTS 单频解 vs 频域直接解，L2 → 1e-6~1e-8；
  - **频域→IFFT 合成波形 vs 时域 `elastic_fdtd.py` 波形**：到时、波形互相关 > 0.95；
  - 迭代数/壁钟时间 vs 纯 CSLP-GMRES 与直接解的对比表；
  - P 模/S 模分别的谱互补图。
- **交付**：弹性 `elastic_helmholtz.py` + 矢量 DeepONet + 交叉验证报告（这是论文主图）。

### Stage 3 — 多频摊销与板中导波/Lamb（2 周）
- 一张 DeepONet 跨整个 AE 频带泛化（ω 作分支输入），在 IFFT 合成里复用同一算子。
- 薄板：复现 Lamb 频散（与解析频散曲线对比）。
- **验证**：跨频带平均迭代数稳定；合成频散曲线与 Rayleigh-Lamb 解析解吻合。
- **交付**：多频求解器 + 频散验证图。

### Stage 4 — 真实 AE 实验验证（3+ 周，依赖你的数据）
见 §5。

---

## 5. 与你的声发射实验对接

**两级验证（先校准源、后真实裂纹）：**

1. **校准源（铅芯折断 / Hsu-Nielsen, ASTM E976）**：
   - 已有 `ae_step_source` + 表面点力源。
   - 用你实测的传感器位置、板材尺寸、材料 (vp, vs, ρ)、采样率建模。
   - **比对量**：首波(P)到时、Rayleigh/Lamb 到时与频散、波形互相关、频谱峰。
   - **标定**：用实测反推/微调材料参数、传感器耦合传递函数、衰减 Q。

2. **真实 AE 源（裂纹/分层）**：
   - 用 moment tensor 表示微裂纹（剪切=double-couple，张开=带体积分量）。
   - 频域 HINTS 正演不同源机制 → 合成波形库 → 与实测匹配做源定位/源表征。

**需要你提供（清单，方便后续填）：**
- [ ] 试件材料与几何（板厚、各向同性/异性、vp/vs/ρ 或弹性常数）
- [ ] 传感器型号、坐标、频响（或标称带宽）
- [ ] 采样率、记录长度、触发设置
- [ ] 校准（铅芯折断）波形原始数据
- [ ] 真实 AE 事件波形 + 已知/疑似源位置（若有）
- [ ] 衰减信息（材料 Q 或经验值）

**数据格式建议**：每事件存 `(传感器×时间)` 数组 + 元数据 JSON（坐标、dt、增益）。

---

## 6. 验证指标汇总（贯穿全程）

| 维度 | 指标 | 目标 |
| --- | --- | --- |
| 正确性 | 与直接解/解析解 L2 相对误差 | 频域 ≤1e-6；标量 ≤1e-8 |
| 加速 | GMRES 迭代数 / 壁钟时间 vs 纯 CSLP | 显著下降，随波数更优 |
| 谱互补 | 模态误差曲线（P/S 分开） | 低频由 DeepONet 主导下降 |
| 物理 | 频域↔时域 FDTD 波形互相关 | >0.95 |
| 物理 | Lamb 频散 vs 解析 | 曲线吻合 |
| 实验 | 到时误差、波形互相关 vs 实测 | 到时 < 1 个波长/采样；相关高 |
| 可靠性 | 不同训练范式/更新比的鲁棒性 | 收敛稳定、不发散 |

---

## 7. 备选 / 扩展方向

- **时域隐式 HINTS**：把 `elastic_fdtd` 改隐式 Newmark/后向欧拉，每步解移位系统，
  HINTS 做步内求解器 —— 把 HINTS 带进瞬态弹性动力学（更大工作量，独立贡献）。
- **3D**：方法不变，组装与训练成本上升；可借 `HINTS_petsc` 路线。
- **PML** 取代 Sommerfeld（更干净的吸收，但组装更复杂）。
- **各向异性 / 衰减(Q)** 材料：贴近真实复合材料 AE。
- **源反演**：正演成熟后接 FWI / moment-tensor inversion（AE 源表征）。

---

## 8. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 高波数弹性系统强不定，HINTS 可能不收敛 | 必接 CSLP 作平滑/内解（HINTS-CSLP），而非裸 Jacobi |
| DeepONet 复数+矢量训练不稳 | re/im+分量分通道；充分无量纲化；先标量 Stage 1 验证管线 |
| 混合解可靠性（2026 警示文） | 把更新比/训练范式作受控实验；监控发散 |
| 频域→时域合成的频谱泄漏/Gibbs | 足够频率采样 + 窗函数；与 FDTD 交叉验证 |
| 实验材料参数不确定 | 先用校准源标定 vp/vs/Q/耦合，再做真实源 |

---

## 9. 参考文献（定位用）

- Zhang, Kahana, Kopaničáková, Turkel, Ranade, Pathak, Karniadakis,
  *Blending Neural Operators and Relaxation Methods in PDE Numerical Solvers*,
  Nature Machine Intelligence (2024). arXiv:2208.13273.
- Kahana et al., *On the geometry transferability of HINTS*, Computational Mechanics (2023).
- *Attention-based hybrid solvers for linear equations that are geometry aware* (2024), arXiv:2411.13341.
- *A Hybrid Iterative Neural Solver Based on Spectral Analysis for Parametric PDEs* (2024), arXiv:2408.08540.
- *Are Deep Learning Based Hybrid PDE Solvers Reliable? ...* (2026), arXiv:2602.06842.
- Azulay & Treister, *Multigrid-Augmented Deep Learning Preconditioners for the
  Helmholtz Equation*, SIAM J. Sci. Comput. (CNN 预条件子基线).
- Zou et al., *Deep neural Helmholtz operators for 3-D elastic wave propagation and
  inversion*, Geophysical Journal International 239(3):1469 (2024).
- *NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers* (2025), arXiv:2511.02481.
- *SciML for Guided Wave and SAW Propagation* 综述 (2025), PMC11902327.
- Erlangga, Vuik, Oosterlee, *On a class of preconditioners for solving the
  Helmholtz equation* (CSLP), Appl. Numer. Math. 50 (2004).
- Virieux, *P-SV wave propagation ... velocity-stress FD*, Geophysics 51 (1986).
- Ohtsu & Ono, *A generalized theory of acoustic emission and Green's functions in a
  half space*, J. Acoustic Emission 3 (1984).
