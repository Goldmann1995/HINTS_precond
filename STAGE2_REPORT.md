# Stage 2 — 弹性矢量 P-SV Navier-Helmholtz 的 HINTS（核心创新）

> 状态：**核心算子严格验证通过；矢量 HINTS 管线端到端打通**
> 日期：2026-06-16 ｜ 对应 `HINTS_AE_ROADMAP.md` §4 Stage 2
> 代码：`wave_solvers/elastic_helmholtz.py`（`ElasticHelmholtz2DProblem`、`freq_to_time`）、
> `wave_solvers/hints_bridge.py`（`VectorComplexDeepONet2D`）、
> 示例 `wave_solvers/examples/hints_elastic_stage2.py`、测试 `wave_solvers/tests/test_elastic_helmholtz.py`

## 0. 结论（TL;DR）

- ✅ **首个矢量/弹性 HINTS 的核心算子已实现并严格验证**：复数 2D P-SV Navier-Helmholtz
  组装（位移 `u=(ux,uz)`，二阶有限差分，component-major 排序）。
- ✅ **算子正确性多重证明（与 ML 无关）**：制造解 **2 阶收敛**（rate 2.00）；截断一致性
  （每加密一倍残差降 ~4×）；**弹性动力学互易性到机器精度（rel 2.1e-14）**；矩张量/点力源辐射
  出可区分的波场。
- ✅ **矢量 HINTS 桥接端到端打通**：复数矢量 DeepONet（4 输出通道 re/im×ux/uz）可训练；
  `HINTSSolver`/`fgmres`/预条件子原样支持 block_size=2；理想算子下 oracle-HINTS 与
  FGMRES(oracle) **1 步收敛**；示例中 FGMRES 迭代数 none/ILU/HINTS = 244/6/8。
- ✅ **频↔时合成机制就绪**：`source_spectrum`/`freq_to_time`（IFFT 往返验证通过）。
- ⚠️ **关键经验**：弹性算子在**有量纲**参数下条件数极差（λ,μ ~ 1e10，矩阵元跨 15 个数量级），
  直接解被舍入误差主导（互易性"假性失败"）。**无量纲化（roadmap §2.3）**后一切恢复机器精度。
- ⏳ **未完成（明确界定）**：频域→时域**声发射波形**与时域 `elastic_fdtd` 的**定量**互校尚未达成
  >0.95 相关——根因是简单对角吸收层(ABL)无法吸收**波长≈域尺寸**的低频分量（驻波→非因果合成），
  需**真正的 PML**（roadmap §7 已列为后续）。算子本身已由制造解+互易性验证为正确。

测试：`test_elastic_helmholtz.py` **7/7 通过**；全仓 wave_solvers 测试 **22/22 通过**，无回归。

---

## 1. 交付物

| 文件 / 类 | 内容 |
| --- | --- |
| `ElasticHelmholtz2DProblem` | 复数 2D P-SV Navier-Helmholtz 组装；Dirichlet（制造解测试）与 absorbing（Dirichlet 边框 + 内部 ABL 阻尼）边界；`point_force`、`moment_tensor`（剪切=double-couple，体积=explosion 的 AE 微裂纹源）、`smooth_random_source`；`block_size=2`。 |
| `source_spectrum` / `freq_to_time` | 震源频谱 + 逐频解 → IFFT 合成时域波形（AE 正演管线）。 |
| `VectorComplexDeepONet2D` | 复数矢量 DeepONet：输入 `[材料场…, Re/Im fx, Re/Im fz]`，输出 4 头 re/im×(ux,uz)；保位置 CNN 分支 + Fourier 特征 trunk（沿用 Stage 1 的抗谱偏置设计）。 |
| `hints_elastic_stage2.py` | 端到端演示：训练矢量 DeepONet → 路线 A（独立混合）/ 路线 B（FGMRES，ILU vs HINTS）。 |

`HINTSSolver`、`fgmres`、`make_hints_preconditioner`、`fft_band_energy`、`generate_dataset`
均已在 Stage 1 实现并**对 block_size 通用**，本阶段把 `shape=(2,nx,nz)` 传入即复用。

---

## 2. 验证结果（真实数值，全部 ML 无关）

| 测试 | 指标 | 结果 |
| --- | --- | --- |
| 制造解收敛 | `ux=uz=sin(πx)sin(πz)`，无量纲，最大误差阶 | **2.00** |
| 截断一致性 | `‖A u_exact − f‖`（有量纲）随加密 | 每倍 ~4× 下降（3.9e-3→9.8e-4）|
| **弹性互易性** | `u_z(B; f_z@A)=u_z(A; f_z@B)` 相对差 | **2.1e-14**（含交叉分量 3e-11）|
| 源辐射 | explosion vs double-couple 波场相关 | \|corr\| 0.01（可区分）|
| 频↔时往返 | IFFT 还原 Ricker 子波 | 误差 <1e-6 |
| 矢量 DeepONet | 短训验证相对误差 | 1.01→0.30（400 ep, 24²）|
| 矢量桥接 | oracle-HINTS / FGMRES(oracle) 迭代数 | 1 / 1 |

演示（无量纲，grid 28，弱网络仅 120 ep，val 0.61）：FGMRES 迭代数 **none 244 / ILU 6 / HINTS 8**。
与 Stage 1 一致：CPU 预算的弱网络给经典预条件子带来轻微开销而非加速；理想算子下机制完美（1 步）。

---

## 3. 关键工程发现

1. **无量纲化是弹性 HINTS 的必要前提**（roadmap §2.3 落地为硬约束）。有量纲 (λ,μ~1e10) 下矩阵元跨
   15 个数量级，直接解舍入受限：互易性、oracle 一致性都会"假性失败"。无量纲 O(1) 参数下，
   互易性 2.1e-14、oracle 1e-14。**所有训练/求解应在无量纲坐标进行。**
2. **吸收边界须为 Dirichlet 边框 + 内部 ABL 阻尼**。最初用"钳位索引的边界 stencil"破坏了算子对称性
   → 互易性失败。改为内部块对称（精确 0 非对称）后互易性恢复机器精度。
3. **混合导数的对称离散**：`∂xz` 四角 stencil 用行节点系数，常系数下内部块**精确复对称**
   （`‖A_II−A_IIᵀ‖=0`），是互易性成立的代数基础。

---

## 4. 未完成项：AE 波形频↔时定量互校（明确界定 + 方案）

**现象**：频域→IFFT 合成波形与时域 `elastic_fdtd` 的波形互相关仅 ~0.1，合成波形首至非因果。
**根因（已诊断，非算子错误）**：
- Ricker 频带的**最低频波长 ≈ 域尺寸**，任何有限吸收层都无法吸收 → 驻波 → 合成非因果；
- 简单对角 ABL 远不及 **PML**；
- 配置点(collocated)频域点源与交错网格(staggered)FDTD 力源的**等效性**需要仔细处理；
- `e^{±iωt}` 约定（numpy IFFT 用 `e^{+iωt}`，需对响应取共轭取因果分支）。

**方案（Stage 2.5 / 进入 Stage 3 前）**：实现频域弹性 **PML**（roadmap §7），增大域≫最长波长，
统一无量纲坐标，再做 FDTD 互校（目标互相关 >0.95、到时误差 < 1 波长）。算子正确性已由制造解 +
互易性独立保证，此项是**正演管线打磨**而非算子修复。

---

## 5. 复现命令

```bash
python wave_solvers/tests/test_elastic_helmholtz.py        # 7/7 通过
python wave_solvers/examples/hints_elastic_stage2.py        # 矢量 HINTS 演示（无量纲）
#   可调：--grid 32 --omega 6 --n-train 1500 --epochs 800
```

---

## 6. 对照 roadmap §6 验证指标

| 指标 | 目标 | 实测 | 结论 |
| --- | --- | --- | --- |
| 正确性（解 vs 直接解 / 解析） | 频域 ≤1e-6 | 制造解 2 阶；互易性 2e-14；FGMRES 收敛 1e-8 | ✅ |
| 频域↔时域 FDTD 波形互相关 | >0.95 | ~0.1（待 PML）| ⏳ 明确界定为后续 |
| P/S 谱互补图 | 成立 | 诊断工具 `fft_band_energy` 支持矢量；理想算子下成立 | ✅（工具就绪）|
| 加速（迭代数 vs 经典预条件子） | 显著下降 | 当前弱网络略增（同 Stage 1）| ❌（受 CPU 网络精度限制）|

---

## 7. Stage 3 交接

- **首要**：频域弹性 PML + 大域，完成 AE 波形定量互校（解锁 roadmap 核心验证图）。
- **网络精度**：GPU + 更大数据/更长训练（弱网络是不超过经典预条件子的唯一瓶颈，机制已由 oracle 证明）。
- **多频摊销（Stage 3）**：把 ω 作为分支输入，一张矢量 DeepONet 跨 AE 频带泛化，复用 `freq_to_time`；
  薄板复现 Lamb 频散。

**判定：Stage 2 的核心创新（矢量/弹性 HINTS 算子）已实现并严格验证；AE 波形定量互校界定为
需 PML 的后续工程项，已给出明确诊断与方案。**
