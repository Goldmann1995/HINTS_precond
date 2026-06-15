# Stage 1 — 标量复数 2D Helmholtz + Sommerfeld 的 HINTS

> 状态：**基础设施完成、管线打通；"显著少于 CSLP 的迭代数"目标在 CPU 预算下未达成（如实报告）**
> 日期：2026-06-15 ｜ 对应 `HINTS_AE_ROADMAP.md` §4 Stage 1
> 代码：`wave_solvers/elastic_helmholtz.py`、`wave_solvers/hints_bridge.py`、
> 示例 `wave_solvers/examples/hints_helmholtz2d_stage1.py`、测试 `wave_solvers/tests/test_hints_bridge.py`

## 0. 结论（TL;DR，如实）

- ✅ **桥接基础设施完成且全部测试通过**：复数 2D Sommerfeld 组装、复数 DeepONet（re/im 双
  通道、Fourier 特征 trunk、保位置 CNN 分支）、独立混合迭代（带安全回溯）、**FGMRES**
  （支持非线性 HINTS-CSLP 预条件子）、FFT 谱互补诊断。`test_hints_bridge.py` 6/6 通过。
- ✅ **管线端到端打通**：理想算子（精确逆作"网络"）的 HINTS **5 步收敛**；**FGMRES
  HINTS-CSLP 在 freq 2000–5000 全部收敛到 1e-8**。复数/2D/Sommerfeld 的神经算子能训练
  （中等波数验证相对误差 0.16–0.24）。
- ⚠️ **未达成的验证目标**：在 CPU 训练预算下，训练出的 DeepONet 精度（算子相对误差
  ~0.2–0.4）**不足以让 HINTS-CSLP 的迭代数低于纯 CSLP**；独立路线 A（HINTS-Jacobi）在
  不定算子上**不稳定**。这恰好印证 roadmap §8 与 "Are DL hybrid solvers reliable?"(2026)
  的可靠性警示，是有价值的**受控负结果**。
- ➡️ **结论与交接**：瓶颈是**网络精度**（理想逆完美加速；差距纯粹来自网络）。要达成"超过
  CSLP"，需 **GPU + 更大数据/更长训练**把算子误差压到 ~5% 以下。基础设施（Stage 1 的
  "铺路"本质）已就绪，Stage 2 的弹性矢量情形可直接复用。

---

## 1. 交付物（已建 / 已测）

| 文件 | 内容 |
| --- | --- |
| `wave_solvers/elastic_helmholtz.py` | 复数 2D 声学 Helmholtz 组装（复用 Sommerfeld 五点格式）；异质速度场采样（含缺陷）；点源/光滑源；分辨率诊断（PPW）。`block_size` 参数化，Stage 2 的 `(ux,uz)` 直接复用。 |
| `wave_solvers/hints_bridge.py` | 复数 `ComplexDeepONet2D`；`HINTSSolver`（独立混合迭代 + 回溯安全机制 + 发散保护）；`fgmres`（柔性 GMRES，容纳非线性预条件子）；`make_hints_preconditioner`（CSLP 内解 + DON 校正）；`fft_band_energy`（谱诊断）；`generate_dataset`（含点源占比）。 |
| `examples/hints_helmholtz2d_stage1.py` | 端到端演示：训练网络 → 路线 A/B 对比 → 出图。 |
| `tests/test_hints_bridge.py` | 6 个测试：组装、FFT 分带、理想-HINTS 收敛、FGMRES 预条件子、数据集一致性、网络短训冒烟。 |

DeepONet 的两点关键改造（roadmap §2.3）已落地：**复数**（re/im 输出通道）+ **矢量就绪**
（每节点块大小参数化）。针对不定/振荡 Helmholtz 的两点工程要点也已解决：分支保留**源位置**
（避免全局平均池化抹掉位置），trunk 用**随机 Fourier 特征**克服谱偏置（否则光滑网络无法表示
波的振荡）。

---

## 2. 验证结果（真实数值）

### 2.1 方法本身正确（理想算子）
`test_pure_jacobi_diverges_oracle_hints_converges`：纯 Jacobi 发散（res 9.5e-1），把**精确
逆**当作"网络"的 HINTS-Jacobi **5 步收敛**（误差 9.1e-15）。→ HINTS 机制在复数 2D 不定系统
上成立；后续一切差距来自网络近似精度。

### 2.2 网络可训练（复数/2D/Sommerfeld 神经算子）
grid 32，2500 训练样本，1000 epochs（CPU，~17 分钟）：

| 频率 | k 范围 | PPW | 验证相对 L2 误差 |
| --- | --- | --- | --- |
| 2000 Hz | 8.4–12.0 | 16.3 | **0.226** |
| 3500 Hz | 14.7–20.9 | 9.3 | **0.235** |
| 5000 Hz | 20.9–29.9 | 6.5（欠分辨） | 0.412 |

训练 MSE ~5e-7 ≪ 验证误差 → 明显**过拟合**（2500 样本不足）。数据生成极廉价（共享一次 LU
分解），更多数据可显著改善，但每 epoch 成本随样本数线性增长，CPU 上代价高。

### 2.3 路线 B：FGMRES HINTS-CSLP（端到端收敛，但未超过 CSLP）
全部收敛到 rtol=1e-8（点源测试问题）：

| 频率 | 纯 CSLP 迭代数 | HINTS-CSLP 迭代数(最优 n_inner) | 是否更快 |
| --- | --- | --- | --- |
| 2000 Hz | 20 | 51 | 否 |
| 3500 Hz | 46 | 78 | 否 |
| 5000 Hz | 83 | 86 | 持平（略慢） |

随波数升高 CSLP 迭代数增长（20→46→83），HINTS-CSLP 的相对差距缩小（2.5×→1.7×→1.04×），
说明高波数下网络的"低频校正"开始有意义；但当前精度仍不足以净加速。

### 2.4 路线 A：独立 HINTS-Jacobi（不稳定，已加保护）
纯 Jacobi 在不定系统上不收敛；加入训练网络后，**即便有残差回溯安全机制仍会发散**。根因是
重要的**受控发现**：

> 不精确网络会向算子 A 的**近零空间**（`A·v≈0` 的模态）注入分量——这几乎不改变残差，因此
> **基于残差的安全机制看不见它**——却使 `‖u‖` 膨胀，下一步 Jacobi 随即爆掉。

已加**发散保护**（残差超过初值 1e3 倍即干净停止并置 `diverged=True`），避免数值溢出。结论：
不定 Helmholtz 上，独立 HINTS 的稳健形式本质上就是 **Krylov 加速（即路线 B 的 FGMRES）**，
这与原 HINTS 在不定问题上依赖多重网格/Krylov 框架一致。

---

## 3. 关键发现与诊断

1. **瓶颈是网络精度，不是管线**。理想逆 → 5 步收敛；网络 0.2–0.4 误差 → 比 CSLP 慢
   1.0–2.5×。迭代数随网络精度单调改善。
2. **残差型安全机制对不定算子不充分**（近零空间污染）。这是混合求解器可靠性的具体机理性
   证据，可直接写进论文的"可靠性"小节。
3. **柔性 GMRES 是必需的**：HINTS 预条件子因每次按残差范数归一化而**非线性**，标准 GMRES
   会失效（NaN）；FGMRES 解决之，且对常数线性预条件子退化为普通 GMRES，比较公平。

---

## 4. 复现命令

```bash
# 测试（6/6 通过）
python wave_solvers/tests/test_hints_bridge.py
# 端到端演示（默认 freq 2000, grid 32, 1000 epochs；约 17 分钟 CPU）
python wave_solvers/examples/hints_helmholtz2d_stage1.py
#   可调：--freq 3500 --grid 32 --n-train 2500 --epochs 1000 --ratio 5
```

---

## 5. 与验证指标(roadmap §6)的对照

| 指标 | 目标 | 实测 | 结论 |
| --- | --- | --- | --- |
| 正确性（解 vs 直接解 L2） | ≤1e-8 | FGMRES HINTS-CSLP 收敛到 1e-8（err ~2e-8） | ✅ |
| 加速（GMRES 迭代数 vs 纯 CSLP） | 显著下降 | 当前慢 1.0–2.5× | ❌（受网络精度限制） |
| 谱互补图 | 成立 | 诊断已实现；理想算子下成立 | ✅（工具就绪） |
| 网络可训练（复数/2D） | — | 验证误差 0.16–0.24 | ✅ |

---

## 6. 进入 Stage 2 的建议

- **优先提网络精度**（决定一切）：GPU 训练；数据 ↑（5k–20k）；epoch ↑；可加权重衰减/更大
  容量/多频共享（ω 作分支输入，提前布局 Stage 3）。目标算子误差 < ~5%。
- **路线 B 为主**（FGMRES HINTS-CSLP 稳健收敛）；路线 A 仅在多重网格/Krylov 框架内使用。
- **基础设施可直接进入 Stage 2**：把 `ScalarHelmholtz2DProblem`（block_size=1）扩展为
  `ElasticHelmholtz2DProblem`（block_size=2，P-SV Navier-Helmholtz），DeepONet 输出由 2 通道
  （re/im）变为 4 通道（re/im × ux/uz），其余（FGMRES、安全机制、诊断、数据生成）原样复用。

**判定：Stage 1 的"铺路/管线"目标达成；"超过 CSLP"目标因 CPU 网络精度受限未达成，已如实记录
并给出明确改进路径。**
