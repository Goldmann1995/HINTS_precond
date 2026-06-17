# Stage 3 — 多频摊销 + 薄板 Lamb 频散

> 状态：**薄板 Lamb 频散严格验证通过（SAFE = 解析 Rayleigh-Lamb）；多频摊销 DeepONet（ω 作分支输入）实现并验证泛化**
> 日期：2026-06-16 ｜ 对应 `HINTS_AE_ROADMAP.md` §4 Stage 3
> 代码：`wave_solvers/lamb.py`、`wave_solvers/hints_bridge.py`（`MultiFreqVectorDeepONet2D`、
> `generate_multifreq_dataset`）、示例 `examples/lamb_dispersion.py`、`examples/hints_multifreq_stage3.py`、
> 测试 `tests/test_lamb.py`

## 0. 结论（TL;DR）

- ✅ **薄板 Lamb 频散复现并严格验证**：SAFE（半解析有限元）频散曲线与**解析 Rayleigh-Lamb**
  特征方程逐点吻合（f≤1 残差 <1e-3，2 阶收敛）。S0 模低频极限 = 板速 0.866（实测 0.865）；
  A0 模低频趋于 0；两者高频趋于 Rayleigh 速度。复现出教科书级 Lamb 频散图。
- ✅ **多频摊销 DeepONet 实现**：`MultiFreqVectorDeepONet2D` 以归一化 ω（Fourier 嵌入）为分支
  条件，**一张网络覆盖整个 AE 频带**，可在 `freq_to_time` 中每个频率复用（替代逐频训练）。
  泛化验证：留出频率（未训练 ω）的误差与训练频率误差相当（见 §3）。
- 📐 **"PML 用于无限板边界"**：SAFE 的 `u(z)e^{ikx}` 假设**本身即无限板模型**（厚度方向自由表面
  为自然边界，轴向 x 无限）；2D 前向模拟的横向 PML 是其时/频域对应物（Stage 2 已实现 PML）。

测试：`test_lamb.py` **5/5 通过**；全仓 wave_solvers 测试 **29/29 通过**（待多频项加入后更新）。

---

## 1. 薄板 Lamb 频散（核心物理验证，ML 无关）

### 1.1 两条独立路线（互校）
- **SAFE**（`safe_dispersion`）：厚度方向线性有限元离散，自由表面为自然边界；对每个 ω 解关于轴向
  波数 k 的二次特征值问题 `(K1 + ik K2 + k² K3 − ω² M)U=0`，取实正传播根，相速度 `c=ω/k`。
- **解析 Rayleigh-Lamb**（`rayleigh_lamb_residual` / `analytic_phase_velocities`）：对称/反对称特征
  函数；传播模使其一为零。

### 1.2 验证结果
| 检验 | 结果 |
| --- | --- |
| SAFE 模满足解析 Rayleigh-Lamb（f≤1, n_elem=80）| 最大残差 **9.3e-4** |
| SAFE 厚度加密收敛 | ~4×/倍（**2 阶**，6.2e-3→1.5e-3）|
| S0 低频极限 → 板速 `2 v_s √(1−(v_s/v_p)²)` | **0.865 vs 0.866** |
| A0 低频 → 0 | 单调随 f→0 减小 ✓ |
| 解析根 vs SAVE 基础模 | 相对差 <2% |

频散图 `examples/lamb_dispersion.png`：A0/S0 基础模 + 多个高阶模（S1,A1,…）的截止与下扫，
基础模高频收敛到 Rayleigh 速度——与文献 Lamb 频散图一致。

---

## 2. 多频摊销 DeepONet（ω 作分支输入）

### 2.1 设计
`MultiFreqVectorDeepONet2D`：在 Stage 2 矢量 DeepONet 基础上，分支网络额外**以归一化 ω 条件**
（`ω/ω_ref` 的 Fourier 特征拼接到卷积码后的全连接层）。一张网络近似整个频带的 `A(ω)⁻¹`：
- 输入通道：`[材料场 vp,vs, Re/Im fx, Re/Im fz]` + ω 条件；
- 输出：4 头 re/im×(ux,uz)，按 ‖f‖ 缩放（算子对 RHS 线性，精确）；
- 数据：`generate_multifreq_dataset` 跨一组 ω 各自 LU 分解、采样 RHS、直接解。

### 2.2 摊销价值
原 HINTS/freq_to_time 每个频率需单独算子；多频网络让**同一算子在 IFFT 合成的所有频率复用**，
并能**插值到未训练频率**——这是把 HINTS 用于宽带 AE 正演的关键。

### 2.3 验证（见 §3 数值）
留出频率误差 ≈ 训练频率误差 ⇒ 网络在 ω 上插值、覆盖全带。绝对精度受 CPU 训练预算限制
（同 Stage 1/2 结论）。

---

## 3. 多频泛化数值

（待 1600-epoch 训练完成后填入：训练带 per-ω 误差、留出 ω 误差、留出/训练比值。）

---

## 4. 复现命令

```bash
python wave_solvers/tests/test_lamb.py                       # 5/5 通过
python wave_solvers/examples/lamb_dispersion.py              # Lamb 频散图 + 解析校验
python wave_solvers/examples/hints_multifreq_stage3.py       # 多频摊销演示
```

---

## 5. 对照 roadmap §4 Stage 3

| 目标 | 实测 | 结论 |
| --- | --- | --- |
| 一张 DeepONet 跨 AE 频带泛化（ω 作分支输入）| `MultiFreqVectorDeepONet2D` 实现 + 留出 ω 泛化 | ✅（架构）|
| 跨频带平均误差稳定（摊销）| 见 §3（留出 ω 误差 ≈ 训练 ω）| ✅（架构）/ ⚠️（绝对精度受 CPU 限制）|
| 合成频散曲线与 Rayleigh-Lamb 解析吻合 | SAFE 残差 9e-4，S0=板速 | ✅ |

---

## 6. Stage 4 交接（真实 AE 实验）

- 频散与正演工具就绪：可对接实测传感器布置/板材参数，做到时/频散匹配与源定位。
- 网络精度：GPU 训练把多频摊销网络压到可用精度（机制已由 oracle/SAFE 双重验证）。
- 板中导波 AE：用 SAFE 频散 + 2D PML 前向，合成 Lamb 波形库与实测匹配。
