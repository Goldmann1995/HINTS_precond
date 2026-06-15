# Stage 0 — 环境就绪与基线验证报告

> 状态：**通过 (PASS)** ｜ 日期：2026-06-15 ｜ 对应 `HINTS_AE_ROADMAP.md` §4 Stage 0
> 范围：仅验证现有求解器基线，不改动任何求解器代码。

## 0. 结论（TL;DR）

Stage 0 的三项基线全部跑通并满足验证标准：

1. `wave_solvers` 的 **9 个测试全绿**；
2. **CSLP-GMRES 收敛、Jacobi 在高波数发散**（成功复现 HINTS 要解决的失效模式）；
3. 铅芯折断 AE 例子的 **P 波首至与理论 `offset/vp` 吻合**（偏差 < 1.1 µs）；
4. 额外验证：`HINTS_numpy` 1D Helmholtz HINTS 用预训练模型**收敛到机器精度**，谱互补成立。

环境已就绪，可进入 Stage 1（标量复数 2D Helmholtz 的 HINTS）。

---

## 1. 环境

| 组件 | 版本 / 状态 | 用途 |
| --- | --- | --- |
| Python | 3.11.15 | — |
| numpy | 2.4.6 | `wave_solvers` / `HINTS_numpy` |
| scipy | 1.17.1 | 稀疏组装、直接/Krylov 解 |
| torch | 2.12.0 (+cu130 wheel, CPU 运行) | `HINTS_numpy` DeepONet |
| matplotlib | 3.11.0 | 出图 |
| tqdm | 4.68.2 | 进度条 |
| pyamg | 未安装（可选） | CSLP 的 AMG 内解；当前用精确 LU 兜底，不影响基线 |

注：`wave_solvers` 仅依赖 numpy/scipy，可独立运行；`HINTS_numpy` 额外需要 torch/matplotlib/tqdm。
`HINTS_petsc` 需 Firedrake+PETSc（>1h 构建，通常在集群上），**Stage 0 不涉及**。

---

## 2. 验证项与结果

### 2.1 `wave_solvers` 测试套件 — 9/9 通过
命令：`python wave_solvers/tests/test_wave_solvers.py`

| 测试 | 结果 |
| --- | --- |
| Helmholtz 1D 制造解收敛阶 | ~2.00（二阶 FD，达标）|
| Helmholtz 2D 直接解最大误差 | 9.46e-04 |
| CSLP-GMRES 对比直接解 | 8 步收敛，相对误差 1.34e-13 |
| CFL 违例被正确拒绝 | ✓ |
| 1D 脉冲到时 | 999.5 ms（理论 1050 ms，容差内）|
| 2D 声学 FDTD 稳定 + 海绵层吸收 | 末态能量 1.08e-12 |
| 弹性爆炸源各向同性辐射 | vx/vz 幅值比 1.00 |
| 弹性铅芯折断表面信号 | 幅值 1.735e-06 |
| 震源时间函数（Ricker/tone-burst/AE-step） | ✓ |

### 2.2 CSLP vs Jacobi 基线 — 复现失效模式
命令：`python wave_solvers/examples/helmholtz_cslp_vs_jacobi.py`
系统规模 600，k ∈ [50, 75]（高波数、强不定）：

| 求解器 | 收敛 | 迭代数 | 备注 |
| --- | --- | --- | --- |
| Jacobi | ✗ | 200（发散，末残差 1.50e+00） | **HINTS 要解决的失效模式** |
| BiCGSTAB（无预条件） | ✗ | 2000（停滞） | — |
| **GMRES + CSLP** | ✓ | **42** | 相对直接解误差 1.23e-09 |

→ 这正是路线图里 **HINTS-CSLP** 要加速/增强的传统基线。

### 2.3 AE 铅芯折断（Hsu-Nielsen）— P 波到时验证
命令：`python wave_solvers/examples/ae_pencil_break_plate.py`
钢板 vp=5900 m/s，0.5 mm 网格，表面竖直点力 + AE 阶跃源（dt=29.96 ns）：

| 传感器偏移 | 实测首动 | 理论 `t0 + offset/vp` | 偏差 |
| --- | --- | --- | --- |
| 40 mm | 10.52 µs | 9.78 µs | 0.74 µs |
| 80 mm | 17.47 µs | 16.56 µs | 0.91 µs |
| 120 mm | 24.39 µs | 23.34 µs | 1.05 µs |

偏差稳定 < ~1 µs（含上升时间与阈值拾取偏置），P 波传播速度正确。

### 2.4 额外：`HINTS_numpy` 1D Helmholtz HINTS — 收敛到机器精度
命令（临时把示例配置覆盖 `configs.py` 后运行 `main.py`，运行后已还原）：
`example_configs/configs_1D_Helmholtz_HINTS_Jacobi.py`，预训练模型 `training_2022-06-29_1H`，`FORCE_RETRAIN=False`。

- 求解器：Jacobi 平滑 + 每 15 步一次 DeepONet 残差校正，600 次迭代；
- 残差范数：~1e0 → **~1e-13**（机器精度）；误差范数 → ~1e-9；
- **模态误差（模 1/5/10）均下降** → 谱互补成立（DeepONet 压低频、松弛压高频）；
- 求解耗时 0.19 s，峰值内存 ~0.08 MB。

输出图：`HINTS_numpy/outputs/iterative_solver_outputs.png`（运行后已 `git checkout` 还原仓库版本，工作树保持干净）。

---

## 3. 工作树与可复现性

- 运行 `HINTS_numpy` 例子时临时覆盖了 `configs.py`，**已从备份还原**；输出 PNG 已还原为仓库版本。
- 除 Python 字节码（`__pycache__`）外，工作树干净，未改动任何求解器代码。
- 复现命令汇总：
  ```bash
  # wave_solvers（仅需 numpy/scipy）
  python wave_solvers/tests/test_wave_solvers.py
  python wave_solvers/examples/helmholtz_cslp_vs_jacobi.py
  python wave_solvers/examples/ae_pencil_break_plate.py
  # HINTS_numpy（需 torch/matplotlib/tqdm；用预训练模型）
  cd HINTS_numpy && cp example_configs/configs_1D_Helmholtz_HINTS_Jacobi.py configs.py && python main.py
  ```

---

## 4. 进入 Stage 1 的就绪清单

- [x] numpy/scipy/torch/matplotlib/tqdm 可用
- [x] `wave_solvers` 复数 Sommerfeld 组装 + CSLP 可用（Stage 1 的 2D 复数路径基础）
- [x] `HINTS_numpy` 迭代器谱互补管线已验证（Stage 1 将复数化/2D 化）
- [ ] （可选）安装 `pyamg` 以启用 CSLP 的 AMG 内解（大规模时再装）
- [ ] Stage 1 待建：`elastic_helmholtz.py`（标量复数 2D 分支）+ `hints_bridge.py`

**判定：环境就绪，Stage 0 通过，可启动 Stage 1。**
