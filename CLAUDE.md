# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Reference code for the HINTS method ("Blending Neural Operators and Relaxation
Methods in PDE Numerical Solvers", Zhang et al., *Nature Machine Intelligence*
2024). HINTS interleaves classical relaxation iterations (Jacobi / Gauss-Seidel,
optionally inside a multigrid V-cycle) with a DeepONet that predicts a solution
update from the current residual — the network damps the low-frequency error
modes that relaxation is slow on.

There are three largely independent subprojects, each with its own environment
and entry points:

| Directory | Stack | Purpose |
| --- | --- | --- |
| `HINTS_numpy/` | NumPy/SciPy + PyTorch | Small 1D/2D/3D Poisson & Helmholtz HINTS examples (dense matrices). The methodology reference. |
| `HINTS_petsc/` | Firedrake + PETSc + PyTorch | Large-scale FEM HINTS as a PETSc preconditioner for Krylov methods. |
| `wave_solvers/` | NumPy/SciPy only | Classical (non-learned) wave/Helmholtz solvers + elastic-wave sources; baselines and building blocks for acoustic-emission (AE) modelling. |

`HINTS_AE_ROADMAP.md` tracks an in-progress effort to extend HINTS to
elastic/Helmholtz forward modelling for acoustic-emission applications.

## HINTS_numpy

Everything is driven by **`configs.py`** (a module of globals, not CLI flags).
Editing `configs.py` is how you select the problem, dimension, and solver.
Run from inside the directory so the flat `import configs` works:

```bash
cd HINTS_numpy
pip install -r requirements.txt
python main.py
```

Key control flow (`main.py`):
- `configs.ITERATION_METHOD` dispatches the run:
  - `'Numerical'` → pure relaxation (`run_numerical`)
  - `'DeepONet'` / `'Numerical_DeepONet_Single'` → train/infer DeepONet (`run_dl`)
  - `'Numerical_DeepONet_Hybrid'` → the HINTS solver (`run_hybrid`)
- `configs.SOLVER_METHOD` picks the relaxation/Krylov scheme: `'Jacobi'`,
  `'GS'`, `'CG'`, `'MG-J'`, `'MG-GS'` (the `MG-*` variants run the multigrid
  V-cycle with that smoother).
- `configs.PROBLEM` (`'poisson'`/`'helmholtz'`) and `configs.DIMENSIONS`
  (1/2/3) select the assembled operator (`utils.poisson`, `utils.helmholtz*`).

**Typical workflow to add/run a HINTS case**: set `ITERATION_METHOD='DeepONet'`
first to create the dataset and train (a model is saved under `models/`,
debug figures under `debug_figs/`), then switch to
`'Numerical_DeepONet_Hybrid'` to run HINTS. To retrain after changing the
problem, set `FORCE_RETRAIN=True` or delete `outputs/results*.npz`.
`MODEL_NAME`/`DATASET_NAME` at the bottom of `configs.py` name the saved
artifacts per dimension — rename these when running a new scenario so you
don't overwrite or accidentally load a stale model.

Module roles:
- `data_handler.py` — generates Gaussian-random-field samples of `k` and `f`,
  caches them as the pickled `DATASET_NAME`.
- `deeponet.py` — `DeepONet` (PyTorch). Branch net is an MLP in 1D, a CNN in
  2D/3D; trunk net over node coordinates. Trains on FEM solutions.
- `iterative_solver.py` — the heart of the method. `NumericalSolver` assembles
  the system and does one relaxation step; `DeepONetSolver` does one network
  correction from the residual; `IterativeSolver` (multiple-inherits both) is
  what interleaves them based on `NUMERICAL_TO_DON_RATIO` (one DeepONet step
  every N relaxation steps). `MultiGrid` wraps `IterativeSolver`s per level and
  uses the chosen solver as its smoother.
- `utils.py` — operator assembly, GRF sampling, eigen-decomposition for
  mode-wise error tracking, plotting helpers.

Results/plots are written to `outputs/`. Pre-canned configs that reproduce the
paper figures live in `example_configs/` — copy one over `configs.py` to use it.

Dimensionality is threaded through `configs.DIMENSIONS` with many
`if DIMENSIONS == 1/2/3` branches (the code is explicit, not generic — note the
several `TODO: make generic` markers). When changing behaviour, expect to touch
all three branches.

## HINTS_petsc

FEM assembly via Firedrake, linear algebra via PETSc; the DeepONet is plugged
in as a custom PETSc preconditioner (`petsc_solvers/JacobiHINTS.py`, subclass of
Firedrake's `PCBase`). This requires a full Firedrake+PETSc build (see the root
`README.md`; building PETSc can take >1 hour) and is normally run on a cluster.

Run pattern (from `HINTS_petsc/`, with the Firedrake venv active and
`PYTHONPATH` including the repo dir; note the flat imports like
`from config import params`):

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/HINTS_petsc
cd example
python3 -u hints_test_HINTSgmg_sampled_k.py --num_samples_total 10000 --num_samples 10000 --dofs_don 8 --num_basis_functions 128 --k_sigma 6
python3 -u hints_test_hypre_sampled_k.py     --num_samples_total 10000 --num_samples 100000 --k_sigma 6.0
```

Here configuration is via argparse (`config.py::get_params`, exposed as
`config.params`), not a globals module. `datasets/` samples FEM problems,
`deeponets/` defines the operator network, `trainers/Trainer.py` trains it,
`petsc_solvers/` holds the HINTS preconditioner and PETSc helpers. The 3D
Helmholtz-cylinder dataset is large and downloaded from Zenodo (link in the
root README).

## wave_solvers

Pure NumPy/SciPy, no Firedrake/PETSc/PyTorch. Imported as a package (`import
wave_solvers as ws`), so run from the **repo root**.

```bash
python wave_solvers/tests/test_wave_solvers.py     # 9 correctness tests (also runs under pytest)
python wave_solvers/examples/helmholtz_cslp_vs_jacobi.py
python wave_solvers/examples/ae_pencil_break_plate.py
```

Modules: `helmholtz.py` (sparse FD assembly, direct/GMRES/BiCGSTAB, ILU and the
complex shifted-Laplacian/CSLP preconditioner), `wave_fdtd.py` (1D/2D scalar
leapfrog FDTD with Mur/Cerjan absorbing BCs), `elastic_fdtd.py` (2D P-SV
velocity-stress staggered grid, Virieux 1986), `sources.py` (Ricker / tone-burst
/ AE-step time functions; point-force / moment-tensor / double-couple / explosion
spatial sources). The CSLP-preconditioned Helmholtz path is the classical
baseline that HINTS is meant to accelerate. The tests are
manufactured-solution / convergence checks — keep them green when touching these
solvers.

## Cross-cutting notes

- The three subprojects do not import each other and use different config
  conventions (`HINTS_numpy` globals module vs `HINTS_petsc` argparse). Don't
  assume shared utilities — each has its own `utils.py`.
- `HINTS_numpy` and `HINTS_petsc` use **flat imports** (`import configs`,
  `from config import params`) that only resolve when the working directory is
  that subproject; `wave_solvers` is a proper package run from the repo root.
- SciPy compatibility: `interp2d` was removed in SciPy ≥1.14 and is guarded with
  `try/except` in several `HINTS_numpy` modules; it's only on the 2D paths.
