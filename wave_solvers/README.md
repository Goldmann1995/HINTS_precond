# wave_solvers: traditional wave / Helmholtz solvers and elastic-wave sources

This package adds **classical (non-learned) solvers** for the wave and
Helmholtz equations, together with a library of **elastic-wave excitation
sources**, to the HINTS repository. It serves two purposes:

1. provide transparent, dependency-light *baselines* alongside the HINTS
   hybrid neural-operator solvers (`HINTS_numpy`, `HINTS_petsc`), and
2. act as building blocks for **acoustic-emission (AE) waveform modelling**,
   which is the intended downstream application.

Everything here is pure NumPy/SciPy — no Firedrake/PETSc needed.

## Contents

| Module | What it provides |
| --- | --- |
| `helmholtz.py` | Sparse 1D/2D finite-difference assembly (Dirichlet & first-order Sommerfeld BCs); direct (SuperLU), GMRES, BiCGSTAB; ILU and **complex shifted-Laplacian (CSLP)** preconditioners; Jacobi/Gauss-Seidel/SOR. |
| `wave_fdtd.py` | 1D/2D scalar acoustic wave equation, 2nd-order leapfrog FDTD, with **Mur** and **Cerjan sponge** absorbing boundaries and a CFL guard. |
| `elastic_fdtd.py` | 2D P-SV elastic wave propagation, **Virieux (1986)** velocity-stress staggered grid, free surfaces (Rayleigh/Lamb waves), absorbing sponge. |
| `sources.py` | Source-time functions (Ricker, Gaussian, **ultrasonic tone burst**, **AE step / pencil-break**) and spatial sources (point force, **moment tensor**, double-couple, explosion). |

## Quick start

```bash
# from the repository root
python wave_solvers/tests/test_wave_solvers.py        # 9 correctness tests
python wave_solvers/examples/helmholtz_cslp_vs_jacobi.py
python wave_solvers/examples/ae_pencil_break_plate.py
```

```python
import numpy as np
import wave_solvers as ws

# --- Helmholtz with the classical shifted-Laplacian preconditioner -------
from wave_solvers import helmholtz as hz
A = hz.helmholtz_matrix_1d(n=600, k=50.0, bc='sommerfeld')
f = np.zeros(600); f[120] = 1.0
M = hz.make_shifted_laplacian_preconditioner(
        hz.helmholtz_matrix_1d, beta=(1.0, 0.5), n=600, k=50.0, bc='sommerfeld')
u, info = hz.solve_gmres(A, f, M=M)          # converges in ~tens of iterations

# --- Acoustic-emission micro-crack on a steel block ----------------------
solver = ws.ElasticWaveFDTD2D(vp=5900, vs=3200, rho=7800, dx=1e-3,
                              free_surface=True, shape=(200, 120))
t = np.arange(600) * solver.dt
stf = ws.ae_step_source(t, rise_time=1e-6, t0=3e-6)       # crack opening
solver.add_source(ws.double_couple_2d(100, 60, stf))      # shear micro-crack
solver.add_receiver(150, 1)                               # surface AE sensor
rec = solver.run(600)
```

## Relevance to acoustic-emission (AE) detection

AE testing detects elastic transients radiated by sudden internal changes
(crack growth, delamination) with surface-mounted sensors. The pieces here
map directly onto that workflow:

* **Source models.** A buried **moment tensor** with a step-like time
  history is the canonical micro-crack AE source (Ohtsu & Ono, *J. Acoustic
  Emission* 3, 1984); `double_couple_2d` (shear crack) and `explosion_2d`
  (volumetric) are provided. The **Hsu-Nielsen pencil-lead break**
  (ASTM E976), the standard AE calibration source, is a vertical surface
  point force with `ae_step_source` — see `examples/ae_pencil_break_plate.py`.
* **Forward modelling.** The elastic solver produces synthetic sensor
  waveforms with correct P / S / Rayleigh (and, for plates, Lamb) arrivals,
  for arrival-time picking, source-location and characterization studies.
* **Frequency domain.** The Helmholtz solvers give the time-harmonic
  response and a classical CSLP-preconditioned baseline that the HINTS
  hybrid preconditioner is designed to accelerate.

## Using established libraries for production runs

These modules are intentionally simple and self-contained. For large 3D
models, higher-order accuracy, GPU execution, or PML boundaries, the same
formulations are available in well-known open-source packages:

* **[Devito](https://www.devitoproject.org/)** — symbolic finite-difference
  DSL with optimized/GPU codegen; standard for seismic acoustic & elastic
  FWI. The leapfrog and Virieux schemes here mirror Devito's operators.
* **[k-Wave](http://www.k-wave.org/)** — pseudo-spectral acoustic/ultrasound
  toolbox; its `toneBurst` matches `sources.tone_burst`.
* **[SPECFEM2D/3D](https://specfem.org/)** — spectral-element elastic waves
  with moment-tensor and force sources and PML; reference for AE/seismic.
* **[pyamg](https://github.com/pyamg/pyamg)** — algebraic multigrid; if
  installed, `make_shifted_laplacian_preconditioner(..., use_amg=True)`
  applies a single AMG V-cycle as the CSLP inner solve (the scalable variant
  of Erlangga–Vuik–Oosterlee).

## References

* J. Virieux, *P-SV wave propagation in heterogeneous media: velocity-stress
  finite-difference method*, Geophysics 51 (1986).
* Y.A. Erlangga, C. Vuik, C.W. Oosterlee, *On a class of preconditioners for
  solving the Helmholtz equation*, Appl. Numer. Math. 50 (2004); SIAM J. Sci.
  Comput. 27 (2006).
* C. Cerjan et al., *A nonreflecting boundary condition for discrete acoustic
  and elastic wave equations*, Geophysics 50 (1985).
* G. Mur, *Absorbing boundary conditions for the FD approximation of the
  time-domain electromagnetic-field equations*, IEEE Trans. EMC 23 (1981).
* M. Ohtsu, K. Ono, *A generalized theory of acoustic emission and Green's
  functions in a half space*, J. Acoustic Emission 3 (1984).
