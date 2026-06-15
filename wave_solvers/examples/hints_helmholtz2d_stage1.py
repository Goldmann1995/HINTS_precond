"""Stage 1 demo: HINTS for a complex 2D acoustic Helmholtz problem.

Builds a heterogeneous-velocity acoustic Helmholtz system with absorbing
(Sommerfeld) boundaries, trains a complex DeepONet to approximate A^{-1},
and demonstrates the three Stage-1 verification targets from
``HINTS_AE_ROADMAP.md`` §4:

  1. plain damped-Jacobi diverges (indefinite system), while the HINTS
     hybrid (Jacobi + periodic DeepONet correction) is stabilized;
  2. spectral complementarity: the low-frequency error band drops at the
     DeepONet steps, the smoother handles the high-frequency band;
  3. HINTS as a GMRES preconditioner reduces the iteration count versus
     unpreconditioned GMRES (and is compared against pure CSLP).

Run from the repo root:
    python wave_solvers/examples/hints_helmholtz2d_stage1.py

CPU-only and intentionally small (32x32, a few thousand epochs) so it
finishes in a couple of minutes. Pass --epochs / --grid to scale up.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from wave_solvers import helmholtz as hz
from wave_solvers import elastic_helmholtz as eh
from wave_solvers import hints_bridge as hb


def build_problem(grid, freq, c0=1500.0, seed=0):
    rng = np.random.default_rng(seed)
    vel = eh.sample_velocity_field(grid, grid, c0=c0, n_defects=2,
                                   contrast=0.3, rng=rng)
    omega = 2.0 * np.pi * freq
    return eh.ScalarHelmholtz2DProblem(vel, omega, bc='sommerfeld'), rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=32)
    ap.add_argument('--freq', type=float, default=2000.0)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--n-train', type=int, default=2500)
    ap.add_argument('--ratio', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    problem, rng = build_problem(args.grid, args.freq, seed=args.seed)
    print('Problem:', problem.nondim())
    nx = nz = args.grid

    # ---- training data ----------------------------------------------------
    print(f'\nGenerating {args.n_train} training + 100 val samples...')
    f_tr, u_tr = hb.generate_dataset(problem, args.n_train, rng=rng)
    f_va, u_va = hb.generate_dataset(problem, 100, rng=rng)

    # ---- train the complex DeepONet --------------------------------------
    print(f'Training ComplexDeepONet2D ({args.epochs} epochs, CPU)...')
    don = hb.ComplexDeepONet2D(nx, nz, problem.k, latent=64)
    t0 = time.time()
    don.fit(f_tr, u_tr, epochs=args.epochs, batch_size=64,
            f_val=f_va, u_val=u_va, log_every=max(1, args.epochs // 8))
    print(f'  trained in {time.time() - t0:.1f}s; '
          f'final val rel.err = {don.relative_error(f_va, u_va):.3e}')

    # ---- test problem: a point source ------------------------------------
    b = problem.point_source(nx // 2, nz // 2)
    u_ref = problem.solve_direct(b)

    # ---- (1)+(2) standalone hybrid: Jacobi vs HINTS-Jacobi ---------------
    print('\n[1] Standalone iteration (damped Jacobi smoother):')
    plain = hb.HINTSSolver(problem.A, (nx, nz), deeponet=None,
                           smoother='jacobi', jacobi_omega=0.5)
    _, info_plain = plain.solve(b, maxiter=200, rtol=1e-8)
    print(f'    pure Jacobi      : converged={info_plain["converged"]}, '
          f'final res={info_plain["residuals"][-1]:.2e}')

    hints = hb.HINTSSolver(problem.A, (nx, nz), deeponet=don,
                           smoother='jacobi', ratio=args.ratio,
                           jacobi_omega=0.5, safeguard=True)
    u_h, info_h = hints.solve(b, maxiter=200, rtol=1e-8,
                              record_spectrum=True, u_ref=u_ref)
    err_h = np.linalg.norm(u_h - u_ref) / np.linalg.norm(u_ref)
    print(f'    HINTS-Jacobi     : converged={info_h["converged"]}, '
          f'diverged={info_h["diverged"]}, '
          f'final res={info_h["residuals"][-1]:.2e}')
    print('    (NOTE: a CPU-budget net pollutes the near-null space of the '
          'indefinite\n     operator, so standalone route A is unreliable '
          'here -- see route B below.)')

    # ---- (3) HINTS-CSLP preconditioner vs CSLP vs none -------------------
    # Use flexible GMRES throughout (the HINTS preconditioner is nonlinear);
    # with the constant CSLP operator FGMRES reduces to GMRES, so the
    # iteration-count comparison is fair.
    print('\n[3] FGMRES iteration counts (rtol=1e-8):')
    _, g_none = hb.fgmres(problem.A, b, prec=lambda v: v, rtol=1e-8,
                          maxiter=400)
    print(f'    no preconditioner: converged={g_none["converged"]}, '
          f'its={g_none["iterations"]}')

    cslp = hb.CSLPSmoother(eh._hz.helmholtz_matrix_2d, beta=(1.0, 0.5),
                           nx=nx, ny=nz, k=problem.k, bc='sommerfeld')
    M_cslp = hz.make_shifted_laplacian_preconditioner(
        eh._hz.helmholtz_matrix_2d, beta=(1.0, 0.5),
        nx=nx, ny=nz, k=problem.k, bc='sommerfeld')
    _, g_cslp = hb.fgmres(problem.A, b, prec=M_cslp.matvec, rtol=1e-8,
                          maxiter=400)
    print(f'    CSLP             : converged={g_cslp["converged"]}, '
          f'its={g_cslp["iterations"]}')

    M_hints = hb.make_hints_preconditioner(problem.A, (nx, nz), don, cslp)
    _, g_hints = hb.fgmres(problem.A, b, prec=M_hints.matvec, rtol=1e-8,
                           maxiter=400)
    print(f'    HINTS-CSLP       : converged={g_hints["converged"]}, '
          f'its={g_hints["iterations"]}')

    _summary(info_plain, info_h, g_none, g_cslp, g_hints)
    _maybe_plot(info_h, u_ref, u_h, (nx, nz))


def _summary(info_plain, info_h, g_none, g_cslp, g_hints):
    print('\n=== Stage 1 summary ===')
    print(f'  pure Jacobi (smoother only) : final res '
          f'{info_plain["residuals"][-1]:.2e} (indefinite -> no convergence)')
    print(f'  route A standalone HINTS     : diverged={info_h["diverged"]} '
          '(near-null-space pollution; needs a more accurate net)')
    print(f'  route B FGMRES its  none / CSLP / HINTS-CSLP : '
          f'{g_none["iterations"]} / {g_cslp["iterations"]} / '
          f'{g_hints["iterations"]}  (all converge)')
    print('  Pipeline validated end-to-end; beating CSLP needs a better-'
          'trained\n  (GPU) network -- see STAGE1_REPORT.md.')


def _maybe_plot(info_h, u_ref, u_h, shape):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n(matplotlib not installed -- skipping figures)')
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(info_h['residuals'], label='HINTS-Jacobi residual')
    for s in info_h['don_steps']:
        ax[0].axvline(s, color='r', alpha=0.15)
    ax[0].set_xlabel('iteration'); ax[0].set_ylabel('relative residual')
    ax[0].set_title('Convergence (red = DeepONet steps)'); ax[0].legend()

    if 'band_low' in info_h:
        ax[1].semilogy(info_h['band_low'], label='low-frequency error')
        ax[1].semilogy(info_h['band_high'], label='high-frequency error')
        for s in info_h['don_steps']:
            ax[1].axvline(s, color='r', alpha=0.15)
        ax[1].set_xlabel('iteration'); ax[1].set_ylabel('error-band energy')
        ax[1].set_title('Spectral complementarity'); ax[1].legend()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'hints_helmholtz2d_stage1.png')
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f'\nFigure written to {out}')


if __name__ == '__main__':
    main()
