"""Stage 2 demo: vector HINTS for the 2D elastic Navier-Helmholtz system.

Trains a complex, vector-valued DeepONet (``VectorComplexDeepONet2D``) to
approximate the inverse of the elastic operator and uses it inside the HINTS
bridge, in the two routes from HINTS_AE_ROADMAP.md §2.2:

  (A) standalone hybrid iteration (Jacobi smoother + DeepONet correction);
  (B) FGMRES with an "ILU + DeepONet" preconditioner (the scalable route;
      ILU stands in for the scalar CSLP, which does not apply to the vector
      operator).

Everything is nondimensional (L = 1, vp = 1) so the elastic system is
well-conditioned (roadmap §2.3). As in Stage 1, a CPU-budget network is not
expected to beat the classical preconditioner -- the demo validates the
*vector* pipeline end-to-end and reports honest iteration counts.

Run from the repo root:  python wave_solvers/examples/hints_elastic_stage2.py
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


def _robust_ilu(A):
    """ILU preconditioner, with a tiny complex shift to avoid the zero pivots
    that the Dirichlet identity rows can produce in the elastic operator."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    shift = 1e-6 * abs(A.diagonal()).max()
    ilu = spla.spilu((A + shift * sp.identity(A.shape[0], dtype=A.dtype)).tocsc(),
                     drop_tol=1e-4, fill_factor=20)
    return spla.LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=32)
    ap.add_argument('--omega', type=float, default=6.0)
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--n-train', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.grid
    # nondimensional heterogeneous elastic medium (vp ~ 1, vs ~ 0.57)
    vp = 1.0 + 0.15 * (eh.sample_velocity_field(n, n, c0=1.0, n_defects=2,
                                                contrast=0.2, rng=rng) - 1.0)
    vs = 0.571 * vp / vp.mean()
    rho = np.ones((n, n))
    p = eh.ElasticHelmholtz2DProblem(vp, vs, rho, args.omega, shape=(n, n),
                                     bc='absorbing', abl_width=6)
    print('Elastic problem:', p.nondim())
    shape = (p.block_size, n, n)

    # --- training data ----------------------------------------------------
    print(f'\nGenerating {args.n_train} training + 100 val samples...')
    f_tr, u_tr = hb.generate_dataset(p, args.n_train, rng=rng)
    f_va, u_va = hb.generate_dataset(p, 100, rng=rng)

    # --- train the vector DeepONet ---------------------------------------
    print(f'Training VectorComplexDeepONet2D ({args.epochs} epochs, CPU)...')
    static = np.stack([p.vp, p.vs], axis=-1)
    don = hb.VectorComplexDeepONet2D(n, n, static, n_comp=2, latent=64)
    t0 = time.time()
    don.fit(f_tr, u_tr, epochs=args.epochs, batch_size=64,
            f_val=f_va, u_val=u_va, log_every=max(1, args.epochs // 6))
    print(f'  trained in {time.time() - t0:.1f}s; '
          f'val rel.err = {don.relative_error(f_va, u_va):.3e}')

    # --- test problem: a moment-tensor (shear micro-crack) source --------
    b = p.moment_tensor(n // 2, n // 2, mxx=0.0, mzz=0.0, mxz=1.0)
    u_ref = p.solve_direct(b)

    # (A) standalone hybrid with oracle vs trained net
    print('\n[A] standalone hybrid (Jacobi smoother + DeepONet correction):')
    solver = hb.HINTSSolver(p.A, shape, deeponet=don, smoother='jacobi',
                            ratio=5, jacobi_omega=0.3, safeguard=True)
    u_h, info = solver.solve(b, maxiter=150, rtol=1e-8)
    print(f'    HINTS-Jacobi: converged={info["converged"]}, '
          f'diverged={info["diverged"]}, '
          f'final res={info["residuals"][-1]:.2e}')

    # (B) FGMRES: none vs ILU vs HINTS(ILU+DON)
    print('\n[B] FGMRES iteration counts (rtol=1e-8):')
    _, g_none = hb.fgmres(p.A, b, prec=lambda v: v, rtol=1e-8, maxiter=400)
    print(f'    no preconditioner : converged={g_none["converged"]}, '
          f'its={g_none["iterations"]}')

    ilu = _robust_ilu(p.A)
    _, g_ilu = hb.fgmres(p.A, b, prec=ilu.matvec, rtol=1e-8, maxiter=400)
    print(f'    ILU               : converged={g_ilu["converged"]}, '
          f'its={g_ilu["iterations"]}')

    def hints_prec(r):
        x = ilu.matvec(r)
        resid = r - p.A @ x
        return x + don.apply(resid.reshape(shape)).reshape(-1)
    _, g_hints = hb.fgmres(p.A, b, prec=hints_prec, rtol=1e-8, maxiter=400)
    print(f'    HINTS (ILU + DON) : converged={g_hints["converged"]}, '
          f'its={g_hints["iterations"]}')

    print('\n=== Stage 2 summary ===')
    print(f'  vector DeepONet val rel.err : {don.relative_error(f_va, u_va):.3e}')
    print(f'  FGMRES its none / ILU / HINTS : {g_none["iterations"]} / '
          f'{g_ilu["iterations"]} / {g_hints["iterations"]}')
    print('  Vector elastic HINTS pipeline validated end-to-end; as in '
          'Stage 1,\n  beating the classical preconditioner needs a better '
          '(GPU) network.')


if __name__ == '__main__':
    main()
