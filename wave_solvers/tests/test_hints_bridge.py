"""Tests for the Stage 1 HINTS bridge (complex 2D Helmholtz).

These exercise the assembly + hybrid-iteration + preconditioner *plumbing*
without depending on a well-trained network: an "oracle" correction operator
(exact A^{-1} applied to the residual) stands in for the DeepONet, so the
hybrid logic can be verified deterministically. A separate, short-training
test of the real ComplexDeepONet2D runs only if torch is installed.
"""

import os
import sys

import numpy as np
import scipy.sparse.linalg as spla

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from wave_solvers import helmholtz as hz
from wave_solvers import elastic_helmholtz as eh
from wave_solvers import hints_bridge as hb


def _small_problem(grid=24, freq=2500.0, seed=1):
    rng = np.random.default_rng(seed)
    vel = eh.sample_velocity_field(grid, grid, c0=1500.0, n_defects=1, rng=rng)
    return eh.ScalarHelmholtz2DProblem(vel, 2 * np.pi * freq, bc='sommerfeld')


class _OracleDeepONet:
    """Exact inverse used as a perfect DeepONet stand-in for plumbing tests."""

    def __init__(self, problem):
        self.lu = spla.splu(problem.A.tocsc())
        self.shape = (problem.nx, problem.nz)

    def apply(self, residual_field):
        r = residual_field.reshape(-1)
        return self.lu.solve(r).reshape(self.shape)


def test_assembly_matches_helmholtz_module():
    p = _small_problem()
    A = p.A
    assert A.shape == (p.nx * p.nz,) * 2
    assert np.iscomplexobj(A.toarray())
    nd = p.nondim()
    assert nd['points_per_wavelength'] > 0
    print(f'  [ok] assembly: ndof={nd["ndof"]}, '
          f'ppw={nd["points_per_wavelength"]:.1f}')


def test_fft_band_energy_split():
    nx = nz = 32
    x = np.linspace(0, 1, nx)
    xx, zz = np.meshgrid(x, x, indexing='ij')
    low = np.sin(np.pi * xx) * np.sin(np.pi * zz)        # smooth
    high = np.sin(8 * np.pi * xx) * np.sin(8 * np.pi * zz)  # oscillatory
    lo_l, hi_l = hb.fft_band_energy(low)
    lo_h, hi_h = hb.fft_band_energy(high)
    assert lo_l > hi_l, 'smooth field should be low-band dominated'
    assert hi_h > lo_h, 'oscillatory field should be high-band dominated'
    print('  [ok] FFT band-energy split separates smooth vs oscillatory')


def test_pure_jacobi_diverges_oracle_hints_converges():
    p = _small_problem()
    b = p.point_source(p.nx // 2, p.nz // 2)
    u_ref = p.solve_direct(b)

    plain = hb.HINTSSolver(p.A, (p.nx, p.nz), deeponet=None,
                           smoother='jacobi', jacobi_omega=0.5)
    _, info_plain = plain.solve(b, maxiter=80, rtol=1e-8)
    assert not info_plain['converged'], 'pure Jacobi unexpectedly converged'

    oracle = _OracleDeepONet(p)
    hints = hb.HINTSSolver(p.A, (p.nx, p.nz), deeponet=oracle,
                           smoother='jacobi', ratio=5, jacobi_omega=0.5)
    u_h, info_h = hints.solve(b, maxiter=80, rtol=1e-8)
    err = np.linalg.norm(u_h - u_ref) / np.linalg.norm(u_ref)
    assert info_h['converged'] and err < 1e-6, \
        f'oracle-HINTS failed to converge (err={err:.2e})'
    print(f'  [ok] pure Jacobi diverges (res {info_plain["residuals"][-1]:.1e}); '
          f'oracle-HINTS converges in {info_h["iterations"]} its '
          f'(err {err:.1e})')


def test_hints_preconditioner_runs_and_matches_direct():
    p = _small_problem()
    b = p.point_source(p.nx // 2, p.nz // 2)
    u_ref = p.solve_direct(b)
    cslp = hb.CSLPSmoother(eh._hz.helmholtz_matrix_2d, beta=(1.0, 0.5),
                           nx=p.nx, ny=p.nz, k=p.k, bc='sommerfeld')
    oracle = _OracleDeepONet(p)
    M = hb.make_hints_preconditioner(p.A, (p.nx, p.nz), oracle, cslp)
    u, info = hz.solve_gmres(p.A, b, M=M, rtol=1e-8, maxiter=200)
    err = np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref)
    assert info['converged'] and err < 1e-6
    print(f'  [ok] HINTS-CSLP preconditioned GMRES: {info["iterations"]} its, '
          f'err {err:.1e}')


def test_dataset_solutions_are_consistent():
    p = _small_problem()
    f, u = hb.generate_dataset(p, 3, rng=np.random.default_rng(0))
    # verify A u = f for the generated pairs
    res = np.linalg.norm(p.A @ u[0].reshape(-1) - f[0].reshape(-1))
    assert res < 1e-8 * (np.linalg.norm(f[0]) + 1e-30)
    print('  [ok] generated dataset pairs satisfy A u = f')


def test_complex_deeponet_trains_a_little():
    """Short smoke-training: torch only; checks the net learns (val error
    drops well below 1.0). Skipped automatically if torch is absent."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print('  [skip] torch not installed')
        return
    p = _small_problem(grid=24)
    rng = np.random.default_rng(0)
    f_tr, u_tr = hb.generate_dataset(p, 200, rng=rng)
    f_va, u_va = hb.generate_dataset(p, 40, rng=rng)
    don = hb.ComplexDeepONet2D(p.nx, p.nz, p.k, latent=48)
    rel0 = don.relative_error(f_va, u_va)            # untrained baseline (~1)
    don.fit(f_tr, u_tr, epochs=400, batch_size=64, log_every=400,
            logger=lambda m: None)
    rel = don.relative_error(f_va, u_va)
    # short CPU smoke-training: just require a clear improvement over the
    # untrained network (full accuracy comes from the demo's longer run).
    assert rel < 0.7 * rel0, \
        f'DeepONet did not improve (before={rel0:.2f}, after={rel:.2f})'
    print(f'  [ok] ComplexDeepONet2D learned (400 epochs, '
          f'val rel.err {rel0:.2f} -> {rel:.2f})')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    print(f'Running {len(tests)} HINTS-bridge tests...\n')
    for fn in tests:
        print(f'- {fn.__name__}')
        fn()
    print(f'\nAll {len(tests)} tests passed.')
