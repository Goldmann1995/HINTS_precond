"""Stage 2 tests: elastic (vector) P-SV Navier-Helmholtz operator + bridge.

The operator-correctness checks (manufactured solution, moment-tensor
radiation, frequency<->time synthesis) are ML-independent. The bridge check
uses an exact-inverse "oracle" so the vector hybrid/FGMRES plumbing is
verified deterministically; a short vector-DeepONet training smoke test runs
only if torch is installed.
"""

import os
import sys

import numpy as np
import scipy.sparse.linalg as spla

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from wave_solvers import elastic_helmholtz as eh
from wave_solvers import hints_bridge as hb


# ---------------------------------------------------------------------------
# Operator correctness
# ---------------------------------------------------------------------------
def _manufactured_error(n, vp=2.0, vs=1.0, rho=1.0, omega=2.0):
    """Solve with ux=uz=sin(pi x)sin(pi z) imposed; return max error.
    Uses nondimensional O(1) parameters so the system is well-conditioned."""
    lam, mu = eh.lame_from_velocities(vp, vs, rho)
    p = eh.ElasticHelmholtz2DProblem(vp, vs, rho, omega, shape=(n, n),
                                     bc='dirichlet')
    x = np.linspace(0, 1, n)
    xx, zz = np.meshgrid(x, x, indexing='ij')
    S = np.sin(np.pi * xx) * np.sin(np.pi * zz)
    C = np.cos(np.pi * xx) * np.cos(np.pi * zz)
    coefS = -(lam + 3 * mu) * np.pi ** 2 + rho * omega ** 2
    coefC = (lam + mu) * np.pi ** 2
    fx = coefS * S + coefC * C
    fz = coefC * C + coefS * S
    ue = np.concatenate([S.reshape(-1), S.reshape(-1)]).astype(complex)
    b = np.concatenate([fx.reshape(-1), fz.reshape(-1)]).astype(complex)
    mask = np.ones((n, n), bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    bm = np.concatenate([mask.reshape(-1), mask.reshape(-1)])
    b = np.where(bm, b, ue)
    u = p.solve_direct(b)
    return np.max(np.abs(u - ue))


def test_elastic_manufactured_second_order():
    errs = [_manufactured_error(n) for n in (21, 41, 81)]
    rate = np.log2(errs[0] / errs[-1]) / 2.0
    assert rate > 1.8, f'observed order {rate:.2f} < 1.8 (errs={errs})'
    print(f'  [ok] elastic Navier-Helmholtz operator: 2nd-order convergence '
          f'(rate {rate:.2f})')


def test_elastic_truncation_consistency():
    """A u_exact - f should be a small (2nd-order) truncation residual even
    with realistic Lame values -- isolates operator consistency from the
    conditioning of the solve."""
    vp, vs, rho, omega = 3000., 1800., 2500., 2 * np.pi * 200.
    lam, mu = eh.lame_from_velocities(vp, vs, rho)
    rels = []
    for n in (21, 41):
        p = eh.ElasticHelmholtz2DProblem(vp, vs, rho, omega, shape=(n, n),
                                         bc='dirichlet')
        x = np.linspace(0, 1, n)
        xx, zz = np.meshgrid(x, x, indexing='ij')
        S = np.sin(np.pi * xx) * np.sin(np.pi * zz)
        C = np.cos(np.pi * xx) * np.cos(np.pi * zz)
        coefS = -(lam + 3 * mu) * np.pi ** 2 + rho * omega ** 2
        coefC = (lam + mu) * np.pi ** 2
        f = np.concatenate([(coefS * S + coefC * C).reshape(-1),
                            (coefC * C + coefS * S).reshape(-1)])
        ue = np.concatenate([S.reshape(-1), S.reshape(-1)]).astype(complex)
        mask = np.ones((n, n), bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
        bm = np.concatenate([mask.reshape(-1), mask.reshape(-1)])
        r = (p.A @ ue - f)[bm]
        rels.append(np.max(np.abs(r)) / np.max(np.abs(f)))
    assert rels[0] / rels[1] > 3.0, f'truncation not 2nd order: {rels}'
    print(f'  [ok] operator consistency: truncation drops ~4x per refine '
          f'({rels[0]:.1e} -> {rels[1]:.1e})')


def test_moment_tensor_and_force_radiate():
    p = eh.ElasticHelmholtz2DProblem(3000., 1800., 2500., 2 * np.pi * 150.,
                                     shape=(64, 64), bc='absorbing')
    # explosion (isotropic) and double-couple should both radiate finite,
    # nonzero, and distinct wavefields
    b_exp = p.moment_tensor(32, 32, mxx=1.0, mzz=1.0, mxz=0.0)
    b_dc = p.moment_tensor(32, 32, mxx=0.0, mzz=0.0, mxz=1.0)
    b_force = p.point_force(32, 32, fx=0.0, fz=1.0)
    for name, b in (('explosion', b_exp), ('double-couple', b_dc),
                    ('point-force', b_force)):
        u = p.solve_direct(b)
        assert np.all(np.isfinite(u)) and np.max(np.abs(u)) > 0, name
    u_exp = p.solve_direct(b_exp)
    u_dc = p.solve_direct(b_dc)
    corr = abs(np.vdot(u_exp, u_dc)) / (np.linalg.norm(u_exp) *
                                        np.linalg.norm(u_dc))
    assert corr < 0.9, f'explosion and double-couple too similar (corr {corr})'
    print(f'  [ok] moment-tensor / point-force sources radiate distinct '
          f'fields (|corr| {corr:.2f})')


def test_freq_to_time_synthesis():
    from wave_solvers.sources import ricker
    dt = 1e-4
    t = np.arange(256) * dt
    stf = ricker(t, f_peak=300.0)
    freqs, spec = eh.source_spectrum(stf, dt)
    # trivial frequency response: identity scalar -> output should be the stf
    out = eh.freq_to_time(lambda omega: np.array([1.0 + 0j]), freqs, spec,
                          n_time=len(t), dt=dt)
    assert out.shape == (len(t), 1)
    err = np.linalg.norm(out[:, 0] - stf) / np.linalg.norm(stf)
    assert err < 1e-6, f'IFFT round-trip failed (err {err:.2e})'
    print('  [ok] frequency<->time synthesis round-trips the source wavelet')


# ---------------------------------------------------------------------------
# Vector HINTS bridge plumbing (oracle = exact inverse)
# ---------------------------------------------------------------------------
class _VectorOracle:
    def __init__(self, problem):
        self.lu = spla.splu(problem.A.tocsc())
        self.shape = (problem.block_size, problem.nx, problem.nz)

    def apply(self, residual_field):
        r = residual_field.reshape(-1)
        return self.lu.solve(r).reshape(self.shape)


def test_vector_oracle_hints_and_fgmres():
    p = eh.ElasticHelmholtz2DProblem(3000., 1800., 2500., 2 * np.pi * 120.,
                                     shape=(40, 40), bc='absorbing')
    b = p.point_force(20, 20, fx=0.0, fz=1.0)
    u_ref = p.solve_direct(b)
    shape = (p.block_size, p.nx, p.nz)
    oracle = _VectorOracle(p)

    # standalone hybrid with oracle correction
    solver = hb.HINTSSolver(p.A, shape, deeponet=oracle, smoother='jacobi',
                            ratio=1, jacobi_omega=0.3, safeguard=True)
    u_h, info = solver.solve(b, maxiter=40, rtol=1e-8)
    err = np.linalg.norm(u_h - u_ref) / np.linalg.norm(u_ref)
    assert info['converged'] and err < 1e-6, \
        f'vector oracle-HINTS failed (conv={info["converged"]}, err={err:.2e})'

    # FGMRES with exact-inverse preconditioner -> ~1 iteration
    _, ginfo = hb.fgmres(p.A, b, prec=lambda v: oracle.apply(
        v.reshape(shape)).reshape(-1), rtol=1e-8, maxiter=10)
    assert ginfo['converged'] and ginfo['iterations'] <= 2
    print(f'  [ok] vector bridge: oracle-HINTS converges in '
          f'{info["iterations"]} its; FGMRES(oracle) in {ginfo["iterations"]}')


def test_vector_deeponet_trains_a_little():
    try:
        import torch  # noqa: F401
    except ImportError:
        print('  [skip] torch not installed')
        return
    p = eh.ElasticHelmholtz2DProblem(2.0, 1.0, 1.0, 6.0, shape=(24, 24),
                                     bc='absorbing')
    rng = np.random.default_rng(0)
    f_tr, u_tr = hb.generate_dataset(p, 200, rng=rng)
    f_va, u_va = hb.generate_dataset(p, 40, rng=rng)
    static = np.stack([p.vp, p.vs], axis=-1)
    don = hb.VectorComplexDeepONet2D(p.nx, p.nz, static, n_comp=2, latent=48)
    rel0 = don.relative_error(f_va, u_va)
    don.fit(f_tr, u_tr, epochs=400, batch_size=64, log_every=400,
            logger=lambda m: None)
    rel = don.relative_error(f_va, u_va)
    assert rel < 0.85 * rel0, f'vector DeepONet did not improve ({rel0:.2f}->{rel:.2f})'
    print(f'  [ok] VectorComplexDeepONet2D learned (val rel {rel0:.2f} -> {rel:.2f})')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    print(f'Running {len(tests)} elastic-Helmholtz (Stage 2) tests...\n')
    for fn in tests:
        print(f'- {fn.__name__}')
        fn()
    print(f'\nAll {len(tests)} tests passed.')
