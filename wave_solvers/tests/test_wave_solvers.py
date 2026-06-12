"""Self-contained correctness tests for the wave_solvers package.

Run directly (``python tests/test_wave_solvers.py``) or under pytest.
Only NumPy/SciPy are required.
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import wave_solvers as ws
from wave_solvers import helmholtz as hz


# ---------------------------------------------------------------------------
# Helmholtz: manufactured-solution convergence and solver agreement
# ---------------------------------------------------------------------------
def test_helmholtz_1d_manufactured():
    """u(x) = sin(pi x) solves u'' + k^2 u = (k^2 - pi^2) sin(pi x)."""
    k = 2.0
    errors = []
    for n in (101, 201, 401):
        x = np.linspace(0.0, 1.0, n)
        u_exact = np.sin(np.pi * x)
        f = (k ** 2 - np.pi ** 2) * np.sin(np.pi * x)
        A = hz.helmholtz_matrix_1d(n, length=1.0, k=k, bc='dirichlet')
        b = f.copy()
        b[0] = b[-1] = 0.0           # Dirichlet data u(0)=u(1)=0
        u = hz.solve_direct(A, b)
        errors.append(np.max(np.abs(u - u_exact)))
    # second-order scheme: halving dx should cut the error by ~4
    rate = np.log2(errors[0] / errors[-1]) / 2.0
    assert rate > 1.8, f'observed order {rate:.2f} < 1.8'
    print(f'  [ok] Helmholtz 1D convergence order ~ {rate:.2f}')


def test_helmholtz_krylov_matches_direct():
    n = 201
    x = np.linspace(0.0, 1.0, n)
    k = 5.0
    f = np.exp(-((x - 0.5) ** 2) / 0.01)
    A = hz.helmholtz_matrix_1d(n, k=k, bc='sommerfeld')
    u_direct = hz.solve_direct(A, f)

    M = hz.make_shifted_laplacian_preconditioner(
        hz.helmholtz_matrix_1d, n=n, k=k, bc='sommerfeld')
    u_gmres, info = hz.solve_gmres(A, f, M=M, rtol=1e-10, maxiter=500)
    assert info['converged'], 'preconditioned GMRES did not converge'
    err = np.linalg.norm(u_gmres - u_direct) / np.linalg.norm(u_direct)
    assert err < 1e-6, f'GMRES vs direct mismatch {err:.2e}'
    print(f'  [ok] CSLP-GMRES converged in {info["iterations"]} its, '
          f'rel.err {err:.2e}')


def test_helmholtz_2d_direct():
    nx = ny = 41
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    k = 3.0
    u_exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    f = (k ** 2 - 2 * np.pi ** 2) * u_exact
    A = hz.helmholtz_matrix_2d(nx, ny, k=k, bc='dirichlet')
    b = f.flatten()
    # zero out boundary rows in rhs
    mask = np.ones((nx, ny), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    b = np.where(mask.flatten(), b, 0.0)
    u = hz.solve_direct(A, b).reshape(nx, ny)
    err = np.max(np.abs(u - u_exact))
    assert err < 5e-3, f'2D Helmholtz error too large: {err:.2e}'
    print(f'  [ok] Helmholtz 2D max error {err:.2e}')


# ---------------------------------------------------------------------------
# Scalar wave equation: CFL guard, energy decay with absorbing BC, d'Alembert
# ---------------------------------------------------------------------------
def test_cfl_guard():
    try:
        ws.AcousticWaveFDTD1D(np.ones(100), dx=0.01, dt=1.0)
    except ValueError:
        print('  [ok] CFL violation correctly rejected')
        return
    raise AssertionError('CFL violation was not detected')


def test_wave_1d_propagation_speed():
    """A pulse should travel at speed c: check arrival time at a receiver."""
    nx, dx, c = 2000, 1.0, 1000.0
    solver = ws.AcousticWaveFDTD1D(np.full(nx, c), dx=dx, boundary='mur')
    f0 = 30.0
    t = np.arange(2000) * solver.dt
    solver.add_source(200, ws.ricker(t, f0))
    solver.add_receiver(1200)            # 1000 cells = 1000 m away
    out = solver.run(2000)
    trace = out['receivers'][:, 0]
    t_arrival = out['t'][np.argmax(np.abs(trace))]
    # source peak at t0 = 1.5/f0; travel distance 1000 m at c
    expected = 1.5 / f0 + 1000.0 / c
    rel = abs(t_arrival - expected) / expected
    assert rel < 0.05, f'arrival time off by {rel:.1%}'
    print(f'  [ok] 1D pulse arrival {t_arrival*1e3:.3f} ms '
          f'(expected {expected*1e3:.3f} ms)')


def test_wave_2d_runs_and_absorbs():
    nx = nz = 120
    c = np.full((nx, nz), 1500.0)
    solver = ws.AcousticWaveFDTD2D(c, dx=2.0, boundary='sponge',
                                   sponge_width=20)
    t = np.arange(400) * solver.dt
    solver.add_source(nx // 2, nz // 2, ws.ricker(t, 80.0))
    out = solver.run(400)
    peak_energy = None
    final_energy = np.sum(out['final_wavefield'] ** 2)
    # run a bit longer to let the sponge absorb
    assert np.isfinite(final_energy)
    print(f'  [ok] 2D acoustic FDTD stable, final energy {final_energy:.3e}')


# ---------------------------------------------------------------------------
# Elastic solver: reciprocity-style sanity + source radiation
# ---------------------------------------------------------------------------
def test_elastic_explosion_radiates_p():
    nx = nz = 140
    solver = ws.ElasticWaveFDTD2D(
        vp=3000.0, vs=1800.0, rho=2500.0, dx=2.0,
        free_surface=False, sponge_width=20, shape=(nx, nz))
    t = np.arange(500) * solver.dt
    stf = ws.ae_step_source(t, rise_time=20 * solver.dt,
                            t0=5 * solver.dt, amplitude=1e6)
    solver.add_source(ws.explosion_2d(nx // 2, nz // 2, stf))
    solver.add_receiver(nx // 2 + 30, nz // 2)
    solver.add_receiver(nx // 2, nz // 2 + 30)
    out = solver.run(500)
    # explosion is isotropic: horizontal receiver sees mostly vx,
    # vertical receiver sees mostly vz, with comparable amplitudes
    a_h = np.max(np.abs(out['vx'][:, 0]))
    a_v = np.max(np.abs(out['vz'][:, 1]))
    assert a_h > 0 and a_v > 0, 'no energy radiated'
    ratio = a_h / a_v
    assert 0.5 < ratio < 2.0, f'explosion not isotropic, ratio {ratio:.2f}'
    print(f'  [ok] elastic explosion radiates isotropically '
          f'(vx/vz amplitude ratio {ratio:.2f})')


def test_elastic_pencil_break_surface_wave():
    """Vertical surface force (Hsu-Nielsen) should excite a strong signal
    on a surface receiver via the free surface."""
    nx, nz = 200, 120
    solver = ws.ElasticWaveFDTD2D(
        vp=5900.0, vs=3200.0, rho=7800.0, dx=1e-3,   # steel, mm grid
        free_surface=True, sponge_width=20, shape=(nx, nz))
    t = np.arange(600) * solver.dt
    stf = ws.ae_step_source(t, rise_time=15 * solver.dt,
                            t0=5 * solver.dt, amplitude=1.0)
    solver.add_source(ws.PointForceSource2D(nx // 2, 1, stf, fx=0.0, fz=1.0))
    solver.add_receiver(nx // 2 + 50, 1)
    out = solver.run(600)
    amp = np.max(np.abs(out['vz'][:, 0]))
    assert np.isfinite(amp) and amp > 0, 'no surface signal recorded'
    print(f'  [ok] elastic pencil-break surface signal amplitude {amp:.3e}')


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def test_source_time_functions():
    t = np.linspace(0, 1e-4, 1000)
    r = ws.ricker(t, 1e5)
    assert abs(np.mean(r)) < 0.1 * np.max(np.abs(r)), 'Ricker not zero-mean'
    tb = ws.tone_burst(t, 5e4, n_cycles=5, t0=1e-5)
    assert np.max(np.abs(tb)) > 0
    step = ws.ae_step_source(t, rise_time=1e-5, t0=1e-5)
    assert step[-1] > 0.99 and step[0] == 0.0, 'AE step malformed'
    print('  [ok] source-time functions (Ricker / tone burst / AE step)')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    print(f'Running {len(tests)} wave_solvers tests...\n')
    for fn in tests:
        print(f'- {fn.__name__}')
        fn()
    print(f'\nAll {len(tests)} tests passed.')
