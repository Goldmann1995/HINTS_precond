"""Stage 3 tests: Lamb-wave dispersion in a traction-free plate.

Validates the SAFE dispersion solver against (a) the analytic Rayleigh-Lamb
characteristic equation and (b) known physical limits (the S0 mode tends to
the plate velocity, the A0 mode tends to zero as frequency -> 0). All ML-
independent.
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from wave_solvers import lamb

VP, VS, RHO, H = 1.0, 0.5, 1.0, 1.0          # nondimensional plate


def test_s0_tends_to_plate_velocity():
    """As frequency -> 0 the symmetric S0 mode approaches the plate
    (extensional) velocity c_p = 2 vs sqrt(1 - (vs/vp)^2)."""
    c_plate = 2 * VS * np.sqrt(1 - (VS / VP) ** 2)
    c = lamb.safe_dispersion([0.03], VP, VS, RHO, H, n_elem=40)[0]
    s0 = c.max()                               # S0 is the fast fundamental
    assert abs(s0 - c_plate) / c_plate < 0.02, \
        f'S0={s0:.3f} vs plate velocity {c_plate:.3f}'
    print(f'  [ok] S0 -> plate velocity at low f ({s0:.3f} ~ {c_plate:.3f})')


def test_a0_tends_to_zero():
    """The antisymmetric A0 (flexural) mode slows toward zero as f -> 0."""
    c_lo = lamb.safe_dispersion([0.03], VP, VS, RHO, H, n_elem=40)[0].min()
    c_hi = lamb.safe_dispersion([0.12], VP, VS, RHO, H, n_elem=40)[0].min()
    assert c_lo < c_hi, 'A0 should slow down toward zero frequency'
    print(f'  [ok] A0 slows toward f=0 (c={c_lo:.3f} < {c_hi:.3f})')


def test_safe_satisfies_rayleigh_lamb():
    """Every SAFE dispersion point must (nearly) zero the analytic
    Rayleigh-Lamb characteristic function -- the rigorous cross-check."""
    max_res = 0.0
    for f in (0.3, 0.6, 1.0):
        omega = 2 * np.pi * f
        for c in lamb.safe_dispersion([f], VP, VS, RHO, H, n_elem=80,
                                      imag_tol=1e-5)[0]:
            k = omega / c
            d_sym, d_anti = lamb.rayleigh_lamb_residual(omega, k, VP, VS, H)
            scale = (abs(k) ** 2 + (omega / VS) ** 2) ** 2 + 1e-30
            max_res = max(max_res, min(abs(d_sym), abs(d_anti)) / scale)
    assert max_res < 5e-3, f'SAFE deviates from Rayleigh-Lamb ({max_res:.2e})'
    print(f'  [ok] SAFE modes satisfy Rayleigh-Lamb (max residual {max_res:.1e})')


def test_safe_converges_with_thickness_refinement():
    """SAFE (linear elements) is 2nd-order: the Rayleigh-Lamb residual at its
    dispersion points drops ~4x when the thickness mesh is doubled."""
    def max_residual(n_elem):
        mr = 0.0
        for f in (0.4, 0.8):
            omega = 2 * np.pi * f
            for c in lamb.safe_dispersion([f], VP, VS, RHO, H, n_elem=n_elem,
                                          imag_tol=1e-5)[0]:
                k = omega / c
                ds, da = lamb.rayleigh_lamb_residual(omega, k, VP, VS, H)
                scale = (abs(k) ** 2 + (omega / VS) ** 2) ** 2 + 1e-30
                mr = max(mr, min(abs(ds), abs(da)) / scale)
        return mr
    r1, r2 = max_residual(40), max_residual(80)
    assert r1 / r2 > 3.0, f'not 2nd order: {r1:.2e} -> {r2:.2e}'
    print(f'  [ok] SAFE converges ~4x per refinement ({r1:.1e} -> {r2:.1e})')


def test_analytic_root_finder_matches_safe_fundamentals():
    """At a low frequency (only A0, S0 exist) the analytic root finder and
    SAFE agree on both fundamental phase velocities."""
    f = 0.1
    safe = lamb.safe_dispersion([f], VP, VS, RHO, H, n_elem=80)[0]
    ana = lamb.analytic_phase_velocities(f, VP, VS, H, c_max=1.2)
    # match each analytic root to the nearest SAFE value
    for a in ana:
        assert np.min(np.abs(safe - a)) / a < 0.02, \
            f'analytic c={a:.3f} not matched by SAFE {np.round(safe,3)}'
    print(f'  [ok] analytic and SAFE agree on fundamentals '
          f'(c={np.round(ana,3)})')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    print(f'Running {len(tests)} Lamb-dispersion (Stage 3) tests...\n')
    for fn in tests:
        print(f'- {fn.__name__}')
        fn()
    print(f'\nAll {len(tests)} tests passed.')
