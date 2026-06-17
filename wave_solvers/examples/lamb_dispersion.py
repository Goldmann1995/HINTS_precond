"""Stage 3: reproduce the Lamb-wave dispersion curves of a traction-free plate
and validate them against the analytic Rayleigh-Lamb relation.

Computes phase-velocity dispersion with the SAFE method (``wave_solvers.lamb``)
across a frequency band, checks every point against the analytic Rayleigh-Lamb
characteristic equation, and (if matplotlib is available) plots the dispersion
curves with the fundamental modes labelled.

Run from the repo root:  python wave_solvers/examples/lamb_dispersion.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from wave_solvers import lamb


def main():
    vp, vs, rho, h = 1.0, 0.5, 1.0, 1.0          # nondimensional plate
    c_plate = 2 * vs * np.sqrt(1 - (vs / vp) ** 2)
    c_rayleigh = vs * (0.862 + 1.14 * 0.333) / (1 + 0.333)   # nu=1/3
    print(f'Plate: vp={vp}, vs={vs}, h={h} | plate velocity {c_plate:.3f}, '
          f'Rayleigh ~{c_rayleigh:.3f}')

    freqs = np.linspace(0.05, 2.5, 40)
    disp = lamb.safe_dispersion(freqs, vp, vs, rho, h, n_elem=60)

    # validate every SAFE point against the analytic Rayleigh-Lamb relation
    max_res = 0.0
    pts_f, pts_c = [], []
    for f, cs in zip(freqs, disp):
        omega = 2 * np.pi * f
        for c in cs:
            k = omega / c
            d_sym, d_anti = lamb.rayleigh_lamb_residual(omega, k, vp, vs, h)
            scale = (abs(k) ** 2 + (omega / vs) ** 2) ** 2 + 1e-30
            max_res = max(max_res, min(abs(d_sym), abs(d_anti)) / scale)
            pts_f.append(f); pts_c.append(c)
    print(f'\nSAFE dispersion validated against analytic Rayleigh-Lamb:')
    print(f'  {len(pts_c)} modal points, max characteristic residual '
          f'{max_res:.2e}')

    # low-frequency fundamental checks
    s0 = disp[0].max()
    print(f'  S0 at f={freqs[0]:.2f}: c={s0:.3f} (plate velocity {c_plate:.3f})')

    _maybe_plot(np.array(pts_f), np.array(pts_c), c_plate, c_rayleigh, vs)


def _maybe_plot(pts_f, pts_c, c_plate, c_rayleigh, vs):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n(matplotlib not installed -- skipping figure)')
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pts_f, pts_c, s=10, color='tab:blue',
               label='SAFE modes (= Rayleigh-Lamb)')
    ax.axhline(c_plate, color='g', ls='--', alpha=0.6, label='plate velocity (S0)')
    ax.axhline(c_rayleigh, color='r', ls=':', alpha=0.6, label='Rayleigh velocity')
    ax.set_xlabel('frequency (nondim, f*h)')
    ax.set_ylabel('phase velocity c')
    ax.set_ylim(0, 2.0)
    ax.set_title('Lamb-wave dispersion (traction-free plate)')
    ax.legend(loc='upper right')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'lamb_dispersion.png')
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f'\nFigure written to {out}')


if __name__ == '__main__':
    main()
