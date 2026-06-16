"""Stage 2 cross-validation: frequency-domain elastic Helmholtz (with a PML)
vs time-domain FDTD (HINTS_AE_ROADMAP.md §4, Stage 2 headline verification).

A vertical point force with a Ricker time history is applied in a homogeneous
2D elastic medium. The out-of-plane particle velocity at a receiver is
computed two independent ways and cross-correlated:

  * time domain  : ``elastic_fdtd.ElasticWaveFDTD2D`` (Virieux staggered grid);
  * frequency domain : for each frequency in the Ricker band, solve the
    complex elastic Navier-Helmholtz system with a perfectly-matched layer
    (``ElasticHelmholtz2DProblem(bc='pml')``), multiply the unit-force
    displacement response by ``i*omega`` (-> velocity) and the source
    spectrum, then inverse-FFT (``freq_to_time``).

Three ingredients make the two agree to >0.99 (vs the roadmap's 0.95 target):
  1. a true PML (the simple diagonal 'absorbing' layer cannot damp wavelengths
     comparable to the domain, leaving acausal standing waves);
  2. the raw (un-conjugated) Green's function, which the PML already makes
     causal under numpy's e^{+i omega t} IFFT convention;
  3. a velocity-to-velocity comparison (multiply by i*omega), avoiding the
     drift of integrating the FDTD velocity to displacement.

Everything is nondimensional (L = 1, vp = 1) so the elastic solve is
well-conditioned (roadmap §2.3).

Run from the repo root:  python wave_solvers/examples/elastic_freq_vs_time.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from wave_solvers import elastic_helmholtz as eh
from wave_solvers.elastic_fdtd import ElasticWaveFDTD2D
from wave_solvers.sources import ricker, PointForceSource2D


def main():
    vp, vs, rho = 1.0, 0.571, 1.0          # nondimensional (vp = 1)
    n, L = 130, 1.0
    dx = L / (n - 1)
    src, rec = (45, 65), (95, 65)          # x-offset receiver: SV-dominated
    offset = abs(rec[0] - src[0]) * dx
    f0 = 4.0
    t_p, t_s = offset / vp, offset / vs

    # --- time-domain reference (velocity) --------------------------------
    fd = ElasticWaveFDTD2D(vp, vs, rho, dx=dx, cfl=0.4, free_surface=False,
                           sponge_width=25, shape=(n, n))
    dt = fd.dt
    nt = int(6.0 / dt)
    t = np.arange(nt) * dt
    t0 = 2.0 / f0
    stf = ricker(t, f_peak=f0, t0=t0)
    fd.add_source(PointForceSource2D(src[0], src[1], stf, fx=0.0, fz=1.0))
    fd.add_receiver(rec[0], rec[1])
    print(f'Time-domain FDTD: dt={dt:.4f}, nt={nt} ...')
    vz_time = fd.run(nt)['vz'][:, 0]

    # --- frequency-domain synthesis (PML) --------------------------------
    freqs, spec = eh.source_spectrum(stf, dt)
    band = (0.3 * f0, 3.0 * f0)
    nband = int(((freqs >= band[0]) & (freqs <= band[1])).sum())
    print(f'Frequency-domain: solving {nband} PML systems in '
          f'[{band[0]:.1f}, {band[1]:.1f}] ...')
    rec_dof = [None]

    def solve_one(omega):
        p = eh.ElasticHelmholtz2DProblem(vp, vs, rho, omega, lx=L, lz=L,
                                         shape=(n, n), bc='pml', pml_width=22)
        if rec_dof[0] is None:
            rec_dof[0] = p._uz(rec[0], rec[1])
        g = p.solve_direct(-p.point_force(src[0], src[1], fx=0.0, fz=1.0))
        return np.array([1j * omega * g[rec_dof[0]]])     # -> velocity

    tic = time.time()
    vz_freq = eh.freq_to_time(solve_one, freqs, spec, n_time=nt, dt=dt,
                              freq_band=band)[:, 0]
    print(f'  ... {nband} solves in {time.time() - tic:.1f}s')

    # --- compare velocity waveforms --------------------------------------
    win = (t > t0) & (t < t0 + t_s + 2.5 / f0)
    corr, lag = _best_xcorr(vz_time[win], vz_freq[win], max_lag=60)
    print('\n=== cross-validation (velocity waveforms) ===')
    print(f'  offset {offset:.3f} (nondim) | P {t_p:.3f}, S {t_s:.3f}')
    print(f'  cross-correlation (freq-PML vs FDTD): {corr:.3f} '
          f'(lag {lag * dt:+.3f})')
    status = 'PASS' if corr > 0.95 else 'below target'
    print(f'  roadmap target > 0.95 : {status}')
    _maybe_plot(t, t0, vz_time, vz_freq, t_p, t_s, corr, win)


def _best_xcorr(a, b, max_lag):
    a = (a - np.mean(a)) / (np.std(a) + 1e-30)
    b = (b - np.mean(b)) / (np.std(b) + 1e-30)
    m = len(a)
    best, blag = -1.0, 0
    for lag in range(-max_lag, max_lag + 1):
        aa = a[max(0, lag):m + min(0, lag)]
        bb = b[max(0, -lag):m + min(0, -lag)]
        if len(aa) < 16:
            continue
        c = abs(np.dot(aa, bb) / len(aa))
        if c > best:
            best, blag = c, lag
    return best, blag


def _maybe_plot(t, t0, vz_time, vz_freq, t_p, t_s, corr, win):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n(matplotlib not installed -- skipping figure)')
        return

    def norm(x):
        return (x - np.mean(x)) / (np.max(np.abs(x - np.mean(x))) + 1e-30)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t[win], norm(vz_time[win]), label='time-domain FDTD (vz)')
    ax.plot(t[win], norm(vz_freq[win]), '--',
            label='frequency-domain PML synthesis')
    ax.axvline(t0 + t_p, color='g', ls=':', label='P arrival')
    ax.axvline(t0 + t_s, color='r', ls=':', label='S arrival')
    ax.set_xlabel('time (nondim)'); ax.set_ylabel('normalized vz')
    ax.set_title(f'Elastic freq-domain (PML) vs time-domain FDTD '
                 f'(xcorr {corr:.3f})')
    ax.legend()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'elastic_freq_vs_time.png')
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f'\nFigure written to {out}')


if __name__ == '__main__':
    main()
