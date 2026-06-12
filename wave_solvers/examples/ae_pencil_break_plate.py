"""Acoustic-emission example: Hsu-Nielsen pencil-lead break on a steel plate.

Models the standard ASTM E976 AE sensor-calibration source -- a vertical
point force with a fast cosine-smoothed step time history applied at the
top free surface of a steel plate -- and records the out-of-plane velocity
at a surface-mounted "sensor" some distance away.  The recorded waveform
shows the characteristic P / S / Rayleigh arrivals that AE analysis
(arrival-time picking, source location) relies on.

Run:  python examples/ae_pencil_break_plate.py
Requires NumPy/SciPy; matplotlib optional (a PNG is written if available).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import wave_solvers as ws


def main():
    # Steel plate: vp ~ 5900 m/s, vs ~ 3200 m/s, rho 7800 kg/m^3
    nx, nz = 400, 200
    dx = 0.5e-3                       # 0.5 mm grid -> 200 mm x 100 mm plate
    solver = ws.ElasticWaveFDTD2D(
        vp=5900.0, vs=3200.0, rho=7800.0, dx=dx,
        free_surface=True, free_bottom=True,   # plate: Lamb-wave regime
        sponge_width=25, shape=(nx, nz))

    nt = 2500
    t = np.arange(nt) * solver.dt
    # pencil-lead break: ~1 us rise time step force, downward (fz)
    rise = 1.0e-6
    stf = ws.ae_step_source(t, rise_time=rise, t0=3 * rise, amplitude=1.0)
    solver.add_source(ws.PointForceSource2D(80, 1, stf, fx=0.0, fz=1.0))

    # surface AE sensors at increasing offsets
    offsets_mm = [40, 80, 120]
    for off in offsets_mm:
        solver.add_receiver(80 + int(off * 1e-3 / dx), 1)

    print(f'dt = {solver.dt*1e9:.2f} ns, simulating {nt} steps '
          f'({t[-1]*1e6:.1f} us)...')
    out = solver.run(nt, record_wavefield_every=0)

    # report first-arrival (P-wave) times vs theory
    print('\nReceiver  offset   t_first_motion   t_P(theory)')
    for r, off in enumerate(offsets_mm):
        trace = out['vz'][:, r]
        thr = 0.02 * np.max(np.abs(trace))
        idx = np.argmax(np.abs(trace) > thr)
        t_fm = out['t'][idx]
        t_p = 3 * rise + (off * 1e-3) / 5900.0
        print(f'  {r}      {off:3d} mm   {t_fm*1e6:8.2f} us     '
              f'{t_p*1e6:8.2f} us')

    _maybe_plot(out, offsets_mm)


def _maybe_plot(out, offsets_mm):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n(matplotlib not installed -- skipping figure)')
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    t_us = out['t'] * 1e6
    for r, off in enumerate(offsets_mm):
        trace = out['vz'][:, r]
        trace = trace / np.max(np.abs(trace) + 1e-30)
        ax.plot(t_us, trace + 1.2 * r, label=f'{off} mm')
    ax.set_xlabel('time [us]')
    ax.set_ylabel('normalized vz (offset for clarity)')
    ax.set_title('AE pencil-lead break: surface waveforms on a steel plate')
    ax.legend()
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'ae_pencil_break_plate.png')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f'\nWaveform figure written to {out_png}')


if __name__ == '__main__':
    main()
