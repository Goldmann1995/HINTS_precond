"""Stage 3 demo: multi-frequency amortization with one vector DeepONet.

Trains a single ``MultiFreqVectorDeepONet2D`` to approximate the inverse of
the elastic Navier-Helmholtz operator ``A(omega)`` across a band of
frequencies (omega is a branch input), then reports the per-frequency
accuracy on the *trained* frequencies and on *held-out* frequencies -- the
point being that one network amortizes the whole acoustic-emission band and
interpolates to unseen frequencies, so it can be reused at every frequency of
an ``freq_to_time`` synthesis instead of training one network per frequency.

Everything is nondimensional (vp = 1) so the elastic solves are
well-conditioned (roadmap §2.3). As in Stages 1-2, a CPU-budget network does
not reach high absolute accuracy; the demo validates the amortization
behaviour (held-out error comparable to trained-frequency error).

Run from the repo root:  python wave_solvers/examples/hints_multifreq_stage3.py
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from wave_solvers import elastic_helmholtz as eh
from wave_solvers import hints_bridge as hb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--n-per-omega', type=int, default=350)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.grid
    vp = np.ones((n, n)); vs = 0.571 * np.ones((n, n)); rho = np.ones((n, n))

    # PML boundaries give clean, non-resonant solutions (far more learnable
    # than the thin diagonal absorbing layer -- see STAGE3_REPORT.md §3).
    def factory(omega):
        return eh.ElasticHelmholtz2DProblem(vp, vs, rho, omega, shape=(n, n),
                                            bc='pml', pml_width=8)

    train_w = [4.0, 5.0, 6.0, 7.0, 8.0]
    held_out = [4.5, 6.5]
    print(f'Training frequencies omega = {train_w}')
    print(f'Held-out frequencies omega = {held_out}')

    print(f'\nGenerating data ({args.n_per_omega}/omega train) ...')
    f_tr, u_tr, w_tr = hb.generate_multifreq_dataset(factory, train_w,
                                                     args.n_per_omega, rng=rng)
    f_va, u_va, w_va = hb.generate_multifreq_dataset(factory, train_w, 30, rng=rng)
    f_te, u_te, w_te = hb.generate_multifreq_dataset(factory, held_out, 30, rng=rng)

    static = np.stack([vp, vs], axis=-1)
    don = hb.MultiFreqVectorDeepONet2D(n, n, static, omega_ref=8.0,
                                       n_comp=2, latent=64)
    print(f'Training one MultiFreqVectorDeepONet2D ({args.epochs} epochs)...')
    t0 = time.time()
    don.fit(f_tr, u_tr, w_tr, epochs=args.epochs, batch_size=64,
            val=(f_va, u_va, w_va), log_every=max(1, args.epochs // 6))
    print(f'  trained in {time.time() - t0:.1f}s')

    tr_err = don.per_frequency_error(f_va, u_va, w_va)
    te_err = don.per_frequency_error(f_te, u_te, w_te)
    print('\n=== Stage 3 multi-frequency amortization ===')
    print('  trained-band per-omega relative error:')
    for w, e in tr_err.items():
        print(f'    omega={w:.1f}: {e:.3f}')
    print('  HELD-OUT per-omega relative error (interpolation in omega):')
    for w, e in te_err.items():
        print(f'    omega={w:.1f}: {e:.3f}')
    tr_mean = np.mean(list(tr_err.values()))
    te_mean = np.mean(list(te_err.values()))
    print(f'\n  mean trained {tr_mean:.3f} | mean held-out {te_mean:.3f} '
          f'(ratio {te_mean / tr_mean:.2f})')
    print('  -> one network covers the whole band; held-out error close to '
          'trained\n     error demonstrates amortization (roadmap §4 Stage 3).')


if __name__ == '__main__':
    main()
