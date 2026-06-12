"""Excitation sources for acoustic / elastic wave simulations.

This module collects the source-time functions (STFs) and spatial source
descriptions that are commonly used in computational seismology,
ultrasonics, and acoustic-emission (AE) modelling:

* ``ricker``              -- Ricker (Mexican-hat) wavelet, the de-facto
                             standard transient source in seismic FDTD codes
                             (e.g. Devito, SPECFEM, Madagascar).
* ``gaussian_pulse``      -- Gaussian pulse and its first derivative.
* ``tone_burst``          -- windowed sine burst, standard excitation in
                             ultrasonic NDT / guided-wave testing (same
                             definition as k-Wave's ``toneBurst``).
* ``ae_step_source``      -- cosine-smoothed step with finite rise time,
                             the classical model of a sudden crack advance
                             or of the Hsu-Nielsen pencil-lead break used to
                             calibrate acoustic-emission sensors.
* ``erf_step_source``     -- error-function smoothed step (infinitely
                             differentiable alternative).

Spatial sources for the 2D P-SV elastic solver:

* ``PointForceSource2D``  -- body force applied to the velocity components.
* ``MomentTensorSource2D``-- symmetric moment tensor injected into the
                             stress components; ``double_couple_2d`` and
                             ``explosion_2d`` are convenience constructors.

A buried moment tensor with a step-like time history is the canonical
representation of an AE event (micro-crack), see e.g. Ohtsu & Ono (1984),
"A generalized theory of acoustic emission and Green's functions in a half
space", Journal of Acoustic Emission 3, 27-40.
"""

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Source-time functions
# ---------------------------------------------------------------------------
def ricker(t, f_peak, t0=None, amplitude=1.0):
    """Ricker wavelet with peak frequency ``f_peak`` [Hz].

    ``r(t) = (1 - 2 pi^2 f^2 (t-t0)^2) exp(-pi^2 f^2 (t-t0)^2)``

    If ``t0`` is None it defaults to ``1.5 / f_peak`` so that the wavelet
    is causal (essentially zero at t = 0).
    """
    t = np.asarray(t, dtype=float)
    if t0 is None:
        t0 = 1.5 / f_peak
    arg = (np.pi * f_peak * (t - t0)) ** 2
    return amplitude * (1.0 - 2.0 * arg) * np.exp(-arg)


def gaussian_pulse(t, f_peak, t0=None, amplitude=1.0):
    """Gaussian pulse; ``f_peak`` controls the width (sigma = 1/(2 pi f))."""
    t = np.asarray(t, dtype=float)
    if t0 is None:
        t0 = 1.5 / f_peak
    sigma = 1.0 / (2.0 * np.pi * f_peak)
    return amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def gaussian_derivative(t, f_peak, t0=None, amplitude=1.0):
    """First derivative of a Gaussian (zero-mean, suitable as force STF)."""
    t = np.asarray(t, dtype=float)
    if t0 is None:
        t0 = 1.5 / f_peak
    sigma = 1.0 / (2.0 * np.pi * f_peak)
    g = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    return -amplitude * (t - t0) / sigma ** 2 * g


def tone_burst(t, f_center, n_cycles, t0=0.0, amplitude=1.0, window='hann'):
    """Windowed sine burst of ``n_cycles`` cycles at ``f_center`` [Hz].

    The standard narrow-band excitation in ultrasonic NDT and guided-wave
    inspection (Lamb-wave testing of plates, which shares its physics with
    acoustic-emission wave propagation).

    window : 'hann', 'rect' or a callable mapping phase in [0, 1] -> weight.
    """
    t = np.asarray(t, dtype=float)
    duration = n_cycles / f_center
    phase = (t - t0) / duration
    active = (phase >= 0.0) & (phase <= 1.0)

    if window == 'hann':
        win = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
    elif window == 'rect':
        win = np.ones_like(phase)
    elif callable(window):
        win = window(phase)
    else:
        raise ValueError("window must be 'hann', 'rect' or a callable")

    burst = amplitude * win * np.sin(2.0 * np.pi * f_center * (t - t0))
    return np.where(active, burst, 0.0)


def ae_step_source(t, rise_time, t0=0.0, amplitude=1.0):
    """Cosine-smoothed step: classical acoustic-emission source model.

    ``s(t)`` ramps from 0 to ``amplitude`` over ``rise_time`` following
    ``0.5 * (1 - cos(pi (t - t0) / rise_time))`` and stays constant
    afterwards.  This is the standard model for a sudden crack advance and
    for the Hsu-Nielsen pencil-lead break (ASTM E976) used to calibrate AE
    sensors; typical rise times are 0.1-3 microseconds for real AE events.
    """
    t = np.asarray(t, dtype=float)
    tau = (t - t0) / rise_time
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip(tau, 0.0, 1.0)))
    return amplitude * np.where(tau > 0.0, ramp, 0.0)


def erf_step_source(t, rise_time, t0=0.0, amplitude=1.0):
    """Error-function smoothed step (smooth alternative to ae_step_source).

    ``rise_time`` is interpreted as the 10-90 percent rise time.
    """
    from scipy.special import erf
    t = np.asarray(t, dtype=float)
    # erf rises from 10% to 90% over ~1.812 sigma * 2
    sigma = rise_time / 1.812
    return amplitude * 0.5 * (1.0 + erf((t - t0) / (np.sqrt(2.0) * sigma)))


# ---------------------------------------------------------------------------
# Spatial sources for the 2D P-SV elastic solver
# ---------------------------------------------------------------------------
@dataclass
class PointForceSource2D:
    """Point body force f = (fx, fz) * stf(t) at grid index (ix, iz).

    A vertical surface force is the textbook model of the Hsu-Nielsen
    pencil-lead break used for AE sensor calibration.
    """
    ix: int
    iz: int
    stf: np.ndarray          # source-time function sampled at the solver dt
    fx: float = 0.0
    fz: float = 1.0


@dataclass
class MomentTensorSource2D:
    """Symmetric moment tensor M = [[mxx, mxz], [mxz, mzz]] * stf(t).

    Injected into the stress components of the staggered grid.  With a
    step-like ``stf`` (see ``ae_step_source``) this is the canonical
    representation of an acoustic-emission micro-crack source.
    """
    ix: int
    iz: int
    stf: np.ndarray
    mxx: float = 1.0
    mzz: float = 1.0
    mxz: float = 0.0


def explosion_2d(ix, iz, stf, m0=1.0):
    """Isotropic (explosive) source: pure P-wave radiation."""
    return MomentTensorSource2D(ix=ix, iz=iz, stf=stf, mxx=m0, mzz=m0, mxz=0.0)


def double_couple_2d(ix, iz, stf, m0=1.0, angle=0.0):
    """In-plane double couple (shear crack) with fault angle ``angle`` [rad].

    angle = 0 gives M = [[0, m0], [m0, 0]], i.e. slip on a horizontal or
    vertical plane -- the classical model of a shear micro-crack AE source.
    """
    c, s = np.cos(2.0 * angle), np.sin(2.0 * angle)
    return MomentTensorSource2D(ix=ix, iz=iz, stf=stf,
                                mxx=-m0 * s, mzz=m0 * s, mxz=m0 * c)
