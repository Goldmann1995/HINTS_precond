"""Traditional (non-learned) wave-equation and Helmholtz solvers, plus
elastic-wave excitation sources, intended as classical baselines next to
the HINTS hybrid solvers and as building blocks for acoustic-emission
(AE) waveform modelling.

Submodules
----------
helmholtz    : sparse FD assembly + direct / Krylov / shifted-Laplacian /
               stationary solvers for the Helmholtz equation.
wave_fdtd    : 1D/2D scalar wave equation, leapfrog FDTD with Mur and
               Cerjan-sponge absorbing boundaries.
elastic_fdtd : 2D P-SV velocity-stress staggered-grid solver
               (Virieux 1986) with free surfaces and AE sources.
sources      : Ricker / tone-burst / AE step source-time functions and
               point-force / moment-tensor spatial sources.
"""

from . import helmholtz, sources, wave_fdtd, elastic_fdtd
from .sources import (ricker, gaussian_pulse, gaussian_derivative,
                      tone_burst, ae_step_source, erf_step_source,
                      PointForceSource2D, MomentTensorSource2D,
                      explosion_2d, double_couple_2d)
from .wave_fdtd import AcousticWaveFDTD1D, AcousticWaveFDTD2D, cfl_timestep
from .elastic_fdtd import ElasticWaveFDTD2D

__all__ = [
    'helmholtz', 'sources', 'wave_fdtd', 'elastic_fdtd',
    'ricker', 'gaussian_pulse', 'gaussian_derivative', 'tone_burst',
    'ae_step_source', 'erf_step_source',
    'PointForceSource2D', 'MomentTensorSource2D',
    'explosion_2d', 'double_couple_2d',
    'AcousticWaveFDTD1D', 'AcousticWaveFDTD2D', 'cfl_timestep',
    'ElasticWaveFDTD2D',
]
