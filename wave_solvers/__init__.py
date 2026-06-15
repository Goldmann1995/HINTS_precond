"""Traditional (non-learned) wave-equation and Helmholtz solvers, plus
elastic-wave excitation sources, intended as classical baselines next to
the HINTS hybrid solvers and as building blocks for acoustic-emission
(AE) waveform modelling.

Submodules
----------
helmholtz       : sparse FD assembly + direct / Krylov / shifted-Laplacian /
                  stationary solvers for the Helmholtz equation.
wave_fdtd       : 1D/2D scalar wave equation, leapfrog FDTD with Mur and
                  Cerjan-sponge absorbing boundaries.
elastic_fdtd    : 2D P-SV velocity-stress staggered-grid solver
                  (Virieux 1986) with free surfaces and AE sources.
sources         : Ricker / tone-burst / AE step source-time functions and
                  point-force / moment-tensor spatial sources.
elastic_helmholtz : frequency-domain (complex) Helmholtz assembly for HINTS
                  (Stage 1: scalar acoustic; Stage 2: elastic P-SV).
hints_bridge    : blend a complex DeepONet with relaxation / CSLP to solve
                  the frequency-domain systems (standalone hybrid + GMRES
                  preconditioner). Requires PyTorch for the network paths.
"""

from . import (helmholtz, sources, wave_fdtd, elastic_fdtd,
               elastic_helmholtz, hints_bridge)
from .sources import (ricker, gaussian_pulse, gaussian_derivative,
                      tone_burst, ae_step_source, erf_step_source,
                      PointForceSource2D, MomentTensorSource2D,
                      explosion_2d, double_couple_2d)
from .wave_fdtd import AcousticWaveFDTD1D, AcousticWaveFDTD2D, cfl_timestep
from .elastic_fdtd import ElasticWaveFDTD2D

__all__ = [
    'helmholtz', 'sources', 'wave_fdtd', 'elastic_fdtd',
    'elastic_helmholtz', 'hints_bridge',
    'ricker', 'gaussian_pulse', 'gaussian_derivative', 'tone_burst',
    'ae_step_source', 'erf_step_source',
    'PointForceSource2D', 'MomentTensorSource2D',
    'explosion_2d', 'double_couple_2d',
    'AcousticWaveFDTD1D', 'AcousticWaveFDTD2D', 'cfl_timestep',
    'ElasticWaveFDTD2D',
]
