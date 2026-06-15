"""Frequency-domain Helmholtz forward modelling for HINTS / acoustic emission.

This module is the assembly half of the "bridge layer" described in
``HINTS_AE_ROADMAP.md`` §3. It builds the *complex* linear systems
``A(omega) u = f`` that the HINTS hybrid solver (see ``hints_bridge.py``)
accelerates.

Stage 1 (this file, current) — **scalar acoustic** Helmholtz:

    Laplacian(u) + (omega / c(x))^2 u = -f,    x in [0, Lx] x [0, Lz]

with a first-order Sommerfeld (absorbing) boundary so the discrete operator
is complex and strongly indefinite at high frequency — exactly the regime
where simple relaxation (Jacobi/Gauss-Seidel) diverges and HINTS is needed.
The discretization reuses ``helmholtz.helmholtz_matrix_2d`` (5-point stencil,
heterogeneous wavenumber, Sommerfeld BC) so the classical CSLP baseline in
``helmholtz.py`` applies unchanged.

Stage 2 (planned) — **elastic (vector) P-SV** Navier-Helmholtz:

    div(sigma(u)) + rho omega^2 u = -f,  sigma = lambda tr(eps) I + 2 mu eps

will be added here as ``ElasticHelmholtz2DProblem`` with a vector unknown
``u = (ux, uz)``; the scalar class below is deliberately structured so the
HINTS bridge (complex DeepONet, hybrid iteration, preconditioner) carries
over with only the per-node block size changing from 1 to 2.

Acoustic-emission workflow (roadmap §0): a source spectrum is solved one
frequency at a time with HINTS, then synthesised back to a time-domain
sensor waveform by inverse FFT and cross-validated against the time-domain
solver in ``wave_fdtd.py`` / ``elastic_fdtd.py``.
"""

import numpy as np

from . import helmholtz as _hz


# ---------------------------------------------------------------------------
# Heterogeneous velocity / wavenumber fields (training-data generation)
# ---------------------------------------------------------------------------
def sample_velocity_field(nx, nz, c0=1500.0, n_defects=1, contrast=0.3,
                          defect_radius=0.12, smooth=2.0, rng=None):
    """Return a (nx, nz) velocity field: homogeneous background ``c0`` with a
    few smooth Gaussian "defects" (low/high-velocity inclusions).

    This mimics a plate/block with localized material changes — the setting
    relevant to acoustic-emission inspection. ``contrast`` is the maximum
    relative velocity perturbation; defects are placed at random interior
    locations with random sign.

    Parameters
    ----------
    contrast : peak |dc|/c0 of each inclusion.
    defect_radius : Gaussian radius as a fraction of the domain.
    smooth : extra Gaussian smoothing (in grid cells) applied at the end.
    rng : ``numpy.random.Generator`` (or None for the default).
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.linspace(0.0, 1.0, nx)
    z = np.linspace(0.0, 1.0, nz)
    xx, zz = np.meshgrid(x, z, indexing='ij')
    field = np.zeros((nx, nz))
    for _ in range(n_defects):
        cx, cz = rng.uniform(0.2, 0.8, size=2)
        sign = rng.choice([-1.0, 1.0])
        r2 = (xx - cx) ** 2 + (zz - cz) ** 2
        field += sign * np.exp(-r2 / (2.0 * defect_radius ** 2))
    if smooth > 0:
        field = _gaussian_blur(field, smooth)
    # normalize so the peak perturbation equals `contrast`
    peak = np.max(np.abs(field))
    if peak > 0:
        field = field / peak * contrast
    return c0 * (1.0 + field)


def _gaussian_blur(a, sigma):
    """Separable Gaussian blur (FFT-free, small kernels) — avoids a SciPy
    ndimage dependency."""
    radius = max(1, int(3 * sigma))
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    out = a.copy()
    for axis in (0, 1):
        out = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='same'), axis, out)
    return out


# ---------------------------------------------------------------------------
# Scalar acoustic Helmholtz problem (Stage 1)
# ---------------------------------------------------------------------------
class ScalarHelmholtz2DProblem:
    """Complex 2D acoustic Helmholtz system with Sommerfeld boundaries.

    The unknown is a scalar pressure field on an ``(nx, nz)`` grid, flattened
    in row-major order ``index = ix * nz + iz`` (matching ``helmholtz.py`` and
    ``HINTS_numpy/utils.py``).

    Parameters
    ----------
    velocity : (nx, nz) array of wave speeds c(x, z) [m/s].
    omega : angular frequency [rad/s].
    lx, lz : physical domain size [m]; defaults to a unit square.
    bc : 'sommerfeld' (default, absorbing/complex) or 'dirichlet'.

    Notes
    -----
    The wavenumber field is ``k = omega / c``. With ``bc='sommerfeld'`` the
    assembled matrix is complex and indefinite. ``nondim()`` reports the
    points-per-wavelength resolution, the key accuracy/conditioning knob.
    """

    def __init__(self, velocity, omega, lx=1.0, lz=1.0, bc='sommerfeld'):
        self.c = np.asarray(velocity, dtype=float)
        self.nx, self.nz = self.c.shape
        self.omega = float(omega)
        self.lx = lx
        self.lz = lz
        self.bc = bc
        self.k = self.omega / self.c                    # wavenumber field
        self.block_size = 1                             # scalar (Stage 2: 2)
        self._A = None

    @property
    def ndof(self):
        return self.nx * self.nz * self.block_size

    def assemble(self, k_scale=1.0):
        """Assemble and cache the system matrix ``A`` (complex CSR).

        ``k_scale`` multiplies ``k^2`` and is used by the CSLP preconditioner
        (complex shift); leave at 1.0 for the true operator.
        """
        A = _hz.helmholtz_matrix_2d(self.nx, self.nz, lx=self.lx, ly=self.lz,
                                    k=self.k, bc=self.bc, k_scale=k_scale)
        if k_scale == 1.0:
            self._A = A
        return A

    @property
    def A(self):
        if self._A is None:
            self.assemble()
        return self._A

    def point_source(self, ix, iz, amplitude=1.0):
        """Right-hand side for a point source at grid node (ix, iz)."""
        f = np.zeros(self.ndof, dtype=complex)
        f[ix * self.nz + iz] = amplitude
        return f

    def smooth_random_source(self, rng=None, n_blobs=3):
        """A smooth complex RHS (sum of Gaussian blobs with random complex
        weights) — the kind of right-hand side used to *train* the HINTS
        DeepONet to approximate ``A^{-1}`` over a function space."""
        rng = np.random.default_rng() if rng is None else rng
        x = np.linspace(0, 1, self.nx)
        z = np.linspace(0, 1, self.nz)
        xx, zz = np.meshgrid(x, z, indexing='ij')
        field = np.zeros((self.nx, self.nz), dtype=complex)
        for _ in range(n_blobs):
            cx, cz = rng.uniform(0.15, 0.85, size=2)
            rad = rng.uniform(0.06, 0.18)
            weight = rng.normal() + 1j * rng.normal()
            field += weight * np.exp(-((xx - cx) ** 2 + (zz - cz) ** 2)
                                     / (2.0 * rad ** 2))
        return field.reshape(self.ndof)

    def solve_direct(self, f):
        """Reference solution via sparse LU (SuperLU)."""
        return _hz.solve_direct(self.A, f)

    def points_per_wavelength(self):
        """Minimum grid points per wavelength (PPW); ~10 is the usual lower
        bound for a second-order scheme to stay accurate."""
        dx = self.lx / (self.nx - 1)
        dz = self.lz / (self.nz - 1)
        lam_min = 2.0 * np.pi / np.max(self.k)
        return lam_min / max(dx, dz)

    def nondim(self):
        """Human-readable summary of the resolution / wavenumber regime."""
        return {
            'ndof': self.ndof,
            'k_min': float(np.min(self.k)),
            'k_max': float(np.max(self.k)),
            'points_per_wavelength': float(self.points_per_wavelength()),
            'bc': self.bc,
        }
