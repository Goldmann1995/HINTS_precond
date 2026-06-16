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

Stage 2 (this file, current) — **elastic (vector) P-SV** Navier-Helmholtz:

    div(sigma(u)) + rho omega^2 u = -f,  sigma = lambda tr(eps) I + 2 mu eps

is implemented as ``ElasticHelmholtz2DProblem`` with a vector unknown
``u = (ux, uz)`` (second-order finite differences, component-major ordering
``dof = component * N + node``). The HINTS bridge (complex DeepONet, hybrid
iteration, FGMRES preconditioner) carries over from the scalar case with the
per-node block size changing from 1 to 2. Sources include point forces and
**moment tensors** (the canonical acoustic-emission micro-crack model).

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


# ---------------------------------------------------------------------------
# Elastic (vector) P-SV Navier-Helmholtz problem (Stage 2)
# ---------------------------------------------------------------------------
import scipy.sparse as _sp  # noqa: E402


def lame_from_velocities(vp, vs, rho):
    """Return (lambda, mu) Lame parameters from (vp, vs, rho)."""
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2.0 * mu
    return lam, mu


class ElasticHelmholtz2DProblem:
    """Frequency-domain 2D P-SV elastic (Navier-Helmholtz) system.

    Solves, for the complex displacement ``u = (ux, uz)`` at angular
    frequency ``omega``,

        div(sigma(u)) + rho omega^2 u = b,
        sigma = lambda (div u) I + mu (grad u + grad u^T),

    discretized with second-order central differences. Degrees of freedom use
    **component-major** ordering ``dof = c * N + (ix*nz + iz)`` with ``c = 0``
    for ``ux`` and ``c = 1`` for ``uz`` (``N = nx*nz``); this lets the HINTS
    bridge reshape a state vector to ``(2, nx, nz)`` directly.

    Boundaries
    ----------
    * ``bc='dirichlet'`` — identity rows (used by the manufactured-solution
      convergence test).
    * ``bc='absorbing'`` — a frequency-domain absorbing layer: a complex mass
      term ``+ i rho omega gamma(x)`` is added near the edges (the Cerjan-style
      analog of the time-domain sponge in ``elastic_fdtd.py``), so outgoing P
      and S waves are damped and the operator is complex/indefinite.

    Parameters
    ----------
    vp, vs, rho : scalars or (nx, nz) arrays.
    omega : angular frequency [rad/s].
    lx, lz : domain size [m].
    bc : 'absorbing' (default) or 'dirichlet'.
    abl_width : absorbing-layer width in cells (bc='absorbing').
    abl_strength : peak damping fraction of omega in the layer.
    """

    def __init__(self, vp, vs, rho, omega, lx=1.0, lz=1.0, shape=None,
                 bc='absorbing', abl_width=12, abl_strength=2.0):
        if np.isscalar(vp):
            if shape is None:
                raise ValueError('shape=(nx, nz) required for scalar media')
            vp = np.full(shape, float(vp))
        self.vp = np.asarray(vp, dtype=float)
        self.nx, self.nz = self.vp.shape
        self.vs = np.broadcast_to(np.asarray(vs, float), self.vp.shape).copy()
        self.rho = np.broadcast_to(np.asarray(rho, float), self.vp.shape).copy()
        self.omega = float(omega)
        self.lx, self.lz = lx, lz
        self.bc = bc
        self.abl_width = abl_width
        self.abl_strength = abl_strength
        self.lam, self.mu = lame_from_velocities(self.vp, self.vs, self.rho)
        self.block_size = 2
        self.N = self.nx * self.nz
        self._A = None

    @property
    def ndof(self):
        return 2 * self.N

    # -- indexing helpers -------------------------------------------------
    def _ux(self, ix, iz):
        return ix * self.nz + iz

    def _uz(self, ix, iz):
        return self.N + ix * self.nz + iz

    def _absorption(self):
        """Complex diagonal damping gamma(x) (>=0), ramped near the edges."""
        gx = np.zeros(self.nx)
        gz = np.zeros(self.nz)
        w = self.abl_width
        if w > 0:
            ramp = (np.arange(w, 0, -1) / w) ** 2
            gx[:w] = ramp
            gx[-w:] = ramp[::-1]
            gz[:w] = ramp
            gz[-w:] = ramp[::-1]
        g = np.maximum(gx[:, None], gz[None, :])       # union of layers
        return self.abl_strength * self.omega * g

    def assemble(self):
        """Assemble and cache the complex system matrix ``A`` (CSR)."""
        nx, nz = self.nx, self.nz
        dx = self.lx / (nx - 1)
        dz = self.lz / (nz - 1)
        lam, mu, rho = self.lam, self.mu, self.rho
        w2 = self.omega ** 2
        absorb = self._absorption() if self.bc == 'absorbing' else None

        A = _sp.lil_matrix((self.ndof, self.ndof), dtype=complex)
        interior = lambda i, j: 0 < i < nx - 1 and 0 < j < nz - 1

        for ix in range(nx):
            for iz in range(nz):
                rx, rz = self._ux(ix, iz), self._uz(ix, iz)
                if not interior(ix, iz):
                    # Both BCs use a Dirichlet (u = 0) frame: identity rows.
                    # For 'absorbing' the interior ABL damps outgoing waves
                    # before they reach this frame, so reflections are small;
                    # crucially this keeps the interior operator symmetric, so
                    # the discrete Green's function satisfies reciprocity (a
                    # clamped-index "boundary stencil" would break both).
                    A[rx, rx] = 1.0
                    A[rz, rz] = 1.0
                    continue
                l2m = lam[ix, iz] + 2.0 * mu[ix, iz]
                lm = lam[ix, iz] + mu[ix, iz]
                m = mu[ix, iz]
                mass = rho[ix, iz] * w2
                if absorb is not None:
                    # numpy IFFT uses the e^{+i omega t} convention, for which
                    # time-domain velocity damping (rho gamma d_t u) maps to a
                    # mass term rho(omega^2 - i omega gamma): the -i sign makes
                    # outgoing waves decay (causal). The opposite sign yields
                    # incoming/time-reversed (acausal) solutions.
                    mass = mass - 1j * rho[ix, iz] * absorb[ix, iz]

                ip, im = ix + 1, ix - 1     # interior: neighbours are valid
                jp, jm = iz + 1, iz - 1

                # --- ux equation: (lam+2mu) ux_xx + mu ux_zz + (lam+mu) uz_xz
                A[rx, self._ux(ip, iz)] += l2m / dx ** 2
                A[rx, self._ux(im, iz)] += l2m / dx ** 2
                A[rx, self._ux(ix, jp)] += m / dz ** 2
                A[rx, self._ux(ix, jm)] += m / dz ** 2
                A[rx, rx] += -2.0 * l2m / dx ** 2 - 2.0 * m / dz ** 2 + mass
                cxz = lm / (4.0 * dx * dz)
                A[rx, self._uz(ip, jp)] += cxz
                A[rx, self._uz(im, jm)] += cxz
                A[rx, self._uz(ip, jm)] -= cxz
                A[rx, self._uz(im, jp)] -= cxz

                # --- uz equation: mu uz_xx + (lam+2mu) uz_zz + (lam+mu) ux_xz
                A[rz, self._uz(ip, iz)] += m / dx ** 2
                A[rz, self._uz(im, iz)] += m / dx ** 2
                A[rz, self._uz(ix, jp)] += l2m / dz ** 2
                A[rz, self._uz(ix, jm)] += l2m / dz ** 2
                A[rz, rz] += -2.0 * m / dx ** 2 - 2.0 * l2m / dz ** 2 + mass
                A[rz, self._ux(ip, jp)] += cxz
                A[rz, self._ux(im, jm)] += cxz
                A[rz, self._ux(ip, jm)] -= cxz
                A[rz, self._ux(im, jp)] -= cxz

        A = A.tocsr()
        self._A = A
        return A

    @property
    def A(self):
        if self._A is None:
            self.assemble()
        return self._A

    # -- sources ----------------------------------------------------------
    def point_force(self, ix, iz, fx=0.0, fz=1.0, amplitude=1.0):
        """RHS for a point body force (fx, fz) at node (ix, iz)."""
        b = np.zeros(self.ndof, dtype=complex)
        b[self._ux(ix, iz)] = amplitude * fx
        b[self._uz(ix, iz)] = amplitude * fz
        return b

    def moment_tensor(self, ix, iz, mxx=1.0, mzz=1.0, mxz=0.0, amplitude=1.0):
        """RHS for a moment-tensor source ``f_i = -d_j (M_ij delta)``,
        discretized by central differences — the canonical AE micro-crack
        source (shear: mxz; volumetric/explosive: mxx=mzz)."""
        nx, nz = self.nx, self.nz
        dx = self.lx / (nx - 1)
        dz = self.lz / (nz - 1)
        b = np.zeros(self.ndof, dtype=complex)
        a = amplitude
        # fx = -(d/dx)(Mxx delta) - (d/dz)(Mxz delta)
        b[self._ux(min(ix + 1, nx - 1), iz)] += -a * mxx / (2 * dx)
        b[self._ux(max(ix - 1, 0), iz)] += a * mxx / (2 * dx)
        b[self._ux(ix, min(iz + 1, nz - 1))] += -a * mxz / (2 * dz)
        b[self._ux(ix, max(iz - 1, 0))] += a * mxz / (2 * dz)
        # fz = -(d/dx)(Mxz delta) - (d/dz)(Mzz delta)
        b[self._uz(min(ix + 1, nx - 1), iz)] += -a * mxz / (2 * dx)
        b[self._uz(max(ix - 1, 0), iz)] += a * mxz / (2 * dx)
        b[self._uz(ix, min(iz + 1, nz - 1))] += -a * mzz / (2 * dz)
        b[self._uz(ix, max(iz - 1, 0))] += a * mzz / (2 * dz)
        return b

    def smooth_random_source(self, rng=None, n_blobs=3):
        """Smooth complex vector RHS (random Gaussian blobs per component) —
        the training distribution for the vector HINTS DeepONet."""
        rng = np.random.default_rng() if rng is None else rng
        x = np.linspace(0, 1, self.nx)
        z = np.linspace(0, 1, self.nz)
        xx, zz = np.meshgrid(x, z, indexing='ij')
        b = np.zeros(self.ndof, dtype=complex)
        for comp, off in ((0, 0), (1, self.N)):
            field = np.zeros((self.nx, self.nz), dtype=complex)
            for _ in range(n_blobs):
                cx, cz = rng.uniform(0.15, 0.85, size=2)
                rad = rng.uniform(0.06, 0.18)
                wt = rng.normal() + 1j * rng.normal()
                field += wt * np.exp(-((xx - cx) ** 2 + (zz - cz) ** 2)
                                     / (2.0 * rad ** 2))
            b[off:off + self.N] = field.reshape(-1)
        return b

    def solve_direct(self, b):
        return _hz.solve_direct(self.A, b)

    def to_fields(self, vec):
        """Reshape a state vector to ``(2, nx, nz)`` complex fields."""
        return np.asarray(vec).reshape(2, self.nx, self.nz)

    def points_per_wavelength(self):
        """PPW for the slower (shear) wave — the binding resolution limit."""
        dx = self.lx / (self.nx - 1)
        dz = self.lz / (self.nz - 1)
        ks_max = self.omega / np.min(self.vs)
        lam_min = 2.0 * np.pi / ks_max
        return lam_min / max(dx, dz)

    def nondim(self):
        return {
            'ndof': self.ndof,
            'vp_over_vs': float(np.mean(self.vp / self.vs)),
            'shear_ppw': float(self.points_per_wavelength()),
            'bc': self.bc,
        }


# ---------------------------------------------------------------------------
# Frequency <-> time synthesis (AE waveform modelling, roadmap §3)
# ---------------------------------------------------------------------------
def source_spectrum(stf, dt):
    """Return (frequencies [Hz], complex spectrum) of a source-time function
    sampled at ``dt`` via the real FFT."""
    spec = np.fft.rfft(stf)
    freqs = np.fft.rfftfreq(len(stf), d=dt)
    return freqs, spec


def freq_to_time(solve_one, freqs, spectrum, n_time, dt, freq_band=None):
    """Synthesize a time-domain response from per-frequency solves.

    For each frequency ``f`` in ``freqs`` (optionally restricted to
    ``freq_band = (fmin, fmax)``), ``solve_one(omega)`` returns the complex
    response to a unit harmonic source; it is scaled by the source spectrum
    and accumulated, then inverse-FFT'd to the time domain.

    Returns a real array of shape ``(n_time,) + response_shape``.
    """
    freqs = np.asarray(freqs)
    sel = np.ones(len(freqs), dtype=bool)
    if freq_band is not None:
        sel &= (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    sel &= freqs > 0.0

    acc = None
    nfreq = len(freqs)
    for idx in np.where(sel)[0]:
        omega = 2.0 * np.pi * freqs[idx]
        resp = np.asarray(solve_one(omega)) * spectrum[idx]
        if acc is None:
            acc = np.zeros((nfreq,) + resp.shape, dtype=complex)
        acc[idx] = resp
    if acc is None:
        raise ValueError('no frequencies selected')
    # inverse real FFT back to the time domain
    time_series = np.fft.irfft(acc, n=n_time, axis=0)
    return time_series
