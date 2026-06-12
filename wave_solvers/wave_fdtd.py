"""Time-domain finite-difference (FDTD) solvers for the scalar wave equation.

Solves the acoustic wave equation

    u_tt = c(x)^2 Laplacian(u) + s(t) delta(x - x_s)

with the classical second-order leapfrog scheme in time and second-order
central differences in space -- the textbook "traditional" solver used
throughout seismology and ultrasonics (Alford, Kelly & Boore, Geophysics
1974).  For high-order/production runs see the dedicated libraries noted
in wave_solvers/README.md (Devito, SPECFEM, k-Wave); this implementation
is pure NumPy and intended as a transparent, dependency-free baseline for
acoustic-emission waveform modelling.

Boundary conditions
-------------------
* 'dirichlet' : u = 0 (perfectly reflecting, sound-soft).
* 'mur'       : first-order Mur absorbing boundary (Mur, IEEE Trans. EMC
                1981) on all edges.
* 'sponge'    : Cerjan exponential-damping sponge layer (Cerjan et al.,
                Geophysics 1985) combined with Dirichlet edges.

Stability
---------
The explicit leapfrog scheme is conditionally stable; the constructor
chooses ``dt`` from the CFL condition

    dt <= cfl * min(dx, dz) / (c_max * sqrt(ndim))

with a safety factor ``cfl < 1`` (default 0.5), and raises if a
user-supplied ``dt`` violates it.
"""

import numpy as np

from . import sources as _sources  # noqa: F401  (re-exported for convenience)


def cfl_timestep(c_max, dx, ndim, cfl=0.5):
    """Largest stable time step for the 2nd-order leapfrog scheme."""
    return cfl * dx / (c_max * np.sqrt(ndim))


def _sponge_profile(n, width, alpha=0.0053):
    """Cerjan et al. (1985) damping profile: 1 in the interior, decaying
    as exp(-(alpha * d)^2) over ``width`` cells near each boundary."""
    profile = np.ones(n)
    if width > 0:
        d = np.arange(width, 0, -1)
        taper = np.exp(-(alpha * d) ** 2)
        profile[:width] *= taper
        profile[-width:] *= taper[::-1]
    return profile


class AcousticWaveFDTD1D:
    """1D scalar wave equation on [0, L] with velocity c(x)."""

    def __init__(self, velocity, dx, dt=None, cfl=0.5, boundary='mur'):
        self.c = np.atleast_1d(np.asarray(velocity, dtype=float))
        self.nx = self.c.shape[0]
        self.dx = dx
        dt_max = cfl_timestep(self.c.max(), dx, ndim=1, cfl=cfl)
        if dt is None:
            dt = dt_max
        elif dt > dt_max:
            raise ValueError(f'dt={dt:g} violates the CFL limit {dt_max:g}')
        self.dt = dt
        if boundary not in ('dirichlet', 'mur'):
            raise ValueError("boundary must be 'dirichlet' or 'mur'")
        self.boundary = boundary
        self._sources = []     # (index, stf array)
        self._receivers = []   # indices

    def add_source(self, ix, stf):
        self._sources.append((ix, np.asarray(stf, dtype=float)))

    def add_receiver(self, ix):
        self._receivers.append(ix)

    def run(self, nt, record_wavefield_every=0):
        c2 = (self.c * self.dt / self.dx) ** 2
        u_prev = np.zeros(self.nx)
        u = np.zeros(self.nx)
        records = np.zeros((nt, len(self._receivers)))
        snapshots = []

        for it in range(nt):
            u_next = np.zeros_like(u)
            u_next[1:-1] = (2.0 * u[1:-1] - u_prev[1:-1]
                            + c2[1:-1] * (u[2:] - 2.0 * u[1:-1] + u[:-2]))
            for ix, stf in self._sources:
                if it < len(stf):
                    u_next[ix] += self.dt ** 2 * stf[it]

            if self.boundary == 'mur':
                # first-order one-way wave equation at each end
                k0 = (self.c[0] * self.dt - self.dx) / (self.c[0] * self.dt + self.dx)
                kN = (self.c[-1] * self.dt - self.dx) / (self.c[-1] * self.dt + self.dx)
                u_next[0] = u[1] + k0 * (u_next[1] - u[0])
                u_next[-1] = u[-2] + kN * (u_next[-2] - u[-1])

            u_prev, u = u, u_next
            for r, ix in enumerate(self._receivers):
                records[it, r] = u[ix]
            if record_wavefield_every and it % record_wavefield_every == 0:
                snapshots.append(u.copy())

        out = {'receivers': records, 'dt': self.dt,
               't': np.arange(nt) * self.dt, 'final_wavefield': u}
        if snapshots:
            out['snapshots'] = np.array(snapshots)
        return out


class AcousticWaveFDTD2D:
    """2D scalar wave equation on a regular grid with velocity c(x, z).

    Parameters
    ----------
    velocity : scalar or (nx, nz) array of wave speeds [m/s].
    dx, dz : grid spacings [m].
    dt : time step [s]; derived from the CFL condition when None.
    boundary : 'dirichlet', 'mur' or 'sponge'.
    sponge_width : width of the Cerjan sponge layer in cells.
    """

    def __init__(self, velocity, dx, dz=None, dt=None, cfl=0.5,
                 boundary='sponge', sponge_width=20, shape=None):
        if np.isscalar(velocity):
            if shape is None:
                raise ValueError('shape=(nx, nz) is required for scalar velocity')
            velocity = np.full(shape, float(velocity))
        self.c = np.asarray(velocity, dtype=float)
        self.nx, self.nz = self.c.shape
        self.dx = dx
        self.dz = dx if dz is None else dz
        h_min = min(self.dx, self.dz)
        dt_max = cfl_timestep(self.c.max(), h_min, ndim=2, cfl=cfl)
        if dt is None:
            dt = dt_max
        elif dt > dt_max:
            raise ValueError(f'dt={dt:g} violates the CFL limit {dt_max:g}')
        self.dt = dt
        if boundary not in ('dirichlet', 'mur', 'sponge'):
            raise ValueError("boundary must be 'dirichlet', 'mur' or 'sponge'")
        self.boundary = boundary
        if boundary == 'sponge':
            gx = _sponge_profile(self.nx, sponge_width)
            gz = _sponge_profile(self.nz, sponge_width)
            self._sponge = np.outer(gx, gz)
        self._sources = []
        self._receivers = []

    def add_source(self, ix, iz, stf):
        self._sources.append((ix, iz, np.asarray(stf, dtype=float)))

    def add_receiver(self, ix, iz):
        self._receivers.append((ix, iz))

    def _laplacian(self, u):
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.dx ** 2
            + (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dz ** 2)
        return lap

    def run(self, nt, record_wavefield_every=0):
        dt = self.dt
        u_prev = np.zeros((self.nx, self.nz))
        u = np.zeros((self.nx, self.nz))
        records = np.zeros((nt, len(self._receivers)))
        snapshots = []

        for it in range(nt):
            u_next = (2.0 * u - u_prev + (self.c * dt) ** 2 * self._laplacian(u))
            for ix, iz, stf in self._sources:
                if it < len(stf):
                    u_next[ix, iz] += dt ** 2 * stf[it]

            if self.boundary == 'mur':
                self._apply_mur(u_next, u)
            elif self.boundary == 'sponge':
                u_next *= self._sponge
                u *= self._sponge

            u_prev, u = u, u_next
            for r, (ix, iz) in enumerate(self._receivers):
                records[it, r] = u[ix, iz]
            if record_wavefield_every and it % record_wavefield_every == 0:
                snapshots.append(u.copy())

        out = {'receivers': records, 'dt': dt,
               't': np.arange(nt) * dt, 'final_wavefield': u}
        if snapshots:
            out['snapshots'] = np.array(snapshots)
        return out

    def _apply_mur(self, u_next, u):
        dt = self.dt
        cx0 = (self.c[0, :] * dt - self.dx) / (self.c[0, :] * dt + self.dx)
        cxN = (self.c[-1, :] * dt - self.dx) / (self.c[-1, :] * dt + self.dx)
        cz0 = (self.c[:, 0] * dt - self.dz) / (self.c[:, 0] * dt + self.dz)
        czN = (self.c[:, -1] * dt - self.dz) / (self.c[:, -1] * dt + self.dz)
        u_next[0, :] = u[1, :] + cx0 * (u_next[1, :] - u[0, :])
        u_next[-1, :] = u[-2, :] + cxN * (u_next[-2, :] - u[-1, :])
        u_next[:, 0] = u[:, 1] + cz0 * (u_next[:, 1] - u[:, 0])
        u_next[:, -1] = u[:, -2] + czN * (u_next[:, -2] - u[:, -1])
