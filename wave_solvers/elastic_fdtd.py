"""2D P-SV elastic wave propagation on a staggered grid.

Implements the classical velocity-stress staggered-grid finite-difference
scheme of Virieux (Geophysics 51, 1986) -- the standard "traditional"
solver for elastic waves and the workhorse behind most seismic and
ultrasonic / acoustic-emission (AE) waveform modelling codes (the same
scheme, at higher order, underlies SOFI2D, SPECFEM's FD relatives, and
Devito-generated elastic kernels).

Physics
-------
Linear isotropic elasticity in the x-z plane (P-SV system):

    rho dv/dt = div(sigma) + f
    dsigma/dt = lambda tr(eps_dot) I + 2 mu eps_dot + M delta(x - x_s)

with Lame parameters computed from (vp, vs, rho).  Both P and S waves
(and, with the free surface enabled, Rayleigh waves -- which dominate AE
signals recorded by surface-mounted sensors) are modelled.

Sources (see wave_solvers.sources)
----------------------------------
* ``PointForceSource2D``   : body force on the velocity grid; a vertical
  surface force with an ``ae_step_source`` time history reproduces the
  Hsu-Nielsen pencil-lead break (ASTM E976) AE calibration source.
* ``MomentTensorSource2D`` : buried moment tensor injected into the stress
  grid; with a step-like time history this is the canonical micro-crack
  AE source model (Ohtsu & Ono, J. Acoustic Emission 3, 1984).  The
  supplied ``stf`` is the *moment time history*; the solver injects its
  discrete time derivative, so a step produces the correct far-field
  displacement step.

Boundaries
----------
* ``free_surface=True`` applies a zero-stress condition at z = 0 (top),
  letting surface/Rayleigh waves develop (a plate with free top and bottom
  -- set ``free_bottom=True`` as well -- supports Lamb waves, the regime
  of AE in plate-like structures).
* The remaining edges use a Cerjan (Geophysics 1985) damping sponge.
"""

import numpy as np

from .sources import PointForceSource2D, MomentTensorSource2D
from .wave_fdtd import _sponge_profile


class ElasticWaveFDTD2D:
    """2D P-SV velocity-stress staggered-grid FDTD solver (Virieux 1986).

    Grid layout (cell index i along x, j along z, z pointing down):
      sxx, szz at (i, j); vx at (i+1/2, j); vz at (i, j+1/2);
      sxz at (i+1/2, j+1/2).

    Parameters
    ----------
    vp, vs, rho : scalars or (nx, nz) arrays -- P velocity [m/s],
        S velocity [m/s], density [kg/m^3].
    dx, dz : grid spacings [m].
    dt : time step [s]; the CFL-stable value is used when None.
    cfl : CFL safety factor (< 1) used when dt is None.
    free_surface, free_bottom : zero-stress condition at z=0 / z=L.
    sponge_width : Cerjan sponge width (cells) on absorbing edges.
    """

    def __init__(self, vp, vs, rho, dx, dz=None, dt=None, cfl=0.5,
                 free_surface=True, free_bottom=False, sponge_width=20,
                 shape=None):
        if np.isscalar(vp):
            if shape is None:
                raise ValueError('shape=(nx, nz) is required for scalar media')
            vp = np.full(shape, float(vp))
        self.vp = np.asarray(vp, dtype=float)
        self.nx, self.nz = self.vp.shape
        self.vs = np.broadcast_to(np.asarray(vs, dtype=float),
                                  self.vp.shape).copy()
        self.rho = np.broadcast_to(np.asarray(rho, dtype=float),
                                   self.vp.shape).copy()
        self.dx = dx
        self.dz = dx if dz is None else dz

        self.mu = self.rho * self.vs ** 2
        self.lam = self.rho * self.vp ** 2 - 2.0 * self.mu

        h_min = min(self.dx, self.dz)
        dt_max = cfl * h_min / (self.vp.max() * np.sqrt(2.0))
        if dt is None:
            dt = dt_max
        elif dt > dt_max:
            raise ValueError(f'dt={dt:g} violates the CFL limit {dt_max:g}')
        self.dt = dt

        self.free_surface = free_surface
        self.free_bottom = free_bottom
        gx = _sponge_profile(self.nx, sponge_width)
        gz = _sponge_profile(self.nz, sponge_width)
        if free_surface:           # do not damp near the free surface
            gz[:sponge_width] = 1.0
        if free_bottom:
            gz[-sponge_width:] = 1.0
        self._sponge = np.outer(gx, gz)

        self._force_sources = []
        self._moment_sources = []
        self._receivers = []

    # -- setup ------------------------------------------------------------
    def add_source(self, source):
        if isinstance(source, PointForceSource2D):
            self._force_sources.append(source)
        elif isinstance(source, MomentTensorSource2D):
            self._moment_sources.append(source)
        else:
            raise TypeError('source must be PointForceSource2D or '
                            'MomentTensorSource2D')

    def add_receiver(self, ix, iz):
        """Record (vx, vz) at grid cell (ix, iz) -- an AE 'sensor'."""
        self._receivers.append((ix, iz))

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _moment_rate(stf, it):
        """Discrete derivative of the moment time history at step ``it``."""
        if it >= len(stf):
            return 0.0
        return stf[it] - (stf[it - 1] if it > 0 else 0.0)

    def _apply_free_surfaces(self, szz, sxz):
        if self.free_surface:
            szz[:, 0] = 0.0
            sxz[:, 0] = 0.0
        if self.free_bottom:
            szz[:, -1] = 0.0
            sxz[:, -1] = 0.0

    # -- time stepping ----------------------------------------------------
    def run(self, nt, record_wavefield_every=0):
        """Advance ``nt`` steps; returns receiver traces and metadata.

        Returns a dict with 't', 'dt', 'vx'/'vz' receiver arrays of shape
        (nt, n_receivers), the final velocity fields, and optional
        'snapshots' of |v| when ``record_wavefield_every`` > 0.
        """
        dt, dx, dz = self.dt, self.dx, self.dz
        nx, nz = self.nx, self.nz
        vx = np.zeros((nx, nz))
        vz = np.zeros((nx, nz))
        sxx = np.zeros((nx, nz))
        szz = np.zeros((nx, nz))
        sxz = np.zeros((nx, nz))

        lam, mu, rho = self.lam, self.mu, self.rho
        lam2mu = lam + 2.0 * mu
        # mu averaged onto the sxz half-node positions
        mu_xz = mu.copy()
        mu_xz[:-1, :-1] = 0.25 * (mu[:-1, :-1] + mu[1:, :-1]
                                  + mu[:-1, 1:] + mu[1:, 1:])

        rec_vx = np.zeros((nt, len(self._receivers)))
        rec_vz = np.zeros((nt, len(self._receivers)))
        snapshots = []
        cell_area = dx * dz

        for it in range(nt):
            # --- update stresses from velocity gradients ----------------
            dvx_dx = np.zeros((nx, nz))
            dvz_dz = np.zeros((nx, nz))
            dvx_dx[1:, :] = (vx[1:, :] - vx[:-1, :]) / dx
            dvz_dz[:, 1:] = (vz[:, 1:] - vz[:, :-1]) / dz
            sxx += dt * (lam2mu * dvx_dx + lam * dvz_dz)
            szz += dt * (lam * dvx_dx + lam2mu * dvz_dz)

            dvx_dz = np.zeros((nx, nz))
            dvz_dx = np.zeros((nx, nz))
            dvx_dz[:, :-1] = (vx[:, 1:] - vx[:, :-1]) / dz
            dvz_dx[:-1, :] = (vz[1:, :] - vz[:-1, :]) / dx
            sxz += dt * mu_xz * (dvx_dz + dvz_dx)

            # moment-tensor sources act on the stress grid
            for s in self._moment_sources:
                m_rate = self._moment_rate(s.stf, it)
                if m_rate != 0.0:
                    sxx[s.ix, s.iz] -= s.mxx * m_rate / cell_area
                    szz[s.ix, s.iz] -= s.mzz * m_rate / cell_area
                    sxz[s.ix, s.iz] -= s.mxz * m_rate / cell_area

            self._apply_free_surfaces(szz, sxz)

            # --- update velocities from stress gradients -----------------
            dsxx_dx = np.zeros((nx, nz))
            dsxz_dz = np.zeros((nx, nz))
            dsxx_dx[:-1, :] = (sxx[1:, :] - sxx[:-1, :]) / dx
            dsxz_dz[:, 1:] = (sxz[:, 1:] - sxz[:, :-1]) / dz
            vx += dt / rho * (dsxx_dx + dsxz_dz)

            dsxz_dx = np.zeros((nx, nz))
            dszz_dz = np.zeros((nx, nz))
            dsxz_dx[1:, :] = (sxz[1:, :] - sxz[:-1, :]) / dx
            dszz_dz[:, :-1] = (szz[:, 1:] - szz[:, :-1]) / dz
            vz += dt / rho * (dsxz_dx + dszz_dz)

            # point-force sources act on the velocity grid
            for s in self._force_sources:
                if it < len(s.stf):
                    amp = dt * s.stf[it] / (rho[s.ix, s.iz] * cell_area)
                    vx[s.ix, s.iz] += s.fx * amp
                    vz[s.ix, s.iz] += s.fz * amp

            # --- absorbing sponge ----------------------------------------
            for fld in (vx, vz, sxx, szz, sxz):
                fld *= self._sponge

            for r, (ix, iz) in enumerate(self._receivers):
                rec_vx[it, r] = vx[ix, iz]
                rec_vz[it, r] = vz[ix, iz]
            if record_wavefield_every and it % record_wavefield_every == 0:
                snapshots.append(np.sqrt(vx ** 2 + vz ** 2))

        out = {'t': np.arange(nt) * dt, 'dt': dt,
               'vx': rec_vx, 'vz': rec_vz,
               'final_vx': vx, 'final_vz': vz}
        if snapshots:
            out['snapshots'] = np.array(snapshots)
        return out
