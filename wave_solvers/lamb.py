"""Lamb-wave dispersion in a traction-free elastic plate.

Two independent routes, used to cross-validate each other (and so validate
the elastic constitutive physics used throughout ``elastic_helmholtz.py``):

* ``safe_dispersion`` -- the Semi-Analytical Finite Element (SAFE) method:
  the displacement through the plate thickness is discretized with finite
  elements (free surfaces are the natural boundary condition), giving a
  quadratic eigenvalue problem in the axial wavenumber ``k`` for each
  frequency ``omega``. Solving it yields the propagating Lamb modes
  ``k(omega)`` and their phase velocities ``c = omega / k``. (Bartoli et al.,
  J. Sound Vib. 295, 2006.)

* ``rayleigh_lamb_residual`` -- the classical analytic Rayleigh-Lamb
  characteristic functions for symmetric and antisymmetric modes; a true
  dispersion point makes one of them vanish. ``analytic_phase_velocities``
  root-finds them directly.

These are the building blocks for the Stage 3 plate validation in
``HINTS_AE_ROADMAP.md`` §4 (reproduce the Lamb dispersion and compare to the
analytic curves). Everything is nondimensional; pass physical ``(vp, vs)``
and plate thickness ``h`` in consistent units.
"""

import numpy as np
import scipy.linalg as sla


# ---------------------------------------------------------------------------
# Analytic Rayleigh-Lamb characteristic equations
# ---------------------------------------------------------------------------
def rayleigh_lamb_residual(omega, k, vp, vs, h):
    """Return ``(D_sym, D_anti)``, the symmetric and antisymmetric
    Rayleigh-Lamb characteristic functions for a plate of thickness ``h``
    (half-thickness ``b = h/2``). A propagating Lamb mode at ``(omega, k)``
    makes one of them zero.

    Uses complex ``p, q`` so the same expression covers all phase-velocity
    regimes (the result is real for real inputs).
    """
    b = 0.5 * h
    p = np.sqrt((omega / vp) ** 2 - k ** 2 + 0j)
    q = np.sqrt((omega / vs) ** 2 - k ** 2 + 0j)
    k2 = k ** 2
    term = (k2 - q ** 2) ** 2
    cross = 4.0 * k2 * p * q
    # D_sym = (k^2 - q^2)^2 cos(pb) sin(qb) + 4 k^2 p q sin(pb) cos(qb)
    d_sym = term * np.cos(p * b) * np.sin(q * b) \
        + cross * np.sin(p * b) * np.cos(q * b)
    # D_anti = (k^2 - q^2)^2 sin(pb) cos(qb) + 4 k^2 p q cos(pb) sin(qb)
    d_anti = term * np.sin(p * b) * np.cos(q * b) \
        + cross * np.cos(p * b) * np.sin(q * b)
    return d_sym.real, d_anti.real


def analytic_phase_velocities(freq, vp, vs, h, c_max=None, n_scan=4000):
    """Root-find the Rayleigh-Lamb phase velocities at ordinary frequency
    ``freq`` (omega = 2*pi*freq). Returns a sorted array of phase velocities
    ``c`` of all symmetric and antisymmetric modes found below ``c_max``."""
    omega = 2.0 * np.pi * freq
    if c_max is None:
        c_max = 6.0 * vs
    c_grid = np.linspace(0.2 * vs, c_max, n_scan)
    roots = []
    for which in (0, 1):
        prev_c, prev_d = None, None
        for c in c_grid:
            k = omega / c
            d = rayleigh_lamb_residual(omega, k, vp, vs, h)[which]
            if prev_d is not None and np.isfinite(d) and np.isfinite(prev_d) \
                    and prev_d * d < 0:
                # bisection refine
                lo, hi = prev_c, c
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    dm = rayleigh_lamb_residual(omega, omega / mid, vp, vs,
                                                h)[which]
                    if dm * rayleigh_lamb_residual(omega, omega / lo, vp, vs,
                                                   h)[which] < 0:
                        hi = mid
                    else:
                        lo = mid
                roots.append(0.5 * (lo + hi))
            prev_c, prev_d = c, d
    return np.array(sorted(roots))


# ---------------------------------------------------------------------------
# SAFE: through-thickness finite-element dispersion
# ---------------------------------------------------------------------------
def _safe_matrices(n_elem, vp, vs, rho, h):
    """Assemble the SAFE matrices K1, K2, K3, M for a plate of thickness h
    discretized with ``n_elem`` linear (2-node, 2-dof) elements through the
    thickness; free surfaces are natural."""
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2.0 * mu
    C = np.array([[lam + 2 * mu, lam, 0.0],
                  [lam, lam + 2 * mu, 0.0],
                  [0.0, 0.0, mu]])
    Lx = np.array([[1.0, 0], [0, 0], [0, 1.0]])
    Lz = np.array([[0, 0], [0, 1.0], [1.0, 0]])

    n_node = n_elem + 1
    ndof = 2 * n_node
    K1 = np.zeros((ndof, ndof))
    K2 = np.zeros((ndof, ndof))
    K3 = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    le = h / n_elem
    # 2-point Gauss on [0, le]
    gp = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)]) * le
    gw = np.array([0.5, 0.5]) * le

    for e in range(n_elem):
        dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        k1 = np.zeros((4, 4)); k2 = np.zeros((4, 4))
        k3 = np.zeros((4, 4)); me = np.zeros((4, 4))
        for z, w in zip(gp, gw):
            N1 = 1.0 - z / le
            N2 = z / le
            dN1, dN2 = -1.0 / le, 1.0 / le
            Nmat = np.array([[N1, 0, N2, 0], [0, N1, 0, N2]])
            dNmat = np.array([[dN1, 0, dN2, 0], [0, dN1, 0, dN2]])
            LxN = Lx @ Nmat            # 3x4
            LzdN = Lz @ dNmat          # 3x4
            k1 += w * LzdN.T @ C @ LzdN
            k3 += w * LxN.T @ C @ LxN
            k2 += w * (LxN.T @ C @ LzdN - LzdN.T @ C @ LxN)
            me += w * rho * Nmat.T @ Nmat
        for a in range(4):
            for bb in range(4):
                K1[dofs[a], dofs[bb]] += k1[a, bb]
                K2[dofs[a], dofs[bb]] += k2[a, bb]
                K3[dofs[a], dofs[bb]] += k3[a, bb]
                M[dofs[a], dofs[bb]] += me[a, bb]
    return K1, K2, K3, M


def safe_dispersion(freqs, vp, vs, rho, h, n_elem=20, c_max=None,
                    imag_tol=1e-3):
    """Lamb-wave phase velocities vs frequency via SAFE.

    For each frequency in ``freqs`` solve the quadratic eigenvalue problem
    ``(K1 + i k K2 + k^2 K3 - omega^2 M) U = 0`` for the wavenumbers ``k``;
    keep real, positive, propagating roots and return their phase velocities.

    Returns a list (one entry per frequency) of sorted phase-velocity arrays.
    """
    K1, K2, K3, M = _safe_matrices(n_elem, vp, vs, rho, h)
    ndof = K1.shape[0]
    if c_max is None:
        c_max = 6.0 * vs
    out = []
    I = np.eye(ndof)
    Z = np.zeros((ndof, ndof))
    for f in freqs:
        omega = 2.0 * np.pi * f
        A0 = K1 - omega ** 2 * M
        A1 = 1j * K2
        A2 = K3
        # generalized linearization: M1 z = k M2 z
        M1 = np.block([[A0, A1], [Z, I]])
        M2 = np.block([[Z, -A2], [I, Z]])
        kvals = sla.eig(M1, M2, right=False)
        cs = []
        for k in kvals:
            if not np.isfinite(k):
                continue
            if abs(k.real) < 1e-9:
                continue
            if abs(k.imag) > imag_tol * abs(k.real):   # propagating only
                continue
            kr = k.real
            if kr <= 0:
                continue
            c = omega / kr
            if 0 < c <= c_max:
                cs.append(c)
        out.append(np.array(sorted(set(np.round(cs, 6)))))
    return out
