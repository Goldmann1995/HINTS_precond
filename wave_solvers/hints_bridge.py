"""HINTS bridge: blend a complex DeepONet with classical relaxation / CSLP
to solve the frequency-domain Helmholtz systems from ``elastic_helmholtz.py``.

This is the methodology half of the bridge layer in ``HINTS_AE_ROADMAP.md``
§3. It mirrors ``HINTS_numpy/iterative_solver.py`` but generalises the two
ingredients the acoustic-emission target needs (roadmap §2.3):

* **complex** fields — the DeepONet predicts the real and imaginary parts of
  the solution update on separate output channels;
* a structure that is **ready for vector unknowns** — the per-node block size
  is a parameter, so Stage 2's elastic ``(ux, uz)`` reuses everything here.

Two ways to inject the network (roadmap §2.2), both provided:

(A) ``HINTSSolver`` — standalone hybrid iteration: ``ratio`` relaxation/CSLP
    smoothing steps, then one DeepONet residual correction. Closest to the
    original HINTS; used to reproduce the spectral-complementarity curves.
(B) ``make_hints_preconditioner`` — wraps "CSLP inner solve + DeepONet
    low-frequency correction" as a SciPy ``LinearOperator`` so it can drive
    GMRES (the scalable **HINTS-CSLP** route).

The DeepONet learns to approximate ``A^{-1}`` as an operator on the
right-hand-side function (trained on random smooth complex sources), so at
solve time it maps the current residual to an error correction.

``torch`` is imported lazily: the diagnostics and the classical-only paths
work with NumPy/SciPy alone; only building/using the network needs torch.
"""

import numpy as np
import scipy.sparse.linalg as spla

from . import helmholtz as _hz


# ===========================================================================
# Spectral diagnostics (NumPy only) — show low/high-frequency complementarity
# ===========================================================================
def fft_band_energy(field2d, n_low=4):
    """Split the energy of a 2D field into a low- and high-frequency band.

    Returns ``(e_low, e_high)`` where ``e_low`` is the energy in the lowest
    ``n_low`` x ``n_low`` Fourier modes (smooth error that relaxation is slow
    on and the DeepONet targets) and ``e_high`` is the rest. A 3D input
    ``(n_comp, nx, nz)`` (vector/elastic field) is summed over components.
    """
    field2d = np.asarray(field2d)
    if field2d.ndim == 3:
        los, his = zip(*(fft_band_energy(c, n_low) for c in field2d))
        return float(sum(los)), float(sum(his))
    F = np.fft.fft2(field2d)
    P = np.abs(F) ** 2
    mask = np.zeros_like(P, dtype=bool)
    mask[:n_low, :n_low] = True
    mask[:n_low, -n_low:] = True
    mask[-n_low:, :n_low] = True
    mask[-n_low:, -n_low:] = True
    e_low = float(P[mask].sum())
    e_high = float(P[~mask].sum())
    return e_low, e_high


# ===========================================================================
# Classical smoothers used inside the hybrid iteration
# ===========================================================================
def damped_jacobi_step(A, b, u, diag, omega=0.8):
    """One damped-Jacobi sweep. Diverges on indefinite Helmholtz alone — the
    failure mode HINTS repairs by interleaving the network correction."""
    return u + omega * (b - A @ u) / diag


class CSLPSmoother:
    """One complex shifted-Laplacian (CSLP) solve as a (robust) smoother.

    Far stronger than Jacobi on indefinite Helmholtz; used as the inner
    solve for the scalable HINTS-CSLP route.
    """

    def __init__(self, assemble, beta=(1.0, 0.5), **assemble_kwargs):
        self.M = _hz.make_shifted_laplacian_preconditioner(
            assemble, beta=beta, **assemble_kwargs)

    def __call__(self, A, b, u):
        return u + self.M.matvec(b - A @ u)


# ===========================================================================
# Standalone hybrid solver (route A)
# ===========================================================================
class HINTSSolver:
    """Blend a smoother with periodic DeepONet residual corrections.

    Parameters
    ----------
    A : complex sparse system matrix.
    shape : (nx, nz) grid shape used to reshape vectors <-> fields for the
        network and the FFT diagnostics.
    deeponet : an object with ``apply(residual_field) -> correction_field``
        (see ``ComplexDeepONet2D``), or ``None`` for smoother-only.
    smoother : 'jacobi' or a callable ``(A, b, u) -> u`` (e.g. ``CSLPSmoother``).
    ratio : DeepONet correction applied every ``ratio`` smoothing steps.
    jacobi_omega : damping for the Jacobi smoother.
    """

    def __init__(self, A, shape, deeponet=None, smoother='jacobi', ratio=10,
                 jacobi_omega=0.8, safeguard=True):
        self.A = A.tocsr()
        self.shape = shape
        self.deeponet = deeponet
        self.ratio = ratio
        self.jacobi_omega = jacobi_omega
        self.safeguard = safeguard
        self._diag = self.A.diagonal()
        if smoother == 'jacobi':
            self.smoother = lambda A, b, u: damped_jacobi_step(
                A, b, u, self._diag, jacobi_omega)
        elif callable(smoother):
            self.smoother = smoother
        else:
            raise ValueError("smoother must be 'jacobi' or a callable")

    def _apply_correction(self, A, b, u, corr):
        """Add the DeepONet correction with a backtracking safeguard.

        A standalone DeepONet that only approximates ``A^{-1}`` can amplify
        the error when applied to residuals unlike its training data, making
        the naive iteration diverge (the reliability caveat in roadmap §8).
        We accept the largest step ``alpha in {1, 1/2, 1/4, ...}`` along the
        correction that does not increase the residual; if none does, the
        correction is rejected (``alpha = 0``). With ``safeguard=False`` the
        raw correction is always taken (useful to *demonstrate* the failure).
        """
        if not self.safeguard:
            return u + corr
        r0 = np.linalg.norm(b - A @ u)
        if not np.all(np.isfinite(corr)):
            return u
        alpha = 1.0
        for _ in range(6):
            cand = u + alpha * corr
            if np.linalg.norm(b - A @ cand) < r0:
                return cand
            alpha *= 0.5
        return u                       # reject: no productive step found

    def solve(self, b, maxiter=400, rtol=1e-8, u0=None, record_spectrum=False,
              u_ref=None):
        """Run the hybrid iteration.

        Returns ``(u, info)`` with ``info`` holding the relative-residual
        history, the iterations at which the DeepONet acted, and (if
        ``record_spectrum``) the low/high-frequency error-band history.
        """
        A, b = self.A, np.asarray(b, dtype=complex)
        u = np.zeros(A.shape[0], dtype=complex) if u0 is None else u0.astype(complex)
        b_norm = max(np.linalg.norm(b), 1e-300)
        res_hist, don_steps = [], []
        band_low, band_high = [], []
        diverged = False

        for it in range(1, maxiter + 1):
            u = self.smoother(A, b, u)
            apply_don = (self.deeponet is not None and it % self.ratio == 0)
            if apply_don:
                r = b - A @ u
                corr = self.deeponet.apply(r.reshape(self.shape)).reshape(-1)
                u = self._apply_correction(A, b, u, corr)
                don_steps.append(it)

            res = np.linalg.norm(b - A @ u) / b_norm
            res_hist.append(res)
            if record_spectrum and u_ref is not None:
                err = (u_ref - u).reshape(self.shape)
                lo, hi = fft_band_energy(err)
                band_low.append(lo)
                band_high.append(hi)
            if not np.isfinite(res) or res < rtol:
                break
            # divergence guard: a standalone net can pollute the near-null
            # space of an indefinite operator without raising the residual,
            # which later blows up the smoother. Stop cleanly when it does.
            if res > 1e3 * res_hist[0]:
                diverged = True
                break

        info = {
            'converged': bool(res_hist and np.isfinite(res_hist[-1])
                              and res_hist[-1] < rtol),
            'diverged': diverged,
            'iterations': len(res_hist),
            'residuals': np.array(res_hist),
            'don_steps': np.array(don_steps),
        }
        if record_spectrum and u_ref is not None:
            info['band_low'] = np.array(band_low)
            info['band_high'] = np.array(band_high)
        return u, info


# ===========================================================================
# HINTS-CSLP preconditioner (route B)
# ===========================================================================
def fgmres(A, b, prec, rtol=1e-8, maxiter=200, x0=None):
    """Flexible GMRES (Saad 1993) — allows a *nonlinear / varying*
    preconditioner ``prec(vec) -> vec``.

    Standard GMRES assumes a constant linear preconditioner; the HINTS
    preconditioner (CSLP inner solve + a nonlinear DeepONet correction, with
    per-call RHS normalisation) violates that, so FGMRES is required. With a
    constant linear ``prec`` this reduces to ordinary GMRES, keeping the
    iteration-count comparison fair.

    Returns ``(x, info)`` with the relative-residual history.
    """
    A = A.tocsr()
    b = np.asarray(b, dtype=complex)
    n = b.shape[0]
    x = np.zeros(n, dtype=complex) if x0 is None else x0.astype(complex)
    b_norm = max(np.linalg.norm(b), 1e-300)

    r = b - A @ x
    V = np.zeros((maxiter + 1, n), dtype=complex)
    Z = np.zeros((maxiter, n), dtype=complex)
    H = np.zeros((maxiter + 1, maxiter), dtype=complex)
    history = []

    beta = np.linalg.norm(r)
    V[0] = r / max(beta, 1e-300)
    converged = False
    j = 0
    for j in range(maxiter):
        Z[j] = prec(V[j])
        w = A @ Z[j]
        for i in range(j + 1):
            H[i, j] = np.vdot(V[i], w)
            w = w - H[i, j] * V[i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] > 1e-300:
            V[j + 1] = w / H[j + 1, j]
        e1 = np.zeros(j + 2, dtype=complex)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
        res = np.linalg.norm(H[:j + 2, :j + 1] @ y - e1) / b_norm
        history.append(res)
        if res < rtol:
            converged = True
            break

    y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], beta * _unit(j + 2), rcond=None)
    x = x + Z[:j + 1].T @ y
    info = {'converged': converged, 'iterations': len(history),
            'residuals': np.array(history)}
    return x, info


def _unit(m):
    e = np.zeros(m, dtype=complex)
    e[0] = 1.0
    return e


def make_hints_preconditioner(A, shape, deeponet, cslp, n_inner=1):
    """Combine a CSLP inner solve with a DeepONet low-frequency correction
    into a ``LinearOperator`` suitable as the GMRES preconditioner ``M``.

    Applying ``M`` to a residual ``r`` performs ``n_inner`` CSLP smoothing
    steps on ``M x = r`` and then adds the DeepONet correction of the updated
    residual — a single multiplicative two-stage preconditioner.
    """
    A = A.tocsr()

    def apply(r):
        r = np.asarray(r, dtype=complex)
        x = np.zeros_like(r)
        for _ in range(n_inner):
            x = cslp(A, r, x)
        if deeponet is not None:
            resid = r - A @ x
            x = x + deeponet.apply(resid.reshape(shape)).reshape(-1)
        return x

    return spla.LinearOperator(A.shape, matvec=apply, dtype=complex)


# ===========================================================================
# Complex DeepONet (route requires torch; imported lazily)
# ===========================================================================
def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'Building/using ComplexDeepONet2D requires PyTorch. '
            'Install it (CPU build is fine) or use the classical-only paths.'
        ) from exc


class ComplexDeepONet2D:
    """A complex-valued DeepONet solution operator on a fixed 2D grid.

    Branch: CNN over input channels ``[k_norm, Re f_n, Im f_n]`` -> latent.
    Trunk:  MLP over node coordinates ``(x, z)`` -> latent.
    Output: ``Re(u), Im(u)`` via two independent branch-trunk contractions,
            rescaled by the RHS norm (the operator is linear in the RHS, so
            normalising the input and multiplying the output back is exact).

    The Dirichlet boundary mask used in ``HINTS_numpy`` is intentionally
    dropped because Sommerfeld solutions are nonzero on the boundary.

    Designed as a thin wrapper exposing ``apply(field) -> field`` so the
    classical solver code above stays NumPy-only.
    """

    def __init__(self, nx, nz, k_field, latent=64, device='cpu'):
        torch = _require_torch()
        self.torch = torch
        self.nx, self.nz = nx, nz
        self.latent = latent
        self.device = device
        # normalization stats for the (fixed) k field
        self.k_mean = float(np.mean(k_field))
        self.k_std = float(np.std(k_field) + 1e-8)
        self._k_norm = ((np.asarray(k_field) - self.k_mean) / self.k_std)

        x = np.linspace(0.0, 1.0, nx)
        z = np.linspace(0.0, 1.0, nz)
        xx, zz = np.meshgrid(x, z, indexing='ij')
        coords = np.stack([xx.reshape(-1), zz.reshape(-1)], axis=1)
        self._coords = torch.tensor(coords, dtype=torch.float32, device=device)
        self.model = _DeepONetModule(latent).to(device)

    # -- internals --------------------------------------------------------
    def _branch_input(self, f_fields_norm):
        """Assemble the (batch, 3, nx, nz) branch tensor from normalized
        complex RHS fields given as (batch, nx, nz) complex arrays."""
        torch = self.torch
        b = f_fields_norm.shape[0]
        k = np.broadcast_to(self._k_norm, (b, self.nx, self.nz))
        chans = np.stack([k, f_fields_norm.real, f_fields_norm.imag], axis=1)
        return torch.tensor(chans, dtype=torch.float32, device=self.device)

    def _forward_fields(self, f_fields):
        """Predict u for a batch of complex RHS fields (batch, nx, nz)."""
        torch = self.torch
        f = np.asarray(f_fields, dtype=complex)
        norm = np.sqrt(np.mean(np.abs(f) ** 2, axis=(1, 2)))
        norm = np.where(norm > 0, norm, 1.0)            # guard zero RHS
        f_n = f / norm[:, None, None]
        branch_in = self._branch_input(f_n)
        trunk = self.model.trunk(self._coords)                  # (N, latent)
        b_re, b_im = self.model.branch(branch_in)               # (batch, latent) each
        u_re = b_re @ trunk.T
        u_im = b_im @ trunk.T
        scale = torch.tensor(norm, dtype=torch.float32, device=self.device)[:, None]
        u_re = (u_re * scale).reshape(-1, self.nx, self.nz)
        u_im = (u_im * scale).reshape(-1, self.nx, self.nz)
        return u_re, u_im

    # -- public API -------------------------------------------------------
    def apply(self, residual_field):
        """Map a single complex residual field -> complex correction field."""
        torch = self.torch
        with torch.no_grad():
            u_re, u_im = self._forward_fields(residual_field[None])
        u_re = u_re[0].cpu().numpy()
        u_im = u_im[0].cpu().numpy()
        return u_re + 1j * u_im

    def fit(self, f_train, u_train, epochs=2000, batch_size=64, lr=1e-3,
            f_val=None, u_val=None, log_every=200, logger=print):
        """Train on complex RHS fields ``f_train`` (N, nx, nz) and their
        solutions ``u_train`` (N, nx, nz). Returns the loss history."""
        torch = self.torch
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.5 ** (1 / 1500))
        n = f_train.shape[0]
        u_re_t = torch.tensor(u_train.real, dtype=torch.float32, device=self.device)
        u_im_t = torch.tensor(u_train.imag, dtype=torch.float32, device=self.device)
        hist = []
        for ep in range(epochs):
            perm = np.random.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                pre_re, pre_im = self._forward_fields(f_train[idx])
                loss = (torch.mean((pre_re - u_re_t[idx]) ** 2)
                        + torch.mean((pre_im - u_im_t[idx]) ** 2))
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(idx)
            sched.step()
            ep_loss /= n
            hist.append(ep_loss)
            if (ep + 1) % log_every == 0 or ep == 0:
                msg = f'  epoch {ep + 1:5d}  train_mse {ep_loss:.3e}'
                if f_val is not None:
                    msg += f'  val_rel {self.relative_error(f_val, u_val):.3e}'
                logger(msg)
        return np.array(hist)

    def relative_error(self, f_fields, u_fields):
        """Mean relative L2 error over a batch (validation metric)."""
        torch = self.torch
        with torch.no_grad():
            u_re, u_im = self._forward_fields(f_fields)
        pred = u_re.cpu().numpy() + 1j * u_im.cpu().numpy()
        num = np.linalg.norm((pred - u_fields).reshape(len(f_fields), -1), axis=1)
        den = np.linalg.norm(u_fields.reshape(len(f_fields), -1), axis=1) + 1e-30
        return float(np.mean(num / den))

    def save(self, path):
        self.torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(
            self.torch.load(path, map_location=self.device))
        self.model.eval()


def _build_module_classes():
    """Define the torch ``nn.Module`` lazily so importing this file without
    torch still works.

    Two design choices matter for indefinite/oscillatory Helmholtz:
    * the branch keeps spatial resolution (stride-2 convs down to 4x4, then a
      flattening FC) instead of global average pooling, so the *location* of
      the source blob is preserved in the latent code;
    * the trunk lifts the coordinates through fixed random **Fourier
      features** before the MLP, defeating the spectral bias that otherwise
      prevents a smooth network from representing the wave's oscillations.
    """
    torch = _require_torch()
    import torch.nn as nn

    class _Branch(nn.Module):
        def __init__(self, latent):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten())
            self.fc = nn.Sequential(
                nn.Linear(128 * 16, 256), nn.ReLU(),
                nn.Linear(256, 2 * latent))
            self.latent = latent

        def forward(self, x):
            h = self.fc(self.conv(x))
            return h[:, :self.latent], h[:, self.latent:]

    class _Trunk(nn.Module):
        def __init__(self, latent, n_freq=48, scale=6.0):
            super().__init__()
            B = torch.randn(2, n_freq) * scale
            self.register_buffer('B', B)               # fixed, saved with state
            self.net = nn.Sequential(
                nn.Linear(2 * n_freq, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, latent))

        def forward(self, c):
            proj = 2.0 * np.pi * (c @ self.B)
            feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
            return self.net(feats)

    class _Module(nn.Module):
        def __init__(self, latent):
            super().__init__()
            self.branch = _Branch(latent)
            self.trunk = _Trunk(latent)

    return _Module


def _build_vector_module_classes():
    """Module factory for the vector/elastic DeepONet: configurable number of
    input channels and output heads (one branch latent pair per component)."""
    torch = _require_torch()
    import torch.nn as nn

    class _VBranch(nn.Module):
        def __init__(self, latent, in_ch, n_heads):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten())
            self.fc = nn.Sequential(
                nn.Linear(128 * 16, 256), nn.ReLU(),
                nn.Linear(256, n_heads * latent))
            self.latent = latent
            self.n_heads = n_heads

        def forward(self, x):
            h = self.fc(self.conv(x))
            return [h[:, i * self.latent:(i + 1) * self.latent]
                    for i in range(self.n_heads)]

    class _VTrunk(nn.Module):
        def __init__(self, latent, n_freq=48, scale=6.0):
            super().__init__()
            self.register_buffer('B', torch.randn(2, n_freq) * scale)
            self.net = nn.Sequential(
                nn.Linear(2 * n_freq, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, latent))

        def forward(self, c):
            proj = 2.0 * np.pi * (c @ self.B)
            return self.net(torch.cat([torch.sin(proj), torch.cos(proj)], 1))

    class _VModule(nn.Module):
        def __init__(self, latent, in_ch, n_heads):
            super().__init__()
            self.branch = _VBranch(latent, in_ch, n_heads)
            self.trunk = _VTrunk(latent)

    return _VModule


# Resolve the module classes at import time only if torch is present, so the
# name exists for type hints / construction; otherwise leave a lazy factory.
try:  # pragma: no cover - exercised indirectly
    _DeepONetModule = _build_module_classes()
    _VectorDeepONetModule = _build_vector_module_classes()
except ImportError:  # torch missing: classical paths still import fine
    _DeepONetModule = None
    _VectorDeepONetModule = None


class VectorComplexDeepONet2D:
    """Complex, **vector-valued** DeepONet for the elastic Navier-Helmholtz
    system (Stage 2). Predicts ``u = (ux, uz)`` (each complex) from a vector
    RHS ``f = (fx, fz)`` on a fixed grid.

    Input channels: ``[static material fields..., Re fx, Im fx, Re fz, Im fz]``
    (normalized). Output heads: ``2 * n_comp`` branch latents giving
    ``Re/Im`` of each displacement component via independent branch-trunk
    contractions, rescaled by the RHS norm (exact, since the operator is
    linear in the RHS).

    Exposes ``apply(field) -> field`` with ``field`` shaped ``(n_comp, nx,
    nz)`` complex, so ``HINTSSolver``/``make_hints_preconditioner`` (which use
    ``shape=(n_comp, nx, nz)``) work unchanged from the scalar case.
    """

    def __init__(self, nx, nz, static_fields, n_comp=2, latent=64,
                 device='cpu'):
        torch = _require_torch()
        self.torch = torch
        self.nx, self.nz, self.n_comp = nx, nz, n_comp
        self.latent = latent
        self.device = device
        static = np.atleast_3d(np.asarray(static_fields, dtype=float))
        if static.shape[:2] == (nx, nz):           # (nx, nz, n_static)
            static = np.moveaxis(static, -1, 0)
        self.n_static = static.shape[0]
        self._static_norm = np.stack(
            [(s - s.mean()) / (s.std() + 1e-8) for s in static], axis=0)
        in_ch = self.n_static + 2 * n_comp
        x = np.linspace(0.0, 1.0, nx)
        z = np.linspace(0.0, 1.0, nz)
        xx, zz = np.meshgrid(x, z, indexing='ij')
        coords = np.stack([xx.reshape(-1), zz.reshape(-1)], axis=1)
        self._coords = torch.tensor(coords, dtype=torch.float32, device=device)
        self.model = _VectorDeepONetModule(latent, in_ch, 2 * n_comp).to(device)

    def _forward_fields(self, f_fields):
        """f_fields: (batch, n_comp, nx, nz) complex -> (re, im) each
        (batch, n_comp, nx, nz)."""
        torch = self.torch
        f = np.asarray(f_fields, dtype=complex)
        b = f.shape[0]
        norm = np.sqrt(np.mean(np.abs(f) ** 2, axis=(1, 2, 3)))
        norm = np.where(norm > 0, norm, 1.0)
        f_n = f / norm[:, None, None, None]
        static = np.broadcast_to(self._static_norm,
                                 (b, self.n_static, self.nx, self.nz))
        chans = [static]
        for c in range(self.n_comp):
            chans.append(f_n[:, c].real[:, None])
            chans.append(f_n[:, c].imag[:, None])
        branch_in = torch.tensor(np.concatenate(chans, axis=1),
                                 dtype=torch.float32, device=self.device)
        trunk = self.model.trunk(self._coords)             # (N, latent)
        heads = self.model.branch(branch_in)               # list of (b, latent)
        scale = torch.tensor(norm, dtype=torch.float32,
                             device=self.device)[:, None]
        u_re = torch.stack(
            [(heads[2 * c] @ trunk.T * scale).reshape(b, self.nx, self.nz)
             for c in range(self.n_comp)], dim=1)
        u_im = torch.stack(
            [(heads[2 * c + 1] @ trunk.T * scale).reshape(b, self.nx, self.nz)
             for c in range(self.n_comp)], dim=1)
        return u_re, u_im

    def apply(self, residual_field):
        torch = self.torch
        with torch.no_grad():
            u_re, u_im = self._forward_fields(residual_field[None])
        return u_re[0].cpu().numpy() + 1j * u_im[0].cpu().numpy()

    def fit(self, f_train, u_train, epochs=2000, batch_size=64, lr=1e-3,
            f_val=None, u_val=None, log_every=200, logger=print):
        torch = self.torch
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.5 ** (1 / 1500))
        n = f_train.shape[0]
        u_re_t = torch.tensor(u_train.real, dtype=torch.float32, device=self.device)
        u_im_t = torch.tensor(u_train.imag, dtype=torch.float32, device=self.device)
        hist = []
        for ep in range(epochs):
            perm = np.random.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                pre_re, pre_im = self._forward_fields(f_train[idx])
                loss = (torch.mean((pre_re - u_re_t[idx]) ** 2)
                        + torch.mean((pre_im - u_im_t[idx]) ** 2))
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(idx)
            sched.step()
            ep_loss /= n
            hist.append(ep_loss)
            if (ep + 1) % log_every == 0 or ep == 0:
                msg = f'  epoch {ep + 1:5d}  train_mse {ep_loss:.3e}'
                if f_val is not None:
                    msg += f'  val_rel {self.relative_error(f_val, u_val):.3e}'
                logger(msg)
        return np.array(hist)

    def relative_error(self, f_fields, u_fields):
        torch = self.torch
        with torch.no_grad():
            u_re, u_im = self._forward_fields(f_fields)
        pred = u_re.cpu().numpy() + 1j * u_im.cpu().numpy()
        num = np.linalg.norm((pred - u_fields).reshape(len(f_fields), -1), axis=1)
        den = np.linalg.norm(u_fields.reshape(len(f_fields), -1), axis=1) + 1e-30
        return float(np.mean(num / den))

    def save(self, path):
        self.torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(self.torch.load(path, map_location=self.device))
        self.model.eval()


def _build_multifreq_module_classes():
    """Module factory for the multi-frequency vector DeepONet: the branch is
    additionally conditioned on the (normalized) angular frequency omega so a
    single network amortizes the whole acoustic-emission band (roadmap §4
    Stage 3). omega is embedded with Fourier features and concatenated to the
    flattened convolutional code."""
    torch = _require_torch()
    import torch.nn as nn

    class _MFBranch(nn.Module):
        def __init__(self, latent, in_ch, n_heads, omega_feats=8):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten())
            self.register_buffer('wfreq', torch.linspace(1, omega_feats,
                                                         omega_feats))
            self.fc = nn.Sequential(
                nn.Linear(128 * 16 + 2 * omega_feats, 256), nn.ReLU(),
                nn.Linear(256, n_heads * latent))
            self.latent = latent
            self.n_heads = n_heads

        def forward(self, x, omega_norm):
            h = self.conv(x)
            proj = omega_norm[:, None] * self.wfreq[None, :]
            wemb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
            h = self.fc(torch.cat([h, wemb], dim=1))
            return [h[:, i * self.latent:(i + 1) * self.latent]
                    for i in range(self.n_heads)]

    class _MFTrunk(nn.Module):
        def __init__(self, latent, n_freq=48, scale=6.0):
            super().__init__()
            self.register_buffer('B', torch.randn(2, n_freq) * scale)
            self.net = nn.Sequential(
                nn.Linear(2 * n_freq, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, latent))

        def forward(self, c):
            proj = 2.0 * np.pi * (c @ self.B)
            return self.net(torch.cat([torch.sin(proj), torch.cos(proj)], 1))

    class _MFModule(nn.Module):
        def __init__(self, latent, in_ch, n_heads):
            super().__init__()
            self.branch = _MFBranch(latent, in_ch, n_heads)
            self.trunk = _MFTrunk(latent)

    return _MFModule


try:  # pragma: no cover
    _MultiFreqModule = _build_multifreq_module_classes()
except ImportError:
    _MultiFreqModule = None


class MultiFreqVectorDeepONet2D:
    """Multi-frequency, complex, vector-valued DeepONet (Stage 3).

    A single network approximates ``A(omega)^{-1}`` across a band of angular
    frequencies: the branch is conditioned on the normalized ``omega`` (so the
    one operator can be reused at every frequency of an ``freq_to_time`` AE
    synthesis, instead of training a separate network per frequency).

    Same field/channel conventions as ``VectorComplexDeepONet2D``; ``apply``
    and ``_forward_fields`` additionally take ``omega``.
    """

    def __init__(self, nx, nz, static_fields, omega_ref, n_comp=2, latent=64,
                 device='cpu'):
        torch = _require_torch()
        self.torch = torch
        self.nx, self.nz, self.n_comp = nx, nz, n_comp
        self.omega_ref = float(omega_ref)
        self.device = device
        static = np.atleast_3d(np.asarray(static_fields, dtype=float))
        if static.shape[:2] == (nx, nz):
            static = np.moveaxis(static, -1, 0)
        self.n_static = static.shape[0]
        self._static_norm = np.stack(
            [(s - s.mean()) / (s.std() + 1e-8) for s in static], axis=0)
        in_ch = self.n_static + 2 * n_comp
        x = np.linspace(0.0, 1.0, nx)
        z = np.linspace(0.0, 1.0, nz)
        xx, zz = np.meshgrid(x, z, indexing='ij')
        coords = np.stack([xx.reshape(-1), zz.reshape(-1)], axis=1)
        self._coords = torch.tensor(coords, dtype=torch.float32, device=device)
        self.model = _MultiFreqModule(latent, in_ch, 2 * n_comp).to(device)

    def _forward_fields(self, f_fields, omegas):
        torch = self.torch
        f = np.asarray(f_fields, dtype=complex)
        b = f.shape[0]
        norm = np.sqrt(np.mean(np.abs(f) ** 2, axis=(1, 2, 3)))
        norm = np.where(norm > 0, norm, 1.0)
        f_n = f / norm[:, None, None, None]
        static = np.broadcast_to(self._static_norm,
                                 (b, self.n_static, self.nx, self.nz))
        chans = [static]
        for c in range(self.n_comp):
            chans.append(f_n[:, c].real[:, None])
            chans.append(f_n[:, c].imag[:, None])
        branch_in = torch.tensor(np.concatenate(chans, axis=1),
                                 dtype=torch.float32, device=self.device)
        omega_norm = torch.tensor(np.asarray(omegas) / self.omega_ref,
                                  dtype=torch.float32, device=self.device)
        trunk = self.model.trunk(self._coords)
        heads = self.model.branch(branch_in, omega_norm)
        scale = torch.tensor(norm, dtype=torch.float32,
                             device=self.device)[:, None]
        u_re = torch.stack(
            [(heads[2 * c] @ trunk.T * scale).reshape(b, self.nx, self.nz)
             for c in range(self.n_comp)], dim=1)
        u_im = torch.stack(
            [(heads[2 * c + 1] @ trunk.T * scale).reshape(b, self.nx, self.nz)
             for c in range(self.n_comp)], dim=1)
        return u_re, u_im

    def apply(self, residual_field, omega):
        torch = self.torch
        with torch.no_grad():
            u_re, u_im = self._forward_fields(residual_field[None], [omega])
        return u_re[0].cpu().numpy() + 1j * u_im[0].cpu().numpy()

    def fit(self, f_train, u_train, omega_train, epochs=2000, batch_size=64,
            lr=1e-3, val=None, log_every=200, logger=print):
        """Train on ``(f, u, omega)`` triples. ``val`` is an optional
        ``(f_val, u_val, omega_val)`` tuple for the logged metric."""
        torch = self.torch
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.5 ** (1 / 1500))
        n = f_train.shape[0]
        omega_train = np.asarray(omega_train)
        u_re_t = torch.tensor(u_train.real, dtype=torch.float32, device=self.device)
        u_im_t = torch.tensor(u_train.imag, dtype=torch.float32, device=self.device)
        hist = []
        for ep in range(epochs):
            perm = np.random.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                pre_re, pre_im = self._forward_fields(f_train[idx],
                                                      omega_train[idx])
                loss = (torch.mean((pre_re - u_re_t[idx]) ** 2)
                        + torch.mean((pre_im - u_im_t[idx]) ** 2))
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(idx)
            sched.step()
            ep_loss /= n
            hist.append(ep_loss)
            if (ep + 1) % log_every == 0 or ep == 0:
                msg = f'  epoch {ep + 1:5d}  train_mse {ep_loss:.3e}'
                if val is not None:
                    msg += f'  val_rel {self.relative_error(*val):.3e}'
                logger(msg)
        return np.array(hist)

    def relative_error(self, f_fields, u_fields, omegas):
        torch = self.torch
        with torch.no_grad():
            u_re, u_im = self._forward_fields(f_fields, omegas)
        pred = u_re.cpu().numpy() + 1j * u_im.cpu().numpy()
        num = np.linalg.norm((pred - u_fields).reshape(len(f_fields), -1), axis=1)
        den = np.linalg.norm(u_fields.reshape(len(f_fields), -1), axis=1) + 1e-30
        return float(np.mean(num / den))

    def per_frequency_error(self, f_fields, u_fields, omegas):
        """Mean relative error grouped by frequency -- shows that accuracy is
        stable across the band (the point of amortization)."""
        omegas = np.asarray(omegas)
        out = {}
        for w in np.unique(omegas):
            m = omegas == w
            out[float(w)] = self.relative_error(f_fields[m], u_fields[m],
                                                omegas[m])
        return out

    def save(self, path):
        self.torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(self.torch.load(path, map_location=self.device))
        self.model.eval()


# ===========================================================================
# Dataset generation helper
# ===========================================================================
def generate_dataset(problem, n_samples, rng=None, n_blobs=3, point_frac=0.3):
    """Sample ``n_samples`` (RHS, solution) pairs for a fixed problem.

    Returns ``(f_fields, u_fields)`` as complex arrays of shape
    ``(n_samples, nx, nz)``. Solutions are computed with the direct solver,
    so training teaches the DeepONet to approximate ``A^{-1}`` for this
    operator over a space of right-hand sides.

    ``point_frac`` of the samples use a localized (point/narrow-Gaussian)
    source instead of smooth blobs. This matters for HINTS: the actual solve
    is driven by a point source and the early residuals are spiky, so the
    network must see localized inputs during training or it extrapolates and
    destabilizes the iteration.
    """
    rng = np.random.default_rng() if rng is None else rng
    nx, nz = problem.nx, problem.nz
    block = getattr(problem, 'block_size', 1)
    field_shape = (nx, nz) if block == 1 else (block, nx, nz)
    f_fields = np.empty((n_samples,) + field_shape, dtype=complex)
    u_fields = np.empty((n_samples,) + field_shape, dtype=complex)
    x = np.linspace(0, 1, nx)
    z = np.linspace(0, 1, nz)
    xx, zz = np.meshgrid(x, z, indexing='ij')
    lu = spla.splu(problem.A.tocsc())              # one factorization, reused
    for i in range(n_samples):
        if block == 1 and rng.random() < point_frac:
            cx, cz = rng.uniform(0.1, 0.9, size=2)
            rad = rng.uniform(0.02, 0.06)          # narrow -> near-delta
            weight = rng.normal() + 1j * rng.normal()
            f = (weight * np.exp(-((xx - cx) ** 2 + (zz - cz) ** 2)
                                 / (2.0 * rad ** 2))).reshape(-1)
        else:
            f = problem.smooth_random_source(rng=rng, n_blobs=n_blobs)
        u = lu.solve(f)
        f_fields[i] = f.reshape(field_shape)
        u_fields[i] = u.reshape(field_shape)
    return f_fields, u_fields


def generate_multifreq_dataset(problem_factory, omegas, n_per_omega, rng=None,
                               n_blobs=3):
    """Build a multi-frequency training set for ``MultiFreqVectorDeepONet2D``.

    ``problem_factory(omega)`` returns a problem at that angular frequency.
    For each omega in ``omegas`` the operator is factorized once and
    ``n_per_omega`` random RHS are solved. Returns ``(f_fields, u_fields,
    omega_array)`` stacked over all frequencies.
    """
    rng = np.random.default_rng() if rng is None else rng
    f_all, u_all, w_all = [], [], []
    for omega in omegas:
        p = problem_factory(omega)
        block = getattr(p, 'block_size', 1)
        shape = (p.nx, p.nz) if block == 1 else (block, p.nx, p.nz)
        lu = spla.splu(p.A.tocsc())
        for _ in range(n_per_omega):
            f = p.smooth_random_source(rng=rng, n_blobs=n_blobs)
            f_all.append(f.reshape(shape))
            u_all.append(lu.solve(f).reshape(shape))
            w_all.append(omega)
    return (np.array(f_all), np.array(u_all), np.array(w_all))
