"""Classical (non-learned) solvers for the Helmholtz equation.

Discretization
--------------
Second-order finite differences for

    u'' + k(x)^2 u = f                  (1D, on [0, Lx])
    Laplacian(u) + k(x,y)^2 u = f       (2D, on [0, Lx] x [0, Ly])

with either homogeneous Dirichlet boundaries (sound-soft) or first-order
Sommerfeld radiation boundaries  du/dn - i k u = 0  (absorbing), the latter
producing a complex-valued system as is standard in frequency-domain
acoustics.

Solvers
-------
* ``solve_direct``        -- sparse LU (SuperLU via ``scipy.sparse.linalg.splu``)
* ``solve_gmres``         -- restarted GMRES, optionally preconditioned
* ``solve_bicgstab``      -- BiCGSTAB, optionally preconditioned
* ``make_ilu_preconditioner``
* ``make_shifted_laplacian_preconditioner``
                          -- the complex shifted-Laplacian preconditioner
                             (CSLP) of Erlangga, Vuik & Oosterlee
                             (Appl. Numer. Math. 50, 2004; SIAM J. Sci.
                             Comput. 27, 2006), the standard "traditional"
                             preconditioner for indefinite Helmholtz
                             problems and the natural baseline for the
                             HINTS hybrid preconditioners in this repo.
* ``jacobi`` / ``gauss_seidel`` / ``sor``
                          -- classical stationary iterations (these
                             generally diverge on indefinite Helmholtz
                             systems; they are provided as reference
                             baselines and as smoothers).

All Krylov drivers return ``(u, info_dict)`` where ``info_dict`` contains
the relative-residual history so convergence can be compared against the
HINTS solvers.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def helmholtz_matrix_1d(n, length=1.0, k=1.0, bc='dirichlet', k_scale=1.0):
    """Assemble the 1D Helmholtz operator on ``n`` nodes.

    Parameters
    ----------
    n : number of grid nodes (including boundary nodes).
    length : domain length.
    k : wavenumber; scalar or array of nodal values (heterogeneous medium).
    bc : 'dirichlet' (rows of identity at the boundary) or 'sommerfeld'
         (first-order radiation condition, complex matrix).
    k_scale : complex factor multiplying k^2; used internally to build the
              shifted-Laplacian preconditioner.

    Returns
    -------
    A : sparse CSR matrix of shape (n, n).
    """
    dx = length / (n - 1)
    k = np.broadcast_to(np.asarray(k, dtype=float), (n,))
    dtype = complex if (bc == 'sommerfeld' or np.iscomplexobj(k_scale)
                        or isinstance(k_scale, complex)) else float

    main = (k_scale * k ** 2 - 2.0 / dx ** 2).astype(dtype)
    off = np.full(n - 1, 1.0 / dx ** 2, dtype=dtype)
    A = sp.diags([off, main, off], [-1, 0, 1], format='lil')

    if bc == 'dirichlet':
        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
    elif bc == 'sommerfeld':
        # left:  -du/dx = i k u  ->  (u0 - u1)/dx - i k u0 = 0
        A[0, :] = 0.0
        A[0, 0] = 1.0 / dx - 1j * k[0]
        A[0, 1] = -1.0 / dx
        # right:  du/dx = i k u  ->  (uN - uN-1)/dx - i k uN = 0
        A[-1, :] = 0.0
        A[-1, -1] = 1.0 / dx - 1j * k[-1]
        A[-1, -2] = -1.0 / dx
    else:
        raise ValueError("bc must be 'dirichlet' or 'sommerfeld'")
    return A.tocsr()


def helmholtz_matrix_2d(nx, ny, lx=1.0, ly=1.0, k=1.0, bc='dirichlet',
                        k_scale=1.0):
    """Assemble the 2D Helmholtz operator (5-point stencil), row-major
    ordering with index = ix * ny + iy (matching HINTS_numpy/utils.py).

    Returns a sparse CSR matrix of shape (nx*ny, nx*ny).
    """
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    k = np.broadcast_to(np.asarray(k, dtype=float), (nx, ny))
    dtype = complex if (bc == 'sommerfeld' or np.iscomplexobj(k_scale)
                        or isinstance(k_scale, complex)) else float
    n = nx * ny

    A = sp.lil_matrix((n, n), dtype=dtype)
    interior_x = lambda ix: 0 < ix < nx - 1
    interior_y = lambda iy: 0 < iy < ny - 1

    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            if interior_x(ix) and interior_y(iy):
                A[row, row] = k_scale * k[ix, iy] ** 2 \
                    - 2.0 / dx ** 2 - 2.0 / dy ** 2
                A[row, row - ny] = 1.0 / dx ** 2
                A[row, row + ny] = 1.0 / dx ** 2
                A[row, row - 1] = 1.0 / dy ** 2
                A[row, row + 1] = 1.0 / dy ** 2
            elif bc == 'dirichlet':
                A[row, row] = 1.0
            elif bc == 'sommerfeld':
                # first-order radiation condition du/dn - i k u = 0 using a
                # one-sided difference toward the interior neighbour
                kij = k[ix, iy]
                if ix == 0:
                    h, nbr = dx, row + ny
                elif ix == nx - 1:
                    h, nbr = dx, row - ny
                elif iy == 0:
                    h, nbr = dy, row + 1
                else:
                    h, nbr = dy, row - 1
                A[row, row] = 1.0 / h - 1j * kij
                A[row, nbr] = -1.0 / h
            else:
                raise ValueError("bc must be 'dirichlet' or 'sommerfeld'")
    return A.tocsr()


# ---------------------------------------------------------------------------
# Direct and Krylov solvers
# ---------------------------------------------------------------------------
def solve_direct(A, b):
    """Sparse LU factorization (SuperLU). Reference 'traditional' solver."""
    lu = spla.splu(A.tocsc())
    return lu.solve(np.asarray(b, dtype=A.dtype))


class _ResidualMonitor:
    """Records the relative residual at every Krylov iteration."""

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.b_norm = np.linalg.norm(b)
        self.history = []

    def from_residual_norm(self, rk_norm):
        self.history.append(float(rk_norm) / max(self.b_norm, 1e-300))

    def from_iterate(self, xk):
        rk = self.b - self.A @ xk
        self.history.append(np.linalg.norm(rk) / max(self.b_norm, 1e-300))


def solve_gmres(A, b, M=None, rtol=1e-8, maxiter=1000, restart=50, x0=None):
    """Restarted GMRES with optional preconditioner ``M`` (LinearOperator).

    Returns (u, info) with info = {'converged', 'iterations', 'residuals'}.
    """
    monitor = _ResidualMonitor(A, b)
    u, flag = spla.gmres(A, b, M=M, rtol=rtol, maxiter=maxiter,
                         restart=restart, x0=x0,
                         callback=monitor.from_residual_norm,
                         callback_type='pr_norm')
    info = {'converged': flag == 0,
            'iterations': len(monitor.history),
            'residuals': np.array(monitor.history)}
    return u, info


def solve_bicgstab(A, b, M=None, rtol=1e-8, maxiter=1000, x0=None):
    """BiCGSTAB with optional preconditioner ``M`` (LinearOperator)."""
    monitor = _ResidualMonitor(A, b)
    u, flag = spla.bicgstab(A, b, M=M, rtol=rtol, maxiter=maxiter, x0=x0,
                            callback=monitor.from_iterate)
    info = {'converged': flag == 0,
            'iterations': len(monitor.history),
            'residuals': np.array(monitor.history)}
    return u, info


def make_ilu_preconditioner(A, drop_tol=1e-4, fill_factor=10):
    """Incomplete-LU preconditioner as a scipy LinearOperator."""
    ilu = spla.spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    return spla.LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype)


def make_shifted_laplacian_preconditioner(assemble, beta=(1.0, 0.5),
                                          use_amg=False, **assemble_kwargs):
    """Complex shifted-Laplacian preconditioner (CSLP).

    Builds  M = -Laplacian - (beta1 - i beta2) k^2  and returns a
    LinearOperator applying M^{-1}.  beta = (1, 0.5) is the classical
    choice of Erlangga, Vuik & Oosterlee (2004, 2006).

    Parameters
    ----------
    assemble : ``helmholtz_matrix_1d`` or ``helmholtz_matrix_2d``.
    beta : (beta1, beta2) real/imaginary shift of k^2.
    use_amg : if True and pyamg is installed, approximate M^{-1} with one
              AMG V-cycle (the large-scale variant); otherwise exact LU.
    assemble_kwargs : forwarded to ``assemble`` (n / nx, ny, k, bc, ...).
    """
    beta1, beta2 = beta
    M_mat = assemble(k_scale=beta1 - 1j * beta2, **assemble_kwargs)

    if use_amg:
        try:
            import pyamg
            ml = pyamg.smoothed_aggregation_solver(M_mat.tocsr())
            return spla.LinearOperator(
                M_mat.shape, matvec=lambda r: ml.solve(r, maxiter=1),
                dtype=complex)
        except ImportError:
            pass  # fall back to the exact LU variant below
    lu = spla.splu(M_mat.tocsc())
    return spla.LinearOperator(M_mat.shape, matvec=lu.solve, dtype=complex)


# ---------------------------------------------------------------------------
# Classical stationary iterations
# ---------------------------------------------------------------------------
def _stationary_iteration(A, b, sweep, rtol, maxiter, x0):
    A = A.tocsr()
    u = np.zeros(A.shape[0], dtype=A.dtype) if x0 is None else x0.astype(A.dtype)
    b = np.asarray(b, dtype=A.dtype)
    b_norm = max(np.linalg.norm(b), 1e-300)
    history = []
    for _ in range(maxiter):
        u = sweep(A, b, u)
        res = np.linalg.norm(b - A @ u) / b_norm
        history.append(res)
        if not np.isfinite(res) or res < rtol:
            break
    info = {'converged': bool(history and history[-1] < rtol),
            'iterations': len(history),
            'residuals': np.array(history)}
    return u, info


def jacobi(A, b, rtol=1e-8, maxiter=500, omega=1.0, x0=None):
    """(Damped) Jacobi iteration.  Note: diverges on indefinite Helmholtz
    systems for medium/high wavenumbers -- this is precisely the failure
    mode that HINTS addresses; provided here as the classical baseline."""
    d = A.diagonal()

    def sweep(A, b, u):
        return u + omega * (b - A @ u) / d

    return _stationary_iteration(A, b, sweep, rtol, maxiter, x0)


def gauss_seidel(A, b, rtol=1e-8, maxiter=500, x0=None):
    """Forward Gauss-Seidel via sparse triangular solve."""
    L = sp.tril(A, format='csr')   # D + L
    U = sp.triu(A, k=1, format='csr')

    def sweep(A, b, u):
        return spla.spsolve_triangular(L, b - U @ u, lower=True)

    return _stationary_iteration(A, b, sweep, rtol, maxiter, x0)


def sor(A, b, omega=1.5, rtol=1e-8, maxiter=500, x0=None):
    """Successive over-relaxation (omega = 1 reduces to Gauss-Seidel)."""
    D = sp.diags(A.diagonal())
    L = sp.tril(A, k=-1, format='csr')
    U = sp.triu(A, k=1, format='csr')
    M = (D / omega + L).tocsr()
    N = ((1.0 / omega - 1.0) * D - U).tocsr()

    def sweep(A, b, u):
        return spla.spsolve_triangular(M, b + N @ u, lower=True)

    return _stationary_iteration(A, b, sweep, rtol, maxiter, x0)
