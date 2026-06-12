"""Compare classical Helmholtz solvers on a heterogeneous medium.

Demonstrates the well-known fact that simple stationary iterations (Jacobi)
diverge on the indefinite Helmholtz system, BiCGSTAB without preconditioning
stalls, and the complex shifted-Laplacian preconditioner (CSLP, Erlangga et
al. 2004/2006) restores fast Krylov convergence.  This is the traditional
baseline against which the HINTS hybrid preconditioner in this repository
is compared.

Run:  python examples/helmholtz_cslp_vs_jacobi.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from wave_solvers import helmholtz as hz


def main():
    n = 600
    x = np.linspace(0.0, 1.0, n)
    # heterogeneous wavenumber: ~8 wavelengths across the domain, with a
    # slow (high-k) inclusion in the middle
    k = 50.0 * (1.0 + 0.5 * np.exp(-((x - 0.5) ** 2) / 0.005))
    f = np.exp(-((x - 0.2) ** 2) / 0.0005)        # localized source

    A = hz.helmholtz_matrix_1d(n, k=k, bc='sommerfeld')

    u_ref = hz.solve_direct(A, f)
    print(f'System size {n}, k in [{k.min():.0f}, {k.max():.0f}]\n')

    # 1) Jacobi -- expected to diverge
    _, jac = hz.jacobi(A, f, maxiter=200)
    print(f'Jacobi          : converged={jac["converged"]}, '
          f'final rel.res={jac["residuals"][-1]:.2e} after '
          f'{jac["iterations"]} its')

    # 2) BiCGSTAB, no preconditioner
    _, bg = hz.solve_bicgstab(A, f, rtol=1e-8, maxiter=2000)
    print(f'BiCGSTAB (none) : converged={bg["converged"]}, '
          f'{bg["iterations"]} its')

    # 3) GMRES + CSLP
    M = hz.make_shifted_laplacian_preconditioner(
        hz.helmholtz_matrix_1d, beta=(1.0, 0.5), n=n, k=k, bc='sommerfeld')
    u_cslp, gm = hz.solve_gmres(A, f, M=M, rtol=1e-8, maxiter=500, restart=50)
    err = np.linalg.norm(u_cslp - u_ref) / np.linalg.norm(u_ref)
    print(f'GMRES + CSLP    : converged={gm["converged"]}, '
          f'{gm["iterations"]} its, rel.err vs direct {err:.2e}')

    print('\nThe CSLP-preconditioned Krylov solver is the traditional '
          'baseline\nthat HINTS aims to accelerate/replace with a neural '
          'operator.')


if __name__ == '__main__':
    main()
