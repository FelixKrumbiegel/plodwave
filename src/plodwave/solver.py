import numpy as np
from scipy.sparse import csr_array
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import cg, SuperLU, LinearOperator
from sksparse.cholmod import cholesky, Factor
import pyamg
import warnings
# type hinting
from numpy.typing import NDArray


class Solver:

    def __init__(self):
        pass

    @staticmethod
    def setupSolver(A: csr_array,
                    solver_type: str
                    ) -> SuperLU | tuple | LinearOperator:
        """ Sets up the solver. solver_type indicates which solver will
        be setup. If the problem is too large or too dense solver_type
        will be overturned.

        Parameters
        ----------
        A : csr_array
            Left-hand side of the system.
        solver_type : str
            Indicates the type of solve that is used, choices are
            'sparse_direct', 'sparse_cg'.

        Returns
        -------
        SuperLU | tuple | LinearOperator
            Returns the inverse of A, that is either a sparse/dense
            cholesky or LU decomposition or a AMG preconditioner.
        """
        if solver_type == 'sparse_direct':
            if np.shape(A)[0] > 1_000_000:
                warnings.warn(
                    f'System matrix is too large with size {np.shape(A)[0]}. '
                    + f'Using iterative solver instead.')
            elif A.nnz / np.prod(np.shape(A)) > .15:
                warnings.warn(
                    f'Matrix has low sparsity. Using dense direct solver.')
                return cho_factor(A.toarray())
            else:
                return cholesky(A.tocsc())

        # Setting up the Preconditioner
        A.indices = A.indices.astype(np.int32)
        A.indptr = A.indptr.astype(np.int32)
        PreC = pyamg.ruge_stuben_solver(A)
        PreC = PreC.aspreconditioner()
        return PreC

    @staticmethod
    def useSolver(A_inv: Factor | tuple | LinearOperator,
                  b: NDArray,
                  A: csr_array = None,
                  tol: float = None,
                  x0: NDArray = None) -> NDArray:
        """ Solves the system of equations.

        Parameters
        ----------
        A_inv : Factor | tuple | LinearOperator
            Object returned from Solver.setupSolver.
        b : NDArray
            Right-hand side(s) of the system.
        A : csr_array, optional
            System matrix, needs to be provided if sparse cg is used, by
            default None.
        tol : float, optional
            Tolerance, needs to be provided if sparse cg is used, by
            default None.
        x0 : NDArray, optional
            Initial guess for sparse cg, by default None.

        Returns
        -------
        NDArray
            Solution.
        """
        if isinstance(A_inv, Factor):
            return A_inv(b)
        if isinstance(A_inv, tuple):
            return cho_solve(A_inv, b)

        num_iter = 0
        def count_num_iter(x): nonlocal num_iter; num_iter += 1

        x, exit_code = cg(A, b, x0=x0, rtol=0., atol=tol, M=A_inv,
                          callback=count_num_iter)
        print(f"Finished reference solve with exit code {exit_code} "
              + f"and {num_iter} iterations.")
        return x
