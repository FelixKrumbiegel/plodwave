from __future__ import annotations
import numpy as np
from scipy.sparse import csc_array, csr_array, diags_array
from scipy.linalg import solve
from .indices import get_vertices
from .legendre import _legendre2nodal
import plodwave
# type hinting
from numpy.typing import NDArray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mesh import Mesh


def get_bubble(Space: Mesh) -> csr_array:
    if Space.d == 1:
        def theta(x): return x[:, [0]]*(1-x[:, [0]])
    elif Space.d == 2:
        def theta(x): return (x[:, [0]]*(1-x[:, [0]])
                              * x[:, [1]]*(1-x[:, [1]]))**2

    # Set up rhs
    p0dofs = np.arange(np.prod(Space.nelems_coarse)*Space.loc_dofs,
                       step=Space.loc_dofs)
    rhs = csc_array((np.ones_like(p0dofs),
                     (p0dofs, np.arange(np.size(p0dofs)))),
                    shape=(np.prod(Space.nelems_coarse)*Space.loc_dofs,
                           np.prod(Space.nelems_coarse))).toarray()

    domain = np.array([[0, 0], [1, 1]], dtype=plodwave.config.PRECISION)
    D = np.tile(_legendre2nodal(Space)*theta(get_vertices(Space.nelems_refine,
                                                          domain)),
                (np.prod(Space.nelems_coarse), 1))
    J = np.arange(np.prod((Space.nelems_refine+1)*Space.nelems_coarse),
                  dtype=np.int32)
    J = np.repeat(J, Space.loc_dofs, axis=0)
    K = np.arange(np.prod(Space.nelems_coarse)*Space.loc_dofs,
                  dtype=np.int32).reshape(np.prod(Space.nelems_coarse),
                                          Space.loc_dofs)
    K = np.repeat(K, np.prod(Space.nelems_refine+1), axis=0)
    bb = Space.cg2dg_coarse_fine.T.dot(csc_array((D.flatten(),
                                                  (J, K.flatten()))))
    LHS = Space.plod_interpolation.dot(bb).toarray()
    c = csc_array(solve(LHS, rhs))
    return bb.dot(c)


def get_IH(Space: Mesh) -> NDArray:
    IH = Space.cg2dg0_coarse.T
    N = np.sum(IH.toarray(), axis=1, dtype=plodwave.config.PRECISION)
    IH = csr_array(diags_array(1/N, format='csr').dot(IH))  # averaging
    p0dofs = np.arange(np.prod(Space.nelems_coarse)*Space.loc_dofs,
                       step=Space.loc_dofs)
    IH = IH.dot(Space.plod_interpolation[p0dofs, :])
    dofs = np.setdiff1d(np.arange(np.prod(Space.nelems_coarse+1)),
                        Space.boundary_nodes_coarse[0])
    E = csr_array((np.ones_like(dofs), (dofs, dofs)),
                  shape=(np.prod(Space.nelems_coarse+1),
                         np.prod(Space.nelems_coarse+1)))
    return E.dot(IH)
