from __future__ import annotations
import numpy as np
from scipy.sparse import csr_array, kron, eye_array
from scipy.special import sh_legendre
from .indices import get_vertices, get_elements
from .fem import assemble_matrices, get_cg2dg
import plodwave
# type hinting
from numpy.typing import NDArray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mesh import Mesh


def get_mass_legendre(Space: Mesh) -> csr_array:
    """ Returns the inverse of the local Legendre mass matrix.

    Parameters
    ----------
    Space : Mesh
        Underlying Mesh.

    Returns
    -------
    csr_array
        Local Legendre inverse Mass matrix.
    """
    values = _legendre2nodal(Space)
    local_mass = assemble_matrices(Space.nelems_refine,
                                   Space.loc_mass_refine,
                                   mesh_type='cg',
                                   cg2dg=get_cg2dg(
                                       get_elements(Space.nelems_refine)))
    D = np.linalg.inv(values.T @ (local_mass @ values))
    mass_inv = csr_array(kron(eye_array(np.prod(Space.nelems_coarse)), D))

    D = values.T @ local_mass
    D = np.tile(D, (np.prod(Space.nelems_coarse), 1))
    J = np.arange(Space.loc_dofs*np.prod(Space.nelems_coarse), dtype=np.int32)
    J = np.repeat(J, np.prod(Space.nelems_refine+1))
    K = np.arange(np.prod(Space.nelems_refine+1)*np.prod(Space.nelems_coarse),
                  dtype=np.int32).reshape(np.prod(Space.nelems_coarse),
                                          np.prod(Space.nelems_refine+1))
    K = np.repeat(K, Space.loc_dofs, axis=0)
    bb = csr_array((D.flatten(), (J, K.flatten())))
    return mass_inv @ (bb @ Space.cg2dg_coarse_fine)


def _legendre2nodal(Space: Mesh) -> NDArray:
    """ Returns an array of coarse Legendre polynomials interpolated on
    the fine mesh.

    Parameters
    ----------
    Space : Mesh
        Underlying mesh.

    Returns
    -------
    NDArray
        Interpolated values of coarse Legendre polynomials.
    """
    domain = np.array([[0, 0], [1, 1]], dtype=plodwave.config.PRECISION)
    vertices = get_vertices(Space.nelems_refine, domain)
    v_x = [None] * (Space.p+1)
    for j in range(Space.p+1):
        v_x[j] = np.polyval(_legendre_coefficients(j), vertices[:, [0]])
    v_x = np.concatenate(v_x, axis=1)
    if Space.d == 1:
        return v_x
    v_y = [None] * (Space.p+1)
    for j in range(Space.p+1):
        v_y[j] = np.polyval(_legendre_coefficients(j), vertices[:, [1]])
    v_y = np.concatenate(v_y, axis=1)

    v_x = np.tile(v_x, (1, Space.p+1))
    v_y = np.repeat(v_y, Space.p+1, axis=1)
    values = v_x * v_y
    return values


def _legendre_coefficients(j: int) -> NDArray:
    """ Returns the coefficients for the Legendre j-th polynomial.

    Parameters
    ----------
    j : int
        Order of the polynomial.

    Returns
    -------
    NDArray
        Coefficients for the Legendre j-th polynomial.
    """
    return _legendre_norm(j)*sh_legendre(j).c


def _legendre_norm(j: int) -> float:
    """ Returns the L2-norm of the j-th shifted Legendre polynomial.

    Parameters
    ----------
    j : int
        Order of the polynomial.

    Returns
    -------
    float
        L2-norm.
    """
    return np.sqrt(2*j+1, dtype=plodwave.config.PRECISION)
