from __future__ import annotations
from scipy.sparse import csc_array
from .legendre import get_mass_legendre
# type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mesh import Mesh


def get_interpolation(Space: Mesh, itype: str = 'lod') -> csc_array:
    """ Returns the interpolation operator for the LOD or pLOD method.

    Parameters
    ----------
    Space : Mesh
        Underlying Mesh.

    Returns
    -------
    csc_array
        Interpolation operator V_h -> V_H.
    """
    assert itype in ['lod', 'plod']
    if itype == 'plod':
        return _plod_interpolation(Space)
    return _lod_interpolation(Space)


def _lod_interpolation(Space: Mesh) -> csc_array:
    """ Returns LOD interpolation operator.

    Parameters
    ----------
    Space : Mesh
        Undelying Mesh.

    Returns
    -------
    csc_array
        LOD interpolation operator V_h -> V_H.
    """
    Prol = csc_array(Space.dg_prolongation.T)
    return Space.averaging @ (Space.dg_mass_inv_coarse
                              @ (Prol @ (Space.dg_mass_fine @ Space.cg2dg_fine)))


def _plod_interpolation(Space: Mesh) -> csc_array:
    """ Returns pLOD interpolation operator.

    Parameters
    ----------
    Space : Mesh
        Undelying Mesh.

    Returns
    -------
    csc_array
        pLOD interpolation operator V_h -> V_H.
    """
    return get_mass_legendre(Space)
