from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array

from plodwave import indices
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from plodwave.space import Space


class Patch():
    """ Class representing one Patch.

    Attributes
    ----------
    Mesh : Space
        Underlying Mesh.
    j : int
        Current element.
    locDoF : int
        Number of local degrees of freedom (per element).
    ell : int
        Localization parameter.
    """

    def __init__(self,
                 Mesh: Space,
                 j: int,
                 locDoF: int) -> None:
        """ Initialize the class.

        Parameters
        ----------
        Mesh : Space
            Underlying Mesh.
        j : int
            Current element.
        locDoF : int
            Number of local degrees of freedom (per element).
        """
        self.Mesh: Space = Mesh
        self.ell: int = Mesh.ell
        self.locDoF: int = locDoF
        self.j: int = j

        self.NWorldCoarse: NDArray = Mesh.NWorldCoarse
        self.NWorldFine: NDArray = Mesh.NWorldFine

    @property
    def PatchCoarse(self) -> csc_array:
        """ Coarse Patch.

        Returns
        -------
        csc_array
            Coarse Patch.
        """
        if not hasattr(self, '_PatchCoarse'):
            self._PatchCoarse = indices.getPatch(
                self.NWorldCoarse, self.j, self.ell)
        return self._PatchCoarse

    @property
    def cgDoFCoarse(self) -> NDArray:
        """ Coarse cg patch indices.

        Returns
        -------
        NDArray
            Coarse cg patch indices.
        """
        if not hasattr(self, '_cgDoFCoarse'):
            self._cgDoFCoarse = indices.getDoFs(
                self.Mesh.NWorldCoarse,
                self.Mesh.Boundary,
                self.Mesh.Domain,
                Patch=np.squeeze(self.PatchCoarse.toarray() > 0))
        return self._cgDoFCoarse

    @property
    def dgDoFCoarse(self) -> NDArray:
        """ Coarse dg patch indices.

        Returns
        -------
        NDArray
            Coarse dg patch indices.
        """
        if not hasattr(self, '_dgDoFCoarse'):
            self._dgDoFCoarse = indices.getDoFs(
                self.Mesh.NWorldCoarse,
                self.Mesh.Boundary,
                self.Mesh.Domain,
                meshtype='dg',
                locDoF=self.locDoF,
                Patch=np.squeeze(self.PatchCoarse.toarray() > 0))
        return self._dgDoFCoarse

    @property
    def PatchFine(self) -> csc_array:
        """ Fine Patch.

        Returns
        -------
        csc_array
            Fine Patch.
        """
        if not hasattr(self, '_PatchFine'):
            self._PatchFine = coarse2finePatch(self.Mesh, self.PatchCoarse)
        return self._PatchFine

    @property
    def cgDoFFine(self) -> NDArray:
        """ Fine cg patch indices.

        Returns
        -------
        NDArray
            Fine cg patch indices.
        """
        if not hasattr(self, '_cgDoFFine'):
            self._cgDoFFine = indices.getDoFs(
                self.Mesh.NWorldFine,
                self.Mesh.Boundary,
                self.Mesh.Domain,
                Patch=np.squeeze(self.PatchFine.toarray() > 0))
        return self._cgDoFFine

    @property
    def dgDoFFine(self) -> NDArray:
        """ Fine dg patch indices.

        Returns
        -------
        NDArray
            Fine dg patch indices.
        """
        if not hasattr(self, '_dgDoFFine'):
            self._dgDoFFine = indices.getDoFs(
                self.Mesh.NWorldFine,
                self.Mesh.Boundary,
                self.Mesh.Domain,
                meshtype='dg',
                locDoF=2**self.Mesh.d,
                Patch=np.squeeze(self.PatchFine.toarray() > 0))
        return self._dgDoFFine

    @property
    def jdgDoFFine(self) -> NDArray:
        """ Fine dg element indices.

        Returns
        -------
        NDArray
            Fine dg element indices.
        """
        if not hasattr(self, '_jdgDoFFine'):
            self._jdgDoFFine = indices.getDoFs(
                self.Mesh.NWorldFine,
                self.Mesh.Boundary,
                self.Mesh.Domain,
                meshtype='dg',
                locDoF=2**self.Mesh.d,
                Patch=np.squeeze(self.PatchFine.toarray() > 1))
        return self._jdgDoFFine


def coarse2finePatch(Mesh: Space,
                     Patch: csc_array) -> csc_array:
    """ Returns the fine Patch.

    Parameters
    ----------
    Mesh : Space
        Underlying Mesh.
    Patch : csc_array
        Current fine Patch.

    Returns
    -------
    csc_array
        Fine Patch.
    """
    return Mesh.dgProjection @ Patch
