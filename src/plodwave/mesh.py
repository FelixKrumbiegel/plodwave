import numpy as np
from scipy.sparse import csc_array, csr_array
from .exceptions import DimensionError, ModeError, BoundaryError, DomainError
from .indices import (get_elements, get_vertices, get_boundary_indices,
                      get_dofs, get_patch, get_patch_shape)
from .fem import (dg_prolongation, cg_prolongation, element_prolongation,
                  get_cg2dg, get_cg2dg0, get_cg2dg_coarse_fine, averaging,
                  cg_dof_projection, loc_stiffness, loc_mass, loc_mass_inv,
                  assemble_matrices)
from .interpolation import get_interpolation
import plodwave
# type hinting
from numpy.typing import NDArray
from collections.abc import Sequence


class Mesh():
    """ Class that implements the underlying mesh.
    """

    def __init__(self,
                 nelems_coarse: NDArray,
                 nelems_refine: NDArray,
                 p: int = 0,
                 mode: str = 'matrix-free',
                 boundary: NDArray | None = None,
                 domain: NDArray | None = None) -> None:
        self._d: int = np.size(nelems_coarse)
        if not (self.d == 1 or self.d == 2):
            raise DimensionError(self.d)

        if mode not in ['cached', 'matrix-free', 'hybrid']:
            raise ModeError()
        self._mode = mode

        # default is 0 Dirichlet
        if boundary is None:
            boundary = np.zeros((2, self.d), dtype=np.int32)
        if not np.shape(boundary) == (2, self.d):
            raise BoundaryError()
        # default is (0,1)**d
        if domain is None:
            domain = np.array([[0], [1]], dtype=plodwave.config.PRECISION)
            domain = np.tile(domain, (1, self.d))
        if not np.shape(domain) == (2, self.d):
            raise DomainError()

        self._nelems_coarse: NDArray = nelems_coarse
        self._nelems_refine: NDArray = nelems_refine
        self._nelems_fine: NDArray = nelems_coarse * nelems_refine

        self._boundary: NDArray = boundary
        self._domain: NDArray = domain

        self._p: int = p
        self._loc_dofs: int = (p+1)**self.d

    # ==================================================================
    # Cached properties
    # These properties are always cached and are required for every
    # property.
    # ==================================================================

    @property
    def mode(self) -> str:
        """ Returns the mode.

        Returns
        -------
        str
            Mode.
        """
        return self._mode

    @mode.setter
    def mode(self, new_mode: str) -> None:
        """ Sets the mode. Also deletes some properties when switched to
        'matrix-free' or 'hybrid'.

        Parameters
        ----------
        new_mode : str
            New mode.
        """
        if new_mode not in ['cached', 'matrix-free', 'hybrid']:
            raise ModeError
        self._mode = new_mode
        if new_mode == 'hybrid':
            self._clear_some_cached()
            return
        if new_mode == 'matrix-free':
            self._clear_all_cached()

    def _clear_all_cached(self) -> None:
        """ Clear all cached values.
        """
        self.__delattr__('_cg_elements_coarse')
        self.__delattr__('_cg_elements_fine')
        self.__delattr__('_cg_vertices_coarse')
        self.__delattr__('_cg_vertices_fine')
        self.__delattr__('_cg_mids_coarse')
        self.__delattr__('_cg_mids_fine')
        self.__delattr__('_dg_elements_coarse')
        self.__delattr__('_dg_elements_fine')
        self.__delattr__('_pdg_elements_coarse')
        self.__delattr__('_loc_stiffness_fine')
        self.__delattr__('_loc_stiffness_coarse')
        self.__delattr__('_loc_mass_fine')
        self.__delattr__('_loc_mass_coarse')
        self.__delattr__('_loc_mass_refine')
        self.__delattr__('_loc_mass_inv_coarse')
        self.__delattr__('_boundary_nodes_coarse')
        self.__delattr__('_boundary_nodes_fine')
        self.__delattr__('_cg_dofs_coarse')
        self.__delattr__('_cg_dofs_fine')
        self._clear_some_cached()

    def _clear_some_cached(self) -> None:
        """ Clear assembled cached values.
        """
        self.__delattr__('_cg_stiffness_fine')
        self.__delattr__('_cg_stiffness_coarse')
        self.__delattr__('_dg_stiffness_fine')
        self.__delattr__('_cg_mass_fine')
        self.__delattr__('_cg_mass_coarse')
        self.__delattr__('_dg_mass_fine')
        self.__delattr__('_dg_mass_inv_coarse')
        self.__delattr__('_cg_prolongation')
        self.__delattr__('_dg_prolongation')
        self.__delattr__('_element_prolongation')
        self.__delattr__('_cg_dof_projection_fine')
        self.__delattr__('_cg2dg_coarse')
        self.__delattr__('_cg2dg0_coarse')
        self.__delattr__('_cg2dg_coarse_fine')
        self.__delattr__('_cg2dg_fine')
        self.__delattr__('_averaging')
        self.__delattr__('_lod_interpolation')
        self.__delattr__('_plod_interpolation')

    def __delattr__(self, name: str) -> None:
        """ Deletes the attribute if it exists.

        Parameters
        ----------
        name : str
            Name of the attribute.
        """
        if name in self.__dict__:
            del self.__dict__[name]

    @property
    def d(self) -> int:
        """ Returns the space dimension.

        Returns
        -------
        int
            Space dimension.
        """
        return self._d

    @property
    def nelems_coarse(self) -> NDArray:
        """ Returns the number of coarse elements per dimension.

        Returns
        -------
        NDArray
            Number of coarse elements per dimension.
        """
        return self._nelems_coarse

    @property
    def nelems_refine(self) -> NDArray:
        """ Returns the number of fine elements per dimension per coarse
        element.

        Returns
        -------
        NDArray
            Number of fine elements per dimension per coarse element.
        """
        return self._nelems_refine

    @property
    def nelems_fine(self) -> NDArray:
        """ Returns the number of fine elements per dimension.

        Returns
        -------
        NDArray
            Number of fine elements per dimension.
        """
        return self._nelems_fine

    @property
    def boundary(self) -> NDArray:
        """ Returns the boundary flag array.

        Returns
        -------
        NDArray
            Boundary flag array.
        """
        return self._boundary

    @property
    def domain(self) -> NDArray:
        """ Returns the interval ends of the domain.

        Returns
        -------
        NDArray
            Interval ends of the domain.
        """
        return self._domain

    @property
    def p(self) -> int:
        """ Return the polynomial degree.

        Returns
        -------
        int
            Polynomial degree.
        """
        return self._p

    @property
    def loc_dofs(self) -> int:
        """ Returns the local number of DoFs.

        Returns
        -------
        int
            Local number of DoFs.
        """
        return self._loc_dofs

    @property
    def mesh_size_coarse(self) -> float:
        """ Returns the mesh size.

        Returns
        -------
        float
            Mesh size.
        """
        if not hasattr(self, '_mesh_size_coarse'):
            self._mesh_size_coarse = _get_mesh_size(self.nelems_coarse,
                                                    self.domain)
        return self._mesh_size_coarse

    @property
    def mesh_size_fine(self) -> float:
        """ Returns the mesh size.

        Returns
        -------
        float
            Mesh size.
        """
        if not hasattr(self, '_mesh_size_fine'):
            self._mesh_size_fine = _get_mesh_size(self.nelems_fine,
                                                  self.domain)
        return self._mesh_size_fine

    # ==================================================================
    # Hybrid properties
    # These properties are cached if 'hybrid' mode is enabled, these
    # properties do not take much space and are required often.
    # ==================================================================

    @property
    def cg_elements_coarse(self) -> NDArray:
        """ Returns the global indices of coarse elements.

        Returns
        -------
        NDArray
            Global indices of coarse elements.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_cg_elements_coarse'):
                self._cg_elements_coarse = get_elements(self.nelems_coarse)
            return self._cg_elements_coarse
        return get_elements(self.nelems_coarse)

    @property
    def cg_elements_fine(self) -> NDArray:
        """ Returns the global indices of fine elements.

        Returns
        -------
        NDArray
            Global indices of fine elements.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_cg_elements_fine'):
                self._cg_elements_fine = get_elements(self.nelems_fine)
            return self._cg_elements_fine
        return get_elements(self.nelems_fine)

    @property
    def cg_vertices_coarse(self) -> NDArray:
        """ Returns the coordinates of coarse Nodes.

        Returns
        -------
        NDArray
            Coordinates of coarse Nodes.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_cg_vertices_coarse'):
                self._cg_vertices_coarse = get_vertices(
                    self.nelems_coarse, self.domain)
            return self._cg_vertices_coarse
        return get_vertices(self.nelems_coarse, self.domain)

    @property
    def cg_vertices_fine(self) -> NDArray:
        """ Returns the coordinates of fine Nodes.

        Returns
        -------
        NDArray
            Coordinates of fine Nodes.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_cg_vertices_fine'):
                self._cg_vertices_fine = get_vertices(
                    self.nelems_fine, self.domain)
            return self._cg_vertices_fine
        return get_vertices(self.nelems_fine, self.domain)

    @property
    def cg_mids_coarse(self) -> NDArray:
        """ Returns the coordinates of coarse Nodes.

        Returns
        -------
        NDArray
            Coordinates of coarse Nodes.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_cg_mids_coarse'):
                self._cg_mids_coarse = get_vertices(
                    self.nelems_coarse, self.domain, mids=True)
            return self._cg_mids_coarse
        return get_vertices(self.nelems_coarse, self.domain, mids=True)

    @property
    def cg_mids_fine(self) -> NDArray:
        """ Returns the coordinates of fine Nodes.

        Returns
        -------
        NDArray
            Coordinates of fine Nodes.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_cg_mids_fine'):
                self._cg_mids_fine = get_vertices(
                    self.nelems_fine, self.domain, mids=True)
            return self._cg_mids_fine
        return get_vertices(self.nelems_fine, self.domain, mids=True)

    @property
    def dg_elements_coarse(self) -> NDArray:
        """ Returns the global indices of coarse elements.

        Returns
        -------
        NDArray
            Global indices of coarse elements.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_dg_elements_coarse'):
                self._dg_elements_coarse = get_elements(self.nelems_coarse,
                                                        mesh_type='dg',
                                                        loc_dof=2**self.d)
            return self._dg_elements_coarse
        return get_elements(self.nelems_coarse,
                            mesh_type='dg',
                            loc_dof=2**self.d)

    @property
    def dg_elements_fine(self) -> NDArray:
        """ Returns the global indices of fine elements.

        Returns
        -------
        NDArray
            Global indices of fine elements.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_dg_elements_fine'):
                self._dg_elements_fine = get_elements(self.nelems_fine,
                                                      mesh_type='dg',
                                                      loc_dof=2**self.d)
            return self._dg_elements_fine
        return get_elements(self.nelems_fine,
                            mesh_type='dg',
                            loc_dof=2**self.d)

    @property
    def pdg_elements_coarse(self) -> NDArray:
        """ Returns the global indices of coarse elements.

        Returns
        -------
        NDArray
            Global indices of coarse elements.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_pdg_elements_coarse'):
                self._pdg_elements_coarse = get_elements(self.nelems_coarse,
                                                         mesh_type='dg',
                                                         loc_dof=self.loc_dofs)
            return self._pdg_elements_coarse
        return get_elements(self.nelems_coarse,
                            mesh_type='dg',
                            loc_dof=self.loc_dofs)

    @property
    def loc_stiffness_fine(self) -> NDArray:
        """ Returns the local stiffness matrix on the fine mesh.

        Returns
        -------
        NDArray
            Local mass matrix on the fine mesh.
        """
        if self.mode == 'cached' or 'hybrid':
            if not hasattr(self, '_loc_stiffness_fine'):
                self._loc_stiffness_fine = loc_stiffness(self.nelems_fine,
                                                         self.domain)
            return self._loc_stiffness_fine
        return loc_stiffness(self.nelems_fine, self.domain)

    @property
    def loc_stiffness_coarse(self) -> NDArray:
        """ Returns the local stiffness matrix on the coarse mesh.

        Returns
        -------
        NDArray
            Local mass matrix on the coarse mesh.
        """
        if self.mode == 'cached' or 'hybrid':
            if not hasattr(self, '_loc_stiffness_coarse'):
                self._loc_stiffness_coarse = loc_stiffness(self.nelems_coarse,
                                                           self.domain)
            return self._loc_stiffness_coarse
        return loc_stiffness(self.nelems_coarse, self.domain)

    @property
    def loc_mass_fine(self) -> NDArray:
        """ Returns the local mass matrix on the fine mesh.

        Returns
        -------
        NDArray
            Local mass matrix on the fine mesh.
        """
        if self.mode == 'cached' or 'hybrid':
            if not hasattr(self, '_loc_mass_fine'):
                self._loc_mass_fine = loc_mass(self.nelems_fine,
                                               self.domain)
            return self._loc_mass_fine
        return loc_mass(self.nelems_fine, self.domain)

    @property
    def loc_mass_coarse(self) -> NDArray:
        """ Returns the local mass matrix on the coarse mesh.

        Returns
        -------
        NDArray
            Local mass matrix on the coarse mesh.
        """
        if self.mode == 'cached' or 'hybrid':
            if not hasattr(self, '_loc_mass_coarse'):
                self._loc_mass_coarse = loc_mass(self.nelems_coarse,
                                                 self.domain)
            return self._loc_mass_coarse
        return loc_mass(self.nelems_coarse, self.domain)

    @property
    def loc_mass_refine(self) -> NDArray:
        """ Returns the local mass matrix on the refine mesh.

        Returns
        -------
        NDArray
            Local mass matrix on the refine mesh.
        """
        if self.mode == 'cached' or 'hybrid':
            if not hasattr(self, '_loc_mass_refine'):
                self._loc_mass_refine = loc_mass(self.nelems_refine,
                                                 self.domain)
            return self._loc_mass_refine
        return loc_mass(self.nelems_refine, self.domain)

    @property
    def loc_mass_inv_coarse(self) -> NDArray:
        """ Returns the local inverse mass matrix on the coarse mesh.

        Returns
        -------
        NDArray
            Local inverse mass matrix on the coarse mesh.
        """
        if self.mode == 'cached' or 'hybrid':
            if not hasattr(self, '_loc_mass_inv_coarse'):
                self._loc_mass_inv_coarse = loc_mass_inv(self.nelems_coarse,
                                                         self.domain)
            return self._loc_mass_inv_coarse
        return loc_mass_inv(self.nelems_coarse, self.domain)

    @property
    def boundary_nodes_coarse(self) -> tuple[NDArray, NDArray, NDArray]:
        """ Returns the indices of coarse nodes on the boundary as tuple
        (Dirichlet, Neumann, Robin).

        Returns
        -------
        NDArray
            Indices of coarse nodes on the boundary as tuple (Dirichlet,
            Neumann, Robin).
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_boundary_nodes_coarse'):
                self._boundary_nodes_coarse = get_boundary_indices(
                    self.nelems_coarse, self.boundary, self.domain, split=True)
            return self._boundary_nodes_coarse
        return get_boundary_indices(self.nelems_coarse, self.boundary,
                                    self.domain, split=True)

    @property
    def boundary_nodes_fine(self) -> tuple[NDArray, NDArray, NDArray]:
        """ Returns the indices of fine nodes on the boundary as tuple
        (Dirichlet, Neumann, Robin).

        Returns
        -------
        NDArray
            Indices of fine nodes on the boundary as tuple (Dirichlet,
            Neumann, Robin).
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_boundary_nodes_fine'):
                self._boundary_nodes_fine = get_boundary_indices(
                    self.nelems_fine, self.boundary, self.domain, split=True)
            return self._boundary_nodes_fine
        return get_boundary_indices(self.nelems_fine, self.boundary,
                                    self.domain, split=True)

    @property
    def cg_dofs_coarse(self) -> NDArray:
        """ Returns the indices of the coarse DoFs.

        Returns
        -------
        NDArray
            Indices of the coarse DoFs.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_cg_dofs_coarse'):
                self._cg_dofs_coarse = get_dofs(
                    self.cg_elements_coarse,
                    boundary_indices=self.boundary_nodes_coarse,
                    rm_domain_bdy=True)
            return self._cg_dofs_coarse
        return get_dofs(self.cg_elements_coarse,
                        boundary_indices=self.boundary_nodes_coarse,
                        rm_domain_bdy=True)

    @property
    def cg_dofs_fine(self) -> NDArray:
        """ Returns the indices of the fine DoFs.

        Returns
        -------
        NDArray
            Indices of the fine DoFs.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_cg_dofs_fine'):
                self._cg_dofs_fine = get_dofs(
                    self.cg_elements_fine,
                    boundary_indices=self.boundary_nodes_fine,
                    rm_domain_bdy=True)
            return self._cg_dofs_fine
        return get_dofs(self.cg_elements_fine,
                        boundary_indices=self.boundary_nodes_fine,
                        rm_domain_bdy=True)

    # ==================================================================
    # Assembled properties
    # These properties are mostly matrices that potentially take much
    # storage and are assembled every time (except if 'cached' mode is
    # enabled).
    # ==================================================================

    @property
    def cg_stiffness_fine(self) -> csc_array:
        """ Returns the cg stiffness matrix on the fine mesh.

        Returns
        -------
        csc_array
            cg stiffness matrix on the fine mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_cg_stiffness_fine'):
                self._cg_stiffness_fine = assemble_matrices(
                    self.nelems_fine,
                    self.loc_stiffness_fine,
                    mesh_type='cg',
                    cg2dg=self.cg2dg_fine)
            return self._cg_stiffness_fine
        return assemble_matrices(self.nelems_fine, self.loc_stiffness_fine,
                                 mesh_type='cg', cg2dg=self.cg2dg_fine)

    @property
    def cg_stiffness_coarse(self) -> csc_array:
        """ Returns the cg stiffness matrix on the coarse mesh.

        Returns
        -------
        csc_array
            cg stiffness matrix on the coarse mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_cg_stiffness_coarse'):
                self._cg_stiffness_coarse = assemble_matrices(
                    self.nelems_coarse,
                    self.loc_stiffness_coarse,
                    mesh_type='cg',
                    cg2dg=self.cg2dg_coarse)
            return self._cg_stiffness_coarse
        return assemble_matrices(self.nelems_coarse, self.loc_stiffness_coarse,
                                 mesh_type='cg', cg2dg=self.cg2dg_coarse)

    @property
    def dg_stiffness_fine(self) -> csc_array:
        """ Returns the dg stiffness matrix on the fine mesh.

        Returns
        -------
        csc_array
            dg stiffness matrix on the fine mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_dg_stiffness_fine'):
                self._dg_stiffness_fine = assemble_matrices(
                    self.nelems_fine,
                    self.loc_stiffness_fine)
            return self._dg_stiffness_fine
        return assemble_matrices(self.nelems_fine,
                                 self.loc_stiffness_fine)

    @property
    def cg_mass_fine(self) -> csc_array:
        """ Returns the cg mass matrix on the fine mesh.

        Returns
        -------
        csc_array
            cg mass matrix on the fine mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_cg_mass_fine'):
                self._cg_mass_fine = assemble_matrices(self.nelems_fine,
                                                       self.loc_mass_fine,
                                                       mesh_type='cg',
                                                       cg2dg=self.cg2dg_fine)
            return self._cg_mass_fine
        return assemble_matrices(self.nelems_fine, self.loc_mass_fine,
                                 mesh_type='cg',
                                 cg2dg=self.cg2dg_fine)

    @property
    def cg_mass_coarse(self) -> csc_array:
        """ Returns the cg mass matrix on the coarse mesh.

        Returns
        -------
        csc_array
            cg mass matrix on the coarse mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_cg_mass_coarse'):
                self._cg_mass_coarse = assemble_matrices(
                    self.nelems_coarse,
                    self.loc_mass_coarse,
                    mesh_type='cg',
                    cg2dg=self.cg2dg_coarse)
            return self._cg_mass_coarse
        return assemble_matrices(self.nelems_coarse, self.loc_mass_coarse,
                                 mesh_type='cg', cg2dg=self.cg2dg_coarse)

    @property
    def dg_mass_fine(self) -> csc_array:
        """ Returns the dg mass matrix on the fine mesh.

        Returns
        -------
        csc_array
            dg mass matrix on the fine mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_dg_mass_fine'):
                self._dg_mass_fine = assemble_matrices(self.nelems_fine,
                                                       self.loc_mass_fine)
            return self._dg_mass_fine
        return assemble_matrices(self.nelems_fine, self.loc_mass_fine)

    @property
    def dg_mass_inv_coarse(self) -> csc_array:
        """ Returns the dg inverse mass matrix on the coarse mesh.

        Returns
        -------
        csc_array
            dg inverse mass matrix on the coarse mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_dg_mass_inv_coarse'):
                self._dg_mass_inv_coarse = assemble_matrices(
                    self.nelems_coarse, self.loc_mass_inv_coarse)
            return self._dg_mass_inv_coarse
        return assemble_matrices(self.nelems_coarse, self.loc_mass_inv_coarse)

    @property
    def cg_prolongation(self) -> csc_array:
        """ Returns the cg prolongation matrix from the coarse to the 
        fine mesh.

        Returns
        -------
        csc_array
            cg prolongation matrix.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_cg_prolongation'):
                self._cg_prolongation = cg_prolongation(self.nelems_coarse,
                                                        self.nelems_refine)
            return self._cg_prolongation
        return cg_prolongation(self.nelems_coarse, self.nelems_refine)

    @property
    def dg_prolongation(self) -> csc_array:
        """ Returns the dg prolongation matrix from the coarse to the 
        fine mesh.

        Returns
        -------
        csc_array
            dg prolongation matrix.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_dg_prolongation'):
                self._dg_prolongation = dg_prolongation(self.nelems_coarse,
                                                        self.nelems_refine)
            return self._dg_prolongation
        return dg_prolongation(self.nelems_coarse, self.nelems_refine)

    @property
    def element_prolongation(self) -> csc_array:
        """ Returns the element-wise prolongation matrix from the coarse
        to the fine mesh.

        Returns
        -------
        csc_array
            Element-wise prolongation matrix.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_element_prolongation'):
                self._element_prolongation = element_prolongation(
                    self.nelems_coarse, self.nelems_refine)
            return self._element_prolongation
        return element_prolongation(self.nelems_coarse, self.nelems_refine)

    @property
    def cg_dof_projection_fine(self) -> csc_array:
        """ Returns the cg prolongation matrix from the coarse to the 
        fine mesh.

        Returns
        -------
        csc_array
            cg prolongation matrix.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_cg_dof_projection_fine'):
                self._cg_dof_projection_fine = cg_dof_projection(
                    self.cg_dofs_fine,
                    self.nelems_fine)
            return self._cg_dof_projection_fine
        return cg_dof_projection(self.cg_dofs_fine, self.nelems_fine)

    @property
    def cg2dg_coarse(self) -> csc_array:
        """ Returns the cg to dg map on the coarse mesh.

        Returns
        -------
        csc_array
            cg to dg map on the coarse mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_cg2dg_coarse'):
                self._cg2dg_coarse = get_cg2dg(self.cg_elements_coarse)
            return self._cg2dg_coarse
        return get_cg2dg(self.cg_elements_coarse)

    @property
    def cg2dg0_coarse(self) -> csc_array:
        """ Returns the cg to dg map on the coarse mesh.

        Returns
        -------
        csc_array
            cg to dg map on the coarse mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_cg2dg0_coarse'):
                self._cg2dg0_coarse = get_cg2dg0(self.nelems_coarse,
                                                 self.cg2dg_coarse)
            return self._cg2dg0_coarse
        return get_cg2dg0(self.nelems_coarse,
                          self.cg2dg_coarse)

    @property
    def cg2dg_coarse_fine(self) -> csc_array:
        """ Returns the cg to dg map on the coarse mesh.

        Returns
        -------
        csc_array
            cg to dg map on the coarse mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_cg2dg_coarse_fine'):
                self._cg2dg_coarse_fine = get_cg2dg_coarse_fine(self.nelems_coarse,
                                                                self.nelems_refine)
            return self._cg2dg_coarse_fine
        return get_cg2dg_coarse_fine(self.nelems_coarse, self.nelems_refine)

    @property
    def cg2dg_fine(self) -> csc_array:
        """ Returns the cg to dg map on the fine mesh.

        Returns
        -------
        csc_array
            cg to dg map on the fine mesh.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_cg2dg_fine'):
                self._cg2dg_fine = get_cg2dg(self.cg_elements_fine)
            return self._cg2dg_fine
        return get_cg2dg(self.cg_elements_fine)

    @property
    def averaging(self) -> csc_array:
        """ Return the classical averaging operator.

        Returns
        -------
        csc_array
            Classical averaging operator.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_averaging'):
                self._averaging = averaging(self.cg2dg_coarse)
            return self._averaging
        return averaging(self.cg2dg_coarse)

    @property
    def lod_interpolation(self) -> csc_array:
        """ Returns the local quasi-interpolation for the LOD method.

        Returns
        -------
        csc_array
            Quasi-interpolation.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_lod_interpolation'):
                self._lod_interpolation = get_interpolation(self, itype='lod')
            return self._lod_interpolation
        return get_interpolation(self, itype='lod')

    @property
    def plod_interpolation(self) -> csc_array:
        """ Returns the local interpolation for the pLOD method.

        Returns
        -------
        csc_array
            Interpolation.
        """
        if self.mode == 'cached':
            if not hasattr(self, '_plod_interpolation'):
                self._plod_interpolation = get_interpolation(
                    self, itype='plod')
            return self._plod_interpolation
        return get_interpolation(self, itype='plod')

    def get_submeshes(self,
                      ell: int) -> Sequence['Patch']:
        return [Patch(j,
                      ell,
                      self.nelems_coarse,
                      self.nelems_refine,
                      self.p,
                      self.mode,
                      self.boundary,
                      self.domain) for j in range(np.prod(self.nelems_coarse))]


class Patch(Mesh):
    """ General class for a single patch, inherits from Mesh.

    Parameters
    ----------
    Mesh : Mesh
        Mesh class.
    """

    def __init__(self,
                 idx: int,
                 ell: int,
                 nelems_coarse: NDArray,
                 nelems_refine: NDArray,
                 p: int = 0,
                 mode: str = 'matrix-free',
                 boundary: NDArray | None = None,
                 domain: NDArray | None = None) -> None:
        assert idx < np.prod(nelems_coarse)
        self._idx = idx
        self._ell: int = ell
        super().__init__(nelems_coarse, nelems_refine, p, mode, boundary,
                         domain)

    # ==================================================================
    # Cached properties
    # These properties are always cached and are required for every
    # property.
    # ==================================================================

    def _clear_all_cached(self) -> None:
        """ Clear all cached values.
        """
        self.__delattr__('_cg_elements_coarse')
        self.__delattr__('_cg_elements_fine')
        self.__delattr__('_cg_vertices_coarse')
        self.__delattr__('_cg_vertices_fine')
        self.__delattr__('_dg_elements_coarse')
        self.__delattr__('_dg_elements_fine')
        self.__delattr__('_pdg_elements_coarse')
        self.__delattr__('_loc_stiffness_fine')
        self.__delattr__('_loc_stiffness_coarse')
        self.__delattr__('_loc_mass_fine')
        self.__delattr__('_loc_mass_coarse')
        self.__delattr__('_loc_mass_refine')
        self.__delattr__('_loc_mass_inv_coarse')
        self.__delattr__('_boundary_nodes_coarse')
        self.__delattr__('_boundary_nodes_fine')
        self.__delattr__('_cg_dofs_coarse')
        self.__delattr__('_cg_dofs_fine')

        self.__delattr__('_patch_array_coarse')
        self.__delattr__('_patch_array_fine')
        self.__delattr__('_cg_dofs_patch_coarse')
        self.__delattr__('_cg_dofs_patch_fine')
        self.__delattr__('_pdg_dofs_patch_coarse')
        self.__delattr__('_dg_dofs_element_coarse')
        self.__delattr__('_dg_dofs_element_fine')
        self.__delattr__('_pdg_dofs_element_coarse')
        self._clear_some_cached()

    def _clear_some_cached(self) -> None:
        """ Clear assembled cached values.
        """
        self.__delattr__('_cg_stiffness_fine')
        self.__delattr__('_cg_stiffness_coarse')
        self.__delattr__('_dg_stiffness_fine')
        self.__delattr__('_cg_mass_fine')
        self.__delattr__('_cg_mass_coarse')
        self.__delattr__('_dg_mass_fine')
        self.__delattr__('_dg_mass_inv_coarse')
        self.__delattr__('_cg_prolongation')
        self.__delattr__('_dg_prolongation')
        self.__delattr__('_element_prolongation')
        self.__delattr__('_cg_dof_projection_fine')
        self.__delattr__('_cg2dg_coarse')
        self.__delattr__('_cg2dg0_coarse')
        self.__delattr__('_cg2dg_coarse_fine')
        self.__delattr__('_cg2dg_fine')
        self.__delattr__('_averaging')
        self.__delattr__('_lod_interpolation')
        self.__delattr__('_plod_interpolation')

    @property
    def idx(self) -> int:
        """ Returns the element index.

        Returns
        -------
        int
            Element index.
        """
        return self._idx

    @property
    def ell(self) -> int:
        """ Returns the localization parameter.

        Returns
        -------
        int
            Localization parameter.
        """
        return self._ell

    @property
    def nelems_patch_coarse(self) -> NDArray:
        """ Returns the number of elements per dimension inside the
        patch.

        Returns
        -------
        NDArray
            Number of elements per dimension inside the patch.
        """
        if not hasattr(self, '_nelems_patch_coarse'):
            self._nelems_patch_coarse = get_patch_shape(self.nelems_coarse,
                                                        self.patch_array_coarse)
        return self._nelems_patch_coarse

    @property
    def nelems_patch_fine(self) -> NDArray:
        """ Returns the number of elements per dimension inside the
        patch.

        Returns
        -------
        NDArray
            Number of elements per dimension inside the patch.
        """
        if not hasattr(self, '_nelems_patch_fine'):
            self._nelems_patch_fine = self.nelems_patch_coarse*self.nelems_refine
        return self._nelems_patch_fine

    # ==================================================================
    # Hybrid properties
    # These properties are cached if 'hybrid' mode is enabled, these
    # properties do not take much space and are required often.
    # ==================================================================

    @property
    def patch_array_coarse(self) -> NDArray:
        """ Returns the indices of coarse elements within the patch.

        Returns
        -------
        NDArray
            Indices of coarse elements within the patch.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_patch_array_coarse'):
                self._patch_array_coarse = get_patch(self.nelems_coarse,
                                                     self.idx,
                                                     self.ell)
            return self._patch_array_coarse
        return get_patch(self.nelems_coarse, self.idx, self.ell)

    @property
    def patch_array_fine(self) -> NDArray:
        """ Returns the indices of fine elements within the patch.

        Returns
        -------
        NDArray
            Indices of fine elements within the patch.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_patch_array_fine'):
                self._patch_array_fine = self.element_prolongation\
                    @ self.patch_array_coarse
            return self._patch_array_fine
        return self.element_prolongation @ self.patch_array_coarse

    @property
    def cg_dofs_patch_coarse(self) -> NDArray:
        """ Returns the indices of coarse nodes within the patch
        removing patch and Dirichlet boundary.

        Returns
        -------
        NDArray
            Indices of coarse nodes within the patch.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_cg_dofs_patch_coarse'):
                self._cg_dofs_patch_coarse = get_dofs(
                    self.cg_elements_coarse,
                    self.patch_array_coarse.flatten() > 0,
                    self.boundary_nodes_coarse,
                    rm_patch_bdy=False,
                    rm_domain_bdy=True)
            return self._cg_dofs_patch_coarse
        return get_dofs(self.cg_elements_coarse,
                        self.patch_array_coarse.flatten() > 0,
                        self.boundary_nodes_coarse,
                        rm_patch_bdy=False,
                        rm_domain_bdy=True)

    @property
    def cg_dofs_patch_fine(self) -> NDArray:
        """ Returns the indices of fine nodes within the patch
        removing patch and Dirichlet boundary.

        Returns
        -------
        NDArray
            Indices of fine nodes within the patch.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_cg_dofs_patch_fine'):
                self._cg_dofs_patch_fine = get_dofs(
                    self.cg_elements_fine,
                    self.patch_array_fine.flatten() > 0,
                    self.boundary_nodes_fine,
                    rm_patch_bdy=True,
                    rm_domain_bdy=False)
            return self._cg_dofs_patch_fine
        return get_dofs(self.cg_elements_fine,
                        self.patch_array_fine.flatten() > 0,
                        self.boundary_nodes_fine,
                        rm_patch_bdy=True,
                        rm_domain_bdy=False)

    @property
    def pdg_dofs_patch_coarse(self) -> NDArray:
        """ Returns the indices of coarse nodes within the patch
        removing patch and Dirichlet boundary.

        Returns
        -------
        NDArray
            Indices of coarse nodes within the patch.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_pdg_dofs_patch_coarse'):
                self._pdg_dofs_patch_coarse = get_dofs(
                    self.pdg_elements_coarse,
                    self.patch_array_coarse.flatten() > 0)
            return self._pdg_dofs_patch_coarse
        return get_dofs(self.pdg_elements_coarse,
                        self.patch_array_coarse.flatten() > 0)

    @property
    def dg_dofs_element_coarse(self) -> NDArray:
        """ Returns the indices of coarse nodes within the element.

        Returns
        -------
        NDArray
            Indices of coarse nodes within the element.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_dg_dofs_element_coarse'):
                self._dg_dofs_element_coarse = get_dofs(
                    self.dg_elements_coarse,
                    self.patch_array_coarse.flatten() > 1)
            return self._dg_dofs_element_coarse
        return get_dofs(self.dg_elements_coarse,
                        self.patch_array_coarse.flatten() > 1)

    @property
    def dg_dofs_element_fine(self) -> NDArray:
        """ Returns the indices of fine nodes within the element.

        Returns
        -------
        NDArray
            Indices of fine nodes within the element.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_dg_dofs_element_fine'):
                self._dg_dofs_element_fine = get_dofs(
                    self.dg_elements_fine,
                    self.patch_array_fine.flatten() > 1)
            return self._dg_dofs_element_fine
        return get_dofs(self.dg_elements_fine,
                        self.patch_array_fine.flatten() > 1)

    @property
    def pdg_dofs_element_coarse(self) -> NDArray:
        """ Returns the indices of coarse nodes within the patch
        removing patch and Dirichlet boundary.

        Returns
        -------
        NDArray
            Indices of coarse nodes within the patch.
        """
        if self.mode in ['cached', 'hybrid']:
            if not hasattr(self, '_pdg_dofs_element_coarse'):
                self._pdg_dofs_element_coarse = get_dofs(
                    self.pdg_elements_coarse,
                    self.patch_array_coarse.flatten() > 1)
            return self._pdg_dofs_element_coarse
        return get_dofs(self.pdg_elements_coarse,
                        self.patch_array_coarse.flatten() > 1)


class TimeDomain():

    def __init__(self,
                 nelems_time: NDArray,
                 final_time: float,
                 ) -> None:
        self._nelems_time: NDArray = nelems_time
        self._final_time: float = final_time

        self._number_time_steps: int = np.ceil(self.nelems_time
                                               * self.final_time
                                               ).astype(np.int32)[0]

    @property
    def nelems_time(self) -> NDArray:
        return self._nelems_time

    @property
    def final_time(self) -> float:
        return self._final_time

    @property
    def number_time_steps(self) -> int:
        return self._number_time_steps


def _get_mesh_size(NElems: NDArray,
                   Domain: NDArray) -> float:
    """ Computation of the mesh size.

    Parameters
    ----------
    NElems : NDArray
        Number of elements per dimension.
    Domain : NDArray
        Interval ends per dimension.

    Returns
    -------
    float
        Mesh size.
    """
    return np.squeeze(np.sqrt(np.sum((Domain/NElems)**2)))


def _restrictCoefficient(Space: Mesh) -> NDArray:
    """ Returns the coefficient averaged over elements, restricted to
    the coarse mesh.

    Parameters
    ----------
    Space : Mesh
        Underlying Mesh.

    Returns
    -------
    NDArray
        Restricted coefficient.
    """
    A_new = csr_array(Space.element_prolongation.T).dot(Space.A)
    return A_new / np.prod(Space.nelems_refine)
