import numpy as np
from scipy.sparse import csc_array
import plodwave
# type hinting
from numpy.typing import NDArray


# ======================================================================
# Creation of Element arrays
# cg version, dg version, patch restrictions
# ======================================================================

def get_elements(nelems: NDArray,
                 mesh_type: str = 'cg',
                 loc_dof: int = 1) -> NDArray:
    """ Returns the indices per element, can be used as local-to-global
    map.

    Parameters
    ----------
    nelems : NDArray
        Number of elements per dimension.
    mesh_type : str, optional
        Choice between 'cg' and 'dg' mesh, by default 'cg'.
    loc_dof : int, optional
        Number of degrees of freedom per element for 'dg', by default 1.

    Returns
    -------
    NDArray
        local-to-global map.
    """
    if mesh_type == 'dg':
        return _dg_elements(nelems, loc_dof=loc_dof)
    return _cg_elements(nelems)


def _cg_elements(nelems: NDArray) -> NDArray:
    """ Returns the local-to-global map for a cg mesh.

    Parameters
    ----------
    NElems : NDArray
        Number of elements per dimension.

    Returns
    -------
    NDArray
        cg local-to-global map.
    """
    d = np.size(nelems)
    L = np.arange(nelems[0], dtype=np.int32)  # left corners
    if d == 1:
        return np.add.outer(L, np.array([0, 1])).astype(np.int32)
    # d == 2
    L2 = (nelems[0]+1) * np.arange(nelems[1], dtype=np.int32)
    L = np.add.outer(L2, L).flatten()  # bottom left corners
    return np.add.outer(L, np.array([0, 1, nelems[0]+2, nelems[0]+1])
                        ).astype(np.int32)


def _dg_elements(nelems: NDArray,
                 loc_dof: int = 1) -> NDArray:
    """ Returns the local-to-global map for a dg mesh. 

    Parameters
    ----------
    nelems : NDArray
        Number of elements per dimension.
    loc_dof : int, optional
        Number of degrees of freedom per element, by default 1

    Returns
    -------
    NDArray
        dg local-to-global map.
    """
    return np.arange(np.prod(nelems)*loc_dof,
                     dtype=np.int32).reshape(-1, loc_dof)


def get_dofs(elements: NDArray,
             patch_array: NDArray | None = None,
             boundary_indices: NDArray | None = None,
             rm_patch_bdy: bool = False,
             rm_domain_bdy: bool = False,
             local_idx: bool = False) -> NDArray:
    """ Return the degrees of freedom for global or local elements, on
    a patch, or with patch or domain boundary removed.

    Parameters
    ----------
    elements : NDArray
        Element array.
    patch_array : NDArray | None, optional
        Patch element flags as bool array, by default None.
    boundary_indices : NDArray | None, optional
        Remove Dirichlet boundary nodes, by default None.
    rm_patch_bdy : bool, optional
        Remove patch boundary nodes (also removes the Dirichlet boundary
        nodes), by default False.
    rm_domain_bdy : bool, optional
        Remove Dirichlet boundary nodes, by default False.
    local_idx : bool, optional
        Return local indices instead of global, by default False.

    Returns
    -------
    NDArray
        Degrees of freedom.
    """
    if patch_array is not None:
        if rm_patch_bdy:
            assert boundary_indices is not None, "boundary_indices must "\
                + "be provided if rm_patch_bdy is True."
            _, global_nneighbours = _get_indices(elements, get_neighbours=True)
            indices, local_nneighbours = _get_indices(elements,
                                                      patch_array=patch_array,
                                                      get_neighbours=True)
            # setting Dirichlet boundary nodes to 0 neighbours, which
            # are then removed since no node can have 0 neighbours
            global_nneighbours[boundary_indices[0]] = 0
            inner_patch_indices = indices[local_nneighbours
                                          == global_nneighbours[indices]]
            if local_idx:
                return np.where(np.isin(indices, inner_patch_indices))[0]
            # return global indices
            return inner_patch_indices
        indices = _get_indices(elements, patch_array=patch_array)
        if rm_domain_bdy:
            assert boundary_indices is not None, "boundary_indices must "\
                + "be provided if rm_domain_bdy is True."
            inner_patch_indices = np.setdiff1d(indices, boundary_indices[0])
            if local_idx:
                return np.where(np.isin(indices, inner_patch_indices))[0]
            return inner_patch_indices
        if local_idx:
            return np.arange(np.size(indices), dtype=np.int32)
        # return only the global patch indices
        return indices
    # return indices without restriction to a patch
    indices = _get_indices(elements)
    if rm_domain_bdy:
        return np.setdiff1d(indices, boundary_indices[0])
    return indices


def _get_indices(elements: NDArray,
                 patch_array: NDArray | None = None,
                 get_neighbours: bool = False
                 ) -> NDArray | tuple[NDArray, NDArray]:
    """ Returns an array containing the indices of the mesh.

    Parameters
    ----------
    elements : NDArray
        Element array.
    patch_array : NDArray | None
        Indices of elements inside a patch, by default None.
    get_neighbours : bool
        Return the number of neighbours, by default False.

    Returns
    -------
    NDArray | tuple[NDArray, NDArray]
        Indices of the mesh (and the number of neighbours).
    """
    if patch_array is None:
        return np.unique(elements, return_counts=get_neighbours)
    return np.unique(elements[patch_array, :],
                     return_counts=get_neighbours)


# ======================================================================
# Creation of Coordinate arrays
# grid points, mid points
# ======================================================================


def get_vertices(nelems: NDArray,
                 domain: NDArray,
                 mids: bool = False) -> NDArray:
    """ Returns an array with coordinates to the mesh.

    Parameters
    ----------
    nelems : NDArray
        Number of elements per dimension.
    domain : NDArray
        Array with the domain boundary values.
    mids : bool, optional
        Return mid points or grid points, by default False.

    Returns
    -------
    NDArray
        Array containing the coordinates for the underlying mesh.
    """
    d = np.size(nelems)
    x = np.linspace(0, 1, nelems[0]+1, dtype=plodwave.config.PRECISION)
    if d == 1:
        if mids:
            return _shift_interval(x[:-1, None]+1/(2*nelems[0]), domain)
        elif not mids:
            return _shift_interval(x[:, None], domain)
    elif d == 2:
        y = np.linspace(0, 1, nelems[1]+1, dtype=plodwave.config.PRECISION)
        if mids:
            x = np.tile(x[:-1]+1/(2*nelems[0]), (nelems[1],))
            y = np.tile(y[:-1]+1/(2*nelems[1]), (nelems[0], 1)).flatten('F')
            return _shift_interval(np.column_stack((x, y)), domain)
        elif not mids:
            x = np.tile(x, (nelems[1]+1,))
            y = np.tile(y, (nelems[0]+1, 1)).flatten('F')
            return _shift_interval(np.column_stack((x, y)), domain)


def _shift_interval(x: NDArray,
                    domain: NDArray) -> NDArray:
    """ Shifts [0, 1] to the given domain.

    Parameters
    ----------
    x : NDArray
        [0, 1] coordinates.
    domain : NDArray
        Array with the domain boundary values.

    Returns
    -------
    NDArray
        Shifted interval.
    """
    return domain[[0], :] + x * (domain[[1], :] - domain[[0], :])


# ======================================================================
# Return the indices for boundary nodes
# ======================================================================

def get_boundary_indices(nelems: NDArray,
                         boundary: NDArray | None = None,
                         domain: NDArray | None = None,
                         split: bool = False,
                         ) -> NDArray | tuple[NDArray, NDArray, NDArray]:
    """ Returns the indices for boundary nodes as array, if split=False,
    or as tuple if split=True (Dirichlet, Neumann, Robin)

    Parameters
    ----------
    nelems : NDArray
        Number of elements per dimension.
    boundary : NDArray | None, optional
        Array with the boundary flags, by default Dirichlet everywhere.
    domain : NDArray | None, optional
        Array with the domain boundary values, by default (0,1)**d.
    split : bool, optional
        Flag if the boundary indices should be split into Dirichlet,
        Neumann and Robin parts (split=True), or the whole boundary
        should be returned(split=False), by default False.

    Returns
    -------
    NDArray | tuple[NDArray, NDArray, NDArray]
        Boundary indices array or tuple (Dirichlet, Neumann, Robin).
    """
    if not split:
        return _full_cg_boundary_indices(nelems)
    d = np.size(nelems)
    # default is 0 Dirichlet
    if boundary is None:
        boundary = np.zeros((2, d), dtype=np.int32)
    # default is (0,1)**d
    if domain is None:
        domain = np.array([[0], [1]], dtype=plodwave.config.PRECISION)
        domain = np.tile(domain, (1, d))
    return _split_cg_boundary_indices(nelems, boundary, domain)


def _full_cg_boundary_indices(nelems: NDArray) -> NDArray:
    """ Returns all boundary indices.

    Parameters
    ----------
    nelems : NDArray
        Number of elements per dimension.

    Returns
    -------
    NDArray
        Boundary indices.
    """
    d = np.size(nelems)
    indices, counts = _get_indices(get_elements(nelems), get_neighbours=True)
    bc = indices[np.where(counts < 2**d)]
    return bc


def _split_cg_boundary_indices(nelems: NDArray,
                               boundary: NDArray,
                               domain: NDArray
                               ) -> tuple[NDArray, NDArray, NDArray]:
    """ Returns the boundary indices split into (Dirichlet, Neumann,
    Robin)

    Parameters
    ----------
    nelems : NDArray
        Number of elements per dimension.
    boundary : NDArray
        Array with the boundary flags.
    domain : NDArray
        Array with the domain boundary values.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Boundary indices tuple (Dirichlet, Neumann, Robin).
    """
    d = np.size(nelems)
    vertices = get_vertices(nelems, domain)
    Dirichlet_bc = np.empty((0,), dtype=np.int32)
    Neumann_bc = np.empty((0,), dtype=np.int32)
    Robin_bc = np.empty((0,), dtype=np.int32)
    for j in range(d):
        # checking if the coordinate coincides with the left endpoint of
        # the interval in dimension j
        bdy_interval_left = np.where(vertices[:, [j]] == domain[[0], [j]])[0]
        # adding the indices to the respective boundary arrays
        if boundary[[0], [j]] == 0:
            Dirichlet_bc = np.append(Dirichlet_bc, bdy_interval_left)
        elif boundary[[0], [j]] == 1:
            Neumann_bc = np.append(Neumann_bc, bdy_interval_left)
        elif boundary[[0], [j]] == 2:
            Robin_bc = np.append(Robin_bc, bdy_interval_left)
        # checking if the coordinate coincides with the left endpoint of
        # the interval in dimension j
        bdy_interval_right = np.where(vertices[:, [j]] == domain[[1], [j]])[0]
        # adding the indices to the respective boundary arrays
        if boundary[[1], [j]] == 0:
            Dirichlet_bc = np.append(Dirichlet_bc, bdy_interval_right)
        elif boundary[[1], [j]] == 1:
            Neumann_bc = np.append(Neumann_bc, bdy_interval_right)
        elif boundary[[1], [j]] == 2:
            Robin_bc = np.append(Robin_bc, bdy_interval_right)
    # removing duplicate indices, i.e., if a corner would belong to two
    # different sets of boundaries, we force them to be Dirichlet if one
    # of the sets is Dirichlet, and Neumann if the two sets are Neumann
    # and Robin
    Neumann_bc = np.setdiff1d(Neumann_bc, Dirichlet_bc)
    Robin_bc = np.setdiff1d(Robin_bc, Dirichlet_bc)
    Robin_bc = np.setdiff1d(Robin_bc, Neumann_bc)
    return (np.unique(Dirichlet_bc).astype(np.int32),
            np.unique(Neumann_bc).astype(np.int32),
            np.unique(Robin_bc).astype(np.int32))


# ======================================================================
# Returns an array indicating the patch indices
# ======================================================================


def get_patch(nelems: NDArray,
              j: int,
              ell: int) -> NDArray:
    """ Returns vector with entry 1 if element is in Patch, 2 for the
    element itself and 0 otherwise.

    Parameters
    ----------
    nelems : NDArray
        Number of elements per dimension.
    j : int
        Element index.
    ell : int
        Patch size.

    Returns
    -------
    NDArray
        Vector that contains information about the patch.
    """
    d = np.size(nelems)
    J = get_elements(nelems).flatten()
    K = np.repeat(np.arange(np.prod(nelems)), (2**d,))
    # nodes_per_element[j,k] = 1 iff node j in element k
    nodes_per_element = csc_array((np.ones_like(K), (J, K)))
    # N1-patch
    N1patch = csc_array(nodes_per_element.T) @ nodes_per_element
    patch = N1patch[:, [j]].copy()
    for _ in range(1, ell):
        patch = N1patch @ patch
    patch.data[:] = 1
    patch[[j], [0]] += 1
    return patch.toarray()


def get_patch_shape(nelems: NDArray,
                    patch_array: NDArray) -> NDArray:
    """ Returns the elements per dimension for the patch.

    Parameters
    ----------
    nelems : NDArray
        Number of elements per dimension.
    patch_array : NDArray
        Patch indices.

    Returns
    -------
    NDArray
        Number of elements per dimension in the patch.
    """
    d = np.size(nelems)
    elems_grid = np.arange(np.prod(nelems)).reshape(nelems, order='F')
    patch_indices = np.where(patch_array > 0)[0]
    nelems_patch = np.where(np.isin(elems_grid, patch_indices))
    if d == 1:
        nelems_patch = np.array([np.size(np.unique(nelems_patch[0]))],
                                dtype=np.int32)
        return nelems_patch
    nelems_patch = np.array([np.size(np.unique(nelems_patch[0])),
                            np.size(np.unique(nelems_patch[1]))],
                            dtype=np.int32)
    return nelems_patch
