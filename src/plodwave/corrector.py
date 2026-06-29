from __future__ import annotations
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csc_array, csr_array
from scipy.linalg import cho_factor, cho_solve
from sksparse.cholmod import cholesky
import multiprocessing
from functools import partial
import plodwave
from .exceptions import MethodError
from .mesh import Mesh, Patch
from .fem import assemble_matrices
from .stabilization import get_bubble, get_IH
# type hinting
from numpy.typing import NDArray
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .problem import Problem


class Basis:

    def __init__(self):
        pass

    @staticmethod
    def calculate_basis(Space: Patch,
                        coefficient: NDArray,
                        corrector_type: str) -> NDArray:
        if corrector_type not in ['lod', 'plod', 'splod', 'eholod']:
            raise MethodError()
        if corrector_type == 'lod':
            return Basis._lod_corrector(Space, coefficient)
        if corrector_type == 'plod':
            return Basis._plod_corrector(Space, coefficient)
        return Basis._splod_corrector(Space, coefficient)

    @staticmethod
    def enriched_basis(Space: Patch,
                       coefficient: NDArray,
                       G: csc_array) -> NDArray:
        return Basis._enriched_corrector(Space, coefficient, G)

    @staticmethod
    def _lod_corrector(Space: Patch,
                       coefficient: csr_array
                       ) -> NDArray:
        # print('Starting corrector ', Space.idx, end='.\n')
        S = assemble_matrices(Space.nelems_fine,
                              Space.loc_stiffness_fine,
                              mesh_type='cg',
                              cg2dg=Space.cg2dg_fine,
                              weight=coefficient)[
                                  np.ix_(Space.cg_dofs_patch_fine,
                                         Space.cg_dofs_patch_fine)]
        B = Space.lod_interpolation[np.ix_(Space.cg_dofs_patch_coarse,
                                           Space.cg_dofs_patch_fine)].toarray()

        S = cholesky(S.tocsc())
        Y = S(B.T)
        Schur = np.matmul(B, Y)
        Schur = cho_factor(Schur)

        r = assemble_matrices(Space.nelems_fine,
                              Space.loc_stiffness_fine,
                              weight=coefficient)
        r = csc_array(Space.cg2dg_fine[np.ix_(Space.dg_dofs_element_fine,
                                              Space.cg_dofs_patch_fine)].T)\
            @ (r[np.ix_(Space.dg_dofs_element_fine,
                        Space.dg_dofs_element_fine)]
               @ Space.dg_prolongation[np.ix_(Space.dg_dofs_element_fine,
                                              Space.dg_dofs_element_coarse)]
               ).toarray()

        r = S(r)
        L = cho_solve(Schur, np.matmul(B, r))
        C = r - Y @ L
        # print('Corrector ', Space.idx, ' completed.')
        return C

    @staticmethod
    def _plod_corrector(Space: Patch,
                        coefficient: csr_array
                        ) -> NDArray:
        # print('Starting corrector ', Space.idx, end='.\n')
        S = assemble_matrices(Space.nelems_fine,
                              Space.loc_stiffness_fine,
                              mesh_type='cg',
                              cg2dg=Space.cg2dg_fine,
                              weight=coefficient)[
                                  np.ix_(Space.cg_dofs_patch_fine,
                                         Space.cg_dofs_patch_fine)]
        B = Space.plod_interpolation[np.ix_(Space.pdg_dofs_patch_coarse,
                                            Space.cg_dofs_patch_fine)].toarray()

        S = cholesky(S.tocsc())
        Y = S(B.T)
        Schur = np.matmul(B, Y)
        Schur = cho_factor(Schur)

        q = np.eye(np.prod(Space.nelems_coarse)*Space.loc_dofs, Space.loc_dofs,
                   k=-Space.idx*Space.loc_dofs,
                   dtype=plodwave.config.PRECISION)[Space.pdg_dofs_patch_coarse, :]

        L = cho_solve(Schur, q)
        R = np.dot(Y, L)
        # print('Corrector ', Space.idx, ' completed.')
        return R

    @staticmethod
    def _splod_corrector(Space: Patch,
                         coefficient: NDArray
                         ) -> tuple[NDArray]:
        # print('Starting corrector ', Space.idx, end='.\n')
        S = assemble_matrices(Space.nelems_fine,
                              Space.loc_stiffness_fine,
                              mesh_type='cg',
                              cg2dg=Space.cg2dg_fine,
                              weight=coefficient)[
                                  np.ix_(Space.cg_dofs_patch_fine,
                                         Space.cg_dofs_patch_fine)]
        B = Space.plod_interpolation[np.ix_(Space.pdg_dofs_patch_coarse,
                                            Space.cg_dofs_patch_fine)].toarray()

        S = cholesky(S.tocsc())
        Y = S(B.T)
        Schur = np.matmul(B, Y)
        Schur = cho_factor(Schur)

        q = np.eye(np.prod(Space.nelems_coarse)*Space.loc_dofs, Space.loc_dofs,
                   k=-Space.idx*Space.loc_dofs,
                   dtype=plodwave.config.PRECISION)[Space.pdg_dofs_patch_coarse, :]

        L = cho_solve(Schur, q)
        R = np.matmul(Y, L)

        r = assemble_matrices(Space.nelems_fine,
                              Space.loc_stiffness_fine,
                              weight=coefficient)
        r = csc_array(Space.cg2dg_fine[np.ix_(Space.dg_dofs_element_fine,
                                              Space.cg_dofs_patch_fine)].T)\
            @ (r[np.ix_(Space.dg_dofs_element_fine,
                        Space.dg_dofs_element_fine)]
               @ Space.dg_prolongation[np.ix_(Space.dg_dofs_element_fine,
                                              Space.dg_dofs_element_coarse)]
               ).toarray()

        r = S(r)
        L = cho_solve(Schur, np.matmul(B, r))
        C = r - Y @ L
        # print('Corrector ', Space.idx, ' completed.')
        return C, R

    @staticmethod
    def _enriched_corrector(Space: Patch,
                            coefficient: NDArray,
                            G: csc_array) -> NDArray:
        # print('Starting corrector ', Space.idx, end='.\n')
        S = assemble_matrices(Space.nelems_fine,
                              Space.loc_stiffness_fine,
                              mesh_type='cg',
                              cg2dg=Space.cg2dg_fine,
                              weight=coefficient)[
                                  np.ix_(Space.cg_dofs_patch_fine,
                                         Space.cg_dofs_patch_fine)]
        M = Space.cg_mass_fine[np.ix_(Space.cg_dofs_patch_fine,
                                      Space.cg_dofs_patch_fine)]
        B = Space.plod_interpolation[np.ix_(Space.pdg_dofs_patch_coarse,
                                            Space.cg_dofs_patch_fine)].toarray()

        S = cholesky(S.tocsc())
        Y = S(B.T)
        Schur = np.matmul(B, Y)
        Schur = cho_factor(Schur)

        e = np.concatenate([np.array([Space.idx]), np.prod(Space.nelems_coarse)
                            + np.arange(Space.idx*(Space.loc_dofs-1),
                                        (Space.idx+1)*(Space.loc_dofs-1))])
        r = M.dot(G[np.ix_(Space.cg_dofs_patch_fine, e)]).toarray()

        r = S(r)
        L = cho_solve(Schur, np.matmul(B, r))
        D = r - np.matmul(Y, L)
        # print('Corrector ', Space.idx, ' completed.')
        return D


class Corrector:

    def __init__(self,
                 mesh: Mesh,
                 ell: int,
                 save_string: str = None):
        self._mesh: Mesh = mesh
        self._ell: int = ell
        self._submeshes: Sequence[Patch] = self.mesh.get_submeshes(self.ell)

        self._save_string: str = save_string

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def ell(self) -> int:
        return self._ell

    @property
    def submeshes(self) -> Sequence[Patch]:
        return self._submeshes

    @property
    def save_string(self) -> str:
        return self._save_string

    @save_string.setter
    def save_string(self, new_string: str) -> None:
        if not isinstance(new_string, str):
            raise ValueError
        self._save_string = new_string
        return

    def collect_corrector(self,
                          problem: Problem,
                          in_parallel: bool = True
                          ) -> Sequence[NDArray]:
        if in_parallel:
            with multiprocessing.Pool(
                    processes=plodwave.config.NUMBER_CPU) as pool:
                C = pool.map(partial(
                    Basis.calculate_basis,
                    coefficient=problem.coefficient,
                    corrector_type=problem.discretization_type),
                    self.submeshes)
        else:
            C = list(map(partial(Basis.calculate_basis,
                                 coefficient=problem.coefficient,
                                 corrector_type=problem.discretization_type),
                     self.submeshes))
        if self.save_string is not None:
            parameter_str = f'H{self.mesh.nelems_coarse[0]}'
            if problem.discretization_type in ['plod', 'splod']:
                parameter_str = f'{parameter_str}_p{self.mesh.p}'
            parameter_str = f'{parameter_str}_ell{self.ell}.npz'

            if problem.discretization_type == 'lod':
                np.savez(f'{self.save_string}/basis/C_{parameter_str}',
                         *C, allow_pickle=False)
            if problem.discretization_type == 'plod':
                np.savez(f'{self.save_string}/basis/R_{parameter_str}',
                         *C, allow_pickle=False)
            if problem.discretization_type == 'splod':
                G, R = zip(*C)
                np.savez(f'{self.save_string}/basis/C_{parameter_str}',
                         *G, allow_pickle=False)
                np.savez(f'{self.save_string}/basis/R_{parameter_str}',
                         *R, allow_pickle=False)
        return C

    def collect_enriched_corrector(self,
                                   problem: Problem,
                                   G: csc_array,
                                   in_parallel: bool = True
                                   ) -> Sequence[NDArray]:
        if in_parallel:
            with multiprocessing.Pool(
                    processes=plodwave.config.NUMBER_CPU) as pool:
                C = pool.map(partial(
                    Basis.enriched_basis,
                    coefficient=problem.coefficient,
                    G=G),
                    self.submeshes)
        else:
            C = list(map(partial(Basis.enriched_basis,
                                 coefficient=problem.coefficient,
                                 G=G),
                     self.submeshes))
        if self.save_string is not None:
            # Note we do not save the enriched corrector here. The
            # reason is, we do not provide j yet, and thus they would be
            # overriden
            pass
        return C

    def assemble_corrector(self,
                           problem: Problem,
                           C: Sequence[NDArray] = None,
                           in_parallel: bool = True
                           ) -> csc_array:
        if C is None:
            C = self.collect_corrector(problem, in_parallel)
        N = len(self.submeshes)
        if problem.discretization_type in ['lod', 'plod']:
            D = np.hstack([C[j].flatten() for j in range(N)])
        if problem.discretization_type in ['splod', 'eholod']:
            D = np.hstack([C[j][1].flatten() for j in range(N)])  # data for R

        if problem.discretization_type in ['plod', 'splod', 'eholod']:
            J = np.hstack([np.repeat(self.submeshes[j].cg_dofs_patch_fine,
                                     self.submeshes[j].loc_dofs)
                           for j in range(N)])
            K = np.hstack([np.tile(self.submeshes[j].pdg_dofs_element_coarse,
                           (np.size(self.submeshes[j].cg_dofs_patch_fine,)))
                           for j in range(N)])
            R = csc_array((D, (J, K)),
                          shape=(np.prod(self.mesh.nelems_fine+1),
                                 np.prod(self.mesh.nelems_coarse)
                                 * self.mesh.loc_dofs))
        if problem.discretization_type == 'plod':
            if self.save_string is not None:
                parameter_str = f'H{self.mesh.nelems_coarse[0]}'
                parameter_str = f'{parameter_str}_p{self.mesh.p}'
                parameter_str = f'{parameter_str}_ell{self.ell}.npz'
                sparse.save_npz(f'{self.save_string}/corrector/corrector_'
                                + f'{problem.discretization_type}_'
                                + f'{parameter_str}', R)
            return R

        if problem.discretization_type in ['splod', 'eholod']:
            if self.save_string is not None:
                parameter_str = f'H{self.mesh.nelems_coarse[0]}'
                parameter_str = f'{parameter_str}_p{self.mesh.p}'
                parameter_str = f'{parameter_str}_ell{self.ell}.npz'
                sparse.save_npz(f'{self.save_string}/corrector/corrector_'
                                + f'plod_'
                                + f'{parameter_str}', R)
            D = np.hstack([C[j][0].flatten() for j in range(N)])  # data for C
        if problem.discretization_type in ['lod', 'splod', 'eholod']:
            J = np.hstack([np.repeat(self.submeshes[j].cg_dofs_patch_fine,
                                     2**self.submeshes[j].d)
                           for j in range(N)])
            K = np.hstack([np.tile(self.submeshes[j].dg_dofs_element_coarse,
                                   (np.size(self.submeshes[j].cg_dofs_patch_fine,)))
                           for j in range(N)])
            C = csc_array((D, (J, K)),
                          shape=(np.prod(self.mesh.nelems_fine+1),
                                 np.prod(self.mesh.nelems_coarse)*2**self.mesh.d))
            C = self.mesh.cg_prolongation - C.dot(self.mesh.cg2dg_coarse)
        if problem.discretization_type == 'lod':
            if self.save_string is not None:
                parameter_str = f'H{self.mesh.nelems_coarse[0]}'
                parameter_str = f'{parameter_str}_ell{self.ell}.npz'
                sparse.save_npz(f'{self.save_string}/corrector/corrector_'
                                + f'{problem.discretization_type}_'
                                + f'{parameter_str}', C)
            return C

        # Stabilization
        IH = get_IH(self.mesh)
        BB = get_bubble(self.mesh)
        iota = IH.dot(BB)
        PH = C.dot(iota)
        Pi = self.mesh.plod_interpolation - (self.mesh.plod_interpolation.dot(
            self.mesh.cg_prolongation).dot(IH))
        PH += R.dot(Pi.dot(BB))

        p0dofs = np.arange(np.prod(self.mesh.nelems_coarse)*self.mesh.loc_dofs,
                           step=self.mesh.loc_dofs)
        dofs = np.setdiff1d(np.arange(np.prod(self.mesh.nelems_coarse)
                                      * self.mesh.loc_dofs),
                            p0dofs)
        R = sparse.hstack([PH, R[:, dofs]], format='csr')
        if self.save_string is not None:
            parameter_str = f'H{self.mesh.nelems_coarse[0]}'
            parameter_str = f'{parameter_str}_p{self.mesh.p}'
            parameter_str = f'{parameter_str}_ell{self.ell}.npz'
            sparse.save_npz(f'{self.save_string}/corrector/corrector_'
                            + f'{problem.discretization_type}_'
                            + f'{parameter_str}', R)
        return R

    def assemble_enriched_corrector(self,
                                    problem: Problem,
                                    G: csc_array = None,
                                    D: Sequence[NDArray] = None,
                                    in_parallel: bool = True
                                    ) -> csc_array:
        if D is None:
            if G is None:
                raise ValueError('Either D or G need to be provided.')
            D = self.collect_enriched_corrector(problem, G, in_parallel)
        N = len(self.submeshes)
        D = np.hstack([D[j].flatten() for j in range(N)])
        J = np.hstack([np.repeat(self.submeshes[j].cg_dofs_patch_fine,
                                 self.submeshes[j].loc_dofs)
                       for j in range(N)])
        K = np.hstack([np.tile(self.submeshes[j].pdg_dofs_element_coarse,
                               (np.size(self.submeshes[j].cg_dofs_patch_fine,)))
                       for j in range(N)])
        D = csc_array((D, (J, K)),
                      shape=(np.prod(self.mesh.nelems_fine+1),
                             np.prod(self.mesh.nelems_coarse)
                             * self.mesh.loc_dofs))
        return D
