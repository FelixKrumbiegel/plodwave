import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_array, csc_array
from .mesh import Mesh
from .fem import assemble_matrices
from .solution import SolutionStrategy
from .corrector import Corrector
# type hinting
from numpy.typing import NDArray
from collections.abc import Callable, Sequence


class Problem:

    def __init__(self,
                 mesh: Mesh,
                 coefficient: NDArray,
                 forcing: Callable,
                 boundary_conditions: Sequence[Callable],
                 solution_strategy: SolutionStrategy,
                 discretization_type: str,
                 ell: int = 1,
                 j: int = 0,
                 initial_conditions: Sequence[Callable] | None = None,
                 parallel_correctors: bool = True
                 ) -> None:
        self._mesh: Mesh = mesh

        self._coefficient: NDArray = coefficient
        self._forcing: Callable = forcing
        self._boundary_conditions: Sequence[Callable] = boundary_conditions

        self._initial_conditions: Sequence[Callable] | None = initial_conditions
        self._parallel_correctors: bool = parallel_correctors

        if discretization_type not in ['ref', 'fem', 'lod', 'plod',
                                       'splod']:
            raise NotImplementedError(f'The discretization type submitted '
                                      + f'is not implemented.')
        self._discretization_type: str = discretization_type

        self._ell: int = ell
        self._j: int = j
        if discretization_type in ['lod', 'plod', 'splod']:
            self._corrector: Corrector = Corrector(self.mesh, self.ell)

        self._solution_strategy: SolutionStrategy = solution_strategy
        self.solution_strategy.set_problem(self)

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def coefficient(self) -> NDArray:
        return self._coefficient

    @property
    def forcing(self) -> Callable:
        return self._forcing

    @property
    def boundary_conditions(self) -> Sequence[Callable]:
        return self._boundary_conditions

    @property
    def discretization_type(self) -> str:
        return self._discretization_type

    @property
    def solution_strategy(self) -> SolutionStrategy:
        return self._solution_strategy

    @property
    def ell(self) -> int:
        return self._ell

    @property
    def j(self) -> int:
        return self._j

    @property
    def corrector(self) -> Corrector:
        if not hasattr(self, '_corrector'):
            raise ValueError(
                f'Corrector is accessed, however the discretization does not '
                + f'require a corrector. Corrector is only implemented for '
                + f'lod, plod, splod, eholod.')
        return self._corrector

    @property
    def initial_conditions(self) -> Sequence[Callable] | None:
        return self._initial_conditions

    @property
    def parallel_correctors(self) -> bool:
        return self._parallel_correctors

    @parallel_correctors.setter
    def parallel_correctors(self, new_parallel_flag: bool) -> None:
        if not isinstance(new_parallel_flag, bool):
            raise ValueError
        self._parallel_correctors = new_parallel_flag
        return

    @property
    def stiffness(self) -> csr_array:
        if self.mesh.mode == 'cached':
            if not hasattr(self, '_stiffness'):
                self._stiffness = assemble_matrices(
                    self.mesh.nelems_fine,
                    self.mesh.loc_stiffness_fine,
                    mesh_type='cg',
                    cg2dg=self.mesh.cg2dg_fine,
                    weight=self.coefficient)
            return self._stiffness
        return assemble_matrices(self.mesh.nelems_fine,
                                 self.mesh.loc_stiffness_fine,
                                 mesh_type='cg',
                                 cg2dg=self.mesh.cg2dg_fine,
                                 weight=self.coefficient)

    @property
    def mass(self) -> csr_array:
        if self.mesh.mode == 'cached':
            if not hasattr(self, '_mass'):
                self._mass = assemble_matrices(
                    self.mesh.nelems_fine,
                    self.mesh.loc_mass_fine,
                    mesh_type='cg',
                    cg2dg=self.mesh.cg2dg_fine)
            return self._mass
        return assemble_matrices(self.mesh.nelems_fine,
                                 self.mesh.loc_mass_fine,
                                 mesh_type='cg',
                                 cg2dg=self.mesh.cg2dg_fine)

    @property
    def projection(self) -> csc_array:
        if not hasattr(self, '_projection'):
            self._projection = self._get_projection()
        return self._projection

    @property
    def solution(self) -> NDArray:
        if not hasattr(self, '_solution'):
            self._solution = self.solution_strategy.run()
        return self._solution

    def _get_projection(self) -> csc_array:
        if self.discretization_type == 'eholod':
            return self._get_enriched_projection()

        if self.discretization_type == 'ref':
            return self.mesh.cg_dof_projection_fine

        if self.discretization_type == 'fem':
            return self.mesh.cg_prolongation[:, self.mesh.cg_dofs_coarse]

        C = self.corrector.assemble_corrector(
            self, in_parallel=self.parallel_correctors)
        if self.discretization_type == 'lod':
            return C[:, self.mesh.cg_dofs_coarse]
        return C

    def _get_enriched_projection(self) -> csc_array:
        R = self.corrector.assemble_corrector(
            self, in_parallel=self.parallel_correctors)
        if self.j == 1:
            D = self.corrector.assemble_enriched_corrector(
                self, R, in_parallel=self.parallel_correctors)
            return sparse.hstack([R, D])
        if self.j == 1:
            D = self.corrector.assemble_enriched_corrector(
                self, R, in_parallel=self.parallel_correctors)
            DD = self.corrector.assemble_enriched_corrector(
                self, D, in_parallel=self.parallel_correctors)
            return sparse.hstack([R, D, DD])
        return R

    def save_projection(self, save_folder: str) -> None:
        method_str = f'{save_folder}/corrector_{self.discretization_type}'
        if self.discretization_type == 'lod':
            parameter_str = f'H{self.mesh.nelems_coarse[0]}_ell{self.ell}.npz'
            sparse.save_npz(f'{method_str}_{parameter_str}', self.projection)
            return
        parameter_str = f'H{self.mesh.nelems_coarse[0]}_p{self.mesh.p}'\
            + f'_ell{self.ell}.npz'
        sparse.save_npz(f'{method_str}_{parameter_str}', self.projection)
        return

    def save_enriched_projection(self, save_folder: str) -> csc_array:
        parameter_str = f'H{self.mesh.nelems_coarse[0]}'\
            + f'_p{self.mesh.p}_ell{self.ell}.npz'
        if self.j == 0:
            R = self.corrector.assemble_corrector(
                self, in_parallel=self.parallel_correctors)
            sparse.save_npz(f'{save_folder}/corrector/'
                            + f'corrector_eholod_j0_{parameter_str}', R)
            return
        if self.j == 1:
            R = sparse.load_npz(f'{save_folder}/corrector/'
                                + f'corrector_eholod_j0_{parameter_str}')
            D = self.corrector.assemble_enriched_corrector(
                self, R, in_parallel=self.parallel_correctors)
            sparse.save_npz(f'{save_folder}/corrector/'
                            + f'corrector_eholod_j1_{parameter_str}', D)
            return
        if self.j == 2:
            D = sparse.load_npz(f'{save_folder}/corrector/'
                                + f'corrector_eholod_j1_{parameter_str}')
            DD = self.corrector.assemble_enriched_corrector(
                self, D, in_parallel=self.parallel_correctors)
            sparse.save_npz(f'{save_folder}/corrector/'
                            + f'corrector_eholod_j2_{parameter_str}', DD)
            return
        return

    def load_projection(self, save_folder: str) -> None:
        method_str = f'{save_folder}/corrector_{self.discretization_type}'
        if self.discretization_type == 'lod':
            parameter_str = f'H{self.mesh.nelems_coarse[0]}_ell{self.ell}.npz'
            self._projection = sparse.load_npz(f'{method_str}_{parameter_str}')
            return
        parameter_str = f'H{self.mesh.nelems_coarse[0]}_p{self.mesh.p}'\
            + f'_ell{self.ell}.npz'
        self._projection = sparse.load_npz(f'{method_str}_{parameter_str}')
        return

    def load_enriched_projection(self, save_folder: str) -> None:
        parameter_str = f'H{self.mesh.nelems_coarse[0]}_p{self.mesh.p}'\
            + f'_ell{self.ell}.npz'
        G = sparse.load_npz(f'{save_folder}/corrector/corrector_splod_'
                            + f'{parameter_str}')
        if self.j == 1:
            D = sparse.load_npz(f'{save_folder}/corrector/corrector_eholod_'
                                + f'j{1}_{parameter_str}')
            self._projection = sparse.hstack([G, D*1e5])
            return
        if self.j == 2:
            D = sparse.load_npz(f'{save_folder}/corrector/corrector_eholod_'
                                + f'j{1}_{parameter_str}')
            DD = sparse.load_npz(f'{save_folder}/corrector/corrector_eholod_'
                                 + f'j{2}_{parameter_str}')
            self._projection = sparse.hstack([G, D*1e5, DD*1e9])
            return
        self._projection = G
        return

    def save_basis(self, save_folder: str) -> None:
        C = self.corrector.collect_corrector(self, self.parallel_correctors)
        if self.discretization_type == 'lod':
            parameter_str = f'H{self.mesh.nelems_coarse[0]}_ell{self.ell}.npz'
            np.savez(f'{save_folder}/C_{parameter_str}',
                     *C, allow_pickle=False)
            return
        parameter_str = f'H{self.mesh.nelems_coarse[0]}_p{self.mesh.p}'\
            + f'_ell{self.ell}.npz'
        if self.discretization_type == 'plod':
            np.savez(f'{save_folder}/R_{parameter_str}',
                     *C, allow_pickle=False)
            return
        if self.discretization_type == 'splod':
            C, R = zip(*C)
            np.savez(f'{save_folder}/C_{parameter_str}',
                     *C, allow_pickle=False)
            np.savez(f'{save_folder}/R_{parameter_str}',
                     *R, allow_pickle=False)
        return

    def projection_from_basis(self, save_folder: str) -> None:
        if self.discretization_type == 'lod':
            parameter_str = f'H{self.mesh.nelems_coarse[0]}_ell{self.ell}.npz'
            C = np.load(f'{save_folder}/C_{parameter_str}')
            G = [C[f'arr_{j}'] for j in range(len(C))]
        parameter_str = f'H{self.mesh.nelems_coarse[0]}_p{self.mesh.p}'\
            + f'_ell{self.ell}.npz'
        if self.discretization_type == 'plod':
            C = np.load(f'{save_folder}/R_{parameter_str}')
            G = [C[f'arr_{j}'] for j in range(len(C))]
        if self.discretization_type == 'splod':
            C = np.load(f'{save_folder}/C_{parameter_str}')
            R = np.load(f'{save_folder}/R_{parameter_str}')
            G = [(C[f'arr_{j}'], R[f'arr_{j}']) for j in range(len(C))]
        self._projection = self.corrector.assemble_corrector(
            self, G, in_parallel=self.parallel_correctors)
        return

    def save_solution(self, save_folder: str) -> None:
        if self.discretization_type == 'ref':
            parameter_str = f'H{self.mesh.nelems_fine[0]}.npy'
        if self.discretization_type == 'fem':
            parameter_str = f'H{self.mesh.nelems_coarse[0]}.npy'
        if self.discretization_type == 'lod':
            parameter_str = f'H{self.mesh.nelems_coarse[0]}_ell{self.ell}.npy'
        if self.discretization_type in ['splod', 'plod']:
            parameter_str = f'H{self.mesh.nelems_coarse[0]}_p{self.mesh.p}'\
                + f'_ell{self.ell}.npy'
        if self.discretization_type == 'eholod':
            parameter_str = f'j{self.j}_H{self.mesh.nelems_coarse[0]}'\
                + f'_p{self.mesh.p}_ell{self.ell}.npy'
        np.save(f'{save_folder}/solution_{self.discretization_type}'
                + f'_{parameter_str}', self.solution)
        return

    def solve(self) -> NDArray:
        return self.solution
