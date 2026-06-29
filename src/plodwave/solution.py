from __future__ import annotations
import numpy as np
from scipy.sparse import csc_array
import plodwave
from .mesh import TimeDomain
from .solver import Solver
# type hinting
from numpy.typing import NDArray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .problem import Problem


class SolutionStrategy:

    def __init__(self,
                 solver_type: str = 'sparse_direct'
                 ) -> None:
        self._solver_type: str = solver_type

    @property
    def solver_type(self) -> str:
        return self._solver_type

    @solver_type.setter
    def solver_type(self, new_solver_type) -> None:
        if new_solver_type not in ['sparse_direct', 'sparse_cg']:
            raise NotImplementedError(
                f'The provided solver type is not implemented. Choose '
                + f'either [sparse_direct, sparse_cg].')
        self._solver_type = new_solver_type
        return

    @property
    def problem(self) -> Problem:
        if not hasattr(self, '_problem'):
            raise ValueError(
                f'Problem has not been set. A problem has to be created'
                + f' first, before the solution strategy is executed.')
        return self._problem

    def set_problem(self,
                    problem: Problem
                    ) -> None:
        if hasattr(self, '_problem'):
            raise AttributeError(
                f'The Problem context has already been set for this '
                + f'SolutionStrategy.'
            )
        self._problem = problem
        return

    def run(self) -> None:
        raise NotImplementedError()


class EllipticSolutionStrategy(SolutionStrategy):

    def __init__(self,
                 solver_type: str = 'sparse_direct'
                 ) -> None:
        super().__init__(solver_type)

    def run(self) -> NDArray:
        # Set tolerances
        if self.problem.discretization_type == 'ref':  # Reference solve
            tol = self.problem.mesh.mesh_size_fine/100
        elif self.problem.discretization_type in ['fem', 'lod', 'plod',
                                                  'splod', 'eholod']:
            tol = self.problem.mesh.mesh_size_coarse/100
            if self.problem.discretization_type == 'lod':
                tol = tol**2/100
            if self.problem.discretization_type in ['plod', 'splod',
                                                    'eholod']:
                tol = tol**(self.problem.mesh.p + 2)/100

        # projection onto DoFs
        G = self.problem.projection
        # stiffness matrix assembly
        K = G.T.dot(self.problem.stiffness.dot(G))
        # right hand side assembly
        fh = self._get_right_hand_side().flatten()
        gh = self._get_boundary_condition().flatten()
        b = G.T.dot(self.problem.mass.dot(fh) - self.problem.stiffness.dot(gh))

        K_inv = Solver.setupSolver(K, self.solver_type)
        x = Solver.useSolver(K_inv, b, K, tol)
        return G.dot(x) + gh

    def _get_right_hand_side(self) -> NDArray:
        return self.problem.forcing(self.problem.mesh.cg_vertices_fine)

    def _get_boundary_condition(self) -> NDArray:
        gD = self.problem.boundary_conditions[0]
        if self.problem.discretization_type == 'ref':
            gh = np.zeros((np.prod(self.problem.mesh.nelems_fine+1), 1),
                          dtype=plodwave.config.PRECISION)
            gh[self.problem.mesh.boundary_nodes_fine[0], :] = gD(
                self.problem.mesh.cg_vertices_fine)[
                self.problem.mesh.boundary_nodes_fine[0], :]
            return gh
        if self.problem.discretization_type == 'fem':
            gh = np.zeros((np.prod(self.problem.mesh.nelems_fine+1), 1),
                          dtype=plodwave.config.PRECISION)
            gh[self.problem.mesh.boundary_nodes_fine[0], :] = gD(
                self.problem.mesh.cg_vertices_fine)[
                    self.problem.mesh.boundary_nodes_fine[0], :]
            return gh
        if self.problem.discretization_type == 'lod':
            gh = np.zeros((np.prod(self.problem.mesh.nelems_fine+1), 1),
                          dtype=plodwave.config.PRECISION)
            gh[self.problem.mesh.boundary_nodes_fine[0], :] = gD(
                self.problem.mesh.cg_vertices_fine)[
                    self.problem.mesh.boundary_nodes_fine[0], :]
            return gh
        if self.problem.discretization_type in ['plod', 'splod', 'eholod']:
            # Note: Only zero Dirichlet implemented.
            gh = np.zeros((np.prod(self.problem.mesh.nelems_fine+1), 1),
                          dtype=plodwave.config.PRECISION)
            gh[self.problem.mesh.boundary_nodes_fine[0], :] = gD(
                self.problem.mesh.cg_vertices_fine)[
                    self.problem.mesh.boundary_nodes_fine[0], :]
            return gh


class HyperbolicSolutionStrategy(SolutionStrategy):

    def __init__(self,
                 time_domain: TimeDomain,
                 solver_type='sparse_direct'
                 ) -> None:
        self._time_domain: TimeDomain = time_domain
        super().__init__(solver_type)

    @property
    def time_domain(self) -> TimeDomain:
        return self._time_domain


class HyperbolicThetaMethod(HyperbolicSolutionStrategy):

    def __init__(self,
                 time_domain: TimeDomain,
                 theta: float,
                 solver_type='sparse_direct'
                 ) -> None:
        if theta < 0. or theta > .5:
            raise ValueError
        self._theta: float = theta
        super().__init__(time_domain, solver_type)

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, new_theta: float) -> None:
        if new_theta < 0 or new_theta > .5:
            raise ValueError
        self._theta = new_theta
        return

    def set_problem(self,
                    problem: Problem
                    ) -> None:
        if hasattr(self, '_problem'):
            raise AttributeError(
                f'The Problem context has already been set for this '
                + f'SolutionStrategy.')
        if len(problem.initial_conditions) < 4:
            raise ValueError(
                f' The theta-method needs the first two derivatives of '
                + f'the right-hand side at t=0.')
        self._problem = problem
        return

    def run(self) -> NDArray:
        # Set tolerances
        if self.problem.discretization_type == 'ref':  # Reference solve
            tol = self.problem.mesh.mesh_size_fine / 10
        elif self.problem.discretization_type in ['fem', 'lod',
                                                  'plod', 'splod']:
            tol = self.problem.mesh.mesh_size_coarse / 10
            if self.problem.discretization_type == 'lod':
                tol = tol**2
            if self.problem.discretization_type in ['plod', 'splod']:
                tol = tol**(self.problem.mesh.p + 2)

        tau = 1./self.time_domain.nelems_time[0]

        # projection onto DoFs
        G = self.problem.projection
        # stiffness matrix assembly
        K = G.T.dot((self.problem.mass
                     + tau*tau*self.theta*self.problem.stiffness).dot(G))
        # right hand side assembly
        b, u0h, _ = self._get_initial_condition(tau)
        # gh = self._get_boundary_condition().flatten()  # Not implemented
        b = G.T.dot(b)

        K_inv = Solver.setupSolver(K, self.solver_type)
        x = Solver.useSolver(K_inv, b, K, tol)  # x0 = G.T.dot(u0)

        S = G.T.dot(self.problem.stiffness.dot(G))
        M = G.T.dot(self.problem.mass.dot(G))
        S_inv = Solver.setupSolver(S, self.solver_type)
        z0H = Solver.useSolver(S_inv, G.T.dot(self.problem.stiffness.dot(u0h)),
                               S, tol)
        z1H = x + z0H

        for k in range(2, self.time_domain.number_time_steps+1):
            # print(k)
            fh = self._get_right_hand_side(tau, k)
            b = (G.T.dot(self.problem.mass.dot(fh))
                 + M.dot(2*z1H - z0H)
                 - tau*tau*S.dot((1-2*self.theta)*z1H + self.theta*z0H))
            x = Solver.useSolver(K_inv, b, K, tol, x0=z1H)
            z0H, z1H = z1H.copy(), x.copy()
        return G.dot(z1H)

    def _get_right_hand_side(self,
                             tau: float,
                             t: int) -> NDArray:
        fh = tau*tau*(self.theta*(
            self.problem.forcing(
                self.problem.mesh.cg_vertices_fine, (t-2)*tau)
            + self.problem.forcing(self.problem.mesh.cg_vertices_fine, t*tau))
            + (1 - 2*self.theta)*self.problem.forcing(
                self.problem.mesh.cg_vertices_fine, (t-1)*tau))
        return fh.flatten()

    def _get_initial_condition(self,
                               tau: float) -> NDArray:
        u0 = self.problem.initial_conditions[0]
        v0 = self.problem.initial_conditions[1]
        ft = self.problem.initial_conditions[2]
        ftt = self.problem.initial_conditions[3]

        u0h = u0(self.problem.mesh.cg_vertices_fine).flatten()
        v0h = v0(self.problem.mesh.cg_vertices_fine).flatten()  # = v0h
        fh = tau*(self.problem.forcing(self.problem.mesh.cg_vertices_fine, 0)/2
                  + tau*(ft(self.problem.mesh.cg_vertices_fine)/6
                         + tau*ftt(self.problem.mesh.cg_vertices_fine)/24)
                  ).flatten()
        b = self.problem.mass.dot(tau*(v0h + fh))
        b -= self.problem.stiffness.dot(tau*tau*(u0h/2 + tau*v0h/12))
        return b, u0h, v0h

    #     tau = 1./self.time_domain.nelems_time[0]

    #     # projection onto DoFs
    #     G = self.problem.projection
    #     # stiffness matrix assembly
    #     K = G.T.dot((self.problem.mass
    #                  + tau*tau*self.theta*self.problem.stiffness).dot(G))
    #     # right hand side assembly
    #     u0h, u1h = self._get_initial_condition(tau)
    #     b = G.T.dot(self.problem.mass.dot(u0h)
    #                 - self.problem.stiffness.dot(u1h))

    #     K_inv = Solver.setupSolver(K, self.solver_type)
    #     x = Solver.useSolver(K_inv, b, K, tol)

    #     S = G.T.dot(self.problem.stiffness.dot(G))
    #     M = G.T.dot(self.problem.mass.dot(G))
    #     S_inv = Solver.setupSolver(S, self.solver_type)
    #     z0H = Solver.useSolver(S_inv, G.T.dot(self.problem.stiffness.dot(u0h)),
    #                            S, tol)
    #     z1H = x + z0H

    #     for k in range(2, self.time_domain.number_time_steps+1):
    #         fh = self._get_right_hand_side(tau, k)
    #         b = (G.T.dot(self.problem.mass.dot(fh))
    #              + M.dot(2*z1H - z0H)
    #              - tau*tau*S.dot((1-2*self.theta)*z1H + self.theta*z0H))
    #         x = Solver.useSolver(K_inv, b, K, tol, x0=z1H)
    #         z0H, z1H = z1H.copy(), x.copy()
    #     return G.dot(z1H)

    # def _get_right_hand_side(self,
    #                          tau: float,
    #                          t: int) -> NDArray:
    #     fh = tau*tau*(self.theta*(
    #         self.problem.forcing(
    #             self.problem.mesh.cg_vertices_fine, (t-2)*tau)
    #         + self.problem.forcing(self.problem.mesh.cg_vertices_fine, t*tau))
    #         + (1 - 2*self.theta)*self.problem.forcing(
    #             self.problem.mesh.cg_vertices_fine, (t-1)*tau))
    #     return fh.flatten()

    # def _get_initial_condition(self,
    #                            tau: float) -> NDArray:
    #     u0 = self.problem.initial_conditions[0]
    #     v0 = self.problem.initial_conditions[1]
    #     ft = self.problem.initial_conditions[2]
    #     ftt = self.problem.initial_conditions[3]

    #     u0h = u0(self.problem.mesh.cg_vertices_fine).flatten()
    #     v0h = v0(self.problem.mesh.cg_vertices_fine).flatten()
    #     fh = tau*(self.problem.forcing(self.problem.mesh.cg_vertices_fine, 0)/2
    #               + tau*(ft(self.problem.mesh.cg_vertices_fine)/6
    #                      + tau*ftt(self.problem.mesh.cg_vertices_fine)/24)
    #               ).flatten()
    #     return tau*(v0h + fh), tau*tau*(u0h/2 + tau*v0h/12)
