""" This package implements the higher-order LOD method.
"""

from .config import set_precision, set_number_of_cpus
from .exceptions import (DimensionError, ModeError, BoundaryError, DomainError,
                         MethodError, SolverTypeError)
from .mesh import Mesh, Patch, TimeDomain
from .problem import Problem
from .solution import (SolutionStrategy, EllipticSolutionStrategy,
                       HyperbolicThetaMethod)
from .corrector import Corrector, Basis


__version__ = "0.1.0"
__author__ = "Felix Krumbiegel"
__all__ = [
    "set_precision", "set_number_of_cpus",
    "DimensionError", "ModeError", "BoundaryError", "DomainError",
    "MethodError", "SolverTypeError",
    "Mesh", "Patch", "TimeDomain",
    "Problem",
    "Corrector", "Basis",
    "SolutionStrategy", "EllipticSolutionStrategy", "HyperbolicThetaMethod"
]
