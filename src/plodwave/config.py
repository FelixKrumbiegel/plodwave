import numpy as np
import multiprocessing
from typing import Type


PRECISION: Type = np.float64
NUMBER_CPU: int = multiprocessing.cpu_count()


def set_precision(precision: Type) -> None:
    """ Sets the precision for the whole package.

    Parameters
    ----------
    precision : Type
        Precision.
    """
    global PRECISION
    PRECISION = precision
    return


def set_number_of_cpus(n: int) -> None:
    """ Sets the number of CPUs used for parallelization for the whole
    package.

    Parameters
    ----------
    n : int
        Number of CPUs.
    """
    global NUMBER_CPU
    NUMBER_CPU = max(min(n, multiprocessing.cpu_count()), 1)
    return
