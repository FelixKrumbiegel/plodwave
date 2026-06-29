

class DimensionError(Exception):
    """ Exception raised for the implementation of wrong space 
    dimension.

    Attributes
    ----------
    d : int
        Dimension.
    message : str
        Error message string.
    """

    def __init__(self, d: int):
        self.d: int = d
        self.message = f"Dimension {self.d} not implemented. Try d=1,2."
        super().__init__(self.message)


class ModeError(Exception):
    """ Exception raised for the wrong implementation of mesh mode.

    Attributes
    ----------
    message : str
        Error message string.
    """

    def __init__(self):
        self.message = f"Try any of the modes 'cached', 'matrix-free'"\
            + ", or 'hybrid'"
        super().__init__(self.message)


class SolverTypeError(Exception):
    """ Exception raised for the wrong implementation of solvertype.

    Attributes
    ----------
    message : str
        Error message string.
    """

    def __init__(self):
        self.message = f"Try any of the solvertypes 'sparse_cg', "\
            + "'sparse_direct'."
        super().__init__(self.message)


class BoundaryError(Exception):
    """ Exception raised if the Boundary array has not the correct 
    shape.

    Attributes
    ----------
    message : str
        Error message string.
    """

    def __init__(self):
        self.message = "The boundary array is not implemented correctly: The "\
            + "shape should be (d,2), where each row represents one "\
            + "dimension and each column the values x_i=0, and x_i=1."
        super().__init__(self.message)


class DomainError(Exception):
    """ Exception raised if the Domain array has not the correct shape.

    Attributes
    ----------
    message : str
        Error message string.
    """

    def __init__(self):
        self.message = "The domain array is not implemented correctly: The "\
            + "shape should be (d,), where each row represents the value at "\
            + "the end of the interval (0,a) for d=1, and (0,a)x(0,b) for d=2."
        super().__init__(self.message)


class MethodError(Exception):
    """ Exception raised for the implementation of wrong mesh method.

    Attributes
    ----------
    message : str
        Error message string.
    """

    def __init__(self):
        self.message = f"Try any of the methods 'lod', 'plod', 'splod', "\
            + f"'eholod', 'fem', 'ref'."
        super().__init__(self.message)
