import numpy as np
from plodwave import Mesh


def setup_coefficient(save_path: str):
    eps, h = 6, 8
    Neps = np.array([2**eps, 2**eps], dtype=np.int32)
    Nr = np.array([2**(h-eps), 2**(h-eps)], dtype=np.int32)
    mesh = Mesh(Neps, Nr)
    M, N = Neps[0], (Neps*Nr)[0]
    X = np.arange(M)/M + 1/(2*M)

    def f(x, y): return np.cos(N*x**2*y-2*N*x**2)\
        * np.sin(N/2*y**3*x+N*x**2)\
        + np.cos(N*x**3-2*N*y**2)
    X1, X2 = np.meshgrid(X, X)
    A = (2*f(X1, X2)+5).reshape((-1,), order='C')

    A = mesh.element_prolongation.dot(A)
    np.save(f'{save_path}/coefficient.npy', A)
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str)
    pargs = parser.parse_args()
    setup_coefficient(pargs.save_path)
