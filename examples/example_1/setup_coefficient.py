import numpy as np
from plodwave import Mesh


def get_coefficient(save_path: str):
    eps, h = 6, 8
    Neps = np.array([2**eps, 2**eps], dtype=np.int32)
    Nh = np.array([2**(h-eps), 2**(h-eps)], dtype=np.int32)
    M, N = Neps[0], (Neps*Nh)[0]
    X = np.arange(M)/M + 1/(2*M)

    def f(x, y): return np.cos(N*x**2*y-2*N*x**2)\
        * np.sin(N/2*y**3*x+N*x**2)\
        + np.cos(N*x**3-2*N*y**2)
    X1, X2 = np.meshgrid(X, X)
    A = (2*f(X1, X2)+5).reshape((-1,), order='C')

    fine = 9
    P = element_prolongation(Neps, np.array([2**(fine-eps), 2**(fine-eps)]))
    A = P @ A
    np.save(f'{save_path}/coefficient_{2**fine}.npy', A)
    np.savetxt(f'{save_path}/coefficient_{2**fine}.txt', A)
    plt.imshow(A.reshape(2**fine, 2**fine, order='C'), origin='lower')
    plt.show()
    return None


def setup_coefficient(save_path: str) -> None:
    eps, h = 6, 8
    alpha, beta = 1, 9
    Neps = np.array([2**eps, 2**eps], dtype=np.int32)
    Nr = np.array([2**(h-eps), 2**(h-eps)], dtype=np.int32)
    mesh = Mesh(Neps, Nr)

    np.random.seed(69)
    A = np.ones((np.prod(Neps), 1))

    for j in range(2, eps + 1):
        n = 2*(eps + 1 - 2)
        omega1 = np.tile(np.random.rand(2**j, 1).round(0),
                         (2**(eps-j), 1))/n
        omega2 = np.tile(np.random.rand(1, 2**j).round(0),
                         (1, 2**(eps-j)))/n
        omega = np.reshape(omega1 + omega2, (-1, 1))
        A += (beta - alpha) * omega

    A = mesh.element_prolongation.dot(A)
    np.save(f'{save_path}/coefficient.npy', A)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str)
    pargs = parser.parse_args()
    setup_coefficient(pargs.save_path)
