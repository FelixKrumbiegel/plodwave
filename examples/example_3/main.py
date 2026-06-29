import numpy as np
from plodwave import Mesh, TimeDomain, HyperbolicThetaMethod, Problem


def g(x): return np.zeros_like(x[..., [0]])


def f(x, t): return np.sin(np.pi*x[..., [0]])**4\
    * np.sin(np.pi*x[..., [1]])**4\
    * np.sin(t)**4


def main(save_path: str) -> None:
    h = 8
    final_time, theta = 1., 1/12
    Hmin, Hmax = 1, 5
    ellmin_p, ellmax_p = 1, 9
    pmin, pmax = 0, 3

    H1error = np.zeros((Hmax-Hmin+1, ellmax_p-ellmin_p+1, pmax-pmin+1))
    L2error = np.zeros((Hmax-Hmin+1, ellmax_p-ellmin_p+1, pmax-pmin+1))

    A = np.load(f'{save_path}/coefficient.npy')

    T = TimeDomain(np.array([2**(h + 5)]), final_time)
    S = HyperbolicThetaMethod(T, theta, 'sparse_direct')

    """Reference solution
    """

    nelems_coarse = np.array([2**(h-1), 2**(h-1)], dtype=np.int32)
    nelems_refine = np.array([2**1, 2**1], dtype=np.int32)
    mesh = Mesh(nelems_coarse, nelems_refine)
    P = Problem(mesh, A, f, [g, g, g], S, 'ref',
                initial_conditions=[g, g, g, g])
    uref = P.solve()

    """pLOD solution
    """

    for H in range(Hmin, Hmax+1):
        T = TimeDomain(np.array([2**(H + 5)]), final_time)
        nelems_coarse = np.array([2**H, 2**H], dtype=np.int32)
        nelems_refine = np.array([2**(h-H), 2**(h-H)], dtype=np.int32)
        for ell in range(ellmin_p, ellmax_p+1):
            for p in range(pmin, pmax+1):
                mesh = Mesh(nelems_coarse, nelems_refine, p=p)
                S = HyperbolicThetaMethod(T, theta, 'sparse_direct')
                P = Problem(mesh, A, f, [g, g, g], S, 'splod', ell=ell,
                            initial_conditions=[g, g, g, g],
                            parallel_correctors=False)
                ulod = P.solve()

                H1error[H-Hmin, ell-ellmin_p, p-pmin] = np.sqrt(
                    np.dot(uref-ulod, mesh.cg_stiffness_fine.dot(uref-ulod)) /
                    np.dot(uref, mesh.cg_stiffness_fine.dot(uref)))
                L2error[H-Hmin, ell-ellmin_p, p-pmin] = np.sqrt(
                    np.dot(uref-ulod, mesh.cg_mass_fine.dot(uref-ulod)) /
                    np.dot(uref, mesh.cg_mass_fine.dot(uref)))
                print(f'H={H}, ell={ell}, p={p}, '
                      + f'H1error={H1error[H-Hmin, ell-ellmin_p, p-pmin]}')

    with open(f'{save_path}/H1errors_plod.csv', 'w') as file:
        file.write('H')
        for ell in range(ellmin_p, ellmax_p+1):
            for p in range(pmin, pmax+1):
                file.write(f',p{p}ell{ell}')

        for H in range(Hmin, Hmax+1):
            file.write(f'\n{1/2**H}')
            for ell in range(ellmin_p, ellmax_p+1):
                for p in range(pmin, pmax+1):
                    file.write(f',{H1error[H-Hmin, ell-ellmin_p, p-pmin]}')

    with open(f'{save_path}/L2errors_plod.csv', 'w') as file:
        file.write('H')
        for ell in range(ellmin_p, ellmax_p+1):
            for p in range(pmin, pmax+1):
                file.write(f',p{p}ell{ell}')

        for H in range(Hmin, Hmax+1):
            file.write(f'\n{1/2**H}')
            for ell in range(ellmin_p, ellmax_p+1):
                for p in range(pmin, pmax+1):
                    file.write(f',{L2error[H-Hmin, ell-ellmin_p, p-pmin]}')
    return

    H1EOC = np.log2(H1error[:-1, ...]/H1error[1:, ...])
    L2EOC = np.log2(L2error[:-1, ...]/L2error[1:, ...])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str)
    pargs = parser.parse_args()

    from setup_coefficient import setup_coefficient

    setup_coefficient(pargs.save_path)
    main(pargs.save_path)
