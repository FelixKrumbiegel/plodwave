import numpy as np
from plodwave import Mesh, TimeDomain, HyperbolicThetaMethod, Problem


def g(x): return np.zeros_like(x[..., [0]])


def f(x, t): return np.sin(np.pi*x[..., [0]]) * np.sin(np.pi*x[..., [1]])\
    * np.sin(t)**4


def main(save_path: str) -> None:
    h, t = 8, 9
    final_time, theta = 1., 1/4
    Hmin, Hmax = 1, 5
    ellmin_p, ellmax_p = 1, 9
    ellmin, ellmax = 1, 3
    pmin, pmax = 0, 3

    H1error = np.zeros((Hmax-Hmin+1, ellmax_p-ellmin_p+1, pmax-pmin+1))
    L2error = np.zeros((Hmax-Hmin+1, ellmax_p-ellmin_p+1, pmax-pmin+1))
    H1error_fem = np.zeros((Hmax-Hmin+1))
    L2error_fem = np.zeros((Hmax-Hmin+1))
    H1error_lod = np.zeros((Hmax-Hmin+1, ellmax-ellmin+1))
    L2error_lod = np.zeros((Hmax-Hmin+1, ellmax-ellmin+1))

    A = np.load(f'{save_path}/coefficient.npy')

    T = TimeDomain(np.array([2**t]), final_time)
    S = HyperbolicThetaMethod(T, theta, 'sparse_direct')

    """Reference solution
    """

    nelems_coarse = np.array([2**(h-1), 2**(h-1)], dtype=np.int32)
    nelems_refine = np.array([2**1, 2**1], dtype=np.int32)
    mesh = Mesh(nelems_coarse, nelems_refine)
    P = Problem(mesh, A, f, [g, g, g], S, 'ref',
                initial_conditions=[g, g, g, g])
    uref = P.solve()

    """FEM solution
    """

    for H in range(Hmin, Hmax+1):
        nelems_coarse = np.array([2**H, 2**H], dtype=np.int32)
        nelems_refine = np.array([2**(h-H), 2**(h-H)], dtype=np.int32)
        mesh = Mesh(nelems_coarse, nelems_refine)
        S = HyperbolicThetaMethod(T, theta, 'sparse_direct')
        P = Problem(mesh, A, f, [g, g, g], S, 'fem',
                    initial_conditions=[g, g, g, g])
        ufem = P.solve()

        H1error_fem[H-Hmin] = np.sqrt(
            np.dot(uref-ufem, mesh.cg_stiffness_fine.dot(uref-ufem)) /
            np.dot(uref, mesh.cg_stiffness_fine.dot(uref)))
        L2error_fem[H-Hmin] = np.sqrt(
            np.dot(uref-ufem, mesh.cg_mass_fine.dot(uref-ufem)) /
            np.dot(uref, mesh.cg_mass_fine.dot(uref)))
        print(f'H={H}, '
              + f'H1error={H1error_fem[H-Hmin]}')

    with open(f'{save_path}/H1errors_fem.csv', 'w') as file:
        file.write('H,fem')
        for ell in range(ellmin_p, ellmax_p+1):
            for p in range(pmin, pmax+1):
                file.write(f',p{p}ell{ell}')

        for H in range(Hmin, Hmax+1):
            file.write(f'\n{1/2**H}')
            file.write(f',{H1error_fem[H-Hmin]}')

    with open(f'{save_path}/L2errors_fem.csv', 'w') as file:
        file.write('H,fem')

        for H in range(Hmin, Hmax+1):
            file.write(f'\n{1/2**H}')
            file.write(f',{L2error_fem[H-Hmin]}')

    """LOD solution
    """

    for H in range(Hmin, Hmax+1):
        nelems_coarse = np.array([2**H, 2**H], dtype=np.int32)
        nelems_refine = np.array([2**(h-H), 2**(h-H)], dtype=np.int32)
        mesh = Mesh(nelems_coarse, nelems_refine)
        for ell in range(ellmin, ellmax+1):
            S = HyperbolicThetaMethod(T, theta, 'sparse_direct')
            P = Problem(mesh, A, f, [g, g, g], S, 'lod', ell=ell,
                        initial_conditions=[g, g, g, g],
                        parallel_correctors=False)
            ulod = P.solve()

            H1error_lod[H-Hmin, ell-ellmin] = np.sqrt(
                np.dot(uref-ulod, mesh.cg_stiffness_fine.dot(uref-ulod)) /
                np.dot(uref, mesh.cg_stiffness_fine.dot(uref)))
            L2error_lod[H-Hmin, ell-ellmin] = np.sqrt(
                np.dot(uref-ulod, mesh.cg_mass_fine.dot(uref-ulod)) /
                np.dot(uref, mesh.cg_mass_fine.dot(uref)))
            print(f'H={H}, ell={ell}, '
                  + f'H1error={H1error_lod[H-Hmin, ell-ellmin]}')

    with open(f'{save_path}/H1errors_lod.csv', 'w') as file:
        file.write('H')
        for ell in range(ellmin, ellmax+1):
            file.write(f',ell{ell}')

        for H in range(Hmin, Hmax+1):
            file.write(f'\n{1/2**H}')
            for ell in range(ellmin, ellmax+1):
                file.write(f',{H1error_lod[H-Hmin, ell-ellmin]}')

    with open(f'{save_path}/L2errors_lod.csv', 'w') as file:
        file.write('H')
        for ell in range(ellmin, ellmax+1):
            file.write(f',ell{ell}')

        for H in range(Hmin, Hmax+1):
            file.write(f'\n{1/2**H}')
            for ell in range(ellmin, ellmax+1):
                file.write(f',{L2error_lod[H-Hmin, ell-ellmin]}')

    """pLOD solution
    """

    for H in range(Hmin, Hmax+1):
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
