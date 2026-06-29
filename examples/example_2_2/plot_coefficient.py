import numpy as np
import matplotlib.pyplot as plt
import matplot2tikz as tikz


def plot_coefficient(save_path: str) -> None:
    h = 8
    alpha, beta = 0, 10

    A = np.load(f'{save_path}/coefficient.npy').flatten()

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(A.reshape(2**h, 2**h), cmap='viridis', origin='lower',
                   vmin=alpha, vmax=beta,
                   extent=[0, 1, 0, 1],
                   aspect='auto')
    ax.tick_params(axis='both', which='major', labelsize=14)
    # x labelling
    ax.set_xticks([0, 1])
    # y labelling
    ax.set_yticks([0, 1])
    # colorbar
    cbar_ax = fig.add_axes([0.933, 0.11, 0.03, .77])
    cbar_ax.tick_params(labelsize=14)
    plt.colorbar(im, cax=cbar_ax, shrink=.5, ticks=[0, 2, 4, 6, 8, 10])

    plt.savefig(save_path + '/coefficient.pdf', bbox_inches='tight')
    # tikz.save(save_path + '/coefficient.tex')
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str)
    pargs = parser.parse_args()
    plot_coefficient(pargs.save_path)
