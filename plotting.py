import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_press(eq, data, vmin_vmax=None, save=False, fname='gas_pore_pressure'):
    cmap = mpl.cm.viridis
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
        # norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh) for sh, s in zip(eq.shape, eq.sides))  # km

    fig, ax = plt.subplots(figsize=(3, 4), layout="constrained")

    ax.imshow(data[:, :, 0].transpose(), extent=[x_ax[0], x_ax[-1], y_ax[-1], y_ax[-0]], origin='lower', cmap=cmap, norm=norm)
    ax.set_title('XZ plane')
    ax.set_xlabel('x, m')
    ax.set_ylabel('Depth, m')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='vertical', label='Pore pressure, MPa')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()


def plot_sat(eq, data, vmin_vmax=None, save=False, fname='gas_saturation'):
    cmap = mpl.cm.viridis
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
        # norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh) for sh, s in zip(eq.shape, eq.sides))  # km

    fig, ax = plt.subplots(figsize=(3, 4), layout="constrained")

    ax.imshow(data[:, :, 0].transpose(), extent=[x_ax[0], x_ax[-1], y_ax[-1], y_ax[-0]], origin='lower', cmap=cmap, norm=norm)
    ax.set_title('XZ plane')
    ax.set_xlabel('x, m')
    ax.set_ylabel('Depth, m')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='vertical', label='Gas saturation')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()



def plot_fracture_prob(eq, data, vmin_vmax=None, save=False, fname='gas_saturation'):
    cmap = mpl.cm.viridis

    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh) for sh, s in zip(eq.shape, eq.sides))  # km

    fig, ax = plt.subplots(figsize=(3, 4), layout="constrained")

    ax.imshow(data[:, :, 0].transpose(), extent=[x_ax[0], x_ax[-1], y_ax[-1], y_ax[-0]], origin='lower', cmap=cmap, norm=norm)
    ax.set_title('XZ plane')
    ax.set_xlabel('x, m')
    ax.set_ylabel('Depth, m')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='vertical', label='Fracture probability')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()