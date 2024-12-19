import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.figure import SubplotParams
import os

dpi = 300
ax_pad = 10
label_fontdict = {'size': 7}
title_fontdict = {'size': 10}
cbar_inset = [1.02, 0, .0125, .96]
cbar_titel_fontdict = {'size': 7}
cbar_labels_fontdict = {'size': 7}
root_fig = plt.rcParams['savefig.directory']


def make_cbar(ax, title, norm, cmap, cmap_labels, im=None, prefix_labels=True):
    """Generate a colorbar for the given axes.

    Parameters
    ----------
    ax: Axes
        axes for which to plot colorbar
    title: str
        title of colorbar
    norm:
        Normalisation instance
    cmap: Colormap
        Colormap
    cmap_labels: dict
        colorbar labels

    Returns
    -------

    """
    # NOTE: The Linercolormap ticks can only be set up to the number of colors. Thus if we do not have linear, sequential
    # values [0,1,2,3] in the cmap_labels dict this will fail. Solution could be to remap.

    inset = ax.inset_axes(cbar_inset)
    fig = ax.get_figure()
    if im is None:
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=inset)
    else:
        cb = fig.colorbar(im, cax=inset)

    cb.ax.set_title(title, loc='left', fontdict=cbar_titel_fontdict)
    if cmap_labels:
        if prefix_labels:
            labs = [f'{key}, {val}' for key, val in cmap_labels.items()]
        else:
            labs = list(cmap_labels.values())
        cb.set_ticks(list(cmap_labels.keys()))
        cb.ax.set_yticklabels(labs, fontdict=cbar_labels_fontdict)
    else:
        cb.ax.tick_params(axis='y', labelsize=cbar_labels_fontdict['size'])

    # TODO
    def linear_mapping(cmap_labels):
        pass


def savefig(fig, save):
    # if only filename is given, add root_fig, convenient to save plots less verbose.
    if save == os.path.basename(save):
        save = os.path.join(plt.rcParams['savefig.directory'], save)
    fig.savefig(save)
    print(f'Figure saved at: {save}')


def colormap(cmap):
    """Visualise a colormap.

    Parameters
    ----------
    cmap: Colormap

    Returns
    -------

    """
    n = len(cmap.colors)
    a = np.zeros((1, n, 4))

    for j in range(n):
        a[0, j,] = cmap(j)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(a)
    ax.tick_params(labelleft=False, left=False)
    fig.show()

    # ax.set_aspect(.25)


def get_layout(nx, ny=None, max_col=5, max_row=None):
    if ny is None:
        ny = 0

    ncol = np.min((np.ceil(np.sqrt(nx)), max_col)).astype(int)
    nrow = int(np.ceil((nx + ny) / ncol))
    return (nrow, ncol)