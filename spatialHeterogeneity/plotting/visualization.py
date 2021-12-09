# %% imports
import logging
import warnings
from ..utils.general import is_numeric, is_categorical, make_iterable

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, NoNorm, to_rgba, to_rgb, Colormap, ListedColormap
import seaborn as sns

# import matplotlib.ticker as mticker  # https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator

import numpy as np
import os
import napari

# %% figure constants
from .utils import dpi, label_fontdict, title_fontdict, make_cbar, savefig

# %%

def spatial(so, spl: str, attr: str, *, mode: str = 'scatter', node_size: float = 4, coordinate_keys: list = ['x', 'y'],
            mask_key: str = 'cellmasks', graph_key: str = 'knn', edges: bool = False, edge_width: float = .5,
            edge_color: str = 'black', edge_zorder: int = 2, background_color: str = 'white', ax: plt.Axes = None,
            norm=None, set_title: bool = True, cmap=None, cmap_labels: list = None, cbar: bool = True,
            cbar_title: bool = True, show: bool = True, save: str = None, tight_layout: bool = True):
    """Visualisation of samples.

    Args:
        so: SpatialOmics instance
        spl: sample to visualise
        attr: feature to visualise
        mode: {scatter, mask}. In `scatter` mode, observations are represented by their centroid, in `mask` mode by their actual segmentation mask
        node_size: size of the node when plotting the graph representation
        coordinate_keys: column names in SpatialOmics.obs[spl] that indicates the x and y coordinates
        mask_key: key for the segmentation masks when in `mask` mode
        graph_key: which graph representation to use
        edges: whether to plot the graph or not
        edge_width: width of edges
        edge_color: color of edges as string
        edge_zorder: z-order of edges
        background_color: background color of plot
        ax: axes object in which to plot
        norm: normalisation instance to normalise the values of `attr`
        set_title: title of plot
        cmap: colormap to use
        cmap_labels: colormap labels to use
        cbar: whether to plot a colorbar or not
        cbar_title: whether to plot the `attr` name as title of the colorbar
        show: whether to show the plot or not
        save: path to the file in which the plot is saved
        tight_layout: whether to apply tight_layout or not.

    Returns:

    """
    # get attribute information
    data = None  # pd.Series/array holding the attr for colormapping
    colors = None  # array holding the colormappig of data

    # try to fetch the attr data
    if attr:
        if attr in so.obs[spl].columns:
            data = so.obs[spl][attr]
        elif attr in so.X[spl].columns:
            data = so.X[spl][attr]
        else:
            raise KeyError(f'{attr} is not an column of X nor obs')
    else:
        colors = 'black'
        cmap = 'black'
        cbar = False

    # broadcast if necessary
    _is_categorical_flag = is_categorical(data)

    loc = so.obs[spl][coordinate_keys]

    # set colormap
    if cmap is None:
        cmap, cmap_labels = get_cmap(so, attr, data)
    elif isinstance(cmap, str) and colors is None:
        cmap = plt.get_cmap(cmap)

    # normalisation
    if norm is None:
        if _is_categorical_flag:
            norm = NoNorm()
        else:
            norm = Normalize()

    # generate figure
    if ax:
        fig = ax.get_figure()
        show = False  # do not automatically show plot if we provide axes
    else:
        fig, ax = plt.subplots(dpi=dpi)
        ax.set_aspect('equal')

    # compute edge lines
    if edges:
        g = so.G[spl][graph_key]
        e = np.array(g.edges, dtype=type(loc.index.dtype))

        tmp1 = loc.loc[e.T[0]]
        tmp2 = loc.loc[e.T[1]]

        x = np.stack((tmp1[coordinate_keys[0]], tmp2[coordinate_keys[0]]))
        y = np.stack((tmp1[coordinate_keys[1]], tmp2[coordinate_keys[1]]))

        # we have to plot sequentially nodes and edges, this takes a bit longer but is more flexible
        im = ax.plot(x, y, linestyle='-', linewidth=edge_width, marker=None, color=edge_color, zorder=edge_zorder)

    # plot
    if mode == 'scatter':
        # convert data to numeric
        data = np.asarray(
            data) if data is not None else None  # categorical data does not work with cmap, therefore we construct an array
        _broadcast_to_numeric = not is_numeric(data)  # if data is now still categorical, map to numeric

        if _broadcast_to_numeric and data is not None:
            if attr in so.uns['cmap_labels']:
                cmap_labels = so.uns['cmap_labels'][attr]
                encoder = {value: key for key, value in cmap_labels.items()}  # invert to get encoder
            else:
                uniq = np.unique(data)
                encoder = {i: j for i, j in zip(uniq, range(len(uniq)))}
                cmap_labels = {value: key for key, value in encoder.items()}  # invert to get cmap_labels

            data = np.asarray([encoder[i] for i in data])

        # map to colors
        if colors is None:
            colors = cmap(norm(data))

        im = ax.scatter(loc[coordinate_keys[0]], loc[coordinate_keys[1]], s=node_size, c=colors, zorder=2.5)
        ax.set_facecolor(background_color)

    elif mode == 'mask':

        # get cell mask
        mask = so.get_mask(spl, mask_key)

        # generate mapping
        if data is not None:
            mapping = data.to_dict()
        elif colors:
            # case in which attr is a color
            uniq = np.unique(mask)
            uniq = uniq[uniq != 0]
            mapping = {i: j for i, j in zip(uniq, np.ones(len(uniq), dtype=int))}  # map everything to 1
            cmap = ListedColormap([to_rgba('white'), colors])
        else:
            raise RuntimeError('Unknown case')
        mapping.update({0: 0})

        # apply mapping vectorized
        otype = ['int'] if _is_categorical_flag else ['float']
        func = np.vectorize(lambda x: mapping[x], otypes=otype)
        im = func(mask)

        # plot
        im = cmap(norm(im))
        im[mask == 0] = to_rgba(background_color)
        imobj = ax.imshow(im)
        ax.invert_yaxis()

    else:
        raise ValueError(f'Invalide plotting mode {mode}')

    # add colorbar
    if cbar:
        title = attr
        if cbar_title is False:
            title = ''
        make_cbar(ax, title, norm, cmap, cmap_labels)

    # format plot
    ax_pad = min(loc['x'].max() * .05, loc['y'].max() * .05, 10)
    ax.set_xlim(loc['x'].min() - ax_pad, loc['x'].max() + ax_pad)
    ax.set_ylim(loc['y'].min() - ax_pad, loc['y'].max() + ax_pad)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_xlabel('spatial x', label_fontdict)
    ax.set_ylabel('spatial y', label_fontdict)
    ax.set_aspect(1)
    if set_title:
        title = f'{spl}' if cbar else f'{spl}, {attr}'
        ax.set_title(title, title_fontdict)

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if save:
        savefig(fig, save)


def napari_viewer(so, spl: str, attrs: list, censor: float = .95, add_masks='cellmasks'):
    """Starts interactive Napari viewer to visualise raw images

    Args:
        so: SpatialOmics instance
        spl: sample to visualise
        attrs: list of attributes to add as channels to the viewer
        censor: percentil to use to censore pixle values in the raw images
        add_masks: segmentation masks to add as channels to the viewer

    Returns:

    """
    attrs = list(make_iterable(attrs))
    var = so.var[spl]
    index = var[var.target.isin(attrs)].fullstack_index
    names = var[var.target.isin(attrs)].target

    img = so.get_image(spl)[index,]
    if censor:
        for j in range(img.shape[0]):
            v = np.quantile(img[j,], censor)
            img[j, img[j] > v] = v
            img[j,] = img[j,] / img[j,].max()

    viewer = napari.Viewer()
    viewer.add_image(img, channel_axis=0, name=names)
    if add_masks:
        add_masks = make_iterable(add_masks)
        for m in add_masks:
            mask = so.masks[spl][m]
            labels_layer = viewer.add_labels(mask, name=m)


def channel(so, spl: str, attrs: str, ax=None, colors=None, censor: float = None, show=True):
    """Plot challnels. Decreapted, will be removed.

    Args:
        so:
        spl:
        attrs:
        ax:
        colors:
        censor:
        show:

    Returns:

    """
    attrs = list(make_iterable(attrs))
    var = so.var[spl]
    i = var[var.target.isin(attrs)].fullstack_index

    img = so.get_image(spl)[i,]

    if censor:
        for j in range(img.shape[0]):
            v = np.quantile(img[j,], censor)
            img[j, img[j] > v] = v
            img[j,] = img[j,] / img[j,].max()

    res = []
    for c in colors:
        tmp = np.ones((*img.shape[1:], 3))
        col = np.array(to_rgb(c))
        res.append(tmp * col)
    res = np.stack(res)

    res = res * img.reshape((*img.shape, 1))
    res = res.sum(axis=0) / len(res)
    res = np.squeeze(res)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        show = False

    ax.imshow(res)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('spatial y')
    ax.set_xlabel('spatial x')
    ax.set_title(spl)
    fig.tight_layout()

    if show:
        fig.show()


def interactions(so, spl, attr, mode='proportion', prediction_type='diff', graph_key='knn', linewidths=.5, cmap=None,
                 norm=None, ax=None, show=True):
    """Visualise interaction results.

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        mode: One of {classic, histoCAT, proportion}, see notes
        prediction_type: prediction_type: One of {observation, pvalue, diff}
        graph_key: Specifies the graph representation to use in so.G[spl]
        linewidths: Space between tiles
        cmap: colormap to use
        norm: normalisation to use
        ax: axes object to use
        show: whether to show the plot

    Returns:

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        show = False

    if cmap is None:
        cmap = 'coolwarm'

    data = so.uns[spl]['interactions'][f'{attr}_{mode}_{prediction_type}_{graph_key}']
    score = 'diff' if prediction_type == 'diff' else 'score'
    data = data.reset_index().pivot('source_label', 'target_label', score)
    data.index = data.index.astype(int)
    data.columns = data.columns.astype(int)
    data.sort_index(0, inplace=True)
    data.sort_index(1, inplace=True)

    if norm is None:
        v = np.abs(data).max().max()
        norm = Normalize(-v, v)

    sns.heatmap(data=data, cmap=cmap, norm=norm, ax=ax, linewidths=linewidths)
    ax.set_aspect(1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if show:
        fig.show()


def get_cmap(so, attr: str, data):
    '''
    Return the cmap and cmap labels for a given attribute if available, else a default

    Parameters
    ----------
    so: IMCData
        so object form which to fetch the data
    spl: str
        spl for which to get data
    attr: str
        attribute for which to get the cmap and cmap labels if available

    Returns
    -------
    cmap and cmap labels for attribute

    '''

    # TODO: recycle cmap if more observations than colors
    cmap, cmap_labels = None, None
    if attr in so.uns['cmaps'].keys():
        cmap = so.uns['cmaps'][attr]
    elif is_categorical(data):
        cmap = so.uns['cmaps']['category']
    else:
        cmap = so.uns['cmaps']['default']

    if attr in so.uns['cmap_labels'].keys():
        cmap_labels = so.uns['cmap_labels'][attr]

    return cmap, cmap_labels
