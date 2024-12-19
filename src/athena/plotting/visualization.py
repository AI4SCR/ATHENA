# %% imports
import logging

import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker  # https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.colors import Normalize, NoNorm, to_rgba, ListedColormap

# %% figure constants
from .utils import dpi, label_fontdict, title_fontdict, make_cbar, savefig
from ..utils.general import get_nx_graph_from_anndata
from ..utils.general import is_categorical


def spatial(ad: AnnData, attr: str, *, mode: str = 'scatter', node_size: float = 4, coordinate_keys: list = ['x', 'y'],
            mask: np.ndarray = None, graph_key: str = 'knn', edges: bool = False, edge_width: float = .5,
            edge_color: str = 'black', edge_zorder: int = 2, background_color: str = 'white', ax: plt.Axes = None,
            norm=None, title: str = None, cmap=None, cmap_labels: list = None, cbar: bool = True,
            cbar_title: bool = True, show: bool = True, save: str = None, tight_layout: bool = True,
            filter_col: str = None, include_labels: list = None):
    """Various functionalities to visualise samples.
    Allows to visualise the samples and color observations according to features in either so.X or so.obs by setting the ``attr`` parameter accordingly.
    Furthermore, observations (cells) within a sample can be quickly visualised using scatter plots (requires extraction of centroids with :func:`~.extract_centroids`)
    or by their actual segmentatio mask by setting ``mode`` accordingly.
    Finally, the graph representation of the sample can be overlaid by setting ``edges=True`` and specifying the ``graph_key`` as in ``so.G[spl][graph_key]``.

    For more examples on how to use this function have a look at the tutorial_ section.

    Args:
        ad: AnnData
        attr: feature to visualise
        mode: {scatter, mask}. In `scatter` mode, observations are represented by their centroid, in `mask` mode by their actual segmentation mask
        node_size: size of the node when plotting the graph representation
        coordinate_keys: column names in SpatialOmics.obs[spl] that indicates the x and y coordinates
        mask: mask of segmented cells when in `mask` mode
        graph_key: which graph representation to use
        edges: whether to plot the graph or not
        edge_width: width of edges
        edge_color: color of edges as string
        edge_zorder: z-order of edges
        background_color: background color of plot
        ax: axes object in which to plot
        norm: normalisation instance to normalise the values of `attr`
        title: title of plot
        cmap: colormap to use
        cmap_labels: colormap labels to use
        cbar: whether to plot a colorbar or not
        cbar_title: whether to plot the `attr` name as title of the colorbar
        show: whether to show the plot or not
        save: path to the file in which the plot is saved
        tight_layout: whether to apply tight_layout or not.
        filter_col: string identifying the column in so.obs to use to filter out cells that should not be plotted.
        include_labels: list of strings that identify the cells that should be included in the plot. These should be entries in filter_col.

    Examples:

        .. code-block:: python

            sh.pl.spatial(so, 'slide_7_Cy2x2', 'meta_id', mode='mask')
            sh.pl.spatial(so, 'slide_7_Cy2x2', 'meta_id', mode='scatter', edges=True)


    .. _tutorial: https://ai4scr.github.io/ATHENA/source/tutorial.html
    """
    # get attribute information
    data = None  # pd.Series/array holding the attr for colormapping
    colors = None  # array holding the colormappig of data

    # try to fetch the attr data
    if attr in ad.obs:
        data = ad.obs[attr].copy()
    elif attr in ad.var.index:
        data = ad.X[:, ad.var.index == attr]
        data = pd.Series(data.flatten(), index=ad.obs.index)
    else:
        raise KeyError(f'{attr} is not an column of X nor obs')

    data.index = data.index.astype(int)

    loc = ad.obs[coordinate_keys].copy()

    # set colormap
    if cmap is None:
        cmap, cmap_labels = get_cmap(ad=ad, attr=attr, data=data)
    elif isinstance(cmap, str) and colors is None:
        cmap = plt.get_cmap(cmap)

    # NOTE: we project the data to a numeric representation if it is categorical
    _is_categorical_flag = is_categorical(data)
    if _is_categorical_flag:
        uniq = np.unique(data)
        n = len(uniq)
        encoder = {i: j for i, j in zip(uniq, range(len(uniq)))}
        decoder = {value: key for key, value in encoder.items()}

        data = data.map(encoder)
        data = data.astype(int)

        if isinstance(cmap, list):
            cmap = ListedColormap([to_rgba(c) for c in cmap[:n]])
            cmap_labels = decoder
        elif isinstance(cmap, dict):
            cmap = ListedColormap([to_rgba(cmap[i]) for i in uniq])
            cmap_labels = decoder

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
        g = get_nx_graph_from_anndata(ad=ad, key=graph_key)
        e = np.array(g.edges, dtype=type(loc.index.dtype))

        tmp1 = loc.loc[e.T[0]]
        tmp2 = loc.loc[e.T[1]]

        x = np.stack((tmp1[coordinate_keys[0]], tmp2[coordinate_keys[0]]))
        y = np.stack((tmp1[coordinate_keys[1]], tmp2[coordinate_keys[1]]))

        # we have to plot sequential nodes and edges, this takes a bit longer but is more flexible
        im = ax.plot(x, y, linestyle='-', linewidth=edge_width, marker=None, color=edge_color, zorder=edge_zorder)

    # plot
    if mode == 'scatter':
        if data is not None:
            no_na = data[~np.isnan(data)]
            _ = norm(no_na)  # initialize norm with no NA data.
            colors = cmap(norm(data))

        im = ax.scatter(loc[coordinate_keys[0]], loc[coordinate_keys[1]], s=node_size, c=colors, zorder=2.5)
        ax.set_facecolor(background_color)

    elif mode == 'mask':
        mask = mask or ad.uns['mask']

        mapping = data.to_dict()
        mapping.update({0: 0})
        mapping.update({np.nan: np.nan})

        # apply mapping vectorized
        otype = ['int'] if _is_categorical_flag else ['float']
        func = np.vectorize(lambda x: mapping[x], otypes=otype)
        im = func(mask)

        # convert to masked array to handle np.nan values
        im = np.ma.array(im, mask=np.isnan(im))

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
    ax_pad = min(loc[coordinate_keys[0]].max() * .05, loc[coordinate_keys[1]].max() * .05, 10)
    ax.set_xlim(loc[coordinate_keys[0]].min() - ax_pad, loc[coordinate_keys[0]].max() + ax_pad)
    ax.set_ylim(loc[coordinate_keys[1]].min() - ax_pad, loc[coordinate_keys[1]].max() + ax_pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('spatial x', label_fontdict)
    ax.set_ylabel('spatial y', label_fontdict)
    ax.set_aspect(1)
    if title is not None:
        ax.set_title(title, title_fontdict)

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if save:
        savefig(fig, save)


def interactions(ad: AnnData, attr, mode='proportion', prediction_type='diff', graph_key='knn', linewidths=.5,
                 cmap=None,
                 norm=None, ax=None, show=True, cbar=True):
    """Visualise results from :func:`~neigh.interactions` results.

    Args:
        ad: AnnData instance
        attr: Categorical feature in ad.obs to use for the grouping
        mode: One of {classic, histoCAT, proportion}, see notes
        prediction_type: prediction_type: One of {observation, pvalue, diff}
        graph_key: Specifies the graph representation to use in so.G[spl]
        linewidths: Space between tiles
        cmap: colormap to use
        norm: normalisation to use
        ax: axes object to use
        show: whether to show the plot
        cbar: wheter to show the colorbar

    Examples:

        .. code-block:: python

            # compute Ripley's K
            so = sh.dataset.imc()
            spl = so.spl.index[0]

            # build graph
            sh.graph.build_graph(so, spl, builder_type='knn', mask_key='cellmasks')

            # compute & plot interactions
            sh.neigh.interactions(so, spl, 'meta_id', mode='proportion', prediction_type='observation')
            sh.pl.interactions(so, spl, 'meta_id', mode='proportion', prediction_type='observation')

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        show = False

    if cmap is None:
        cmap = 'coolwarm'

    data = ad.uns['interactions'][f'{attr}_{mode}_{prediction_type}_{graph_key}']
    score = 'diff' if prediction_type == 'diff' else 'score'
    data = data.reset_index().pivot(index='source_label', columns='target_label', values=score)
    # data.index = data.index.astype(int)
    # data.columns = data.columns.astype(int)
    data.sort_index(axis=0, inplace=True)
    data.sort_index(axis=1, inplace=True)

    if norm is None:
        v = np.abs(data).max().max()
        norm = Normalize(-v, v)

    sns.heatmap(data=data, cmap=cmap, norm=norm, ax=ax, linewidths=linewidths, cbar=cbar)
    ax.set_aspect(1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if show:
        fig.tight_layout()
        fig.show()


def get_cmap(ad: AnnData, attr: str, data):
    '''
    Return the cmap and cmap labels for a given attribute if available, else a default

    Parameters
    ----------
    ad: AnnData from which to extract cmap
    attr: str
        attribute for which to get the cmap and cmap labels if available

    Returns
    -------
    cmap and cmap labels for attribute

    '''

    cmap, cmap_labels = None, None
    if attr in ad.uns['cmaps'].keys():
        cmap = ad.uns['cmaps'][attr]
    elif is_categorical(data):
        from matplotlib.colors import to_rgba
        import colorcet as cc
        n = len(set(data))
        cmap = [to_rgba(c) for c in cc.glasbey_bw[:n]]
    elif 'default' in ad.uns['cmaps']:
        cmap = ad.uns['cmaps']['default']
    else:
        import matplotlib
        cmap = matplotlib.colormaps['Reds']

    return cmap, cmap_labels


def ripleysK(ad: AnnData, *, attr: str, ids: list, mode='K', correction='ripley',
             key=None, ax=None, legend='auto', cmap=None, cmap_labels=None):
    """Visualise results from :func:`~neigh.ripleysK` results.

    Args:
        ad: AnnData instance
        attr: Categorical feature in ad.obs to use for the grouping
        ids: The category in the categorical feature `attr`, for which Ripley's K should be plotted
        mode: {K, csr-deviation}. If `K`, Ripley's K is estimated, with `csr-deviation` the deviation from a poission process is computed.
        correction: Correction method to use to correct for boarder effects, see [1].
        key: key to use in so.uns['ripleysK'] for the plot, if None it is constructed from attr,ids,mode and correction
        ax: axes to use for the plot

    Examples:

        .. code-block:: python

            # compute Ripley's K
            so = sh.dataset.imc()
            sh.neigh.ripleysK(so, so.spl.index[0], 'meta_id', 1, mode='csr-deviation', radii=radii)
            sh.pl.ripleysK(so, so.spl.index[0], 'meta_id', [1], mode='csr-deviation')

    """

    if key is None:
        assert isinstance(ids, list), 'ids should be a list'
        keys = [f'{i}_{attr}_{mode}_{correction}' for i in ids]

    res = []
    for i in keys:
        res.append(ad.uns['ripleysK'][i])

    res = pd.concat(res, axis=1)
    if key is None:
        colnames = [i.split('_')[0] for i in keys]
    else:
        colnames = keys
    res = res.reset_index()
    res.columns = ['radii'] + colnames
    radii = res.radii.values
    res = res.melt(id_vars='radii', var_name=attr)
    res[attr] = res[attr].astype('category')

    cmap, labels = get_cmap(ad=ad, attr=attr, data=res[attr])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    sns.lineplot(data=res, x='radii', y='value', hue=attr, palette=cmap, ax=ax, legend=legend)
    ax.plot(radii, np.repeat(0, len(radii)), color='black', linestyle=':')
    fig.tight_layout()
    fig.show()


def infiltration(ad: AnnData, attr: str = 'infiltration', step_size: int = 10,
                 interpolation: str = 'gaussian',
                 obsm_key: str = 'spatial',
                 cmap: str = 'plasma',
                 collision_strategy='mean',
                 ax=None,
                 show=True):
    """Visualises a heatmap of the featuer intensity.

    Approximates the sample with a grid representation and colors each grid square according to the
    value of the attribute. If multiple observations map to the same grid square a the aggregation specified
    in `collision_strategy` is employed (any value accepted by pandas aggregate function, i.e. 'mean', 'max', ...)

    Args:
        ad: AnnData instance
        attr: feature in ad.obs to plot
        step_size: grid step size
        interpolation: interpolation method to use between grid values, see [1]
        cmap: colormap to use
        collision_strategy: aggragation strategy to use if multiple obseravtion values map to the same grid value
        ax: axes to use for the plot
        show: whether to show the plot or not. Will be set to False if axes is provided.

    Examples:

        .. code-block:: python

            so = sh.dataset.imc()
            spl = so.spl.index[0]

            # build graph
            sh.graph.build_graph(so, spl, builder_type='knn', mask_key='cellmasks')

            sh.neigh.infiltration(so, spl, 'meta_id', graph_key='knn')
            sh.pl.infiltration(so, spl, step_size=10)
    Notes:
        .. [1] https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
    """

    dat = ad.obs[[attr]]
    loc = ad.obs[['y', 'x']]
    dat = pd.concat((dat, loc), axis=1)
    dat = dat[~dat.infiltration.isna()]

    # we add step_size to prevent out of bounds indexing should the `{x,y}_img` values be rounded up.
    x, y = np.arange(0, ad.uns['mask'].shape[1], step_size), np.arange(0, ad.uns['mask'].shape[0], step_size)
    img = np.zeros((len(y), len(x)))

    dat['x_img'] = np.round(dat.x / step_size).astype(int)
    dat['y_img'] = np.round(dat.y / step_size).astype(int)

    if dat[['x_img', 'y_img']].duplicated().any():
        logging.warning(
            f'`step_size` is to granular, {dat[["x_img", "y_img"]].duplicated().sum()} observed infiltration values mapped to same grid square')

    if collision_strategy is not None:
        logging.warning(f'computing {collision_strategy} for collisions')
        dat = dat.groupby(['x_img', 'y_img']).infiltration.agg(collision_strategy).reset_index()

    for i in range(dat.shape[0]):
        img[dat.y_img.iloc[i], dat.x_img.iloc[i]] = dat.infiltration.iloc[i]

    # generate figure
    if ax:
        fig = ax.get_figure()
        show = False  # do not automatically show plot if we provide axes
    else:
        fig, ax = plt.subplots()

    ax.imshow(img, interpolation=interpolation, cmap=cmap)
    ax.invert_yaxis()

    if show:
        fig.show()
