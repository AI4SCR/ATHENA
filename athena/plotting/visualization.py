# %% imports
import logging
from ..utils.general import is_numeric, is_categorical, make_iterable
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, NoNorm, to_rgba, to_rgb, ListedColormap
import seaborn as sns
import pandas as pd
# import matplotlib.ticker as mticker  # https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator
import numpy as np
import napari
# %% figure constants
from .utils import dpi, label_fontdict, title_fontdict, make_cbar, savefig


def spatial(so, spl: str, attr: str, *, mode: str = 'scatter', node_size: float = 4, coordinate_keys: list = ['x', 'y'],
            mask_key: str = 'cellmasks', graph_key: str = 'knn', edges: bool = False, edge_width: float = .5,
            edge_color: str = 'black', edge_zorder: int = 2, background_color: str = 'white', ax: plt.Axes = None,
            norm=None, set_title: bool = True, cmap=None, cmap_labels: list = None, cbar: bool = True,
            cbar_title: bool = True, show: bool = True, save: str = None, tight_layout: bool = True,
            filter_col: str = None, labels: list = None):
    """Various functionalities to visualise samples.
    Allows to visualise the samples and color observations according to features in either so.X or so.obs by setting the ``attr`` parameter accordingly.
    Furthermore, observations (cells) within a sample can be quickly visualised using scatter plots (requires extraction of centroids with :func:`~.extract_centroids`)
    or by their actual segmentatio mask by setting ``mode`` accordingly.
    Finally, the graph representation of the sample can be overlaid by setting ``edges=True`` and specifying the ``graph_key`` as in ``so.G[spl][graph_key]``.

    For more examples on how to use this function have a look at the tutorial_ section.

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
        filter_col: string identifying the column in so.obs to use to filter out cells that should not be plotted.
        labels: list of strings that identify the cells that should be included in the plot. These should be entries in filter_col.

    Examples:

        .. code-block:: python

            so = sh.dataset.imc()
            sh.pl.spatial(so, 'slide_7_Cy2x2', 'meta_id', mode='mask')
            sh.pl.spatial(so, 'slide_7_Cy2x2', 'meta_id', mode='scatter', edges=True)


    .. _tutorial: https://ai4scr.github.io/ATHENA/source/tutorial.html
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

    # filter cells
    if filter_col is not None:
        data = so.obs[spl].query(f'{filter_col} in @labels')[attr]
        loc = so.obs[spl].query(f'{filter_col} in @labels')[coordinate_keys].copy()
    else:
        loc = so.obs[spl][coordinate_keys].copy()

    # broadcast if necessary
    _is_categorical_flag = is_categorical(data)

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
            no_na = data[~np.isnan(data)]
            _ = norm(no_na)  # initialise norm with no NA data.

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


def napari_viewer(so, spl: str, attrs: list, censor: float = .95, add_masks='cellmasks', attrs_key='target', index_key: str = 'fullstack_index'):
    """Starts interactive Napari viewer to visualise raw images and explore samples.
    ``attrs`` are measured features in the high dimensional images in ``so.images[spl]``.
    All specified ``attrs`` should be in ``so.var[spl][attrs_key]`` along with the index in the high dimensional images.
    The column with the index in the high dimensional image where the measurement of an attribute is stored.

    Args:
        so: SpatialOmics instance
        spl: sample to visualise
        attrs: list of attributes/features to add as channels to the viewer
        censor: percentil to use to censore pixle values in the raw images
        add_masks: segmentation masks to add as channels to the viewer
        attrs_key: key in ``so.var[spl]`` that defines the ``attrs`` names
        index_key: key in ``so.var[spl]`` that specifies the layer index in the high dimensional image in ``so.images[spl]``.


    Examples:

    .. code-block:: python

        so = sh.dataset.imc()
        spl = so.spl.index[0]

        # add all measured features to the napari viewer
        sh.pl.napari_viewer(so, spl, attrs=so.var[spl]['target'], add_masks=so.masks[spl].keys())

        # specify the column containing the ``attrs`` names
        sh.pl.napari_viewer(so, spl, attrs=so.var[spl]['target'], attrs_key='target')

        # specify the column containing the ``attrs`` names and the columns that specifies the layer index
        sh.pl.napari_viewer(so, spl, attrs=so.var[spl]['target'], attrs_key='target', index_key='fullstack_index')

    """
    attrs = list(make_iterable(attrs))
    var = so.var[spl]
    index = var[var[attrs_key].isin(attrs)][index_key]
    names = var[var[attrs_key].isin(attrs)][attrs_key]

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


def _channel(so, spl: str, attrs: str, ax=None, colors=None, censor: float = None, show=True):
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
    i = var[var.target.isin(attrs)][index_key]

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
                 norm=None, ax=None, show=True, cbar=True):
    """Visualise results from :func:`~neigh.interactions` results.

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

    sns.heatmap(data=data, cmap=cmap, norm=norm, ax=ax, linewidths=linewidths, cbar=cbar)
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


def ripleysK(so, spl: str, attr: str, ids, *, mode='K', correction='ripley',
             key=None, ax=None, legend='auto'):
    """Visualise results from :func:`~neigh.ripleysK` results.

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        ids: The category in the categorical feature `attr`, for which Ripley's K should be plotted
        mode: {K, csr-deviation}. If `K`, Ripley's K is estimated, with `csr-deviation` the deviation from a poission process is computed.
        correction: Correction method to use to correct for boarder effects, see [1].
        key: key to use in so.uns['ripleysK'] for the plot, if None it is constructed from spl,attr,ids,mode and correction
        ax: axes to use for the plot

    Examples:

        .. code-block:: python

            # compute Ripley's K
            so = sh.dataset.imc()
            sh.neigh.ripleysK(so, so.spl.index[0], 'meta_id', 1, mode='csr-deviation', radii=radii)
            sh.pl.ripleysK(so, so.spl.index[0], 'meta_id', [1], mode='csr-deviation')

    """

    if key is None:
        if isinstance(ids, list):
            keys = [f'{i}_{attr}_{mode}_{correction}' for i in ids]
        else:
            keys = [f'{ids}_{attr}_{mode}_{correction}']
    else:
        keys = [key]

    res = []
    for i in keys:
        res.append(so.uns[spl]['ripleysK'][i])

    res = pd.concat(res, 1)
    if key is None:
        colnames = [i.split('_')[0] for i in keys]
    else:
        colnames = keys
    res = res.reset_index()
    res.columns = ['radii'] + colnames
    radii = res.radii.values
    res = res.melt(id_vars='radii', var_name=attr)
    res[attr] = res[attr].astype('category')

    cmap, labels = get_cmap(so, attr, res[attr])
    cmap_dict = {j: i for i, j in zip(cmap.colors, labels.values())}
    if labels:
        res[attr] = res[attr].astype(type(list(labels.keys())[0])).map(labels)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    sns.lineplot(data=res, x='radii', y='value', hue=attr, palette=cmap_dict, ax=ax, legend=legend)
    ax.plot(radii, np.repeat(0, len(radii)), color='black', linestyle=':')
    fig.tight_layout()
    fig.show()


def infiltration(so, spl: str, attr: str = 'infiltration', step_size: int = 10,
                 interpolation: str = 'gaussian',
                 cmap: str = 'plasma',
                 collision_strategy='mean',
                 ax=None,
                 show=True):
    """Visualises a heatmap of the featuer intensity.

    Approximates the sample with a grid representation and colors each grid square according to the
    value of the attribute. If multiple observations map to the same grid square a the aggregation specified
    in `collision_strategy` is employed (any value accepted by pandas aggregate function, i.e. 'mean', 'max', ...)

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: feature in SpatialOmics.obs to plot
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

    dat = so.obs[spl][[attr] + ['x', 'y']]
    dat = dat[~dat.infiltration.isna()]

    # we add step_size to prevent out of bounds indexing should the `{x,y}_img` values be rounded up.
    x, y = np.arange(0, so.images[spl].shape[2], step_size), np.arange(0, so.images[spl].shape[1], step_size)
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
