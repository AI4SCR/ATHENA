import logging

import pandas as pd

import spatialHeterogeneity as sh
from spatialHeterogeneity.neighborhood.utils import get_node_interactions
from networkx import Graph
import numpy as np
import spatialHeterogeneity as sh
from spatialHeterogeneity.graph_builder.constants import GRAPH_BUILDER_DEFAULT_PARAMS
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
import copy
from skimage.measure import regionprops
# %%

so = sh.dataset.imc()
spl = list(so.G.keys())[0]
attr = 'cell_type'
G: Graph = so.G[spl]['contact']

node = 2
neigh = list(G[node])
g = G.subgraph(neigh)
data = so.obs[spl][attr]

nint = get_node_interactions(g, data)

# %%


so = sh.dataset.imc()
spl = list(so.G.keys())[1]
attr = 'cell_type'

sh.pp.extract_centroids(so, spl)
config = GRAPH_BUILDER_DEFAULT_PARAMS['radius']
config['builder_params']['radius'] = 36
sh.graph.build_graph(so, spl, builder_type='radius', config=config)
sh.neigh.infiltration(so, spl, attr, graph_key='radius', local=True)

# %%
cmap = ['white', 'darkgreen', 'gold', 'steelblue', 'darkred', 'coral']
cmap_labels = {0: 'background', 1: 'immune', 2: 'endothelial', 3: 'stromal', 4: 'tumor', 5: 'myoepithelial'}
cmap = ListedColormap(cmap)
so.uns['cmaps'].update({'cell_type_id': cmap})
so.uns['cmap_labels'].update({'cell_type_id': cmap_labels})

cmap = copy.copy(get_cmap('BuGn'))
cmap = copy.copy(get_cmap('plasma'))
cmap.set_bad('gray')
so.uns['cmaps']['infiltration'] = cmap

sh.pp.extract_centroids(so, spl)

radi = [20, 36, 50]
fig, axs = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
sh.pl.spatial(so, spl, attr='cell_type_id', ax=axs[0, 0])
for r, ax in zip(radi, axs.flat[1:]):
    config['builder_params']['radius'] = r
    sh.graph.build_graph(so, spl, builder_type='radius', config=config)
    sh.neigh.infiltration(so, spl, attr, graph_key='radius', local=True)
    sh.pl.spatial(so, spl, attr='infiltration', ax=ax, background_color='black')
fig.show()
fig.savefig('/Users/art/Downloads/infiltration_plasma.pdf')

# %%
config = GRAPH_BUILDER_DEFAULT_PARAMS['radius']
config['builder_params']['radius'] = 50
sh.graph.build_graph(so, spl, builder_type='radius', config=config)

G = so.G[spl]['radius']

# %%

dat: pd.DataFrame = so.obs[spl][['infiltration', 'x', 'y']]
dat = dat[~dat.infiltration.isna()]

# mask size
mask = so.masks[spl]['cellmasks']


rg = regionprops(mask)
areas = []
for r in rg:
    areas.append(r.area)

plt.hist(areas, bins=100)
plt.show()

step_size = 20
x, y = np.arange(0, mask.shape[1], step_size), np.arange(0, mask.shape[0], step_size)
xv, yv = np.meshgrid(mask.shape[1], mask.shape[0])
img = np.zeros((len(y), len(x)))

dat['x_img'] = np.round(dat.x / step_size).astype(int)
dat['y_img'] = np.round(dat.y / step_size).astype(int)

for i in range(dat.shape[0]):
    img[dat.y_img.iloc[i], dat.x_img.iloc[i]] = dat.infiltration.iloc[i]

methods = [None, 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

plt.imshow(img, interpolation='gaussian', cmap='plasma');
plt.show()


# %%

def infiltration(so, spl: str, attr: str ='infiltration', step_size: int = 10,
                 interpolation: str = 'gaussian',
                 cmap: str ='plasma',
                 collision_strategy='mean',
                 ax=None,
                 show=True):

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

    if show:
        fig.show()

# %%

infiltration(so, spl, step_size=20, collision_strategy='max')

# %%
attr = 'infiltration'
step_size = 10

dat = so.obs[spl][[attr] + ['x', 'y']]
dat = dat[~dat.infiltration.isna()]


# we add step_size to prevent out of bounds indexing should the `{x,y}_img` values be rounded up.
x, y = np.arange(0, mask.shape[1], step_size), np.arange(0, mask.shape[0], step_size)
img = np.zeros((len(y), len(x)))

# %%
import spatialHeterogeneity as sh
from spatialHeterogeneity.graph_builder.constants import GRAPH_BUILDER_DEFAULT_PARAMS
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

so = sh.dataset.imc()
spl = list(so.G.keys())[1]

cmap = ['white', 'darkgreen', 'gold', 'steelblue', 'darkred', 'coral']
cmap_labels = {0: 'background', 1: 'immune', 2: 'endothelial', 3: 'stromal', 4: 'tumor', 5: 'myoepithelial'}
cmap = ListedColormap(cmap)
so.uns['cmaps'].update({'cell_type_id': cmap})
so.uns['cmap_labels'].update({'cell_type_id': cmap_labels})

sh.pp.extract_centroids(so, spl)

config = GRAPH_BUILDER_DEFAULT_PARAMS['radius']
config['builder_params']['radius'] = 36
sh.graph.build_graph(so, spl, builder_type='radius', config=config)

attr = 'cell_type'
sh.neigh.infiltration(so, spl, attr, graph_key='radius', local=True)

step_sizes = [5, 10, 20]
fig, axs = plt.subplots(2,2, dpi=300)
sh.pl.spatial(so, spl, 'cell_type_id', ax=axs[0,0])
for ax, step_size in zip(axs.flat[1:], step_sizes):
    ax: plt.Axes
    sh.pl.infiltration(so, spl, step_size=step_size, ax=ax)
    ax.set_title(f'step_size = {step_size}')
    ax.set_axis_off()
fig.show()
fig.savefig('/Users/art/Downloads/infiltration.pdf')

methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6), dpi=300,
                        subplot_kw={'xticks': [], 'yticks': []})
step_size = 20
for ax, interpolation in zip(axs.flat, methods):
    sh.pl.infiltration(so, spl, step_size=step_size, interpolation=interpolation, ax=ax)
    ax.set_title(interpolation)
fig.show()
fig.savefig('/Users/art/Downloads/infiltration_interpolation.pdf')