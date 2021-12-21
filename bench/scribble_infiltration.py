import spatialHeterogeneity as sh
from spatialHeterogeneity.neighborhood.utils import get_node_interactions
from networkx import Graph

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
import spatialHeterogeneity as sh
from spatialHeterogeneity.graph_builder.constants import GRAPH_BUILDER_DEFAULT_PARAMS
so = sh.dataset.imc()
spl = list(so.G.keys())[1]
attr = 'cell_type'

config = GRAPH_BUILDER_DEFAULT_PARAMS['radius']
config['builder_params']['radius'] = 50
sh.graph.build_graph(so, spl, builder_type='radius', config=config)
sh.neigh.infiltration(so, spl, attr, graph_key='radius', local=True)

# %%
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
import copy

cmap = ['white', 'darkgreen', 'gold', 'steelblue', 'darkred', 'coral']
cmap_labels = {0: 'background', 1: 'immune',  2: 'endothelial', 3: 'stromal', 4: 'tumor', 5: 'myoepithelial'}
cmap = ListedColormap(cmap)
so.uns['cmaps'].update({'cell_type_id': cmap})
so.uns['cmap_labels'].update({'cell_type_id': cmap_labels})

cmap = copy.copy(get_cmap('BuGn'))
cmap = copy.copy(get_cmap('plasma'))
cmap.set_bad('gray')
so.uns['cmaps']['infiltration'] = cmap

sh.pp.extract_centroids(so, spl)

radi = [20, 36, 50]
fig, axs = plt.subplots(2,2, figsize=(8,8), dpi=300)
sh.pl.spatial(so, spl, attr='cell_type_id', ax=axs[0,0])
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