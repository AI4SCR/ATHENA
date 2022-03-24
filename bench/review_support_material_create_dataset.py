# %%
import spatialHeterogeneity as sh
from spatialOmics import SpatialOmics
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np

# %%
f_imc = Path('~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/imc_full.pkl').expanduser()
with open(f_imc, 'rb') as f:
    imc = pickle.load(f)

# f_mibi = Path('~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/mibi-final-no-images.h5py').expanduser()
# mibi = SpatialOmics.from_h5py(f_mibi)

# %% cell count statistics
isImmune = list(range(7))
mapping = {i:0 for i in isImmune}
isTumor = list(range(14,27))
mapping.update({i:1 for i in isTumor})

for i in imc.spl.index:
    imc.obs[i]['tumorYN'] = imc.obs[i].meta_id.map(mapping).fillna(2).astype(int).astype('category')

for i in imc.spl.index:
    imc.spl.loc[i, 'nCells'] = len(imc.obs[i])
    n = imc.spl.loc[i, 'nImmuneCells'] = (imc.obs[i].tumorYN == 0).sum()
    imc.spl.loc[i, 'nTumorCells'] = (imc.obs[i].tumorYN == 1).sum()

# %%  FIND SUITABLE CORES TO SHOW

spl = imc.spl
spl['frac_immune'] = spl.nImmuneCells / spl.nCells
spl['frac_tumor'] = spl.nTumorCells / spl.nCells

spl = imc.spl
sel = spl[(spl.nCells > 1500) & (spl.frac_immune > .15) & (spl.frac_tumor > 0.15)]
sel = sel[sel.nImmuneCells > 250*1500/2000] # exclude are cold tumors (i.e. only few immune cells)

# %% compute infiltration
for spl in tqdm(sel.index):
    sh.neigh.infiltration(imc, spl, 'tumorYN', interaction1=(1, 0), interaction2=(0, 0), graph_key='contact')

from spatialHeterogeneity.graph_builder.constants import GRAPH_BUILDER_DEFAULT_PARAMS
params = GRAPH_BUILDER_DEFAULT_PARAMS['radius']
params['builder_params']['radius'] = 36
for spl in tqdm(sel.index):
    sh.graph.build_graph(imc, spl, 'radius')
    sh.neigh.infiltration(imc, spl, 'tumorYN', interaction1=(1, 0), interaction2=(0, 0), graph_key='radius', local=True)

# %% compute modularity, entropy
for spl in tqdm(sel.index):
    sh.metrics.graph.modularity(imc, spl, 'tumorYN', graph_key='contact')
    sh.metrics.shannon(imc, spl, 'tumorYN', graph_key='contact', local=False)
    sh.metrics.quadratic_entropy(imc, spl, 'tumorYN', graph_key='contact', local=False)

for spl in tqdm(sel.index):
    sh.metrics.shannon(imc, spl, 'tumorYN', graph_key='contact', local=True)
    sh.metrics.quadratic_entropy(imc, spl, 'tumorYN', graph_key='contact', local=True)

for spl in tqdm(sel.index):
    sh.metrics.shannon(imc, spl, 'meta_id', graph_key='contact', local=False)
    sh.metrics.shannon(imc, spl, 'meta_id', graph_key='contact', local=True)

params['builder_params']['radius'] = 20
for spl in tqdm(sel.index):
    sh.metrics.shannon(imc, spl, 'meta_id', graph_key='contact', local=False)

    sh.metrics.shannon(imc, spl, 'meta_id', graph_key='contact', local=True)
    sh.graph.build_graph(imc, spl, 'radius', config=params)
    sh.metrics.shannon(imc, spl, 'meta_id', graph_key='radius', local=True)

params['builder_params']['radius'] = 20
for spl in tqdm(sel.index):
    sh.metrics.shannon(imc, spl, 'tissue_type', graph_key='contact', local=False)

    sh.metrics.shannon(imc, spl, 'tissue_type', graph_key='contact', local=True)
    sh.graph.build_graph(imc, spl, 'radius', config=params)
    sh.metrics.shannon(imc, spl, 'tissue_type', graph_key='radius', local=True)

# %%
from matplotlib.colors import ListedColormap
cmap = plt.get_cmap('Paired')
newCmap = []
newCmap.extend([cmap(i) for i in (2,4,0)])
newCmap = ListedColormap(newCmap)
imc.uns['cmaps']['tumorYN'] = newCmap
imc.uns['cmap_labels']['tumorYN'] = {0:'immune', 1: 'tumor', 2: 'other'}

# %%
with open(f_imc, 'wb') as f:
    pickle.dump(imc, f)

# %% interactions
inter = ['SP41_191_X15Y7']
loc_inter = ['SP41_220_X10Y5']
comp = ['SP43_116_X3Y4']
spls = inter + loc_inter + comp

for spl in spls:
    sh.neigh.interactions(imc, spl, 'meta_id', mode='proportion', graph_key='contact')

# %%
isImmune = list(range(7))
isEndothelial = [7]
isStromal = list(range(8,14))
isTumor = list(range(14,27))

mapping = {i:0 for i in isImmune}
mapping.update({i:1 for i in isEndothelial})
mapping.update({i:2 for i in isStromal})
mapping.update({i:3 for i in isTumor})

for i in imc.spl.index:
    imc.obs[i]['tissue_type'] = imc.obs[i].meta_id.map(mapping).fillna(2).astype(int).astype('category')

from matplotlib.colors import ListedColormap
cmap = plt.get_cmap('Paired')
newCmap = []
newCmap.extend([cmap(i) for i in (2,10,0,4,)])
newCmap = ListedColormap(newCmap)
imc.uns['cmaps']['tissue_type'] = newCmap
imc.uns['cmap_labels']['tissue_type'] = {0:'immune', 1: 'endothelial', 2: 'stromal', 3:'tumor'}

with open(f_imc, 'wb') as f:
    pickle.dump(imc, f)