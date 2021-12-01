
from spatialOmics.spatialOmics import SpatialOmics
import spatialHeterogeneity as sh
import pickle as pk
import os

# %%

f = '/Users/art/Documents/spatial-omics/spatialOmics.hdf5'
f = '/Users/art/Documents/spatial-omics/spatialOmics.pkl'
# so = SpatialOmics.form_h5py(f)
with open(f, 'rb') as f:
    so = pk.load(f)

so.spl_keys = list(so.X.keys())
spl = so.spl_keys[0]

# sh.metrics.richness(so, spl, 'meta_id')
# sh.metrics.richness(so, spl, 'meta_id', local=False)
# sh.metrics.abundance(so, spl, 'meta_id', local=False)
# sh.metrics.abundance(so, spl, 'meta_id', local=True)
# sh.metrics.shannon(so, spl, 'meta_id')
# sh.metrics.shannon(so, spl, 'meta_id', local=False)
# sh.metrics.simpson(so, spl, 'meta_id')
# sh.metrics.simpson(so, spl, 'meta_id', local=False)
# sh.metrics.hill_number(so, spl, 'meta_id', q=2)
# sh.metrics.hill_number(so, spl, 'meta_id', q=2, local=False)
# sh.metrics.renyi_entropy(so, spl, 'meta_id', q=1)
# sh.metrics.renyi_entropy(so, spl, 'meta_id', q=2)
# sh.metrics.renyi_entropy(so, spl, 'meta_id', q=2, local=False)

# %%
import spatialHeterogeneity as sh

# %%
so = sh.dataset.imc()
spl = list(so.obs.keys())[0]
sh.metrics.shannon(so, spl, 'meta_id', graph_key='contact')
so.obs[spl].columns
sh.metrics.shannon(so, spl, 'meta_id', graph_key='contact')
so.obs[spl].columns

sh.metrics.shannon(so, spl, 'meta_id', local=False)
so.spl.columns
sh.metrics.shannon(so, spl, 'meta_id', local=False)
so.spl.columns

