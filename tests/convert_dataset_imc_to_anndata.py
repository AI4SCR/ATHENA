# %%
from anndata import AnnData
import athena as ath

# %%
so = ath.dataset.imc()

# %%
from networkx import to_scipy_sparse_array
sample_names = so.G.keys()
dataset = {}
for sample_name in sample_names:
    x = so.X[sample_name]
    obs = so.obs[sample_name]
    mask = so.masks[sample_name]['cellmasks']
    var = so.var[sample_name].set_index('target')
    metadata = so.spl.loc[sample_name]

    x, obs = x.align(obs, axis=0)

    obsp = {}
    for topology in ['knn', 'radius', 'contact']:
        adj = to_scipy_sparse_array(so.G[sample_name][topology], nodelist=x.index)
        obsp[topology] = adj

    x.index = x.index.astype(str)
    obs.index = obs.index.astype(str)

    ad = AnnData(X=x.values, obs=obs, var=var, obsp=obsp)
    ad.uns['mask'] = mask
    ad.uns['metadata'] = metadata

    dataset[sample_name] = ad

# %%
import pickle
from pathlib import Path
with open(Path('~/Downloads/imc.pkl').expanduser(), 'wb') as f:
    pickle.dump(dataset, file=f)
