#%%
import spatialHeterogeneity as sh
from spatialOmics import SpatialOmics
from pathlib import Path
from bench.imc_data_athena import IMCData
import pandas as pd
#%%
root = Path('/Users/art/Library/CloudStorage/Box-Box/documents/thesis/data/SingleCellPathologyLandscapeOfBreastCancer/')

#%%
b = IMCData.from_pickle('ad_basel_contact.pkl',cohort='basel',root=root)
z = IMCData.from_pickle('ad_zurich_contact.pkl',cohort='zurich',root=root)

b.set_root(str(root) + 'basel')
z.set_root(str(root) + 'zurich')

b.meta.file_fullstack = b.meta.file_fullstack.map(lambda x: x.replace('/home/ubuntu/thesis/data/SingleCellPathologyLandscapeOfBreastCancer/', str(root)+'/'))
b.meta.file_cellmask = b.meta.file_cellmask.map(lambda x: x.replace('/home/ubuntu/thesis/data/SingleCellPathologyLandscapeOfBreastCancer/', str(root)+'/'))
b.meta.file_tumor_stroma_mask = b.meta.file_tumor_stroma_mask.map(lambda x: x.replace('/home/ubuntu/thesis/data/SingleCellPathologyLandscapeOfBreastCancer/', str(root)+'/'))

z.meta.file_fullstack = z.meta.file_fullstack.map(lambda x: x.replace('/home/ubuntu/thesis/data/SingleCellPathologyLandscapeOfBreastCancer/', str(root)+'/'))
z.meta.file_cellmask = z.meta.file_cellmask.map(lambda x: x.replace('/home/ubuntu/thesis/data/SingleCellPathologyLandscapeOfBreastCancer/', str(root)+'/'))
z.meta.file_tumor_stroma_mask = z.meta.file_tumor_stroma_mask.map(lambda x: x.replace('/home/ubuntu/thesis/data/SingleCellPathologyLandscapeOfBreastCancer/', str(root)+'/'))

#%% add images and masks
b.add_cellmasks()
#b.add_tiffstacks()
#b.add_tumor_stroma_mask()

z.add_cellmasks()
#z.add_tiffstacks()
#z.add_tumor_stroma_mask()

#%%
so = SpatialOmics()

so.obs.update(b.obs)
so.obs.update(z.obs)

so.var.update(b.var)
so.var.update(z.var)

so.X.update(b.X)
so.X.update(z.X)

so.spl = pd.concat((b.meta, z.meta))

for s in b.G.keys():
    so.G[s] = {'contact': b.G[s]}
for s in z.G.keys():
    so.G[s] = {'contact': z.G[s]}

for s in b.cellmasks.keys():
    so.masks[s] = {'cellmasks': b.cellmasks[s]}
    # so.masks[s] = {'tumor_stroma': b.tumor_stroma_mask[s]}
for s in z.cellmasks.keys():
    so.masks[s] = {'cellmasks': z.cellmasks[s]}
    # so.masks[s] = {'tumor_stroma': z.tumor_stroma_mask[s]}
sh.pl.spatial()
# so.images.update(b.tiffstacks)
# so.images.update(z.tiffstacks)

so.uns['cmaps'] = b.uns['cmaps']
so.uns['cmap_labels'] = b.uns['cmap_labels']

# %%
root = Path('/Users/art/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/')
so.to_h5py(root / 'imc_full.h5py')

import pickle
with open(root / 'imc_full.pkl', 'wb') as f:
    pickle.dump(so, f)

import pickle
with open('/Users/art/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/imc_full.pkl', 'rb') as f:
    so = pickle.load(f)