# %%
from spatialOmics import SpatialOmics
import pickle as pk
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm

# %%

path = Path('/Users/art/Box/internship_adriano/AI4Mechanisms/ATHENA-tutorial/data.pkl')
with open(path, 'rb') as f:
    data = pk.load(f)

print(data.keys())
spl = list(data['X'].keys())
print(data['G'][spl[0]].keys())

# %%
so = SpatialOmics()

# load meta data
spl = pd.read_csv(os.path.join(path.parent, 'meta_data.csv')).set_index('core')
spl = spl.replace('/Users/art/Documents/thesis/data', '/Users/art/Box/thesis/data', regex=True)

so.spl = spl

# samples we will work with
samples = ['slide_7_Cy2x2',
           'slide_7_Cy2x3',
           'slide_7_Cy2x4',
           'slide_59_Cy7x7',
           'slide_59_Cy7x8',
           'slide_59_Cy8x1',
           'SP43_116_X3Y4',
           'slide_49_By2x5']

# add image / mask data to instance
for s in tqdm(samples):
    so.add_image(s, so.spl.file_fullstack.loc[s], to_store=False)
    so.add_mask(s, 'cellmasks', so.spl.file_cellmask.loc[s], to_store=False)
    so.add_mask(s, 'tumor_stroma', so.spl.file_tumor_stroma_mask.loc[s], to_store=False)

    so.X[s] = data['X'][s]
    so.obs[s] = data['obs'][s]
    so.var[s] = data['var'][s]
    so.G[s] = data['G'][s]

del data

fpath = Path(os.path.expanduser('~/Documents/imc_dataset.h5py'))
so.to_h5py(fpath)

# %%
# import spatialHeterogeneity as sh

# high_var_samples = ['slide_7_Cy2x2', 'slide_7_Cy2x3', 'slide_7_Cy2x4', 'slide_7_Cy2x5']
# low_var_samples = ['slide_59_Cy7x7', 'slide_59_Cy7x8', 'slide_59_Cy8x1', 'slide_59_Cy8x2']
# spls = high_var_samples+low_var_samples
#
# for s in spls:
#     so.obs[s].drop(['x','y'], 1, inplace=True)
#     sh.pp.extract_centroids(so, s)
#
# from matplotlib.colors import ListedColormap
# cmap = ['white', 'darkgreen', 'gold', 'steelblue', 'darkred', 'coral']
# cmap_labels = {0:'background', 4:'tumor', 5:'myoepithelial', 1:'immune', 2:'endothelial', 3:'stromal'}
# cmap = ListedColormap(cmap)
#
# if 'cmaps' not in so.uns: so.uns['cmaps']={}
# if 'cmap_labels' not in so.uns: so.uns['cmap_labels']={}
# so.uns['cmaps'].update({'cell_type_id': cmap})
# so.uns['cmap_labels'].update({'cell_type_id': cmap_labels})
#
# for s in spls:
#     sh.pl.spatial(so, s, 'cell_type_id')