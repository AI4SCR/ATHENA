import spatialHeterogeneity as sh
import pickle as pk
import os
import matplotlib.pyplot as plt

hdf = '/Users/art/Documents/spatial-omics/spatialOmics.hdf5'
f = '/Users/art/Documents/spatial-omics/spatialOmics.pkl'
# so = SpatialOmics.form_h5py(f)
with open(f, 'rb') as f:
    so = pk.load(f)

# with open(f, 'wb') as file:
#     pk.dump(so, file)

so.h5py_file = hdf
so.spl_keys = list(so.X.keys())
spl = so.spl_keys[0]

sh.pl.spatial(so,spl, 'meta_id')
sh.pl.spatial(so,spl, 'meta_id', mode='mask')
fig, ax = plt.subplots()
sh.pl.spatial(so,spl, 'meta_id', mode='mask', ax=ax)
fig.show()
sh.pl.spatial(so,spl, 'EGFR')
sh.pl.spatial(so,spl, 'EGFR', mode='mask')

sh.pl.spatial(so,spl, 'meta_id', edges=True)
sh.pl.spatial(so,spl, 'meta_id', mode='mask', edges=True)

# %%
sh.pl.napari_viewer(so, spl, ['DNA2', 'EGFR', 'H3'], add_masks='cellmask')

#%%

import numpy as np
import napari
from skimage import data

viewer = napari.view_image(data.astronaut(), rgb=True)

img = [so.images[spl][i,...] for i in [8,39]]
a0 = np.stack(img, axis=0)
a2 = np.stack(img, axis=2)

img = so.images[spl][[8,39]]
mask = so.masks[spl]['cellmask']
viewer = napari.view_image(img[0,], name='adsf')
labels_layer = viewer.add_labels(mask, name='assd')

viewer = napari.Viewer()
viewer.add_image(img, channel_axis=0, name=['Ich', 'Du'])
labels_layer = viewer.add_labels(mask, name='assd')

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(data.astronaut())

