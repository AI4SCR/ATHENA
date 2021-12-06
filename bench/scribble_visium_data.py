# %%
import pandas as pd
import squidpy as sq
import scanpy as sc
from spatialOmics import SpatialOmics
import spatialHeterogeneity as sh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
ad = sq.datasets.visium_fluo_adata_crop()
img = sq.datasets.visium_fluo_image_crop()

# %%
sc.pl.spatial(ad, color="cluster")
sc.pl.spatial(ad, color="leiden")

img.show(channelwise=True)

# %% image segmentation
sq.im.process(
    img=img,
    layer="image",
    method="smooth",
)

sq.im.segment(img=img, layer="image_smooth", method="watershed", channel=0, chunks=1000)
np.unique(img['segmented_watershed'])

# plot the resulting segmentation
fig, ax = plt.subplots(1, 2)
img_crop = img.crop_corner(2000, 2000, size=500)
img_crop.show(layer="image", channel=0, ax=ax[0])
img_crop.show(
    layer="segmented_watershed",
    channel=0,
    ax=ax[1],
)
plt.show()

fig, ax = plt.subplots(1, 2)
img.show(layer="image", channel=0, ax=ax[0])
img.show(
    layer="segmented_watershed",
    channel=0,
    ax=ax[1],
)
fig.show()

# %%
so = SpatialOmics()

# %%
spl = 'mouseBrain'
X = pd.DataFrame(ad.X.A, index=ad.obs.index, columns=ad.var.index)

so.X.update({spl:X})
so.var.update({spl:ad.var})

obs = ad.obs[['array_row', 'array_col', 'cluster']]
obs.columns = ['y', 'x', 'cluster']
obs['group_id'] = obs.groupby('cluster').ngroup()
obs['group_id'] = obs.group_id.astype('category')

so.obs.update({spl:obs})
so.spl = pd.DataFrame(0, index=[spl], columns=['sampleID'])

# %%
from matplotlib.colors import ListedColormap


so.uns['cmaps'] = {'category': plt.get_cmap('Set3')}
so.uns['cmaps']['default'] = plt.get_cmap('gist_heat')
so.uns['cmap_labels'] = {}
sh.pl.spatial(so,spl, 'cluster')