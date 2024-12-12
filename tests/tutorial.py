import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

import athena as ath

pd.set_option('display.max_columns', 5)

# ad = ath.dataset.imc(force_download=True)
data = ath.dataset.imc()
data

sample_name = list(data.keys())[0]
ad = data[sample_name]

print(ad.uns['metadata'])

print(ad.obs.columns)  # see all available sample annotations
ad.obs.head(3)

print(ad.uns['mask'].shape)
assert set(ad.uns['mask'].flatten()) - set([0]) == set(ad.obs.index.astype(int))
plt.imshow(ad.uns['mask'] > 0, cmap='gray')

import colorcet

# set up colormaps for meta_label
cmap_meta_label = {
    'Background': (1.0, 1.0, 1.0),
    'B cells': (0.0392156862745098, 0.5529411764705883, 0.25882352941176473),
    'B and T cells': (0.24313725490196078,
                      0.7098039215686275,
                      0.33725490196078434),
    'T cell': (0.7058823529411765, 0.8313725490196079, 0.19607843137254902),
    'Macrophages': (0.09019607843137255, 0.396078431372549, 0.21176470588235294),
    'Endothelial': (0.9725490196078431, 0.9098039215686274, 0.050980392156862744),
    'Vimentin Hi': (0.00392156862745098, 0.9686274509803922, 0.9882352941176471),
    'Small circular': (0.7450980392156863,
                       0.9882352941176471,
                       0.9882352941176471),
    'Small elongated': (0.5882352941176471,
                        0.8235294117647058,
                        0.8823529411764706),
    'Fibronectin Hi': (0.592156862745098, 0.996078431372549, 1.0),
    'Larger elongated': (0.0, 1.0, 0.9803921568627451),
    'SMA Hi Vimentin': (0.6039215686274509,
                        0.9568627450980393,
                        0.9921568627450981),
    'Hypoxic': (0.07450980392156863, 0.2980392156862745, 0.5647058823529412),
    'Apopotic': (0.0, 0.00784313725490196, 0.984313725490196),
    'Proliferative': (0.5764705882352941, 0.8352941176470589, 0.7764705882352941),
    'p53 EGFR': (0.2627450980392157, 0.5490196078431373, 0.4470588235294118),
    'Basal CK': (0.9333333333333333, 0.27450980392156865, 0.2784313725490196),
    'CK7 CK hi Cadherin': (0.3137254901960784,
                           0.17647058823529413,
                           0.5607843137254902),
    'CK7 CK': (0.5294117647058824, 0.2980392156862745, 0.6196078431372549),
    'Epithelial low': (0.9803921568627451,
                       0.7019607843137254,
                       0.7647058823529411),
    'CK low HR low': (0.9647058823529412, 0.4235294117647059, 0.7019607843137254),
    'HR hi CK': (0.8117647058823529, 0.403921568627451, 0.16862745098039217),
    'CK HR': (0.8784313725490196, 0.6352941176470588, 0.00784313725490196),
    'HR low CK': (0.9647058823529412, 0.5137254901960784, 0.43137254901960786),
    'Myoepithalial': (0.5294117647058824,
                      0.3607843137254902,
                      0.16862745098039217),
    'CK low HR hi p53': (0.6980392156862745,
                         0.12941176470588237,
                         0.10980392156862745)}

# cell_type colormap
cmap_cell_type = {
    'background': 'white',
    'immune': 'darkgreen',
    'endothelial': 'gold',
    'stromal': 'steelblue',
    'tumor': 'darkred',
    'myoepithelial': 'coral'
}

for ad in data.values():
    # define default colormap
    ad.uns['cmaps'] = {}
    ad.uns['cmaps'].update({'default': cm.Reds})
    ad.uns['cmaps']['category'] = colorcet.glasbey_bw
    ad.uns['cmaps'].update({'meta_label': cmap_meta_label})
    ad.uns['cmaps'].update({'cell_type': cmap_cell_type})
    ad.uns['cmap_labels'] = {}

from skimage.measure import regionprops_table
import pandas as pd

for spl in data.keys():
    ad = data[spl]
    centroids = pd.DataFrame(regionprops_table(ad.uns['mask'], properties=('label', 'centroid')))
    centroids.columns = ['object_id', 'y', 'x']
    centroids = centroids.set_index('object_id')
    centroids.index = centroids.index.astype(str)  # we need to convert to `str` to match the index of `ad.obs`
    ad.obs = pd.concat((ad.obs, centroids), axis=1)

# # %%
# spl = 'slide_7_Cy2x4'
# fig, axs = plt.subplots(2, 3, figsize=(15, 6), dpi=300)
# ath.pl.spatial(ad=ad, attr='cell_type', ax=axs.flat[0])
# ath.pl.spatial(ad=ad, attr='cell_type', mode='mask', ax=axs.flat[1])
# ath.pl.spatial(ad=ad, attr='meta_label', mode='mask', ax=axs.flat[2])
# ath.pl.spatial(ad=ad, attr='CytokeratinPan', mode='mask', ax=axs.flat[3])
# ath.pl.spatial(ad=ad, attr='Cytokeratin5', mode='mask', ax=axs.flat[4])
# ath.pl.spatial(ad=ad, attr='SMA', mode='mask', ax=axs.flat[5])
# fig.show()
#
# # %%
# spl = 'slide_49_By2x5'
# ad = data[spl]
#
# ad.uns['cmaps'].update({'default': cm.plasma})
# fig, axs = plt.subplots(1, 3, figsize=(15, 3), dpi=100)
# ath.pl.spatial(ad=ad, attr='CytokeratinPan', mode='mask', ax=axs.flat[0], background_color='black')
# ath.pl.spatial(ad=ad, attr='Cytokeratin5', mode='mask', ax=axs.flat[1], background_color='black')
# ath.pl.spatial(ad=ad, attr='SMA', mode='mask', ax=axs.flat[2], background_color='black')
# fig.show()
#
# # %%
# # We provide some default topologies. Under the hood we use the sklearn.neighbors implementation.
# # This means you can pass additional arguments to the function according to the sklearn.neighbors documentation.
# mask = ad.uns['mask']
# _ = ath.graph.build_graph(ad, topology='knn', n_neighbors=6, copy=True)
# _ = ath.graph.build_graph(ad, topology='knn', graph_key='knn_n10', n_neighbors=10, include_self=True, copy=True)
# ath.graph.build_graph(ad, topology='radius', radius=36, include_self=True)
# _ = ath.graph.build_graph(ad, topology='contact', include_self=True, copy=True)
#
# # %%
# fig, axs = plt.subplots(1, 3, figsize=(15, 6), dpi=100)
# for topology, ax in zip(['knn', 'radius', 'contact'], axs):
#     ath.pl.spatial(ad=ad, attr='meta_id', edges=True, graph_key=topology, ax=ax, cbar=False)
#     ax.set_title(topology)
# fig.show()
#
# # %%
# # compute cell counts
# ad.uns['metadata']['cell_count']=len(ad.obs)
# ad.uns['metadata']['immune_cell_count']=(ad.obs.cell_type == 'immune').sum()
#
# # compute metrics at a sample level
# for spl in data.keys():
#     ad = data[spl]
#     ath.metrics.richness(ad=ad, attr='meta_label', local=False)
#     ath.metrics.shannon(ad=ad, attr='meta_label', local=False)
#
# # estimated values are saved in ad.uns
# sample_level_scores = pd.DataFrame()
# for spl in data.keys():
#     ad = data[spl]
#     df = pd.DataFrame(
#         {'richness_meta_label': ad.uns['richness_meta_label'],
#          'shannon_meta_label': ad.uns['shannon_meta_label']}, index=[spl])
#     sample_level_scores = pd.concat((sample_level_scores, df))
# sample_level_scores
#
# # %%
# fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
# ath.pl.spatial(ad=data['slide_7_Cy2x2'], attr='meta_label', mode='mask', ax=axs.flat[0])
# ath.pl.spatial(ad=data['slide_7_Cy2x4'], attr='meta_label', mode='mask', ax=axs.flat[1])
# fig.show()
#
# # %%
# import seaborn as sns
# fig = plt.figure(figsize=(15, 3))
# plt.subplot(1, 2, 1)
# sns.histplot(data['slide_7_Cy2x4'].obs['meta_label'])
# plt.xticks(rotation=90)
# plt.subplot(1, 2, 2)
# sns.histplot(data['slide_7_Cy2x2'].obs['meta_label'])
# plt.xticks(rotation=90)
# plt.show()

# # %%
# from tqdm import tqdm
#
# #
# # # compute metrics at a cell level for all samples - this will take some time
# for spl in tqdm(data.keys()):
#     ath.metrics.richness(ad=data[spl], attr='meta_label', local=True, graph_key='contact')
#     ath.metrics.shannon(ad=data[spl], attr='meta_label', local=True, graph_key='contact')
#     ath.metrics.quadratic_entropy(ad=data[spl], attr='meta_label', local=True, graph_key='contact', metric='cosine')
#
# # estimated values are saved in so.obs
# ad.obs.columns
#
# # %%
# # visualize cell-level scores
# spl = 'slide_49_By2x5'
# ad = data[spl]

# ad.uns['cmaps'].update({'default': cm.plasma})
# fig, axs = plt.subplots(1, 4, figsize=(25, 12), dpi=300)
# axs = axs.flat
# ath.pl.spatial(ad=ad, attr='meta_label', mode='mask', ax=axs[0], title=spl)
# ath.pl.spatial(ad=ad, attr='richness_meta_label_contact', mode='mask', ax=axs[1], cbar_title=False,
#                background_color='black', title=spl)
# ath.pl.spatial(ad=ad, attr='shannon_meta_label_contact', mode='mask', ax=axs[2], cbar_title=False,
#                background_color='black', title=spl)
# ath.pl.spatial(ad=ad, attr='quadratic_meta_label_contact', mode='mask', ax=axs[3], cbar_title=False,
#                background_color='black', title=spl)
# fig.show()

# # %%
# spl = 'slide_49_By2x5'
# ad = data[spl]
# ad.uns['cmaps'].update({'default': cm.plasma})
#
# # try out different graph topologies
# ath.metrics.quadratic_entropy(ad=ad, attr='meta_label', local=True, graph_key='contact', metric='cosine')
# ath.metrics.quadratic_entropy(ad=ad, attr='meta_label', local=True, graph_key='radius')
# ath.metrics.quadratic_entropy(ad=ad, attr='meta_label', local=True, graph_key='knn')
#
# # visualize results
# fig, axs = plt.subplots(1, 4, figsize=(25, 12), dpi=300)
# axs = axs.flat
# ath.pl.spatial(ad=ad, attr='meta_label', mode='mask', ax=axs[0])
# ath.pl.spatial(ad=ad, attr='quadratic_meta_label_contact', mode='mask', ax=axs[1], cbar_title=False,
#                background_color='black')
# ath.pl.spatial(ad=ad, attr='quadratic_meta_label_radius', mode='mask', ax=axs[2], cbar_title=False,
#                background_color='black')
# ath.pl.spatial(ad=ad, attr='quadratic_meta_label_knn', mode='mask', ax=axs[3], cbar_title=False,
#                background_color='black')
# fig.show()
#
# # %%
# import seaborn as sns
#
# fig = plt.figure(figsize=(25, 12))
# for i, spl in enumerate(data.keys()):
#     plt.subplot(2, 4, i + 1)
#     g = sns.histplot(data[spl].obs['quadratic_meta_label_contact'], stat='probability')
#     g.set_title(
#         spl + ', median quad entropy = ' + str(round(data[spl].obs['quadratic_meta_label_contact'].median(), 3)))
#     plt.ylim([0, 0.32])
#     plt.xlim([0, 1])
# fig.show()
#
# # %%
# fig, axs = plt.subplots(1, 4, figsize=(20, 8), dpi=100)
# ath.pl.spatial(ad=data['slide_7_Cy2x4'], attr='meta_label', mode='mask', ax=axs.flat[0])
# ath.pl.spatial(ad=data['slide_7_Cy2x4'], attr='quadratic_meta_label_contact', mode='mask', ax=axs.flat[1], cbar=False,
#                background_color='black')
# ath.pl.spatial(ad=data['slide_59_Cy7x7'], attr='meta_label', mode='mask', ax=axs.flat[2])
# ath.pl.spatial(ad=data['slide_59_Cy7x7'], attr='quadratic_meta_label_contact', mode='mask', ax=axs.flat[3], cbar=False,
#                background_color='black')
# fig.show()
#
# # %%
# data[spl].obs.loc[:, ['richness_meta_label_contact',
#                       'shannon_meta_label_contact',
#                       'quadratic_meta_label_contact']].head(5)
#
# # %%
# infiltration = pd.DataFrame()
# for spl in tqdm(data.keys()):
#     ath.neigh.infiltration(ad=data[spl], attr='cell_type',
#                            interaction1=('tumor', 'immune'), interaction2=('immune', 'immune'),
#                            graph_key='contact')
#     val = data[spl].uns['infiltration']
#     pid = data[spl].uns['metadata']['pid']
#     infiltration.loc[spl, 'infiltration'] = val
#     infiltration.loc[spl, 'pid'] = int(pid)
#
# infiltration
#
# # %%
#
# infiltration = infiltration.sort_values('infiltration', ascending=True)
#
# fig, axs = plt.subplots(2, 4, figsize=(14, 7), dpi=300)
# for i, spl in enumerate(infiltration.index):
#     ath.pl.spatial(ad=data[spl], attr='cell_type', mode='mask', ax=axs.flat[i])
#     d = infiltration.loc[spl]
#     axs.flat[i].set_title(f'Patient {d.pid}, infiltration: {d.infiltration:.2f}', fontdict={'size': 10})
# axs.flat[-1].set_axis_off()
# fig.show()

# # %%
# spl = 'slide_49_By2x5'
# ath.graph.build_graph(data[spl], topology='radius', radius=36, include_self=True)
#
# attr = 'cell_type'
# ath.neigh.infiltration(ad=data[spl], attr=attr, graph_key='radius', local=True)
#
# # %%
# fig, axs = plt.subplots(1, 3, figsize=(16, 8))
# ath.pl.spatial(ad=data[spl], attr='cell_type', ax=axs[0])
# ath.pl.infiltration(ad=data[spl], step_size=10, ax=axs[1])
# ath.pl.infiltration(ad=data[spl], step_size=5, ax=axs[2])
# fig.show()

# %%
# import logging
#
# logging.getLogger().setLevel(logging.ERROR)  # set logger to logging.INFO if you want to see more progress information
#
# for spl in ['slide_7_Cy2x4', 'slide_7_Cy2x2']:
#     ath.neigh.interactions(ad=data[spl], attr='meta_label', mode='proportion', prediction_type='diff',
#                            graph_key='contact')
#     ath.neigh.interactions(ad=data[spl], attr='cell_type', mode='proportion', prediction_type='diff',
#                            graph_key='contact')

# %%
# from matplotlib.colors import Normalize
#
# norm = Normalize(-.3, .3)
# fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
# ath.pl.spatial(ad=data['slide_7_Cy2x4'], attr='meta_label', mode='mask', ax=axs[0])
# ath.pl.interactions(ad=data['slide_7_Cy2x4'], attr='meta_label', mode='proportion',
#                     prediction_type='diff', graph_key='contact',
#                     ax=axs[1], norm=norm)
# fig.tight_layout()
# fig.show()
#
# fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
# ath.pl.spatial(ad=data['slide_7_Cy2x2'], attr='meta_label', mode='mask', ax=axs[0])
# ath.pl.interactions(ad=data['slide_7_Cy2x2'], attr='meta_label',
#                     mode='proportion', prediction_type='diff', graph_key='contact', ax=axs[1], norm=norm)
#
# fig.tight_layout()
# fig.show()
#
# # %%
#
# fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
# ath.pl.spatial(ad=data['slide_7_Cy2x4'], attr='cell_type', mode='mask', ax=axs[0])
# norm = Normalize(-.3, .3)
# ath.neigh.interactions(ad=data['slide_7_Cy2x4'], attr='cell_type', mode='proportion', prediction_type='diff',
#                        graph_key='contact')
# ath.pl.interactions(ad=data['slide_7_Cy2x4'], attr='cell_type', mode='proportion', prediction_type='diff',
#                     graph_key='contact', ax=axs[1])
#
# fig.tight_layout()
# fig.show()

# %%
for spl in data.keys():
    height = data[spl].uns['mask'].shape[0]
    width = data[spl].uns['mask'].shape[1]
    area = width * height
    data[spl].uns['area'] = area
    data[spl].uns['height'] = height
    data[spl].uns['width'] = width

# %% compute estimated deviation from random poisson process L(t)-t
import numpy as np

spl = 'slide_7_Cy2x2'
spl = 'slide_49_By2x5'
attr = 'meta_label'
radii = np.linspace(0, 400, 100)
ids = data[spl].obs[attr].unique().tolist()
for _id in ids:
    ath.neigh.ripleysK(data[spl], attr=attr, id=_id, mode='csr-deviation', radii=radii)

# %%
# plot estimated deviation from random poisson process L(t)-t
fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
ath.pl.spatial(ad=data[spl], attr=attr, ax=axs[0])
ath.pl.ripleysK(ad=data[spl], attr=attr, ids=ids, mode='csr-deviation', ax=axs[1], legend=False)
fig.show()