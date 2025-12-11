#%%
from clustering import *

# %%
from ai4bmr_datasets import Keren2018
from pathlib import Path
import pandas as pd
from skimage.measure import regionprops_table
import anndata as ad
from matplotlib import cm
import colorcet
import itertools
# %%
base_dir = Path('/users/achiozza/workspace') 
pd.set_option('display.max_columns', 5)

#%%
dataset = Keren2018(base_dir=base_dir,
                  image_version="published",
                  mask_version="published",
                  feature_version="published",
                  load_intensity=True,
                  metadata_version="published",
                  load_metadata=True
                  )
dataset.setup()
#%%
def dict_adata(dataset):
    labels = list(np.unique(dataset.metadata.label_name))
    var = dataset.panel
    var["channel"] = dataset.panel.index
    var = var.set_index(var.columns[0])

    sample_ids = list(dataset.masks.keys() & set(dataset.intensity.index.get_level_values("sample_id")))
    ds_athena = {sample_id: ad.AnnData(X=dataset.intensity.xs(sample_id, level="sample_id"),
                                        obs = dataset.metadata.xs(sample_id, level="sample_id"), 
                                        var = var) for sample_id in sample_ids}
    # .uns
    # metadata
    for sample_id in ds_athena.keys():
        ds_athena[sample_id].uns['metadata'] = dataset.clinical.loc[sample_id]
    # original mask
    for sample_id in ds_athena.keys():
        ds_athena[sample_id].uns['mask_original'] = dataset.masks[sample_id].data
    # corrected mask (with the object id mismatches set as 0 (background))
    for sample_id in ds_athena.keys():
        mismatch = list(set(set(np.unique(ds_athena[sample_id].uns['mask_original']))-set(map(int, set(ds_athena[sample_id].obs.index)))-set([np.uint16(0)])))
        mask = dataset.masks[sample_id].data
        mask[np.isin(mask, mismatch)] = np.uint16(0)
        ds_athena[sample_id].uns['mask'] = mask
    # set obs as categorical
    for sample_id in ds_athena.keys():
        ds_athena[sample_id].obs = ds_athena[sample_id].obs.astype("category")

    for sample_id in ds_athena.keys():
        adata = ds_athena[sample_id]
        centroids = pd.DataFrame(regionprops_table(adata.uns['mask'], properties=('label', 'centroid'))) # extract centroids of each cell cegmentation mask
        centroids.columns = ['object_id', 'y', 'x']
        centroids = centroids.set_index('object_id')
        centroids.index = centroids.index.astype(str)  # we need to convert to `str` to match the index of `ad.obs`
        adata.obs = adata.obs.join(centroids, how="left") # add centroids to ad.obs
    # color maps
    cmap_label_name = {
        'Background': (1.0, 1.0, 1.0),
        'Macrophages': (0.0392156862745098, 0.5529411764705883, 0.25882352941176473),
        'keratin_positive_tumor': (0.24313725490196078,0.7098039215686275, 0.33725490196078434),
        'DC/Mono': (0.7058823529411765, 0.8313725490196079, 0.19607843137254902),
        'DC' : (0.3137254901960784, 0.17647058823529413, 0.5607843137254902),
        'CD8 T': (0.09019607843137255, 0.396078431372549, 0.21176470588235294),
        'CD4 T': (0.9725490196078431, 0.9098039215686274, 0.050980392156862744),
        'Other immune': (0.00392156862745098, 0.9686274509803922, 0.9882352941176471),
        'Mono/Neu': (0.7450980392156863, 0.9882352941176471, 0.9882352941176471),
        'CD3 T': (0.5882352941176471, 0.8235294117647058, 0.8823529411764706),
        'B': (0.592156862745098, 0.996078431372549, 1.0),
        'unidentified': (0.0, 1.0, 0.9803921568627451),
        'NK': (0.6039215686274509,  0.9568627450980393, 0.9921568627450981),
        'Neutrophils': (0.07450980392156863, 0.2980392156862745, 0.5647058823529412),
        'Tregs': (0.0, 0.00784313725490196, 0.984313725490196),
        'endothelial': (0.5764705882352941, 0.8352941176470589, 0.7764705882352941),
        'mesenchymal_like': (0.2627450980392157, 0.5490196078431373, 0.4470588235294118),
        'tumor': (0.9333333333333333, 0.27450980392156865, 0.2784313725490196),
        }

    # cell_type colormap
    cmap_group_name = {'background': 'white', 'immune': 'darkgreen', 'keratin_positive_tumor': 'gold', 'endothelial': 'steelblue',
        'mesenchymal_like': 'coral', 'tumor': 'darkred'}

    cmap_cluster = {0: 'plum',1: 'goldenrod', 2: 'darkgreen', 3: 'gold', 4: 'steelblue', 5: 'coral', 6: 'darkred', 7: 'purple',
        8: 'orange', 9: 'pink', 10: 'lightblue', 11: 'lightgreen', 12: 'brown', 13: 'gray', 14: 'cyan', 15: 'magenta', 16: 'yellow',
        17: 'olive', 18: 'teal', 19: 'navy', 20: 'maroon', 21: 'lime', 22: 'indigo', 23: 'salmon'}
    cmap_cluster_ann = {'Tumor enriched': 'plum','B enriched': 'goldenrod', 'K+ tumor enriched': 'darkgreen', 'Immune enriched': 'gold'}
    
    for adata in ds_athena.values():
        # define default colormap
        adata.uns['cmaps'] = {}
        adata.uns['cmaps'].update({'default': cm.Reds})
        adata.uns['cmaps']['category'] = colorcet.glasbey_bw
        adata.uns['cmaps'].update({'label_name': cmap_label_name})
        adata.uns['cmaps'].update({'group_name': cmap_group_name})
        adata.uns['cmaps'].update({'k_means4': cmap_cluster_ann})
        adata.uns['cmaps'].update({'k_means12': cmap_cluster})
        adata.uns['cmaps'].update({'k_means24': cmap_cluster})
        adata.uns['cmap_labels'] = {}
    
    return ds_athena

#%%
ad_dict = dict_adata(dataset)

#%%
from athena.graph import build_graph
#%%
for sample_id in ad_dict.keys():
    build_graph(ad_dict[sample_id], topology='radius', graph_key='radius_80', radius = 80, include_self = True)

# %%
merged = neighborhood_label_type_merged(ad_dict, label_type='label_name', graph_key='radius_80', neighborhood_repr='proportions')
# %%
k_means_clustering(ad_dict, label_type='label_name', graph_key='radius_80', neighborhood_repr='proportions', k=4, random_seeds=[42, 7], merged=merged)

# %%
z_score = z_scores(ad_dict, clustering_key='k_means_4_proportions_label_name_radius_80_seed_42', label_type='label_name')


#%%
interactions = cell_type_interactions(ad_dict['1'], cell_type_column='label_name', group='per_group', obs_key = 'k_means_4_proportions_label_name_radius_80_seed_42')


# %%
from interactions import*

#%%
cell_type_interactions_dictionary(ad_dict, cell_type_column='label_name', graph_key='radius_80', group='per_group', obs_key='k_means_4_proportions_label_name_radius_80_seed_42')
#%%
keys = ['cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_42_0','cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_42_1' , 'cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_42_2', 'cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_42_3']
dict_merged= aggregate_interactions(ad_dict, interaction_key=keys, aggregator='mean')
# %%
merged_0 = aggregate_interactions(ad_dict, interaction_key='cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_42_0', aggregator='mean')
#%%
cell_type_interactions_dictionary(ad_dict, cell_type_column='label_name', graph_key='radius_80', group='whole_sample')
#%%
fraction_above_median = above_median_fraction(ad_dict, interaction_key_group='cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_42_0', interaction_key_whole_sample='cell_type_interactions_whole_sample')
