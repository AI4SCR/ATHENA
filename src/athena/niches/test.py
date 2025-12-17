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
import numpy as np
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
from neighborhood_representation import *
#%%
neigh_repr = compute_neighborhood_representations(ad_dict['1'], attr='label_name', mode='proportion', graph_key='radius_80', min_neighbors=5)
#%%
neighborhood_representations_ad_dict(ad_dict, attr='label_name', mode='proportion', graph_key='radius_80', min_neighbors=5, inplace=True)
#%%
for ad in ad_dict.values():
    ad.obs.drop('neighbors_radius_80', axis=1, inplace=True)
#%%
import pickle

# Define your path for the dictionary specifically
pickle_path = "/users/achiozza/workspace/niche-learning/adict_Keren.pkl"

# Save the dictionary
with open(pickle_path, 'wb') as f:
    pickle.dump(ad_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Successfully saved ad_dict to {pickle_path}")
#%%
with open("/users/achiozza/workspace/niche-learning/adict_Keren.pkl", 'rb') as f:
    ad_dict_loaded = pickle.load(f)

# Check the keys to verify
print(ad_dict_loaded.keys())

#%%
filtered_merged = retrieve_merged_neighborhood_representations(ad_dict, attr='label_name', mode='proportion', graph_key='radius_80', filtered=True)
# %%
k_means_clustering(ad_dict, label_type='label_name', graph_key='radius_80', neighborhood_repr='proportions', k=4, random_seeds=[42, 7], merged=filtered_merged)

# %%
z_score = z_scores(ad_dict, clustering_key='k_means_4_proportions_label_name_radius_80_seed_42', label_type='label_name')
from plots import *
plot_z_scores_heatmap(ad_dict, z_score=z_score, clustering_key='k_means_4_proportions_label_name_radius_80_seed_42', label_type='label_name')

#%%
interactions = cell_type_interactions(ad_dict['1'], cell_type_column='label_name', group='per_group', obs_key = 'k_means_4_proportions_label_name_radius_80_seed_42')


# %%
from interactions import*

#%%
cell_type_interactions_dictionary(ad_dict, cell_type_column='label_name', graph_key='radius_80', group='per_group', obs_key='k_means_4_proportions_label_name_radius_80_seed_7')

# %%
merged_0 = aggregate_interactions(ad_dict, interaction_key='cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_7_0', aggregator='mean')
merged_1 = aggregate_interactions(ad_dict, interaction_key='cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_7_1', aggregator='mean')
merged_2 = aggregate_interactions(ad_dict, interaction_key='cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_7_2', aggregator='mean')
merged_3 = aggregate_interactions(ad_dict, interaction_key='cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_7_3', aggregator='mean')
#%%
cell_type_interactions_dictionary(ad_dict, cell_type_column='label_name', graph_key='radius_80', group='whole_sample')
#%%
fraction_above_median0 = above_median_fraction(ad_dict, interaction_key_group='cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_7_0')
fraction_above_median1 = above_median_fraction(ad_dict, interaction_key_group='cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_7_1')            
fraction_above_median2 = above_median_fraction(ad_dict, interaction_key_group='cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_7_2')
fraction_above_median3 = above_median_fraction(ad_dict, interaction_key_group='cell_type_interactions_k_means_4_proportions_label_name_radius_80_seed_7_3')
#%%
from pycirclize import Circos
from matplotlib import cm, colors
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def check_triangular_zero_xor(df: pd.DataFrame):
    """
    Checks if the strictly upper triangle OR the strictly lower triangle of a 
    square DataFrame is composed entirely of zeros, but NOT both.

    Args:
        df: The input Pandas DataFrame (assumed to be square).

    Returns:
        Boolean result for check.
    """
    if df.shape[0] != df.shape[1]:
        return False, "DataFrame is not square. Cannot perform triangular check."

    # Convert the DataFrame to a NumPy array for efficient checks
    data = df.values

    # Upper Triangle 
    is_upper_zero = np.all(np.triu(data, k=1) == 0) #k=1 to exclude the diagonal itself.
    # Lower Triangle
    is_lower_zero = np.all(np.tril(data, k=-1) == 0) #k=-1 to exclude the diagonal itself.

    # 3. Apply the XOR (Exclusive OR) logic: A OR B, but NOT (A AND B)
    result_xor = is_upper_zero ^ is_lower_zero

    return result_xor


def get_color_map(df: pd.DataFrame) -> Dict[str, str]:
    '''
    Generate a color map for cell types in the dataframe.
    Args:
        df : pandas.DataFrame -> dataframe containing input data.
        It has to contain:
                - MultiIndex with 'cell_type_1' and 'cell_type_2'(str).
    returns:
        Dict[str, str]: Dictionary mapping cell types to colors.
    '''
    assert df.index.nlevels == 2, "DataFrame must have a MultiIndex "

    if df.index.names != ['cell_type_1', 'cell_type_2']:
        df.index.set_names(['cell_type_1', 'cell_type_2'], inplace=True)

    df_new = df.reset_index()
    labels = list(set(df_new['cell_type_1'].unique()).union(set(df_new['cell_type_2'].unique())))
    cmap = cm.get_cmap('tab10')
    color_indices = {label: i for i, label in enumerate(labels)}
    color_map = {ct: cmap(color_indices[ct]) for ct in labels}
    return color_map


def data_for_circos_plot(df: pd.DataFrame, color: str):
    '''
    Create color and width dictionaries for circos plot links. 
        they will contain:
            labels as keys (tuples) and color/width values as values.    
    Creates sectors dataframe with width values.
        It will contain:
            labels as index and columns, and width values as values.
            The dataframe has 0 values on the top (or bottom) of the matrix -> not count interactions twice.

    Args:
        df : pandas.DataFrame -> dataframe containing input data.
            It has to contain:
                - MultiIndex with 'cell_type_1' and 'cell_type_2'(str).
                - column 'above_median_fraction' (float): Fraction of samples with interaction above median.
                - column 'aggregated_interaction' (float): Aggregated interaction value.
        color: String indicating which color values to use 'aggregated_interaction' or 'above_median_fraction'
    Returns:
        Dict[str, Dict[Tuple[str, str], float]]: Dictionary with 'color_dict' and 'width_dict'.
    '''
    assert color in ['aggregated_interaction', 'above_median_fraction'], "color must be either 'aggregated_interaction' or 'above_median_fraction'"
    assert 'above_median_fraction' and 'aggregated_interaction' in df.columns, "DataFrame must contain 'above_median_fraction' and 'aggregated_interaction' columns"
    assert df.index.nlevels == 2, "DataFrame must have a MultiIndex "

    if df.index.names != ['cell_type_1', 'cell_type_2']:
        df.index.set_names(['cell_type_1', 'cell_type_2'], inplace=True)

    #WIDTH
    width = [col for col in df.columns.tolist() if col != color][0]
    width_df = df[width]
    width_df = width_df.reset_index()
    width_df.columns = ['from', 'to', 'Value']
    width_df['Value'].dropna()
    width_dict = {(row['from'], row['to']): row['Value'] for _, row in width_df.iterrows()}

    # COLOR
    color_df = df[color]
    color_df = color_df.reset_index()
    color_df.columns = ['from', 'to', 'Value']
    color_df['Value'].dropna()
    color_dict = {(row['from'], row['to']): row['Value'] for _, row in color_df.iterrows()}

    #SECTORS 
    sectors_df = df[width].copy()
    sectors_df = sectors_df.reset_index()
    sectors_df.columns = ['cell_type_1', 'cell_type_2', 'width']
    sectors_df = sectors_df.pivot(index='cell_type_2', columns='cell_type_1', values='width')
    sectors_df = sectors_df.fillna(0)
    assert check_triangular_zero_xor(sectors_df), "Either the upper or lower triangle values of sectors_df must be 0, but not both."

    return {'color_dict': color_dict, 'width_dict': width_dict, 'sectors_df': sectors_df}


def get_circos(df: pd.DataFrame, color:str = 'aggregated_interaction', color_map: Dict[str,str] = None):
    '''
    Create circos object for aggregated interactions across samples.

    Args:
        df : pandas.DataFrame -> dataframe containing input data.
            It has to contain:
                - MultiIndex with 'cell_type_1' and 'cell_type_2'(str).
                - column 'above_median_fraction' (float): Fraction of samples with interaction above median.
                - column 'aggregated_interaction' (float): Aggregated interaction value.
        color: String indicating which color values to use 'aggregated_interaction' or 'above_median_fraction'
        color_map: Dictionary mapping cell types to colors. Default is None, a default colormap is generated.
    Returns:
        Circos object
    '''
    assert color in ['aggregated_interaction', 'above_median_fraction'], "color must be either 'aggregated_interaction' or 'above_median_fraction'"
    assert 'above_median_fraction' and 'aggregated_interaction' in df.columns, "DataFrame must contain 'above_median_fraction' and 'aggregated_interaction' columns"
    assert df.index.nlevels == 2, "DataFrame must have a MultiIndex "

    dicts = data_for_circos_plot(df = df, color = color)

    color_dict = dicts['color_dict']
    width_dict = dicts['width_dict']
    sectors_df = dicts['sectors_df']
    
    if color_map is None:
        cell_color_map= get_color_map(df)
    
    color_var = color
    val_min = min(color_dict.values())
    val_max = max(color_dict.values())

    # define color and with of the links
    def link_handler(from_label, to_label):
        # Get external value
        val = color_dict.get((from_label, to_label)) or color_dict.get((to_label, from_label), 0)
        lw = width_dict.get((from_label, to_label)) or width_dict.get((to_label, from_label), 1)

        if color_var == 'aggregated_interaction':
            filt = val
        else:
            filt = lw
        
        if filt <= 0.001:
            # Filter = if the value used for the color (either aggregated interaction or above median fraction) is 0.001 or less, make it invisible
            color = 'none'
            lw = 0
        else:    
            # If the value used for the color is bigger than 0.001 => assign it to a color using a colormap
            cmap = cm.get_cmap("Reds")  # or "Reds", "coolwarm", etc.
            val_min = min(color_dict.values())
            val_max = max(color_dict.values())
            norm = colors.Normalize(vmin=val_min, vmax=val_max)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            color = sm.to_rgba(val)  # val should be in 0-1

        # Return styling dictionary
        return dict(ec='none', lw=lw, fc=color, alpha=0.7)   
    
    circos = Circos.chord_diagram(
        sectors_df,
        space=2,
        cmap = cell_color_map,
        label_kws=dict(
            size=14,          # larger font
            color="black",    # dark text
            r=110,            # radial distance from circle
            orientation="vertical",  # or 'horizontal'
        ),
        link_kws_handler=link_handler
        )
    return circos
   

def plot_circos_plot(df: pd.DataFrame, color = 'aggregated_interaction', color_map: Dict[str,str] = None, niche_name: str = None, aggregator: str = None, save_path: str = None):
    '''
    Create circos plots for aggregated interactions across samples.

    Args:
        df : pandas.DataFrame -> dataframe containing input data.
            It has to contain:
                - MultiIndex with 'cell_type_1' and 'cell_type_2'(str).
                - column 'above_median_fraction' (float): Fraction of samples with interaction above median.
                - column 'aggregated_interaction' (float): Aggregated interaction value.
        color: String indicating which color values to use 'aggregated_interaction' or 'above_median_fraction'
        color_map: Dictionary mapping cell types to colors. Default is None, a default colormap is generated.
        niche_name: String indicating the name of the niche to put in the title if wanted. Default is None.
        aggregator: String indicating the aggregation method used to put in the title if wanted. Default is None.
        save_path: String indicating the path to save the figure. If None, figure is not saved. Default is None.
    Returns:
        None: Displays dot plot.
        Saves figure if save_path is provided.
    '''
    assert color in ['aggregated_interaction', 'above_median_fraction'], "color must be either 'aggregated_interaction' or 'above_median_fraction'"
    assert 'above_median_fraction' and 'aggregated_interaction' in df.columns, "DataFrame must contain 'above_median_fraction' and 'aggregated_interaction' columns"
    assert df.index.nlevels == 2, "DataFrame must have a MultiIndex "

    circos= get_circos(df=df, color=color, color_map=color_map) #circo object

    fig = circos.plotfig(figsize=(20, 15))

    # Adjust spacing to make room for title and colorbar
    fig.subplots_adjust(top=3, right=3)  # leave margin on top and right
    # Title 
    if niche_name==None:
        niche_name=''
    if aggregator==None:
        aggregator=''
    else:
        aggregator=f'({aggregator}) '
    fig.suptitle(
        f"Circos Plot of Aggregated Interactions {aggregator}in Niche {niche_name}\nColor = {color}, Line Width = {[col for col in df.columns.tolist() if col != color][0]}",
        fontsize=16,
        fontweight="bold",
        y=0.99  # vertical position: 1.0 is top of figure
    )
    

    # Create colorbar
    val_min = min(df[color])
    val_max = max(df[color])
    norm = colors.Normalize(vmin=val_min, vmax=val_max)
    cmap = cm.get_cmap("Reds")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.91, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label=color)
    cbar.set_label(color, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    fig.show()    

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
#%%
df = pd.concat([merged_3, fraction_above_median3], axis=1)
df.columns = ['aggregated_interaction', 'above_median_fraction']

plot_circos_plot(df=df, color='aggregated_interaction', aggregator='mean')

#%%
dictionary = data_for_circos_plot(df=df, color='aggregated_interaction')
