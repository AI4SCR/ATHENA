from clustering import z_scores, label_type_proportions
from interactions import aggregate_interactions, above_median_fraction
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Dict, Union, List
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import cm
import matplotlib.colors as colors

#%%

def plot_z_scores_heatmap(ad_dict: Dict[str, AnnData], clustering_key: str, label_type: str, z_score: pd.DataFrame = None, save_path: str = None):
    '''
    Plot important heatmap of z-scores of label_type enrichment in each cluster.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        clustering_key (str): Key in AnnData.obs representing the clustering.
        label_type (str): Key in AnnData.obs representing the labels to assess enrichment.
        z_score (pd.DataFrame, optional): Precomputed z-scores DataFrame. If None, it will be computed.
    
    Returns:
        None: Displays a heatmap plot.
    '''
    if z_score is None:
        z_score = z_scores(ad_dict, clustering_key, label_type)
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(z_score, annot=True, cmap='vlag', center=0, ax=ax)
    ax.set_title(f'Z-scores of Cell Type Enrichment in {clustering_key} Niches')
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Niche Cluster')
    fig.show()
    if save_path:
        dot_plot_name = save_path + f'z_scores_{clustering_key}_{label_type}.png'
        fig.savefig(dot_plot_name, dpi=300, bbox_inches="tight")

def get_color_map(labels: List[str]) -> Dict[str, str]:
    cmap = cm.get_cmap('tab10')
    color_indices = {label: i for i, label in enumerate(labels)}
    color_map = {ct: cmap(color_indices[ct]) for ct in labels}
    return color_map

def stacked_bar_plots(ad_dict: Dict[str, AnnData], label_type:str, clustering_key: str, color_map: Dict[str,str] = None, save_path: str = None):
    '''
    Create stacked bar plots for cell type proportions in each niche cluster.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        label_type (str): Key in AnnData.obs representing the labels to assess proportions.
        clustering_key (str): Key in AnnData.obs representing the clustering.
        color_map (Dict[str,str], optional): Color map for cell types. If None, a default will be generated.
        save_path (str, optional): Path to save the plot.png if desired
    Returns:
        None: Displays stacked bar plots.
    '''
    proportions = label_type_proportions(ad_dict, label_type, group='per_group', obs_key=clustering_key)
    if color_map is None:
        labels = sorted(list(set(proportions.index.unique()))) 
        color_map= get_color_map(labels, )
    
    number_of_groups = len(proportions.columns)   
    fig, axes = plt.subplots(1, number_of_groups, figsize=(20, 10), sharey=True)
    
    for ax, (title, df_series) in zip(axes, proportions.items()):
    
        bottom = 0  # bottom of the stack
    
        for ct, prop in df_series.items():
        
            ax.bar(0, prop, bottom=bottom, color=color_map[ct], edgecolor='black', width=0.6, label=ct)
            bottom += prop
        
        ax.set_title(title)
        ax.set_xticks([])  # remove x tick for single bar
        ax.set_ylim(0, 1)  # since proportions

    # Add a single legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[ct]) for ct in labels]
    fig.legend(handles, labels, loc='right', title='Cell Type')
    axes[0].set_ylabel("Proportion")
    fig.suptitle("Stacked Bar Charts of Cell Type Proportions in Niches", fontsize=16, weight='bold')
    fig.show()
    if save_path:
        dot_plot_name = save_path + f'stacked_bar_plots_{label_type}_{clustering_key}.png'
        fig.savefig(dot_plot_name, dpi=300, bbox_inches="tight")

def dot_plots(ad_dict: Dict[str,AnnData], interaction_key_group:str, aggregator: str = 'mean', save_path: str = None):
    '''
    Create dot plots for aggregated interactions across samples.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key: Key or list of keys in ad.uns where the interactions DataFrame is stored
        aggregator: Aggregation method - 'mean' or 'median'
    
    Returns:
        None: Displays dot plot.
    '''
    assert aggregator in ['mean', 'median'], "aggregator must be either 'mean' or 'median'"

    merged_group = aggregate_interactions(ad_dict, interaction_key_group, aggregator)
    above_median_fr = above_median_fraction(ad_dict, interaction_key_group)

    fig, ax = plt.subplots(figsize=(15, 12))

    scatter = ax.scatter(
    x=merged_group['Cell_Type_2'],
    y=merged_group['Cell_Type_1'],
    c=merged_group[f'{aggregator}_interaction'],        # color by aggregated interaction value 
    s=above_median_fr['above_median_fraction'] *1000,   # dot size by fraction above median value
    cmap='Reds', #'YlGn'
    alpha=0.8,
    edgecolor='k'
    )
    # --- Colorbar ---
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{aggregator} interaction value', fontsize=12)

    k = interaction_key_group[interaction_key_group.find('interactions_') + len('interactions_') + 1:] # remove cell_type_interactions_ to get the niche name

    # --- Titles and labels ---
    ax.set_title(
        f'Dot Plot of Interactions in {k}\n'
        f'(Size = above median fraction, Color = {aggregator} interaction)',
        fontsize=16,
        pad=20,
        weight='bold'
    )
    ax.set_xlabel('Cell type', fontsize=13, labelpad=10)
    ax.set_ylabel('Cell type', fontsize=13, labelpad=10)

    # --- Ticks ---
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=11)

    # --- Grid and layout ---
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.show()
    if save_path:
        dot_plot_name = save_path + f'dot_plot_interactions_{interaction_key_group}_{aggregator}.png'
        fig.savefig(dot_plot_name, dpi=300, bbox_inches="tight")

def data_for_circos_plot(color_df: pd.DataFrame, width_df: pd.DataFrame, c):
    '''
    Create color and width dictionaries for circos plot links and pd.DataFrame for sectors.
    Args:
        color_df (pd.DataFrame): DataFrame with color values for interactions.
        width_df (pd.DataFrame): DataFrame with width values for interactions.
    Returns:
        Dict[str, Dict[Tuple[str, str], float]] and Dict[str, pd.DataFrame]: Dictionary with 'color_dict' and 'width_dict' and 'sectors_df'.
    '''
    #WIDTH
    width_df.reset_index(inplace=True)
    width_df.columns = ['from', 'to', 'Value']
    width_df = width_df.dropna(subset=['Value'])
    width_dict = {(row['from'], row['to']): row['Value'] for _, row in width_df.iterrows()}
    #width_dict = {k: v for k, v in width_dict.items()}

    # COLOR
    color_df.reset_index(inplace=True)
    color_df.columns = ['from', 'to', 'Value']
    color_df = color_df.dropna(subset=['Value'])
    color_dict = {(row['from'], row['to']): row['Value'] for _, row in color_df.iterrows()}

    #SECTORS 
    sectors_df = width_df.copy()
    sectors_df.columns = ['width']
    sectors_df = width_df.pivot(index='cell_type_1', columns='cell_type_2', values='above_median_fraction')
    sectors_df = sectors_df.fillna(0)

    return {'color_dict': color_dict, 'width_dict': width_dict, 'sectors_df': sectors_df}

def interactions_circos_plots(ad_dict: Dict[str,AnnData], interaction_key_group:str, aggregator: str = 'mean', color = 'interaction_values', color_map: Dict[str,str] = None, save_path: str = None):
    '''
    Create circos plots for aggregated interactions across samples.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key: Key or list of keys in ad.uns where the interactions DataFrame is stored
        aggregator: Aggregation method - 'mean' or 'median'
        color: String indicating which color values to use 'interaction_values' or 'above_median_fraction'
        save_path: Path to save the plot.png if desired

    Returns:
        None: Displays dot plot.
    '''
    assert aggregator in ['mean', 'median'], "aggregator must be either 'mean' or 'median'"
    assert color in ['interaction_values', 'above_median_fraction'], "color must be either 'interaction_values' or 'above_median_fraction'"
    
    from pycirclize import Circos
    merged_group = aggregate_interactions(ad_dict, interaction_key_group, aggregator)
    above_median_fr = above_median_fraction(ad_dict, interaction_key_group)

    if color == 'interaction_values':
        dicts = data_for_circos_plot(color_df=merged_group, width_df=above_median_fr)
    else: 
        dicts = data_for_circos_plot(color_df=above_median_fr, width_df=merged_group)

    color_dict = dicts['color_dict']
    width_dict = dicts['width_dict']
    sectors_df = dicts['sectors_df']
    
    if color_map is None:
        labels = sorted(list(set(sectors_df.index['cell_type_1'].unique()).union(set(sectors_df.index['cell_type_2'].unique())))) 
        cell_color_map= get_color_map(labels)

    def link_handler(from_label, to_label):
        # Get external value
        val = color_dict.get((from_label, to_label)) or color_dict.get((to_label, from_label), 0)
        lw = width_dict.get((from_label, to_label)) or width_dict.get((to_label, from_label), 1)

        if color == 'interaction_values':
            filt = val
        else:
            filt = lw
        
        if filt <= 0.001:
            color = 'none'
            lw = 0
        else:    
            # Map value to a color using a colormap
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
    
    fig = circos.plotfig(figsize=(20, 15))
    ax = fig.axes[0]

    # Adjust spacing to make room for title and colorbar
    plt.subplots_adjust(top=3, right=3)  # leave margin on top and right

    k = interaction_key_group[interaction_key_group.find('interactions_') + len('interactions_') + 1:] # remove cell_type_interactions_ to get the niche name

    if color == 'interaction_values':
        col = f'{aggregator} interaction value'
        wid = 'above median fraction'
    else:
        col = 'above median fraction'
        wid = f'{aggregator} interaction value'
    
    fig.suptitle(
        f"Circos Plot of Interactions in Niche {k}\nColor = {col}, Line Width = {wid}",
        fontsize=16,
        fontweight="bold",
        y=0.99  # vertical position: 1.0 is top of figure
    )

    # Add colorbar legend
    val_min = min(color_dict.values())
    val_max = max(color_dict.values())
    norm = colors.Normalize(vmin=val_min, vmax=val_max)
    cmap = cm.get_cmap("Reds")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.91, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label=col)
    cbar.set_label(col, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    fig.show()    

    if save_path:
        dot_plot_name = save_path + f'circos_plot_interactions_{interaction_key_group}_{aggregator}_color_{color}.png'
        fig.savefig(dot_plot_name, dpi=300, bbox_inches="tight")


def interaction_heatmaps(ad_dict: Dict[str,AnnData], interaction_key_group:str, aggregator: str = 'mean', save_path: str = None):
    '''
    Create heatmaps for aggregated interactions across samples.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key: Key or list of keys in ad.uns where the interactions DataFrame is stored
        aggregator: Aggregation method - 'mean' or 'median'
    Returns:
        None: Displays heatmap plot.
    '''
    assert aggregator in ['mean', 'median'], "aggregator must be either 'mean' or 'median'"
    merged_group = aggregate_interactions(ad_dict, interaction_key_group, aggregator)
    df = merged_group.pivot(index='cell_type_1', columns='cell_type_2', values=f'{aggregator}_interaction')
    df = df.fillna(0)

    fig, ax = plt.subplots(figsize=(10,8))

    sns.heatmap(df, annot=True, cmap='Reds', ax=ax)
    
    k = interaction_key_group[interaction_key_group.find('interactions_') + len('interactions_') + 1:] # remove cell_type_interactions_ to get the niche name
    ax.set_title(f'Heatmap of {aggregator} Interactions in Niche {k}')
    fig.show()

    if save_path:
        dot_plot_name = save_path + f'heatmap_interactions_{interaction_key_group}_{aggregator}.png'
        fig.savefig(dot_plot_name, dpi=300, bbox_inches="tight")



def radar_plots():
    '''needs also a functions to compute marker proportions for each niche'''
    return





    

