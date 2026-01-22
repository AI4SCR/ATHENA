from athena.niches.clustering.cluster_analysis import z_scores, attr_proportions
from athena.niches.interactions.interactions import aggregate_interactions, above_median_fraction
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Dict, Union, List
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from athena.plotting.utils import savefig, dpi, label_fontdict, title_fontdict
from athena.niches.plotting.utils import get_color_map, data_for_circos_plot
#%%

def plot_ARIs(aris_df:pd.DataFrame, best_avg: bool = False, title: str = None, save: str = None, ax: int = None, tight_layout: bool = False, show: bool = True):
    '''
    Args:
        ax: axes object in which to plot
        title: title of plot
        show: whether to show the plot or not
        save: path to the file in which the plot is saved
    
    '''
    if ax:
        fig = ax.get_figure()
        show = False # do not automatically show plot if we provide axes
    else:
        fig, ax = plt.subplots(dpi=dpi)
        ax.set_aspect('equal')
    
    means = aris_df.mean()
    max_idx = means.argmax()

    # 3. Set default colors and highlight the max
    if best_avg:
        bp = aris_df.boxplot(grid=False, ax=ax, patch_artist=True, return_type='dict')
        for i, box in enumerate(bp['boxes']):
            box.set_edgecolor('black')
            if i == max_idx:
                box.set_facecolor('yellow')  # Highlight the best seed
            else:
                box.set_facecolor('white') # Default color for others
        yellow_patch = mpatches.Patch(color='yellow', label='Highest Mean ARI')
    else:
        aris_df.boxplot(grid=False, ax=ax)

    ax.plot(range(1, len(means) + 1), means, 
        color='red',           
        linestyle='-',    
        linewidth=1,
        label='Mean ARI')      # For a legend if you want on
    
    if best_avg:
        ax.legend(handles=[plt.Line2D([0], [0], color='red', label='Mean ARI'), yellow_patch])
    else:
        ax.legend(handles=[plt.Line2D([0], [0], color='red', label='Mean ARI')])

    ax.set_facecolor('white')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Adjusted Rand Index', label_fontdict)
    ax.set_xticklabels(labels=aris_df.columns, rotation=45, ha='right', fontsize= label_fontdict['size'])
    if title is not None:
        ax.set_title(title, title_fontdict)

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if save:
        savefig(fig, save)

    return ax


def plot_z_scores_heatmap(ad_dict: Union[Dict[str, AnnData], None]=None, group_key: str=None, attr: str = None, zscores: pd.DataFrame = None, title: str = None, save: str = None, tight_layout: bool = False, show: bool = True, ax = None, val_min:int = None, val_max:int=None, colormap = None):
    '''
    Plot important heatmap of z-scores of attr enrichment in each cluster.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        group_key (str): Key in AnnData.obs representing the clustering.
        attr (str): Key in AnnData.obs representing the labels to assess enrichment.
        zscores (pd.DataFrame, optional): Precomputed z-scores DataFrame. If None, it will be computed.
        ax: axes object in which to plot
        title: title of plot
        show: whether to show the plot or not
        save: path to the file in which the plot is saved
    
    Returns:
        None: Displays a heatmap plot.
    '''
    assert (ad_dict is None) != (zscores is None), 'either provide zscores or ad_dict'
    
    if ax:
        fig = ax.get_figure()
        show = False # do not automatically show plot if we provide axes
    else:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.set_aspect('equal')
    if zscores is None:
        zscores = z_scores(ad_dict=ad_dict, attr=attr, group_key=group_key)
    
    if val_min is None:
        val_min = zscores.values.min()
    if val_max is None:
        val_max = zscores.values.max()
    
    if colormap is None:
        colormap = 'vlag'
    
    sns.heatmap(zscores, annot=True, cmap=colormap, center=0, ax=ax, vmax=val_max, vmin=val_min)
    ax.set_xlabel(attr, label_fontdict)
    ax.set_ylabel('Group', label_fontdict)
    
    if title is not None:
        ax.set_title(title, title_fontdict)

    
    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if save:
        savefig(fig, save)
    
    return ax

def stacked_bar_plots(ad_dict: Dict[str, AnnData], attr:str, group_key: str, color_map: Dict[str,str] = None, save: str = None, tight_layout: bool = False, show: bool = True, title: str = None):
    '''
    Create stacked bar plots for cell type proportions in each niche cluster.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr (str): Key in AnnData.obs representing the labels to assess proportions.
        group_key (str): Key in AnnData.obs representing the clustering.
        color_map (Dict[str,str], optional): Color map for cell types. If None, a default will be generated.
        save (str, optional): Path to save the plot.png if desired
    Returns:
        None: Displays stacked bar plots.
    '''
    proportions = attr_proportions(ad_dict=ad_dict, attr=attr, group_key=group_key)
    if color_map is None:
        labels = list(proportions.columns)
        color_map= get_color_map(labels)
    
    number_of_groups = len(proportions.index)   
    fig, axes = plt.subplots(1, number_of_groups, figsize=(20, 10), sharey=True)
    
    for ax, (i, (row_name, row_series)) in zip(axes, enumerate(proportions.iterrows())):
    
        bottom = 0  # bottom of the stack
    
        for ct, prop in row_series.items():
        
            ax.bar(0, prop, bottom=bottom, color=color_map[ct], edgecolor='black', width=0.6, label=ct)
            bottom += prop
        
        ax.set_title(row_name)
        ax.set_xticks([])  # remove x tick for single bar
        ax.set_ylim(0, 1)  # since proportions

    # Add a single legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[ct]) for ct in labels]
    fig.legend(handles, labels, loc='right', title='Cell Type')
    axes[0].set_ylabel("Proportion")
    if title:
        fig.suptitle(title, fontsize=16, weight='bold')
    
    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if save:
        savefig(fig, save)
    
    return axes
    
def plot_stacked_bars_on_ax(ad_dict, attr: str, group_key: str, ax, color_map: Dict[str, str] = None, title: str = None):
    '''
    Modular version: Plots stacked bars for all clusters onto a single provided Axes object.
    '''
    
    proportions = attr_proportions(ad_dict=ad_dict, attr=attr, group_key=group_key)
    labels = list(proportions.columns)
    
    if color_map is None:
        color_map = get_color_map(labels)

    # Plotting using pandas logic on the specific ax
    # This plots all clusters side-by-side on the same axis
    proportions.plot(kind='bar', stacked=True, ax=ax, color=[color_map[c] for c in labels], 
                     edgecolor='black', width=0.8, legend=False)

    if title:
        ax.set_title(title, fontsize=12, weight='bold')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    return labels # Return labels for the legend


def dot_plots(ad_dict: Dict[str,AnnData], interaction_key_group:str,interaction_key_overall:str, aggregator: str = 'mean', color: str = 'interaction_values',color_map: Dict[str,str] = None, title: str = None, save: str = None, ax: int = None, tight_layout: bool = False, show: bool = True, val_min: int=None, val_max:int = None, return_data: bool = False):
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
    assert color in ['interaction_values', 'above_median_fraction'], "color must be either 'interaction_values' or 'above_median_fraction'"

    merged_group = aggregate_interactions(ad_dict, interaction_key_group, aggregator)[interaction_key_group]
    above_median_fr = above_median_fraction(ad_dict, interaction_key_group=interaction_key_group, interaction_key_overall=interaction_key_overall)
    if color == 'interaction_values':
        color_df = merged_group.copy().reset_index()
        width_df = above_median_fr.copy().reset_index()
    else: 
        width_df = merged_group.copy().reset_index()
        color_df = above_median_fr.copy().reset_index()
    
    if ax:
        fig = ax.get_figure()
        show = False # do not automatically show plot if we provide axes
        return_data = False
    else:
        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_aspect('equal')
    
    if color == 'interaction_values':
        col = f'{aggregator}_{color}'
        cw_legend = f'Color = {aggregator}_{color}, Line Width = above_median_fraction'
    else:
        col = color
        cw_legend = f'Color = {color}, Line_Width = {aggregator}_interaction_value'
    
    if color_map is None:
        color_map = 'Reds'
    
    if val_min is None:
        val_min = color_df['score'].min()
    if val_max is None:
        val_max= color_df['score'].max()

    scatter = ax.scatter(
    x=color_df['attr_1'],
    y=color_df['attr_2'],
    c=color_df['score'],        
    s=width_df['score'] *1000,   
    cmap=color_map, #'YlGn'
    alpha=0.8,
    edgecolor='k',
    vmin=val_min,    # Set your minimum value here
    vmax=val_max     # Set your maximum value here
    )
    # --- Colorbar ---
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(col, fontsize=12)

    # --- Titles and labels ---
    if title is None:
        title = ''
    ax.set_title(
        f'{title} \n'
        f'{cw_legend}',
        fontsize=16,
        pad=20,
        weight='bold'
    )
    ax.set_xlabel('attr', fontsize=13, labelpad=10)
    ax.set_ylabel('attr', fontsize=13, labelpad=10)

    # --- Ticks ---
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=11)

    # --- Grid and layout ---
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if save:
        savefig(fig, save)
    
    if return_data:
        return {'width': width_df, 'color': color_df}
    
    return ax



def interactions_circos_plots(ad_dict: Dict[str,AnnData], interaction_key_group:str, interaction_key_overall:str, aggregator: str = 'mean', color: str = 'interaction_values', color_map: Dict[str,str] = None, title: str = None, save: str = None, ax: int = None, tight_layout: bool = False, show: bool = True, val_min: int=None, val_max:int = None, return_data: bool = False):
    '''
    Create circos plots for aggregated interactions across samples.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key: Key or list of keys in ad.uns where the interactions DataFrame is stored
        aggregator: Aggregation method - 'mean' or 'median'
        color: String indicating which color values to use 'interaction_values' or 'above_median_fraction'
        save: Path to save the plot.png if desired

    Returns:
        None: Displays dot plot.
    '''
    assert aggregator in ['mean', 'median'], "aggregator must be either 'mean' or 'median'"
    assert color in ['interaction_values', 'above_median_fraction'], "color must be either 'interaction_values' or 'above_median_fraction'"
    
    from pycirclize import Circos
    merged_group = aggregate_interactions(ad_dict, interaction_key= interaction_key_group, aggregator=aggregator)[interaction_key_group]
    above_median_fr = above_median_fraction(ad_dict, interaction_key_group=interaction_key_group, interaction_key_overall=interaction_key_overall)

    if color == 'interaction_values':
        dicts = data_for_circos_plot(color_df=merged_group, width_df=above_median_fr)
    else: 
        dicts = data_for_circos_plot(color_df=above_median_fr, width_df=merged_group)

    color_dict = dicts['color_dict']
    width_dict = dicts['width_dict']
    sectors_df = dicts['sectors_df']

    if val_min is None:
        val_min = min(color_dict.values())
    if val_max is None:
        val_max = max(color_dict.values())
    
    if color_map is None:
        labels = sorted(list(set(sectors_df.index.unique()).union(set(sectors_df.columns.unique())))) 
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
            link_color = 'none'
            lw = 0
        else:    
            # Map value to a color using a colormap
            cmap = cm.get_cmap("Reds")  # or "Reds", "coolwarm", etc.
            norm = colors.Normalize(vmin=val_min, vmax=val_max)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            link_color = sm.to_rgba(val)  # val should be in 0-1

        # Return styling dictionary
        return dict(ec='none', lw=lw, fc=link_color, alpha=0.7)   
    
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
    
    if ax:
        fig = ax.get_figure()
        show = False # do not automatically show plot if we provide axes
        return_data = False
    else:
        fig, ax = plt.subplots(
        figsize=(20, 15), 
        dpi=dpi, 
        subplot_kw={'projection': 'polar'})
        #ax.set_aspect('equal')


    circos.plotfig(ax=ax) # Plot directly on the handle

    #k = interaction_key_group[interaction_key_group.find('interactions_') + len('interactions_') + 1:] # remove cell_type_interactions_ to get the niche name

    if color == 'interaction_values':
        col = f'{aggregator}_{color}'
        cw_legend = f'Color = {aggregator}_{color}, Line Width = above_median_fraction'
    else:
        col = color
        cw_legend = f'Color = {color}, Line_Width = {aggregator}_interaction_value'
    
    if title is None:
        title = 'Circos plot'
    fig.suptitle(
        f"{title} \n {cw_legend}",
        fontsize=16,
        fontweight="bold",
        y=0.99  # vertical position: 1.0 is top of figure
    )

    # Add colorbar legend
    norm = colors.Normalize(vmin=val_min, vmax=val_max)
    cmap = cm.get_cmap("Reds")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.91, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label=col)
    cbar.set_label(col, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if save:
        savefig(fig, save)
    
    if return_data: 
        return dicts
    
    return ax

def interaction_heatmaps(ad_dict: Dict[str,AnnData], interaction_key_group:str, aggregator: str = 'mean',color_map: Dict[str,str] = None, title: str = None, save: str = None, ax: int = None, tight_layout: bool = False, show: bool = True, val_min: int=None, val_max:int = None, return_data: bool = False):
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
    merged_group = aggregate_interactions(ad_dict, interaction_key_group, aggregator)[interaction_key_group]
    df = merged_group.reset_index().pivot(index='attr_1', columns='attr_2', values=f'score')
    df = df.fillna(0)

    if ax:
        fig = ax.get_figure()
        show = False # do not automatically show plot if we provide axes
        return_data = False
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')

    if color_map is None:
        color_map = 'Reds'

    if val_max is None:
        val_max = df.values.max()
    if val_min is None:
        val_min = df.values.min()

    sns.heatmap(df, annot=True, cmap=color_map, ax=ax, vmax=val_max, vmin=val_min)
    
    if title: 
        ax.set_title(title)
    fig.show()

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if save:
        savefig(fig, save)

    if return_data:
        return df

    return ax


def radar_plots():
    '''needs also a functions to compute marker proportions for each niche'''
    return





    

