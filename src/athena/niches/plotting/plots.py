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

def plot_ARIs(aris_df:pd.DataFrame, best_avg: bool = False, title: str = None, save: str = None, ax: int = None, tight_layout: bool = False, xlabel:str = None,  show: bool = True):
    '''
    Args:
        aris_df: Dataframe with ARIs for each column (Seed)
        ax: axes object in which to plot
        title: title of plot
        show: whether to show the picture or not
        save: Path to save the plot.png if desired
        xlabel: x axis label
        tight_layout
        best_avg: whether to highlight the seed with the best average ARI with a different color in the boxplot

    
    Return:
        ax
    
    
    '''
    if ax:
        fig = ax.get_figure()
        show = False # do not automatically show plot if we provide axes
    else:
        fig, ax = plt.subplots(figsize=(12, 8),dpi=dpi)
        #sax.set_aspect('equal')
    
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
        ax.legend(handles=[plt.Line2D([0], [0], color='red', label='Mean ARI'), yellow_patch], bbox_to_anchor=(1.0, 0.5))
    else:
        ax.legend(handles=[plt.Line2D([0], [0], color='red', label='Mean ARI')], bbox_to_anchor=(1.0, 0.5))

    ax.set_facecolor('white')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Adjusted Rand Index', label_fontdict)
    ax.set_xticklabels(labels=aris_df.columns, rotation=45, ha='right', fontsize= label_fontdict['size'])
    if xlabel is not None: 
        ax.set_xlabel(xlabel, label_fontdict)
    if title is not None:
        ax.set_title(title, title_fontdict)

    if tight_layout:
        fig.tight_layout()

    if save:
            savefig(fig, save)
    if show:
        fig.show()
    else:
        plt.close(fig)

    return ax


def plot_zscores_heatmap(ad_dict: Union[Dict[str, AnnData], None]=None, group_key: str=None, attr: str = None, zscores: pd.DataFrame = None, title: str = None, save: str = None, tight_layout: bool = False, show: bool = True, ax = None, val_min:int = None, val_max:int=None, colormap = None, return_data:bool = False, fig_size:tuple=None):
    '''
    Plot important heatmap of z-scores of attr enrichment in each cluster.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        group_key (str): Key in AnnData.obs representing the clustering.
        attr (str): Key in AnnData.obs representing the labels to assess enrichment.
        zscores (pd.DataFrame, optional): Precomputed z-scores DataFrame. If None, it will be computed.
        ax: axes object in which to plot
        title: title of the plot
        tight_layout
        show: whether to show the picture or not
        val_min, val_max: minimum and maximum values of the color scale
        return_data: whether to return the data used to make the plot
        save: Path to save the plot.png if desired
        colormap: colormap to use for the heatmap. If None, 'vlag' will be used.
        fig_size: tuple with figure size. If None, it will be automatically determined based on the number of clusters and attributes.
    
    Returns:
        ax or data if return_data
    '''
    assert (ad_dict is None) != (zscores is None), 'either provide zscores or ad_dict'
    
    if zscores is None:
        zscores = z_scores(ad_dict=ad_dict, attr=attr, group_key=group_key)
    
    if ax:
        fig = ax.get_figure()
        show = False # do not automatically show plot if we provide axes
    else:
        if fig_size is None:
            num_cols = len(zscores.columns)
            num_rows = len(zscores.index)
            # Logic: 1 unit of size for every X items, but never smaller than min_size
            width = max(num_cols * 0.5, 5)
            height = max(num_rows * 0.5, 5)
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig, ax = plt.subplots(figsize=fig_size)
        ax.set_aspect('equal')
    
    
    if val_max is None:
        val_max = max([abs(zscores.values.min()),abs(zscores.values.max())])
    if val_min is None:
        val_min = -val_max

    if colormap is None:
        colormap = 'vlag'
    
    sns.heatmap(zscores, annot=True, cmap=colormap, center=0, ax=ax, vmax=val_max, vmin=val_min)
    ax.set_xlabel(attr, label_fontdict)
    ax.set_ylabel('Group', label_fontdict)
    
    if title is not None:
        ax.set_title(title, title_fontdict)

    
    if tight_layout:
        fig.tight_layout()

    if save:
            savefig(fig, save)
    if show:
        fig.show()
    else:
        plt.close(fig)
    
    if return_data:
        return zscores
    
    return ax

def plot_stacked_bars(ad_dict:Union[ Dict[str, AnnData], None]=None, proportions: Union[pd.DataFrame, None]=None, attr:str=None, group_key: str=None, color_map: Dict[str,str] = None, save: str = None, tight_layout: bool = False, show: bool = True, title: str = None, return_data:bool = False, fig_size:tuple=None):
    '''
    Create stacked bar plots for cell type proportions in each niche cluster.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr (str): Key in AnnData.obs representing the labels to assess proportions.
        group_key: Key in AnnData.obs representing the clustering.
        color_map: color map for cell types. If None, a default will be generated.
        save: path to save the plot
        title: title of the plot
        tight_layout
        show: whether to show the picture or not
        return_data: whether to return the data used to make the plot
        save: Path to save the plot.png if desired
        colormap: colormap to use for the heatmap. If None, 'vlag' will be used.
        fig_size: tuple with figure size. If None, it will be automatically determined based on the number of clusters and cell types.


    Returns:
        ax or data if return_data
    '''
    assert (ad_dict is None) != (proportions is None), 'either provide proportions or ad_dict'
    if ad_dict is not None:
        assert attr is not None and group_key is not None, 'if providing ad_dict, attr and group_key must be provided'
        proportions = attr_proportions(ad_dict=ad_dict, attr=attr, group_key=group_key)
    labels = list(proportions.columns)
    if color_map is None:
        color_map= get_color_map(labels)
    
    number_of_groups = len(proportions.index)  
    if fig_size is None:
         fig_size=(25, 10)
    fig, axes = plt.subplots(1, number_of_groups, figsize=fig_size, sharey=True)
    
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

    if save:
            savefig(fig, save)
    if show:
        fig.show()
    else:
        plt.close(fig)
    
    if return_data:
        return proportions
    
    return axes
    
def plot_stacked_bars_on_ax(ad_dict: Dict[str, AnnData], attr:str, group_key: str, ax, color_map: Dict[str,str] = None, title: str = None):
    '''
    Modular version: Plots stacked bars for all clusters onto a single provided Axes object.
    
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr (str): Key in AnnData.obs representing the labels to assess proportions.
        group_key: Key in AnnData.obs representing the clustering.
        ax: axes object in which to plot
        color_map: color map for cell types. If None, a default will be generated.
        title: title of the plot

    Returns:
        labels for the legend
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


def plot_interactions_dot_plots(ad_dict: Dict[str,AnnData]=None, interaction_key_group:str=None,interaction_key_overall:str=None, aggregator: str = 'mean', above_median_fr:pd.Series=None, merged_group:pd.Series=None, color: str = 'interaction_values',color_map: Dict[str,str] = None, title: str = None, save: str = None, ax: int = None, tight_layout: bool = False, show: bool = True, val_min: int=None, val_max:int = None, return_data: bool = False):
    '''
    Create dot plots for aggregated interactions across samples.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key_group: Key in ad.uns where the interactions DataFrame of the group/niche is stored
        interaction_key_overall: Key in ad.uns where the overall interactions DataFrame is stored
        aggregator: Aggregation method for the ineteractions in different samples -'mean' or 'median'
        color: String indicating which color values to use 'interaction_values' or 'above_median_fraction'
        title: title of the plot
        save: path to save the plot
        ax: axes object in which to plot
        tight_layout
        show: whether to show the picture or not
        val_min, val_max: minimum and maximum values of the color scale
        return_data: whether to return the data used to make the plot
        save: Path to save the plot.png if desired
        color_map: colormap to use for the heatmap. If None, 'Reds' will be used.
        above_median_fr: pd.Series with the above median fraction values for each pair of attributes. If None, it will be computed using ad_dict, interaction_key_group and interaction_key_overall
        merged_group: pd.Series with the aggregated interaction values for each pair of attributes. If None, it will be computed using ad_dict and interaction_key_group
        fig_size: tuple with figure size. If None, it will be automatically determined based on the number of clusters and attributes.
        color: String indicating which color values to use 'interaction_values' or 'above_median_fraction' (the other will be used as size of the dots)

    Returns:
        ax or data if return_data
    '''
    assert color in ['interaction_values', 'above_median_fraction'], "color must be either 'interaction_values' or 'above_median_fraction'"

    if merged_group is None or above_median_fr is None:
        assert (ad_dict is not None) and (interaction_key_group is not None) and (interaction_key_overall is not None), 'you must provide either the merged_goup and above_median fraction, or either the ad_dict, interaction_key_group, interaction_key_overall'
        assert aggregator in ['mean', 'median'], "aggregator must be either 'mean' or 'median'"
        if merged_group is None:
            merged_group = aggregate_interactions(ad_dict, interaction_key= interaction_key_group, aggregator=aggregator)[interaction_key_group]
        if above_median_fr is None:
            above_median_fr = above_median_fraction(ad_dict, interaction_key_group=interaction_key_group, interaction_key_overall=interaction_key_overall)


    if color == 'interaction_values':
        color_df = merged_group.copy().reset_index()
        width_df = above_median_fr.copy().reset_index()
    else: 
        width_df = merged_group.copy().reset_index()
        color_df = above_median_fr.copy().reset_index()
    
    width_df.columns = ['attr_1', 'attr_2', 'width']
    color_df.columns = ['attr_1', 'attr_2', 'color']
    combined_df = color_df.merge(width_df)

    if ax:
        fig = ax.get_figure()
        show = False # do not automatically show plot if we provide axes
        return_data = False
    else:
        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_aspect('equal')
    
    if color == 'interaction_values':
        col = f'{aggregator}_{color}'
        cw_legend = f'Color = {aggregator}_{color}\nLine Width = above_median_fraction'
    else:
        col = color
        cw_legend = f'Color = {color}\nLine_Width = {aggregator}_interaction_value'
    
    if color_map is None:
        color_map = 'Reds'
    
    if val_min is None:
        val_min = color_df['color'].min()
    if val_max is None:
        val_max= color_df['color'].max()

    scatter = ax.scatter(
    x=combined_df['attr_1'],
    y=combined_df['attr_2'],
    c=combined_df['color'],        
    s=combined_df['width'] *1000,   
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
    if title:
        ax.set_title(
            title,
            fontsize=16,
            pad=20,
            weight='bold'
        )
    
    # Add text box at the top right
    ax.text(1.75, 1.10, 
        cw_legend, 
        transform=ax.transAxes, 
        fontsize=14,
        verticalalignment='top', 
        horizontalalignment='right', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    # IMPORTANT: Shrink the main plot area to make room for the box on the right/top
    #plt.subplots_adjust(right=0.8, top=0.8)
    ax.set_xlabel('attr', fontsize=13, labelpad=10)
    ax.set_ylabel('attr', fontsize=13, labelpad=10)

    # --- Ticks ---
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='both', labelsize=11)

    # --- Grid and layout ---
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- Dot Size Legend (Bigger & More Steps) ---
    from matplotlib.lines import Line2D

    # 1. Define 5 steps from the min to max of your width data
    w_min, w_max = combined_df['width'].min(), combined_df['width'].max()
    sample_widths = np.linspace(w_min, w_max, 5) 
    
    # 2. Create the "handles"
    size_handles = [
        Line2D([0], [0], 
               marker='o', 
               color='w', 
               label=f'{round(w, 2)}',
               markerfacecolor='gray', 
               markersize=np.sqrt(w * 1000), 
               alpha=0.8) 
        for w in sample_widths
    ]

    # 3. Increase the x-coordinate (1.45) to move it past the colorbar
    size_legend = ax.legend(
        handles=size_handles, 
        title="Width Value", 
        loc='upper left',          # Changed to upper left for easier anchoring
        bbox_to_anchor=(1.35, 0.9), # Move further right and slightly down
        frameon=True,
        fontsize=12,         
        title_fontsize=14,   
        labelspacing=1.8,    # Even more space for a "bigger" feel
        borderpad=1.5,       
        handletextpad=1.5    
    )
    
    ax.add_artist(size_legend)

    plt.subplots_adjust(right=0.65) # Shrink the plot more to make room on the right

    if tight_layout:
        fig.tight_layout()

    if save:
            savefig(fig, save)
    if show:
        fig.show()
        return ax
    else:
        return fig

    
    
    if return_data:
        return combined_df
    
    return ax



def plot_interactions_circos_plots(ad_dict: Dict[str,AnnData]=None, interaction_key_group:str=None, interaction_key_overall:str=None, aggregator: str = 'mean',merged_group:pd.Series=None,above_median_fr:pd.Series=None,  color: str = 'interaction_values', color_map: Dict[str,str] = None, title: str = None, save: str = None, ax: int = None, tight_layout: bool = False, show: bool = True, val_min: int=None, val_max:int = None, figsize: tuple = None, return_data: bool = False):
    '''
    Create circos plots for aggregated interactions across samples.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key_group: Key in ad.uns where the interactions DataFrame of the group/niche is stored
        interaction_key_overall: Key in ad.uns where the overall interactions DataFrame is stored
        aggregator: Aggregation method for the ineteractions in different samples -'mean' or 'median'
        color: String indicating which color values to use 'interaction_values' or 'above_median_fraction'
        title: title of the plot
        save: path to save the plot
        ax: axes object in which to plot
        tight_layout
        show: whether to show the picture or not
        val_min, val_max: minimum and maximum values of the color scale
        return_data: whether to return the data used to make the plot
        save: Path to save the plot.png if desired
        color_map: colormap to use for the heatmap. If None, 'Reds' will be used.
        above_median_fr: pd.Series with the above median fraction values for each pair of attributes. If None, it will be computed using ad_dict, interaction_key_group and interaction_key_overall
        merged_group: pd.Series with the aggregated interaction values for each pair of attributes. If None, it will be computed using ad_dict and interaction_key_group
        figsize: tuple with figure size. If None, it will be set to (20, 15) by default
        color: String indicating which color values to use 'interaction_values' or 'above_median_fraction' (the other will be used as thickness of the links)

    Returns:
        ax or data if return_data
    '''
    assert color in ['interaction_values', 'above_median_fraction'], "color must be either 'interaction_values' or 'above_median_fraction'"
    from pycirclize import Circos
    
    if merged_group is None or above_median_fr is None:
        assert (ad_dict is not None) and (interaction_key_group is not None) and (interaction_key_overall is not None), 'you must provide either the merged_goup and above_median fraction, or either the ad_dict, interaction_key_group, interaction_key_overall'
        assert aggregator in ['mean', 'median'], "aggregator must be either 'mean' or 'median'"
        if merged_group is None:
            merged_group = aggregate_interactions(ad_dict, interaction_key= interaction_key_group, aggregator=aggregator)[interaction_key_group]
        if above_median_fr is None:
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
    else:
        cell_color_map = color_map

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
    
    if ax is not None:
        fig = ax.get_figure()
        show = False # do not automatically show plot if we provide axes
        return_data = False
    else:
        if figsize is None:
            figsize = (20, 15)
        fig, ax = plt.subplots(
        figsize=figsize, 
        dpi=dpi, 
        subplot_kw={'projection': 'polar'})
        #ax.set_aspect('equal')


    circos.plotfig(ax=ax) # Plot directly on the handle

    if color == 'interaction_values':
        col = f'{aggregator}_{color}'
        cw_legend = f'Color = {aggregator}_{color} \nLine Width = above_median_fraction'
    else:
        col = color
        cw_legend = f'Color = {color} \nLine_Width = {aggregator}_interaction_value'
    
    # Add this to your code to create a clean legend box in the corner
    ax.text(1.05, 1.0, 
        cw_legend, 
        transform=ax.transAxes, 
        fontsize=16,
        verticalalignment='top', 
        horizontalalignment='left', # 'left' anchors it to the right of the plot
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    if title:    
        fig.suptitle(
            title,
            fontsize=20,
            fontweight="bold",
            y=1.05  # vertical position: 1.0 is top of figure
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

    if save:
            savefig(fig, save)
    if show:
        fig.show()
        return ax
    else:
        return fig
    
    if return_data: 
        return dicts
    
    return ax

def plot_interaction_heatmaps(ad_dict: Dict[str,AnnData]=None, interaction_key_group:str=None, aggregator: str = 'mean',  merged_group:pd.Series=None, color_map: Union[Dict[str,str], str] = None, title: str = None, save: str = None, ax: int = None, tight_layout: bool = False, show: bool = True, val_min: int=None, val_max:int = None, return_data: bool = False):
    '''
    Create heatmaps for aggregated interactions across samples.
    
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key_group: Key in ad.uns where the interactions DataFrame of the group/niche is stored
        aggregator: Aggregation method for the ineteractions in different samples -'mean' or 'median'
        color_map: cmap for the plot
        title: title of the plot
        save: path to save the plot
        ax: axes object in which to plot
        tight_layout
        show: whether to show the picture or not
        val_min, val_max: minimum and maximum values of the color scale
        return_data: whether to return the data used to make the plot
        save: Path to save the plot.png if desired
        
    Returns:
        ax or data if return_data
    '''
    if merged_group is None :
        assert (ad_dict is not None) and (interaction_key_group is not None), 'you must provide either the merged_goup, or the ad_dict, interaction_key_group, interaction_key_overall'
        assert aggregator in ['mean', 'median'], "aggregator must be either 'mean' or 'median'"
        merged_group = aggregate_interactions(ad_dict, interaction_key= interaction_key_group, aggregator=aggregator)[interaction_key_group]
       
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

    if save:
            savefig(fig, save)
    if show:
        fig.show()
    else:
        plt.close(fig)

    if return_data:
        return df

    return ax






    

