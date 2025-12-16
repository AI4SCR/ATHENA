from athena.metrics.heterogeneity.metrics import abundance
from random import seed
import numpy as np
import pandas as pd
from anndata import AnnData
from athena.utils.general import get_nx_graph_from_anndata
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, Union, List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#%%
def get_neighborhoods(ad: AnnData, graph_key: str = 'radius_80', inplace: bool = True):
    """Extract neighbor cells for each cell in the AnnData object based on the specified topology.

    Args:
        ad: AnnData instance
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.

    Returns: 
        add to .obs for each cell (key in the neighbors dict) a list of its direct neighbors (value in the neighbors dict) in the column neighbors_graph_key.

    """
    if f'neighbors_{graph_key}' in ad.obs.columns:
        return
    assert graph_key in ad.obsp.keys(), f'Graph key {graph_key} not found in ad.obsp.'

    g = get_nx_graph_from_anndata(ad=ad, key=graph_key) 
    neigh = defaultdict(list)
    for u, v in g.edges:
        neigh[u].append(v)
        neigh[v].append(u)  
    
    ad.obs[f'neighbors_{graph_key}'] = neigh
    if not inplace:
        return neigh
    return

def save_neighborhood_representations(representations: pd.Dataframe): 

    return


def compute_neighborhood_representations(ad: AnnData, attr: str, mode: str = 'proportion', graph_key: str = 'radius_80', min_neighbors: int = 5, inplace: bool=True):
    """Compute count and proportions of label_type for each cell in the AnnData object based on the specified topology.

    Args:
        ad: AnnData instance
        attr: Categorical feature in ad.obs to use for the neighborhood representation. 
        mode: 'proportion' or 'counts' to specify the type of neighborhood representation to compute.
        graph_key: Specifies the graph representation to use in ad.obsp.
        min_neighbors: Minimum number of neighbors required to compute the neighborhood representation.
        inplace: Whether to add the metric to the current AnnData instance or to return a new one.

    Returns: 
        if not inplace -> pd.DataFrame of neighborhood representations 
        id inplace -> None
            ad.obsm[f'neighborhood_representation_{attr}_{mode}_{graph_key}']: np.ndarray of shape (n_cells, n_unique_attr) 
                    0 or 0.0 in all attr if cell has less than min_neighbors neighbors.
            ad.uns[f'neighborhood_representation_{attr}_{mode}_{graph_key}_columns']: List of attribute categories corresponding to the columns in the obsm matrix.
        
        

    """
    assert attr in ad.obs.columns, f'Attribute {attr} not found in ad.obs.'
    assert mode in ['proportion', 'counts'], f'Mode {mode} not recognized. Use "proportion" or "counts".'

    if f'neighbors_{graph_key}' not in ad.obs.columns:
        get_neighborhoods(ad=ad, graph_key=graph_key, inplace=True)

    attrs = ad.obs[attr].unique()
    if mode == 'proportion':
        neighborhood_representations = pd.DataFrame(0.0, index=ad.obs.index, columns=attrs)
    elif mode == 'counts':
        neighborhood_representations = pd.DataFrame(0, index=ad.obs.index, columns=attrs)

    for cell_id in ad.obs.index:
        neighbors = ad.obs.loc[cell_id, f'neighbors_{graph_key}']
        if cell_id not in neighbors:
            neighbors.append(cell_id) #include self
        if len(neighbors) < min_neighbors:
            continue #leave as zeros
        else:
            ad_neighborhood = ad[neighbors]
            neighborhood_repr = abundance(ad=ad_neighborhood, attr=attr, mode=mode, inplace=False)
            neighborhood_representations.loc[cell_id, neighborhood_repr.index] = neighborhood_repr.values
    
    assert neighborhood_representations.index.equals(ad.obs_names), "Row indices of neighborhood representations do not match ad.obs_names."
    if inplace:
        ad.obsm[f'neighborhood_representation_{attr}_{mode}_{graph_key}'] = neighborhood_representations.values
        ad.uns[f'neighborhood_representation_{attr}_{mode}_{graph_key}_columns'] = neighborhood_representations.columns.tolist()
    else:
        return neighborhood_representations

    return

def get_neighborhood_representations(ad: AnnData, attr: str, mode: str = 'proportion', graph_key: str = 'radius_80', min_neighbors: int = 5):
    '''Retrieve neighborhood representations from the AnnData object or compute them if not present.

    Args:
        ad: AnnData instance
        attr: Categorical feature in ad.obs used for the neighborhood representation. 
        mode: 'proportion' or 'counts' specifying the type of neighborhood representation.
        graph_key: Specifies the graph representation used in ad.obsp.
        min_neighbors: Minimum number of neighbors required to compute the neighborhood representation.
    
    Returns:   
        filtered pd.DataFrame of neighborhood representations
    '''
    if f'neighborhood_representation_{attr}_{mode}_{graph_key}' in ad.obsm.keys() and f'neighborhood_representation_{attr}_{mode}_{graph_key}_columns' in ad.uns.keys():
        neighborhood_representation_name = f'{attr}_{mode}_{graph_key}'

        neighborhood_representations_data = ad.obsm[f'neighborhood_representation_{neighborhood_representation_name}']
        row_names = ad.obs_names.tolist()
        columns_names = ad.uns[f'neighborhood_representation_{neighborhood_representation_name}_columns']
        
        if neighborhood_representations_data.shape[0] != len(row_names):
            raise ValueError("Row count mismatch between .obsm matrix and .obs_names.")
        if neighborhood_representations_data.shape[1] != len(columns_names):
            raise ValueError("Column count mismatch between .obsm matrix and .uns column names.")
        
        neighborhood_representation = pd.DataFrame(
            data=neighborhood_representations_data, 
            index=row_names, 
            columns=columns_names
        )
    else:
        neighborhood_representation = compute_neighborhood_representations(ad=ad, attr=attr, mode=mode, graph_key=graph_key, min_neighbors=min_neighbors, inplace=False)
    
    # filter out the rows that are all zeros
    if type(neighborhood_representation.values) == int:
        neighborhood_representation = neighborhood_representation.loc[~(neighborhood_representation==0).all(axis=1)]
    elif type(neighborhood_representation.values) == float:
        neighborhood_representation = neighborhood_representation.loc[~(neighborhood_representation==0.0).all(axis=1)]

    return neighborhood_representation


def get_merged_neighborhood_representations(ad_dict: Dict[str, AnnData], attr: str, mode: str = 'proportion', graph_key: str = 'radius_80', min_neighbors: int = 5):
    '''Retrieve and merge neighborhood representations from the AnnData objects.

    Args:
         ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr: Categorical feature in ad.obs used for the neighborhood representation. 
        mode: 'proportion' or 'counts' specifying the type of neighborhood representation.
        graph_key: Specifies the graph representation used in ad.obsp.
        min_neighbors: Minimum number of neighbors required to compute the neighborhood representation.
    
    Returns:   
        filtered pd.DataFrame of neighborhood representations merged
            with multiindex (sample_id, cell_id)
    '''

    merged_neighborhood_representations_list = []
    for sample_id, ad in ad_dict.items():
        neighborhood_representation = get_neighborhood_representations(ad=ad, attr=attr, mode=mode, graph_key=graph_key, min_neighbors=min_neighbors)
        neighborhood_representation.index = pd.MultiIndex.from_product([[sample_id], neighborhood_representation.index], names=['sample_id', 'cell_id'])
        assert neighborhood_representation.index.get_level_values('cell_id').equals(ad.obs_names), 'cell_id level of MultiIndex does not match ad.obs_names.'
        merged_neighborhood_representations_list.append(neighborhood_representation)
    
    assert len(merged_neighborhood_representations_list) == len(ad_dict.keys()), 'Number of neighborhood representations to merge does not match number of samples in ad_dict.'
    
    merged_neighborhood_representations = pd.concat(merged_neighborhood_representations_list, axis=0)
    if mode == 'proportion':
        merged_neighborhood_representations.fillna(0.0, inplace=True)
    elif mode == 'counts':
        merged_neighborhood_representations.fillna(0, inplace=True)
    

    
    return merged_neighborhood_representations