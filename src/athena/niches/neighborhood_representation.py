from athena.metrics.heterogeneity.metrics import abundance
import pandas as pd
from anndata import AnnData
from athena.utils.general import get_nx_graph_from_anndata
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, Union, List
import copy

#%%
def n_rep_filtering(ad: AnnData, graph_key: str, n_rep_key: str, min_neigh:int, key_added:str = None, inplace:bool = True):
    '''filter n_rep based on min_neigh
    Args:
        ad: AnnData instance.
        graph_key: Specifies the graph representation to use in ad.obsp.
        n_rep_key: Specifies the neighborhood representation to use in ad.uns.
        min_neigh: filtering thereshold.
        inplace: Whether to add the metric to the current AnnData instance or to return a new one.

    Returns: 
        if inplace -> add ad.uns[key_added]
        if not inplace -> copy of the AnnData instance with the above added.
    '''
    # generate a copy if necessary
    ad = ad if inplace else ad.copy()

    if key_added is None:
        key_added = f'{n_rep_key}_filter_{min_neigh}'

    assert min_neigh>0, 'min_neigh has to be set to an int >0'
    assert n_rep_key in ad.obsm.keys(), 'n_rep_key not in ad.obsm.keys()'


    n_rep_filt = ad.obsm[n_rep_key].copy()
    
    g = get_nx_graph_from_anndata(ad=ad, key=graph_key)  
    assert set(nx.nodes_with_selfloops(g)) == set(g.nodes()), 'the graph does not include self-loops'
    observation_ids = ad.obs.index
    for observation_id in observation_ids:
        n = g.degree(observation_id)
        if n < min_neigh:
            n_rep_filt.loc[observation_id] *= 0
    
    ad.obsm[key_added] = n_rep_filt
        
    if not inplace:
        return ad
    
    return


def n_rep_ad(ad: AnnData, attr_rep: str, mode_rep: str = 'proportion', graph_key: str = 'radius_80', key_added: str = None, n_filtering: bool = False, min_neigh: int = 0, inplace: bool=True):
    """Compute count or proportions of attr for each cell (with at least min_neighbors neighbors) in the AnnData object based on the specified topology.

    Args:
        ad: AnnData instance.
        attr_rep: Categorical feature in ad.obs to use for the neighborhood representation. 
        mode_rep: 'proportion' or 'counts' to specify the type of neighborhood representation to compute.
        graph_key: Specifies the graph representation to use in ad.obsp.
        key_added: Key added to ad.uns with the neighborhood representation.
        inplace: Whether to add the metric to the current AnnData instance or to return a new one.

    Returns: 
        if inplace -> ad.uns[key_added] with neighborhood representation of the anndata object
        if not inplace -> copy of AnnData instance with the above added.
    """
    g = get_nx_graph_from_anndata(ad=ad, key=graph_key)  
    assert set(nx.nodes_with_selfloops(g)) == set(g.nodes()), 'the graph does not include self-loops'

    assert attr_rep in ad.obs.columns, f'Attribute {attr_rep} not found in ad.obs.'
    assert mode_rep in ['proportion', 'counts'], f'Mode {mode_rep} not recognized. Use "proportion" or "counts".'
    assert (n_filtering==True and min_neigh>0) or (n_filtering==False and min_neigh==0), 'if n_filtering is True, min_neigh has to be provided and has to be set to an int >0. if n_filtering is False, min_neigh has to be set to 0'

    if key_added is None:
        key_added = f'n_rep_{attr_rep}_{mode_rep}_{graph_key}'

    # generate a copy if necessary
    ad = ad if inplace else ad.copy()
    
    # compute abundance 
    abundance(ad=ad, attr=attr_rep, mode=mode_rep, key_added=key_added, graph_key=graph_key, local=True, inplace=True)# inplace kept default True because a copy of ad has already been generated if inplace=False
    # filter cells with 
    if n_filtering:
        n_rep_filtering(ad=ad, graph_key=graph_key, n_rep_key=key_added, min_neigh=min_neigh) # inplace kept default True because a copy of ad has already been generated if inplace=False

    if not inplace:
        return ad

    return

def n_rep_ad_dict(ad_dict: Dict[str,AnnData], attr_rep: str, mode_rep: str = 'proportion', graph_key: str = 'radius_80', key_added: str = None, n_filtering: bool = False, min_neigh: int = 0, inplace: bool=True):
    """Compute count or proportions of attr for each cell (with at least min_neighbors neighbors) in the AnnData object based on the specified topology.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr_rep: Categorical feature in ad.obs to use for the neighborhood representation. 
        mode_rep: 'proportion' or 'counts' to specify the type of neighborhood representation to compute.
        graph_key: Specifies the graph representation to use in ad.obsp.
        key_added: Key added to ad.uns with the neighborhood representation.
        inplace: Whether to add the metric to the current AnnData instance or to return a new one.

    Returns: 
        if inplace -> ad_dict with ad.uns[key_added] with neighborhood representation of the anndata object added to each anndata object
        if not inplace -> copy of ad_dict instance with the above added.
    """
    assert (n_filtering==True and min_neigh>0) or (n_filtering==False and min_neigh==0), 'if n_filtering is True, min_neigh has to be provided and has to be set to an int >0. if n_filtering is False, min_neigh has to be set to 0'
    

    if key_added is None:
        key_added = f'n_rep_{attr_rep}_{mode_rep}_{graph_key}'

    # generate a copy if necessary
    ad_dict = ad_dict if inplace else copy.deepcopy(ad_dict)

    for ad in ad_dict.values():
        n_rep_ad(ad=ad, attr_rep=attr_rep, mode_rep=mode_rep, graph_key=graph_key, key_added=key_added, n_filtering=n_filtering, min_neigh=min_neigh) # inplace kept default True because ad_dict copy of ad has already been generated if inplace=False

    if not inplace:
        return ad_dict
    
    return

def aggregate_n_rep(ad_dict: Dict[str,AnnData], n_rep_key: str = None, attr_rep: str = None, mode_rep: str = None, graph_key: str = None, key_added: str = None, n_filtering: bool = False, min_neigh: int = 0, inplace: bool=True):
    """Compute count or proportions of attr for each cell (with at least min_neighbors neighbors) in the AnnData object based on the specified topology.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        n_rep_key: Specifies the neighborhood representation to use in ad.obsm.
        attr_rep: Categorical feature in ad.obs to use for the neighborhood representation. 
        mode_rep: 'proportion' or 'counts' to specify the type of neighborhood representation to compute.
        graph_key: Specifies the graph representation to use in ad.obsp.
        key_added: Key added to ad.obsm with the neighborhood representation.
        n_filtering: whether to filter out cells with less than min_neigh neighbors.
        min_neigh: if n_filtering is True -> cells with less than min_neigh neighbors are filtered out of the neighborhood representation
        inplace: Whether to add the metric to the current AnnData instance or to return a new one.

    Returns: 
        pd.Dataframe with 
            - aggregated neighborhood representations from all samples
            - multiindex 'sample_id' and 'observation_id'
    """
    if n_rep_key is None:
        assert (attr_rep and mode_rep and graph_key is not None), f"Missing required components: attr={attr_rep}, mode={mode_rep}, key={graph_key}"
        assert (n_filtering==True and min_neigh>0) or (n_filtering==False and min_neigh==0), 'if n_filtering is True, min_neigh has to be provided and has to be set to an int >0. if n_filtering is False, min_neigh has to be set to 0'
        # generate a copy if necessary
        ad_dict = ad_dict if inplace else copy.deepcopy(ad_dict)
        if key_added is None:
            key_added = f'n_rep_{attr_rep}_{mode_rep}_{graph_key}'
        n_rep_ad_dict(ad_dict=ad_dict, attr_rep=attr_rep, mode_rep=mode_rep, graph_key=graph_key, key_added=key_added, n_filtering=n_filtering, min_neigh=min_neigh)
        if n_filtering is None:
            n_rep_key = key_added
        else:
            n_rep_key = f'{key_added}_filter_{min_neigh}'
    
    aggr_n_rep_list = list()
    for sample_id, ad in ad_dict.items():
        n_rep_sample = ad.obsm[n_rep_key].copy()
        n_rep_sample.index = pd.MultiIndex.from_product([[sample_id], n_rep_sample.index], names=['sample_id', 'observation_id'])
        aggr_n_rep_list.append(n_rep_sample)
    
    aggr_n_rep = pd.concat(aggr_n_rep_list, axis=0)    
    # fill NaNs with 0 because some samples do not have all attrs
    aggr_n_rep = aggr_n_rep.fillna(0) 
    # drops all rows that consist entirely of zeros = rows of observations that have been filtered out due to low number of neighbors
    aggr_n_rep = aggr_n_rep[~(aggr_n_rep == 0).all(axis=1)]
    

    return aggr_n_rep
    


