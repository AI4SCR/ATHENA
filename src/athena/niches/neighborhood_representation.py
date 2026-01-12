from athena.metrics.heterogeneity.metrics import abundance
import pandas as pd
from anndata import AnnData
from athena.utils.general import get_nx_graph_from_anndata
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, Union, List
import copy

#%%
def neigh_rep_filtering(ad: AnnData, graph_key: str, neigh_rep_key: str, min_neigh:int, key_added:str = None, inplace:bool = True):
    '''filter neigh_rep based on min_neigh
    Args:
        ad: AnnData instance.
        graph_key: Specifies the graph representation to use in ad.obsp.
        neigh_rep_key: Specifies the neighborhood representation to use in ad.uns.
        min_neigh: filtering thereshold.
        inplace: Whether to add the metric to the current AnnData instance or to return a new one.

    Returns: 
        if inplace -> add ad.uns[key_added]
        if not inplace -> copy of the AnnData instance with the above added.
    '''
    # generate a copy if necessary
    ad = ad if inplace else ad.copy()

    if key_added is None:
        key_added = f'{neigh_rep_key}_filter_{min_neigh}'

    assert min_neigh>0, 'min_neigh has to be set to an int >0'
    assert neigh_rep_key in ad.obsm.keys(), 'neigh_rep_key not in ad.obsm.keys()'


    neigh_rep_filt = ad.obsm[neigh_rep_key].copy()
    
    g = get_nx_graph_from_anndata(ad=ad, key=graph_key)  
    assert set(nx.nodes_with_selfloops(g)) == set(g.nodes()), 'the graph does not include self-loops'
    print('here')
    observation_ids = ad.obs.index
    for observation_id in observation_ids:
        print('check filtering')
        n = g.degree(observation_id)
        if n < min_neigh:
            neigh_rep_filt.loc[observation_id] *= 0
    
    ad.obsm[key_added] = neigh_rep_filt
        
    if not inplace:
        return ad
    
    return


def neigh_rep_ad(ad: AnnData, attr_rep: str, mode_rep: str = 'proportion', graph_key: str = 'radius_80', key_added: str = None, neigh_filtering: bool = False, min_neigh: int = 0, inplace: bool=True):
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
    assert (neigh_filtering==True and min_neigh>0) or (neigh_filtering==False and min_neigh==0), 'if neigh_filtering is True, min_neigh has to be provided and has to be set to an int >0. if neigh_filtering is False, min_neigh has to be set to 0'

    if key_added is None:
        key_added = f'neigh_rep_{attr_rep}_{mode_rep}_{graph_key}'

    # generate a copy if necessary
    ad = ad if inplace else ad.copy()
    
    # compute abundance 
    abundance(ad=ad, attr=attr_rep, mode=mode_rep, key_added=key_added, graph_key=graph_key, local=True, inplace=True)# inplace kept default True because a copy of ad has already been generated if inplace=False
    print(ad.obsm.keys())
    # filter cells with 
    if neigh_filtering:
        neigh_rep_filtering(ad=ad, graph_key=graph_key, neigh_rep_key=key_added, min_neigh=min_neigh) # inplace kept default True because a copy of ad has already been generated if inplace=False

    if not inplace:
        return ad

    return

def neigh_rep_ad_dict(ad_dict: Dict[str,AnnData], attr_rep: str, mode_rep: str = 'proportion', graph_key: str = 'radius_80', key_added: str = None, neigh_filtering: bool = False, min_neigh: int = 0, inplace: bool=True):
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
    assert (neigh_filtering==True and min_neigh>0) or (neigh_filtering==False and min_neigh==0), 'if neigh_filtering is True, min_neigh has to be provided and has to be set to an int >0. if neigh_filtering is False, min_neigh has to be set to 0'
    

    if key_added is None:
        key_added = f'neigh_rep_{attr_rep}_{mode_rep}_{graph_key}'

    # generate a copy if necessary
    ad_dict = ad_dict if inplace else copy.deepcopy(ad_dict)

    for ad in ad_dict.values():
        neigh_rep_ad(ad=ad, attr_rep=attr_rep, mode_rep=mode_rep, graph_key=graph_key, key_added=key_added, neigh_filtering=neigh_filtering, min_neigh=min_neigh) # inplace kept default True because ad_dict copy of ad has already been generated if inplace=False

    if not inplace:
        return ad_dict
    
    return

def aggregate_neigh_rep(ad_dict: Dict[str,AnnData], neigh_rep_key: str = None, attr_rep: str = None, mode_rep: str = None, graph_key: str = None, key_added: str = None, neigh_filtering: bool = False, min_neigh: int = 0, inplace: bool=True):
    """Compute count or proportions of attr for each cell (with at least min_neighbors neighbors) in the AnnData object based on the specified topology.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        neigh_rep_key: Specifies the neighborhood representation to use in ad.uns.
        attr_rep: Categorical feature in ad.obs to use for the neighborhood representation. 
        mode_rep: 'proportion' or 'counts' to specify the type of neighborhood representation to compute.
        graph_key: Specifies the graph representation to use in ad.obsp.
        key_added: Key added to ad.uns with the neighborhood representation.
        inplace: Whether to add the metric to the current AnnData instance or to return a new one.

    Returns: 
        pd.Dataframe with 
            - aggregated neighborhood representations from all samples
            - multiindex 'sample_id' and 'observation_id'
    """
    if neigh_rep_key is None:
        assert (attr_rep and mode_rep and graph_key is not None), f"Missing required components: attr={attr_rep}, mode={mode_rep}, key={graph_key}"
        assert (neigh_filtering==True and min_neigh>0) or (neigh_filtering==False and min_neigh==0), 'if neigh_filtering is True, min_neigh has to be provided and has to be set to an int >0. if neigh_filtering is False, min_neigh has to be set to 0'
        # generate a copy if necessary
        ad_dict = ad_dict if inplace else copy.deepcopy(ad_dict)
        if key_added is None:
            key_added = f'neigh_rep_{attr_rep}_{mode_rep}_{graph_key}'
        neigh_rep_ad_dict(ad_dict=ad_dict, attr_rep=attr_rep, mode_rep=mode_rep, graph_key=graph_key, key_added=key_added, neigh_filtering=neigh_filtering, min_neigh=min_neigh)
        if neigh_filtering is None:
            neigh_rep_key = key_added
        else:
            neigh_rep_key = f'{key_added}_filter_{min_neigh}'
    
    aggr_neigh_rep_list = list()
    for sample_id, ad in ad_dict.items():
        neigh_rep_sample = ad.obsm[neigh_rep_key].copy()
        neigh_rep_sample.index = pd.MultiIndex.from_product([[sample_id], neigh_rep_sample.index], names=['sample_id', 'observation_id'])
        aggr_neigh_rep_list.append(neigh_rep_sample)
    
    aggr_neigh_rep = pd.concat(aggr_neigh_rep_list, axis=0)    
    # fill NaNs with 0 because some samples do not have all attrs
    aggr_neigh_rep = aggr_neigh_rep.fillna(0) 
    # drops all rows that consist entirely of zeros = rows of observations that have been filtered out due to low number of neighbors
    aggr_neigh_rep = aggr_neigh_rep[~(aggr_neigh_rep == 0).all(axis=1)]
    

    return aggr_neigh_rep
    


