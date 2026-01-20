# %%
import itertools
import numpy as np
import pandas as pd
from anndata import AnnData
from athena.utils.general import get_nx_graph_from_anndata
import networkx as nx
from typing import Dict, Union, List
from collections import defaultdict
import copy

#%%

def compute_interactions(ad: AnnData, graph_key: str, attr: str, mode: str = 'proportion') -> Dict[tuple, int]:
    '''
    Compute the proportion of interactions between attr types based on edges of graph g.

    Args:   
        ad: AnnData object
        graph_key: Key in ad.obsp where the graph adjacency matrix is stored
        attr: The obs column in adata.obs that contains the attr types among which count interactions 
    
    Returns:
        interactions_df: DataFrame with columns ['cell_type_1', 'cell_type_2', 'interaction_proportion']
    '''
    assert graph_key in ad.obsp, f'Graph key {graph_key} not found in ad.obsp.' 
    assert attr in ad.obs.columns, 'attr is not in ad.obs.column'
    assert mode in ['counts', 'proportion'], 'mode has to be "counts" or "proportion"'

    # get graph
    g = get_nx_graph_from_anndata(ad, key=graph_key)
    # remove self-edges
    g.remove_edges_from(nx.selfloop_edges(g))
    assert nx.number_of_selfloops(g)==0, 'the graph still has self-loops'

    labels = ad.obs[attr].unique().tolist()
    pairs = [tuple(sorted(t)) for t in list(itertools.combinations_with_replacement(labels, 2))]
    interactions = {pair: 0 for pair in pairs}
    for u,v in g.edges(): # (u,v) and (v,u) are counted as one interaction and appear once in the iteration because g is undirected
        label_u = ad.obs.loc[u, attr]
        label_v = ad.obs.loc[v, attr]
        pair = tuple(sorted((label_u, label_v)))
        interactions[pair] +=1
    
    if mode == 'counts':
        interactions_df = pd.DataFrame.from_dict(interactions, orient='index', columns=['interaction_counts'])
        interactions_df.index = pd.MultiIndex.from_tuples(interactions_df.index).set_names(['cell_type_1', 'cell_type_2'])
        interactions_df.columns = ['interaction_counts']

    else:    
        total = sum(interactions.values())
        interactions_proportions = {pair: count/total for pair, count in interactions.items()}
        interactions_df = pd.DataFrame.from_dict(interactions_proportions, orient='index', columns=['interaction_proportion'])
        interactions_df.index = pd.MultiIndex.from_tuples(interactions_df.index).set_names(['cell_type_1', 'cell_type_2'])
        interactions_df.columns = ['interaction_proportion']

    return interactions_df
    
def attr_interactions_ad(ad:AnnData,  attr: str, graph_key: str = 'radius_80', group_key: str = None, overall: bool = False, min_obs: int = 0, mode: str = 'proportion', key_added: str = None, inplace: bool = True):
    '''
    Compute interactions between cell types based on a graph stored in adata.obsp[graph_key].
    The interactions can be computed for the whole sample (overall) or/and per group in group_key.

    Args:   
        ad: AnnData object
        attr: The obs column in adata.obs that contains labels
        graph_key: Key in adata.obsp where the graph adjacency matrix is stored
        group_key: If group is 'per_group', the obs column in adata.obs to group 
        overall: True if the interactions have to be computed for the whole sample 
        min_obs: minimum numbr of observations in a group to compute the interactions
        mode: whether to compute the counts or proprortion of interactions. must be 'counts' or 'proportion'
        key_added: key added to the ad.uns
        inplace: whether to add the interaction computation to the provided ad or to a copy of ad
    
    Returns:
        Adds to ad.uns: DataFrame with columns ['cell_type_1', 'cell_type_2', 'interaction_proportion'] for each group

    '''
    assert group_key is not None or overall, 'Must provide group_key or set overall=True'
    assert group_key in ad.obs.columns, 'group_key is not in ad.obs.column'
    assert attr in ad.obs.columns, 'attr is not in ad.obs.column'
    assert graph_key in ad.obsp, 'graph_key is not in ad.obsp'
    assert mode in ['counts', 'proportion'], 'mode has to be "counts" or "proportion"'

    # generate a copy if necessary
    ad = ad if inplace else ad.copy()

    if key_added is None:
        key_added = f'{attr}_interactions'

    if overall: 
        interactions_df = compute_interactions(ad=ad, graph_key=graph_key, attr=attr, mode=mode)
        if f'{key_added}_overall' in ad.uns.keys():
            del ad.uns[f'{attr}_overall']
        ad.uns[f'{attr}_overall'] = interactions_df

    if group_key:
        groups = ad.obs[group_key].dropna().unique().tolist()
        for group in groups: 
            ad_group = ad[ad.obs.index[ad.obs[group_key] == group], :]
            if ad_group.n_obs > min_obs:  
                interactions_df = compute_interactions(ad= ad_group, graph_key=graph_key, attr=attr, mode=mode)
                if f'{key_added}_{group}' in ad.uns.keys():
                    del ad.uns[f'{key_added}_{group}']
                ad.uns[f'{key_added}_{group}'] = interactions_df

    return

def attr_interactions_ad_dict(ad_dict: Dict[str, AnnData],   attr: str, graph_key: str = 'radius_80', group_key: str = None, overall: bool = False, min_obs: int = 0, mode: str = 'proportion', key_added: str = None, inplace: bool = True):
    '''
    Compute interactions between cell types based on a graph stored in adata.obsp[graph_key].
    The interactions can be computed for the whole sample (overall) or/and per group in group_key.

    Args:   
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr: The obs column in adata.obs that contains labels
        graph_key: Key in adata.obsp where the graph adjacency matrix is stored
        group_key: If group is 'per_group', the obs column in adata.obs to group 
        overall: True if the interactions have to be computed for the whole sample 
        min_obs: minimum numbr of observations in a group to compute the interactions
        mode: whether to compute the counts or proprortion of interactions. must be 'counts' or 'proportion'
        key_added: key added to the ad.uns
        inplace: whether to add the interaction computation to the provided ads in the provided ad_dict or to a copy of them
    
    Returns:
        to each ad, adds to ad.uns: DataFrame with columns ['cell_type_1', 'cell_type_2', 'interaction_proportion'] for each group
    '''
    assert group_key is not None or overall, 'Must provide group_key or set overall=True'
    assert mode in ['counts', 'proportion'], 'mode has to be "counts" or "proportion"'
    # generate a copy if necessary
    ad_dict = ad_dict if inplace else copy.deepcopy(ad_dict)

    for ad in ad_dict.values():
        attr_interactions_ad(ad=ad, attr=attr, graph_key=graph_key, group_key=group_key, overall=overall, min_obs=min_obs, mode=mode, key_added=key_added)
    return

def aggregate_interactions(ad_dict: Dict[str,AnnData], interaction_key:Union[str, List[str]], aggregator: str = 'mean'):
    '''
    Aggregate interactions across multiple AnnData objects stored in a dictionary.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key: Key or list of keys in ad.uns where the interactions DataFrame is stored
        aggregator: Aggregation method - 'mean' or 'median'
    
    Returns:
       Dictionary of pd.Series with aggregated interactions across samples for each key/for one key if one interaction_key is provided
    '''
    assert aggregator in ['mean', 'median'], 'Aggregator must be "mean" or "median"'
    
    collection = defaultdict(list)
    
    for sample_id, ad in ad_dict.items():
        keys = [interaction_key] if isinstance(interaction_key, str) else interaction_key
        for key in keys:
            if key in ad.uns.keys():    # can not be there bc of filtering in previous steps
                int_df = ad.uns[key].copy()
                int_df.columns = [sample_id]     
                collection[key].append(int_df)
                
    if not collection:
        raise ValueError(f"None of the keys {interaction_key} were found in ad_dict.")
                    
    results = {}

    for key, dfs in collection.items():
        agg_df = pd.concat(dfs, axis=1).fillna(0)
        results[key] = agg_df.agg(aggregator, axis=1)

    return results

def above_median_fraction(ad_dict: Dict[str,AnnData], interaction_key_group:str, interaction_key_overall:str):
    '''
    Fraction of samples where the interaction proportion in a group is above the median interaction proportion across all samples. This is computed for each pair of cell types.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key_group: Key in ad.uns where the group interactions pd.Series is stored
        interaction_key_overall: Key in ad.uns where the overall interactions pd.Series is stored 

    Returns:
        pd.Series with fraction of samples above median interaction proportion for each pair of cell types
    '''
    median_interaction_overall = aggregate_interactions(ad_dict, interaction_key_overall, aggregator = 'median')[interaction_key_overall]
    above_median_count = pd.Series(0, index=median_interaction_overall.index, name='above_median_count')
    samples_with_group = 0
    for ad in ad_dict.values():
        if interaction_key_group in ad.uns.keys():
            samples_with_group +=1
            interactions_df = ad.uns[interaction_key_group]
            for idx in interactions_df.index:
                if interactions_df.loc[idx].item() > median_interaction_overall[idx].item():
                    above_median_count[idx] +=1

    if samples_with_group == 0:
        raise ValueError(f"{interaction_key_group} is not in any AnnData object in ad_dict.")
    
    above_median_fraction = above_median_count / samples_with_group
    above_median_fraction.columns = ['above_median_fraction']
    return above_median_fraction



