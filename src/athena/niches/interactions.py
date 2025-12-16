# %%
import itertools
import numpy as np
import pandas as pd
from anndata import AnnData
from athena.utils.general import get_nx_graph_from_anndata
import networkx as nx
from typing import Dict, Union, List

#%%
def fix_adjacency_matrix(ad: AnnData, graph_key: str = 'radius_80'):
    '''
    Fix adjacency matrix to ensure zero diagonal => to avoid self-interactions
    Args:
        ad: AnnData object
        graph_key: Key in ad.obsp where the graph adjacency matrix is stored
    Returns:
        Undirected graph of graph_key without self-loops

    '''
    assert graph_key in ad.obsp.keys(), f'Graph key {graph_key} not found in ad.obsp.' 
    g = get_nx_graph_from_anndata(ad, key=graph_key) # get radius graph
    A = nx.to_numpy_array(g)
    assert np.allclose(A, A.T), "The adjacency matrix of the graph is not symmetric."
    np.fill_diagonal(A, 0)
    g_new = nx.from_numpy_array(A)
    mapping = {i: node for i, node in enumerate(list(g.nodes()))}
    g_new = nx.relabel_nodes(g_new, mapping)

    return g_new

def compute_interactions(ad: AnnData, graph_key: str, label_type: str) -> Dict[tuple, int]:
    '''
    Compute the proportion of interactions between cell types based on the directed graph g.

    Args:   
        ad: AnnData object
        graph_key: Key in ad.obsp where the graph adjacency matrix is stored
        label_type: The obs column in adata.obs that contains labels
    
    Returns:
        interactions_df: DataFrame with columns ['cell_type_1', 'cell_type_2', 'interaction_proportion']
    '''
    g =  fix_adjacency_matrix(ad, graph_key=graph_key)
    labels = ad.obs[label_type].unique().tolist()
    pairs = [tuple(sorted(t)) for t in list(itertools.combinations_with_replacement(labels, 2))]
    interactions = {pair: 0 for pair in pairs}
    for u,v in g.edges(): # (u,v) and (v,u) are counted as one interaction and appear once in the iteration because g is undirected
        if u==v:
            continue  # skip self-loops if any
        label_u = ad.obs.loc[u, label_type]
        label_v = ad.obs.loc[v, label_type]
        pair = tuple(sorted((label_u, label_v)))
        interactions[pair] +=1
    
    total = sum(interactions.values())
    interactions_proportions = {pair: count/total for pair, count in interactions.items()}
    interactions_df = pd.DataFrame.from_dict(interactions_proportions, orient='index', columns=['interaction_proportion'])
    interactions_df.index = pd.MultiIndex.from_tuples(interactions_df.index).set_names(['cell_type_1', 'cell_type_2'])
    interactions_df.columns = ['interaction_proportion']

    return interactions_df
    


def label_type_interactions(ad:AnnData,  label_type: str, graph_key: str = 'radius_80', group: Union[str, int, list] = 'whole_sample', obs_key: str = None):
    '''
    Compute interactions between cell types based on a graph stored in adata.obsp[graph_key].
    The interactions can be computed for the whole sample or per group in obs_key.

    Args:   
        ad: AnnData object
        label_type: The obs column in adata.obs that contains labels
        graph_key: Key in adata.obsp where the graph adjacency matrix is stored
        group: cells onto which compute interactions 
                - 'whole_sample' 
                - group name found in obs_key column to select cells from that group
                - 'per_group' to compute interactions for each group found in obs_key column
                - list of cell_ids
                - 'whole_sample_and_per_group' to compute both whole sample and per group interactions
        obs_key: If group is 'per_group', the obs column in adata.obs to group 
    
    Returns:
        Adds to ad.uns: DataFrame with columns ['cell_type_1', 'cell_type_2', 'interaction_proportion'] for each group

    '''
    if f'{graph_key}_directed' not in ad.obsp.keys():
        fix_adjacency_matrix(ad, graph_key=graph_key)
    
    ### Whole sample ###
    if group == 'whole_sample':
        if 'cell_type_interactions_whole_sample' not in ad.uns.keys():
            interactions_df = compute_interactions(ad, graph_key, label_type)
            ad.uns['cell_type_interactions_whole_sample'] = interactions_df
    
    ### Per group and optionally whole sample ###
    elif group == 'per_group'or group == 'whole_sample_and_per_group':
        assert obs_key is not None, "obs_key must be provided when group is 'per_group'."
        groups = ad.obs[obs_key].unique().tolist()

        ### if also whole sample ###
        if group == 'whole_sample_and_per_group':
            if 'cell_type_interactions_whole_sample' not in ad.uns.keys():
                interactions_df = compute_interactions(ad, graph_key, label_type)
                ad.uns['cell_type_interactions_whole_sample'] = interactions_df

        for group in groups:
            if f'cell_type_interactions_{obs_key}_{group}' not in ad.uns.keys():
                ad_group = ad[ad.obs[obs_key] == group, :]
                if ad_group.n_obs >13:                
                    interactions_df = compute_interactions(ad_group, graph_key, label_type)
                    ad.uns[f'cell_type_interactions_{obs_key}_{group}'] = interactions_df

    ### List of cell ids ###    
    elif isinstance(group, list):
        if f'cell_type_interactions_{obs_key}_{group}' not in ad.uns.keys():
            ad_group = ad[group, :]
            interactions_df = compute_interactions(ad_group, graph_key, label_type)
            ad.uns[f'cell_type_interactions_{obs_key}_{group}'] = interactions_df
    
    ### Specific group ###
    else:
        assert obs_key is not None, f"obs_key must be provided when group is {group}."
        if f'cell_type_interactions_{obs_key}_{group}' not in ad.uns.keys():
            ad_group = ad[ad.obs[obs_key] == group, :]
            if ad_group.n_obs >13:
                interactions_df = compute_interactions(ad_group, graph_key, label_type)
                ad.uns[f'cell_type_interactions_{obs_key}_{group}'] = interactions_df

    return

def label_type_interactions_dictionary(ad_dict: Dict[str, AnnData],  label_type: str, graph_key: str = 'radius_80', group: Union[str, int, list] = 'whole_sample', obs_key: str = None):
    '''
    Compute interactions between cell types for multiple AnnData objects stored in a dictionary.
    The interactions can be computed for the whole sample or per group in obs_key.

    Args:   
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        label_type: The obs column in adata.obs that contains labels
        graph_key: Key in adata.obsp where the graph adjacency matrix is stored
        group: cells onto which compute interactions 
                - 'whole_sample' 
                - group name found in obs_key column to select cells from that group
                - 'per_group' to compute interactions for each group found in obs_key column
                - list of cell_ids
                - 'whole_sample_and_per_group' to compute both whole sample and per group interactions
        obs_key: If group is 'per_group', the obs column in adata.obs to group 
    
    Returns:
        Adds to each AnnData.uns: DataFrame with columns ['cell_type_1', 'cell_type_2', 'interaction_proportion'] for each group

    '''
    for ad in ad_dict.values():
        label_type_interactions(ad, label_type, graph_key, group, obs_key)
    return

def aggregate_interactions(ad_dict: Dict[str,AnnData], interaction_key:Union[str, List[str]], aggregator: str = 'mean'):
    '''
    Aggregate interactions across multiple AnnData objects stored in a dictionary.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key: Key or list of keys in ad.uns where the interactions DataFrame is stored
        aggregator: Aggregation method - 'mean' or 'median'
    
    Returns:
        If interaction_key is str: pd.Series with aggregated interactions across samples
        If interaction_key is list: Dictionary of pd.Series with aggregated interactions across samples for each key
    '''
    assert aggregator in ['mean', 'median'], "Aggregator must be 'mean' or 'median'."
    
    ### Single interaction key ###
    if isinstance(interaction_key, str):
        aggregation = None
        for sample_id, ad in ad_dict.items():
            if interaction_key in ad.uns.keys():    # can not be there bc of filtering in previous steps
                interactions_df = ad.uns[interaction_key].copy()
                interactions_df = interactions_df.reset_index(['cell_type_1', 'cell_type_2'])
                interactions_df.columns = ['cell_type_1', 'cell_type_2', sample_id]               
                if aggregation is None:
                    aggregation = interactions_df
                else:
                    aggregation = pd.merge(aggregation, interactions_df, on=['cell_type_1', 'cell_type_2'], how='outer')
        aggregation = aggregation.set_index(['cell_type_1', 'cell_type_2']) 
        aggregation.fillna(0, inplace=True)
        if aggregator == 'mean':
             aggregation[f'{aggregator}_interaction']  = aggregation.mean(axis=1)
        elif aggregator == 'median':
            aggregation[f'{aggregator}_interaction'] = aggregation.median(axis=1)
        
        if aggregation is None:
            raise ValueError(f'No interactions found for key {interaction_key} in any AnnData object.')
        
        return aggregation[f'{aggregator}_interaction'] 
    
    
    ### Multiple interaction keys ###        
    elif isinstance(interaction_key, list):
        aggregation_dict = {}
        
        for sample_id, ad in ad_dict.items():
            for key in interaction_key:
                if key in ad.uns.keys():    # can not be there bc of filtering in previous steps
                    interactions_df = ad.uns[key].copy()
                    interactions_df = interactions_df.reset_index(['cell_type_1', 'cell_type_2'])
                    interactions_df.columns = ['cell_type_1', 'cell_type_2', sample_id]
                    if f'aggregation_{key}' not in aggregation_dict.keys():
                        aggregation_dict[f'aggregation_{key}'] = interactions_df
                    else:
                        aggregation_dict[f'aggregation_{key}'] = pd.merge(aggregation_dict[f'aggregation_{key}'], interactions_df, on=['cell_type_1', 'cell_type_2'], how='outer')
                
        for key in interaction_key:
            aggregation_dict[f'aggregation_{key}'] = aggregation_dict[f'aggregation_{key}'].set_index(['cell_type_1', 'cell_type_2']) 
            aggregation_dict[f'aggregation_{key}'].fillna(0, inplace=True)
            if aggregator == 'mean':
                aggregation_dict[f'aggregation_{key}'][f'{aggregator}_interaction'] = aggregation_dict[f'aggregation_{key}'].mean(axis=1)
            elif aggregator == 'median':
                aggregation_dict[f'aggregation_{key}'][f'{aggregator}_interaction'] = aggregation_dict[f'aggregation_{key}'].median(axis=1)
        
        return {key: aggregation_dict[f'aggregation_{key}'][f'{aggregator}_interaction'] for key in interaction_key}


def above_median_fraction(ad_dict: Dict[str,AnnData], interaction_key_group:str):
    '''
    Fraction of samples where the interaction proportion in a group is above the median interaction proportion across all samples. This is computed for each pair of cell types.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        interaction_key_group: Key in ad.uns where the group interactions DataFrame is stored

    Returns:
        pd.Series with fraction of samples above median interaction proportion for each pair of cell types
    '''
    interaction_key_whole_sample = interaction_key_group[:interaction_key_group.find('interactions') + len('interactions')]+ '_whole_sample'
    median_interaction_overall = aggregate_interactions(ad_dict, interaction_key_whole_sample, aggregator = 'median')
    above_median_count = pd.Series(0, index=median_interaction_overall.index, name='above_median_count')
    print(above_median_count)
    for sample_id, ad in ad_dict.items():
        if interaction_key_group in ad.uns.keys():
            interactions_df = ad.uns[interaction_key_group]
            for idx in interactions_df.index:
                if interactions_df.loc[idx].item() > median_interaction_overall[idx].item():
                    above_median_count[idx] +=1
    above_median_fraction = above_median_count / len(ad_dict)
    above_median_fraction.columns = ['above_median_fraction']
    return above_median_fraction



