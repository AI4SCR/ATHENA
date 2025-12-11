# %%
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
# %%
def get_neighborhoods(ad: AnnData, graph_key: str = 'radius_80', return_results: bool = False):
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
    if return_results:
        return neigh
    return

def neighborhood_representation(ad: AnnData, label_type: str, graph_key: str = 'radius_80'):
    """Compute count and proportions of label_type for each cell in the AnnData object based on the specified topology.

    Args:
        ad: AnnData instance
        label_type: The label type to filters neighbors by.
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.

    Returns: 
        add two columns in .obs = f'proportions_{label_type}_{graph_key}' and f'counts_{label_type}_{graph_key}'
            -> for each cell there is a dict with key labels (of the chosen label_type) and values as proportions/counts of those labels among the cell's direct neighbors

    """    
    if graph_key not in ad.obsp.keys():
        raise ValueError(f'Graph key {graph_key} not found in ad.obsp.')
    if label_type not in ad.obs.columns:
        raise ValueError(f'Label type {label_type} not found in ad.obs.')

    if f'counts_{label_type}_{graph_key}' in ad.obs.columns and f'proportions_{label_type}_{graph_key}' in ad.obs.columns:
        return
    if f'neighbors_{graph_key}' not in ad.obs.columns:
        get_neighborhoods(ad, graph_key=graph_key)
    
    unique_labels = ad.obs[label_type].unique()
    label_series = ad.obs[label_type]
    neighbors = ad.obs[f'neighbors_{graph_key}']

    ad.obs[f'counts_{label_type}_{graph_key}'] = None
    ad.obs[f'proportions_{label_type}_{graph_key}'] = None
    for cell_id in ad.obs.index:
        neigh = neighbors.loc[cell_id]
        neigh_labels = label_series.loc[neigh]
        counts_series = neigh_labels.value_counts()
        counts = counts_series.reindex(unique_labels, fill_value=0).to_dict()
        proportions = {label: counts[label]/len(neigh) if len(neigh) > 0 else 0 for label in counts}
        ad.obs.at[cell_id, f'counts_{label_type}_{graph_key}'] = counts
        ad.obs.at[cell_id, f'proportions_{label_type}_{graph_key}'] = proportions
    return

def neighborhood_representation_merged(ad_dict: Dict[str, AnnData], label_type: str, graph_key: str = 'radius_80', neighborhood_repr: str = 'proportions'):
    """Merge the count or proportions of label_type for each cell in the AnnData object based on the specified topology.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        label_type: The label type to filters neighbors by -> has to be present in ad.obs.
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        neighborhood_repr: 'proportions' or 'counts' to specify which representation to use.

    Returns: 
        dataframe containing the merged neighborhood representation for all samples in ad_dict (count/proportions of label_type among neighbors for each cell)
    """
    merged = None    
    for sample_id in ad_dict.keys():
        ad = ad_dict[sample_id]
        if f'{neighborhood_repr}_{label_type}_{graph_key}' not in ad.obs.columns:
            neighborhood_representation(ad, label_type=label_type, graph_key=graph_key)
        sample_repr = ad.obs[f'{neighborhood_repr}_{label_type}_{graph_key}']
        
        sample_df = pd.DataFrame.from_records(
        data=sample_repr.tolist(),  # The data to flatten
        index=sample_repr.index)
        # Create MultiIndex for the merged DataFrame with sample_id and cell_id as levels
        sample_ids = [sample_id] * len(sample_df)
        sample_df.index = pd.MultiIndex.from_arrays(
            arrays=[sample_ids, sample_df.index], # [Outer Level (Sample ID), Inner Level (Cell ID)]
            names=['sample_id', 'cell_id']
        )
        if merged is None:
            merged = sample_df
        else:
            merged = pd.concat([merged, sample_df], axis=0)
    
    # Fill NaN values with 0 and convert to int
    merged = merged.fillna(0).astype(float)
    assert np.isclose(merged.sum(axis=1), 1.0).all(), "The sum of all cells' neighborhood label_type proportions is not equal to 1."
    return merged


def k_means_clustering(ad_dict: Dict[str, AnnData], label_type: str, graph_key: str = 'radius_80', neighborhood_repr: str = 'proportions', k: Union[int, List[int]] = 4, random_seeds: Union[int, List[int]] = 42, merged: pd.DataFrame = None  ):
    """K-means clustering of the merged (all samples together) local neighborhood representation of cells.

    if there are more than one random seed provided -> for each k the clustering with the lowest inertia will be chosen and added to the anndata objects.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        label_type: The label type to filters neighbors by -> has to be present in ad.obs.
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        neighborhood_representation: 'proportions' or 'counts' to specify which representation to use.
        k: Number of clusters or list of number of clusters for k-means.
        random_seeds: Random seed or list of random seeds to run k-means with each k.
        merged: Precomputed merged DataFrame of neighborhood representations. If None, it will be computed.

    Returns: 
        add a columns in .obs f'k_means_{i}_{neighborhood_representation}_{label_type}_{graph_key}' for each k 
    """
    if merged is None:
        merged = neighborhood_representation_merged(ad_dict, label_type=label_type, graph_key=graph_key, neighborhood_repr=neighborhood_repr)
    merged_results = merged.copy()
    if type(k) == int:
        k = [k]
    clustering_evaluation = pd.DataFrame(columns=['clustering_run', 'inertia', 'silhouette'])
    print('start:done')
    for i in k:
        # if any ad.obs does not have the clustering result yet, compute it
        #check = any(f'k_means_{i}_{neighborhood_repr}_{label_type}_{graph_key}' not in ad.obs.columns for ad in ad_dict.values())
        #if check:
        if type(random_seeds)==int:
            print('one seed -> clustering started')
            kmeans = KMeans(n_clusters=i, random_state=seed, n_init='auto')
            kmeans.fit(merged.values) 
            inertia = kmeans.inertia_
            silhouette = silhouette_score(merged.values, labels)
            labels = kmeans.labels_
            merged_results[f'k_means_{i}_{neighborhood_repr}_{label_type}_{graph_key}_seed_{seed}'] = labels
            new_row = {'clustering_run': f'k_means_{i}_{neighborhood_repr}_{label_type}_{graph_key}_seed_{seed}','inertia':kmeans.inertia_, 'silhouette': silhouette_score(merged.values, labels)}
            clustering_evaluation = pd.concat([clustering_evaluation, pd.DataFrame([new_row])], ignore_index=True)
        else: 
            print('more seeds -> clustering started')
            seed_metrics = {}   
            merged_seeds = merged.copy()         
            for seed in (random_seeds if type(random_seeds) == list else [random_seeds]):
                print(f'seed {seed}')
                kmeans = KMeans(n_clusters=i, random_state=seed, n_init='auto')
                kmeans.fit(merged.values) 
                labels = kmeans.labels_
                merged_seeds[f'k_means_{i}_{neighborhood_repr}_{label_type}_{graph_key}_seed_{seed}'] = labels
                seed_metrics[seed] = kmeans.inertia_ 
            best_seed = min(seed_metrics, key=lambda x: seed_metrics[x])
            merged_results[f'k_means_{i}_{neighborhood_repr}_{label_type}_{graph_key}_seed_{best_seed}'] = merged_seeds[f'k_means_{i}_{neighborhood_repr}_{label_type}_{graph_key}_seed_{best_seed}']
            silhouette = silhouette_score(merged.values, merged_results[f'k_means_{i}_{neighborhood_repr}_{label_type}_{graph_key}_seed_{best_seed}'])
            new_row = {'clustering_run': f'k_means_{i}_{neighborhood_repr}_{label_type}_{graph_key}_seed_{best_seed}','inertia':seed_metrics[best_seed], 'silhouette': silhouette}
            clustering_evaluation = pd.concat([clustering_evaluation, pd.DataFrame([new_row])], ignore_index=True)

    merged_res = merged_results.drop(columns=[col for col in merged_results.columns if 'k_means' not in col])
    print('adding the results to anndata objs')
    for sample_id in ad_dict.keys():
        ad = ad_dict[sample_id]
        merged_sample = merged_res.loc[sample_id] 
        ad.obs = ad.obs.join(merged_sample, how='left')
        if 'clustering_evaluation' not in ad.uns.keys():
            ad.uns['clustering_evaluation'] = clustering_evaluation
        else:
            ad.uns['clustering_evaluation'] = pd.concat([ad.uns['clustering_evaluation'], clustering_evaluation], ignore_index=True)

    return

def merged_info_anndata_dict(ad_dict: Dict[str, AnnData], obs_key: Union[str, List[str]]):
    """Merge the count or proportions of label_type for each cell in the AnnData object based on the specified topology.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        obs_key: .obs column or list of columns to merge in one dataframe

    Returns: 
        dataframe containing the merged .obs columns of the anndata objs in the ad_dict 
    """
    merged = None    
    for sample_id in ad_dict.keys():
        ad = ad_dict[sample_id]
        sample_df = ad.obs[obs_key]
        # Create MultiIndex for the merged DataFrame with sample_id and cell_id as levels
        sample_ids = [sample_id] * len(sample_df)
        sample_df.index = pd.MultiIndex.from_arrays(
            arrays=[sample_ids, sample_df.index], # [Outer Level (Sample ID), Inner Level (Cell ID)]
            names=['sample_id', 'cell_id']
        )
        if merged is None:
            merged = sample_df
        else:
            merged = pd.concat([merged, sample_df], axis=0)
    
    return merged


### CLUSTER ANALYSIS FUNCTIONS ###

def z_scores(ad_dict, clustering_key = str, label_type = str):
    '''
    Compute z-scores of label_type enrichment in each cluster.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        clustering_key (str): Key in AnnData.obs representing the clustering.
        label_type (str): Key in AnnData.obs representing the labels to assess enrichment.

    Returns:
        pd.DataFrame: DataFrame of z-scores with clusters as rows and labels as columns.
    '''
    merged = merged_info_anndata_dict(ad_dict, obs_key=[clustering_key, label_type])
    clusters = np.unique(merged[clustering_key])
    labels = np.unique(merged[label_type])
    df = pd.DataFrame(0, index=clusters, columns=labels)
    for cluster in clusters:
        n = Counter(list(merged[merged[clustering_key]==cluster][label_type]))
        for label in n.keys():
            df.loc[cluster, label] = n[label]
    
    # Compute column-wise z-scores
    
    scaler = StandardScaler()
    z_scores = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns )
        
    return z_scores


def label_type_proportions(ad_dict: Dict[str, AnnData], label_type:str, group: Union[str, List[str]], obs_key: str):
    '''
    Calculate cell type proportions within a specific niche cluster across multiple samples.

    Args:
        ad_dict (Dict[str, AnnData]): Dictionary of AnnData instances with keys as sample names.
        label_type (str): Key in AnnData.obs representing the labels to calculate proportions for.
        group (Union[str, List[str]]): 'whole_sample', 'per_group', 'per_group and whole_sample', or specific group value.
        obs_key (str): Key in AnnData.obs representing the grouping.   
    '''
    if group == 'whole_sample':
        label_counts = None
        for sample_id, ad in ad_dict.items():
            label_counts_sample =ad.obs[label_type].value_counts()
            if label_counts is None:
                label_counts = label_counts_sample
            else:
                label_counts = label_counts.add(label_counts_sample, fill_value=0)

        proportions = label_counts / label_counts.sum()
        return proportions

    elif group == 'per_group' or 'per_group and whole_sample':
        assert obs_key is not None, f'obs_key must be provided when group is {group}'
        label_counts = None
        for sample_id, ad in ad_dict.items():
            for group_id in ad.obs[obs_key].unique():
                ad_group = ad[ad.obs[obs_key] == group_id,]
                label_counts_group = ad_group.obs[label_type].value_counts()
                if label_counts is None:
                    label_counts = pd.DataFrame(label_counts_group, columns=[group_id])
                else:
                    if group_id in label_counts.columns:
                        label_counts[group_id] = label_counts[group_id].add(label_counts_group, fill_value=0)
                    else:
                        label_counts[group_id] = label_counts_group
            if group == 'per_group and whole_sample':
                label_counts_sample =ad.obs[label_type].value_counts()
                if 'whole_sample' in label_counts.columns:
                    label_counts['whole_sample'] = label_counts['whole_sample'].add(label_counts_sample, fill_value=0)
                else:
                    label_counts['whole_sample'] = label_counts_sample
            
        proportions = label_counts.div(label_counts.sum(axis=0), axis=1)
        
        return proportions
    
    else: 
        assert obs_key is not None, f'obs_key must be provided when group is {group}'
        label_counts = None
        for sample_id, ad in ad_dict.items():
            ad_group = ad[ad.obs[obs_key] == group,]
            label_counts_group = ad_group.obs[label_type].value_counts()
            if label_counts is None:
                label_counts = label_counts_group
            else:
                label_counts = label_counts.add(label_counts_group, fill_value=0)
        proportions = label_counts / label_counts.sum()
        return proportions


