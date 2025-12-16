# %%
from random import seed
import numpy as np
import pandas as pd
from anndata import AnnData
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, Union, List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from neighborhood_representation import get_merged_neighborhood_representations
from sklearn.metrics import adjusted_rand_score
#%%

def calculate_ari_vs_reference(df, reference_col_name):
    """
    Calculates the Adjusted Rand Index (ARI) for all clustering runs 
    in the DataFrame against a specified reference run.

    Args:
        df (pd.DataFrame): DataFrame where index is cell_ids and columns 
                           are clustering runs (cluster labels).
        reference_col_name (str): The name of the column to use as the 
                                  "ground truth" or reference clustering.

    Returns:
        pd.Series: A Series containing the ARI for each clustering run 
                   against the reference, excluding the reference itself.
    """
    aris = {}
    reference_labels = df[reference_col_name]
    for seed in df.columns:
        if seed != reference_col_name:
            ari = adjusted_rand_score(reference_labels, df[seed])
            aris[seed] = ari
    return aris

def compute_avg_ARI(labels_df: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]])-> float:
    avg_ARI = {}
    for seed in labels_df.columns:
        aris = calculate_ari_vs_reference(labels_df, reference_col_name=seed)
        avg_ARI[seed] = np.mean(list(aris.values()))
        
    return avg_ARI

def assert_index_series(labels_dict: Dict[int, pd.Series]) -> bool:
    '''Check if all pd.Series in the labels_dict have the same index.
    '''
    indices = [labels_dict[seed].index for seed in labels_dict.keys()]
    first_index = indices[0]
    for idx in indices[1:]:
        if not first_index.equals(idx):
            return False
    return True

def robustness_analysis(multiple_seeds_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]])-> dict:
    '''Select the best seed based on average ARI with other seeds.
    '''
    labels = {}
    for seed in multiple_seeds_dict.keys():
        labels[seed] = multiple_seeds_dict[seed]['labels']
    
    assert assert_index_series(labels), 'Indices of all label series do not match.'
    
    labels_df = pd.DataFrame(labels)
    avg_aris = compute_avg_ARI(labels_df)
    best_seed = max(avg_aris, key=avg_aris.get)   

    selected_seed_dict = multiple_seeds_dict[best_seed]
    selected_seed_dict['metrics'] = { 'avg_ari': avg_aris[best_seed] }

    return selected_seed_dict

def k_selection(robustness_analysis_dict: Dict[str,Dict[str, str]], select_k_metric: str ):
    ''' '''
    metrics = pd.Series
    for k in robustness_analysis_dict.keys():
        robustness_analysis_dict[k]['selected'] = False
        k_dict = robustness_analysis_dict[k]
        metrics[k] = k_dict['metrics'][select_k_metric]
        if select_k_metric == 'inertia':
            best_k = metrics.idxmin()
        else:
            best_k = metrics.idxmax()
    
    robustness_analysis_dict[best_k]['selected'] = True
    
    return robustness_analysis_dict

def neighborhood_filtering(merged: pd.DataFrame, neighborhood_filtering_percentage: int):
    assert 1< neighborhood_filtering_percentage <= 100, "neighborhood_filtering_percentage must be between 1 and 100"
    merged_initial_index = merged.index.copy()
    total_filter_entities = merged['neighborhood_filter'].nunique()
    threshold = (neighborhood_filtering_percentage / 100) * total_filter_entities
    neighborhood_counts = merged.groupby('label')['neighborhood_filter'].nunique()
    neighborhoods_to_filter = neighborhood_counts[neighborhood_counts < threshold].index.tolist()
    merged.loc[merged['label'].isin(neighborhoods_to_filter), 'labels'] = pd.NA
    assert merged_initial_index.equals(merged.index), 'Indices of merged DataFrame have changed after neighborhood filtering.'
    
    return merged['labels']

def filtered_inertia(filtered_merged: pd.DataFrame, filtered_labels: pd.Series, centers: np.ndarray):
    remaining_clusters = np.unique(filtered_labels)
    centers_map = {i: centers[i] for i in remaining_clusters}
    assigned_centers = np.array([centers_map[label] for label in filtered_labels])
    squared_distances = np.sum((filtered_merged.values - assigned_centers)**2, axis=1)
    inertia = np.sum(squared_distances)
    return inertia



def get_metrics(merged: pd.DataFrame,  labels: pd.Series, centers: np.ndarray = None, inertia: float = None):
    assert inertia != None or centers != None, "Either inertia or centers must be provided to compute metrics."
    tot_cells = len(merged.index)
    if inertia != None:
        silhouette = silhouette_score(merged.values, labels,sample_size=0.50*tot_cells)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette}
        return metrics
    elif centers != None:
        filtered_labels = labels.dropna()
        filtered_merged = merged.loc[filtered_labels.index]
        inertia = filtered_inertia(filtered_merged, filtered_labels, centers)
        silhouette = silhouette_score(filtered_merged.values, filtered_labels,sample_size=0.50*tot_cells)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette}
    
        return metrics

def k_means_clustering_singlek_one_seed( merged: pd.DataFrame, k: int, seed: int=None, neighborhood_filtering_percentage: int = None, **kmeans):
    if seed is None:
        seed = np.random.randint(0, 2**32)

    kmeans = KMeans(n_clusters=k, random_state=seed, **kmeans)
    if 'neighborhood_filter' in merged.columns:
        values_df = merged.drop(columns=['neighborhood_filter'])
    else:
        values_df = merged.copy()
    kmeans.fit(values_df.values) 
    
    labels = kmeans.labels_
    
    assert len(merged) == len(labels), 'Length of merged DataFrame and labels do not match.'
    merged[f'labels'] = labels
    if 'neighborhood_filter' in merged.columns:
        filter_df = merged[['neighborhood_filter', 'labels']]
        filtered_labels = neighborhood_filtering(filter_df, neighborhood_filtering_percentage)
        assert filtered_labels.index.equals(merged.index), 'Indices of filtered labels and merged DataFrame do not match.'
        merged['labels'] = filtered_labels
    
    
    if merged['labels'].hasnans:
        centers = kmeans.cluster_centers_
        metrics = get_metrics(merged = values_df, labels = merged['labels'], centers = centers)
    else:
        inertia = kmeans.inertia_
        metrics = get_metrics(merged = values_df, labels = merged['labels'], inertia = inertia)

    k_dict = {'labels': merged[f'labels'], 'seed': seed, 'k': k, 'metrics': metrics}
    

    return k_dict
    

def k_means_singlek_multiple_seeds(merged: pd.DataFrame, k:int, seeds: List[int], neighborhood_filtering_percentage: int, **kmeans ):
    res_dict = {}
    centers = {}
    inertias = {}
    
    if 'neighborhood_filter' in merged.columns:
        values_df = merged.drop(columns=['neighborhood_filter'])
    else:
        values_df = merged.copy()
    
    for seed in seeds:
        kmeans = KMeans(n_clusters=k, random_state=seed, **kmeans)
        kmeans.fit(values_df.values) 
        labels = kmeans.labels_
        assert len(merged) == len(labels), 'Length of merged DataFrame and labels do not match.'
        merged[f'labels'] = labels
        if 'neighborhood_filter' in merged.columns:
            filter_df = merged[['neighborhood_filter', 'labels']]
            filtered_labels = neighborhood_filtering(filter_df, neighborhood_filtering_percentage)
            assert filtered_labels.index.equals(merged.index), 'Indices of filtered labels and merged DataFrame do not match.'
            merged['labels'] = filtered_labels

        if merged['labels'].hasnans:
            centers[seed] = kmeans.cluster_centers_
        else:
            inertias[seed] = kmeans.inertia_
        
        seed_dict = {'labels': merged[f'labels'], 'seed': seed, 'k': k}
        res_dict[seed] = seed_dict    
    
    best_seed_dict = robustness_analysis(res_dict)
    best_seed =  best_seed_dict['seed']
    if best_seed in centers.keys():
        metrics = get_metrics(merged = values_df, labels = merged['labels'], centers = centers[best_seed])
    elif best_seed in inertias.keys():
        metrics = get_metrics(merged = values_df, labels = merged['labels'], inertia = inertias[best_seed])
    
    best_seed_dict['metrics']['inertia'] = metrics['inertia']
    best_seed_dict['metrics']['silhouette_score'] = metrics['silhouette_score']
    
    return best_seed_dict
    



def k_means_clustering_multik_one_seed(merged: pd.DataFrame, k:list, select_k:bool , select_k_metric: str , neighborhood_filtering_percentage: int, **kmeans ):
    if select_k == True:
        assert select_k_metric in ['silhouette_score', 'inertia'], f'select_k_metric {select_k_metric} not recognized. Use "silhouette_score" or "inertia" when 1 random seed is used.'

    seed = np.random.randint(0, 2**32)
    
    ks_dicts = {}
    for i in k:
        k_dict = k_means_clustering_singlek_one_seed(merged, i, seed=seed, **kmeans)
        ks_dicts[i] = k_dict
    
    
    if select_k:
        ks_dicts = k_selection(ks_dicts, select_k_metric=select_k_metric)

    return ks_dicts


def k_means_clustering_multik_multiple_seeds(merged: pd.DataFrame, k:list, random_seeds: int, select_k:bool , select_k_metric: str , neighborhood_filtering_percentage: int, **kmeans ):
    if select_k == True:
        assert select_k_metric in ['silhouette_score', 'inertia', 'average_ARI'], f'select_k_metric {select_k_metric} not recognized. Use "silhouette_score" or "inertia" or "average_ARI".'
    assert random_seeds > 1, 'random_seeds must be greater than 1'

    seeds = np.random.randint(0, 2**32, size=random_seeds).tolist()
    ks_dicts = {}
    for i in k:
        best_seed_dict = k_means_singlek_multiple_seeds(merged, i, seeds, neighborhood_filtering=neighborhood_filtering, neighborhood_filtering_percentage=neighborhood_filtering_percentage, **kmeans)        
        ks_dicts[i] = best_seed_dict
    
    if select_k:
        ks_dicts = k_selection(ks_dicts, select_k_metric=select_k_metric) #adds 'selected' key to ks_dict
    
    return ks_dicts

def get_neighborhood_filter_column(ad_dict, merged, neighborhood_filtering_entity: str):
    ''' Adds neighborhood filtering column to merged DataFrame.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        merged: Merged DataFrame of neighborhood representations.
        neighborhood_filtering_entity: entity to use for neighborhood filtering. Options are 'sample_id' or any categorical column in ad.obs.

    Returns:
        pd.Series: Series containing the neighborhood filtering entity for each neighborhood in the merged DataFrame.
        '''
    neighborhood_filtering_column = list()
    if neighborhood_filtering_entity == 'sample_id':
        merged['neighborhood_filter'] = merged.index.get_level_values('sample_id')
    else: 
        for sample_id, ad in ad_dict.items():
        
            neighborhood_filtering = ad.obs[f'{neighborhood_filtering_entity}']
            neighborhood_filtering.index = pd.MultiIndex.from_product([[sample_id], neighborhood_filtering.index], names=['sample_id', 'cell_id'])
            assert neighborhood_filtering.index.get_level_values('cell_id').equals(ad.obs_names), 'cell_id level of MultiIndex does not match ad.obs_names.'
            neighborhood_filtering_column.append(neighborhood_filtering)
    neighborhood_filtering_column = pd.concat(neighborhood_filtering_column, axis=0)
    return neighborhood_filtering_column
     
def inplace_kmeans_multik(ad_dict: Dict[str, AnnData], k_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]], clustering_key:str, select_k: bool = False):
    '''Inplace addition of clustering results to AnnData objects in ad_dict.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        k_dict: Dictionary containing clustering results for multiple k values.
        clustering_key: str.
        select_k: Whether to add only the selected k clustering results or all k results.
    '''
    if select_k:
        for k_value in k_dict.keys():
            if k_dict[k_value]['selected'] == True:
                obs_add = k_dict[k_value]['labels']
    else:
        obs_add_dict = {}
        for k_value in k_dict.keys():
            obs_add_dict[f'k_{k_value}_{clustering_key}'] = k_dict[k_value]['labels']
        obs_add = pd.DataFrame(obs_add_dict)

        for sample_id, ad in ad_dict.items():
            obs_add_sample = obs_add.loc[sample_id] 
            only_in_clusters = obs_add_sample.index.difference(ad.obs.index).tolist() #check because of previous filtering steps
            if len(only_in_clusters) > 0:
                complete_index = ad.obs.index.to_list()
                obs_add_sample = obs_add_sample.reindex(complete_index)
                assert obs_add_sample.loc[only_in_clusters].isna().all().all(), 'Reindexed rows that were not in ad.obs are not all NaN.'           
            assert obs_add_sample.index.equals(ad.obs.index), 'Indices of obs_add_sample and ad.obs do not match.'
            if select_k:
                ad.obs[f'selected_k_{clustering_key}'] = obs_add_sample     
            else:
                ad.obs = ad.obs.join(obs_add_sample)
            ad.uns[f'k_multik_{clustering_key}_metrics'] = k_dict
    return    
        

def inplace_kmeans_singlek(ad_dict: Dict[str, AnnData], k_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]], clustering_key:str):
    '''Inplace addition of clustering results to AnnData objects in ad_dict.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        k_dict: Dictionary containing clustering results for multiple k values.
        clustering_key: str.
    '''
    obs_add = k_dict['labels']
    k = k_dict['k']
    for sample_id, ad in ad_dict.items():
        obs_add_sample = obs_add.loc[sample_id] 
        only_in_clusters = obs_add_sample.index.difference(ad.obs.index).tolist() #check because of previous filtering steps
        if len(only_in_clusters) > 0:
            complete_index = ad.obs.index.to_list()
            obs_add_sample = obs_add_sample.reindex(complete_index)
            assert obs_add_sample.loc[only_in_clusters].isna().all().all(), 'Reindexed rows that were not in ad.obs are not all NaN.'           
        assert obs_add_sample.index.equals(ad.obs.index), 'Indices of obs_add_sample and ad.obs do not match.'
        ad.obs[f'k_{k}_{clustering_key}'] = obs_add_sample     
        ad.uns[f'k_{k}_{clustering_key}'] = k_dict
           
    return

def k_means_clustering(ad_dict: Dict[str, AnnData], attr: str, graph_key: str = 'radius_80', mode: str = 'proportions', k: Union[int, List[int]] = 4, random_seeds: int= 1, clustering_key:str = None, select_k:bool = False, select_k_metric: str = 'silhouette_score', neighborhood_filtering: bool = True, neighborhood_filtering_percentage: int = 30, neighborhood_filtering_entity: str = 'sample_id',merged: pd.DataFrame = None, inplace:bool=True, **kwargs ):
    """K-means clustering of the merged (all samples together) local neighborhood representation of cells.

    if there are more than one random seed provided -> for each k the clustering with the lowest inertia will be chosen and added to the anndata objects.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr: The label type to filter neighbors by -> has to be present in ad.obs.
        clustering_key: key in .obs to store the clustering results.
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        mode: 'proportion' or 'counts' to specify which representation to use.
        k: Number of clusters or list of number of clusters for k-means.
        random_seeds: number of random seeds to generate. 
        select_k: whether to select the best k based on the select_k_metric. 
        select_k_metric: metric to use for selecting the best k. Options are 'silhouette_score' or 'inertia' or 'average_ARI'.
                    it cannot be average_ARI if random_seeds == 1 is False.
        neighborhood_filtering: whether to filter neighborhoods based on the neighborhood_filtering_entity.
        neighborhood_filtering_percentage: percentage of neighborhoods to keep based on the neighborhood_filtering_entity.
        neighborhood_filtering_entity: entity to use for neighborhood filtering. Options are 'sample_id' or any categorical column in ad.obs.
        merged: Precomputed merged DataFrame of neighborhood representations. If None, it will be computed.
        inplace: Whether to add the clustering results to the current AnnData instances or to return a new one.
        **kwargs: Additional arguments for KMeans and silhouette_score. -> if not provided default values will be used.

    Returns: 
        
    """
    assert mode in ['proportion', 'counts'], f'Mode {mode} not recognized. Use "proportion" or "counts".'
    if inplace:
        assert clustering_key!= None, 'clustering_key must be provided when inplace is True.'
    
    if type(k) == list:
        assert len(k) > 1, 'If k is a list it must contain more than one value.' 
        
    if merged is None:
        merged = get_merged_neighborhood_representations(ad_dict, attr=attr, graph_key=graph_key, mode=mode)
    
    if neighborhood_filtering:
        neighborhood_filtering_column = get_neighborhood_filter_column(ad_dict, neighborhood_filtering_entity)
        only_in_merged = merged.index.difference(neighborhood_filtering_column.index).tolist() #check because of previous filtering steps
        if len(only_in_merged) > 0:
            neighborhood_filtering_column = neighborhood_filtering_column.drop(index=only_in_merged)
        assert merged.index.equals(neighborhood_filtering_column.index), 'Indices of merged DataFrame and neighborhood filter column do not match.'      
        merged['neighborhood_filter'] = neighborhood_filtering_column

    if type(k) == int:
        if random_seeds == 1:
            k_dict = k_means_clustering_singlek_one_seed( merged, k, neighborhood_filtering_percentage=neighborhood_filtering_percentage, **kwargs)
        else:
            k_dict = k_means_singlek_multiple_seeds(merged, k, seeds=np.random.randint(0, 2**32, size=random_seeds).tolist(), neighborhood_filtering_percentage=neighborhood_filtering_percentage, **kwargs)
        k_dict['clustering_key'] = clustering_key
        if inplace:
            inplace_kmeans_singlek(ad_dict, {k: k_dict}, clusterring_key=clustering_key)
            return
        

    if type(k) == list:
        if random_seeds == 1:
            k_dict = k_means_clustering_multik_one_seed( merged, k, select_k, select_k_metric, neighborhood_filtering_percentage, **kwargs)
        else:
            k_dict = k_means_clustering_multik_multiple_seeds(merged, k, random_seeds, select_k, select_k_metric, neighborhood_filtering_percentage, **kwargs)
        for k_value in k_dict.keys():
            k_dict[k_value]['clustering_key'] = clustering_key
        if inplace:
            inplace_kmeans_multik(ad_dict, k_dict, select_k=select_k, clusterring_key=clustering_key)
            return
    
    return k_dict

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