# %%
from random import seed
import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Dict, Union, List
from sklearn.cluster import KMeans
from athena.niches.neighborhood_representation import retrieve_merged_neighborhood_representations
from athena.niches.cluster_filtering import cluster_filter_to_merge, perform_cluster_filtering
from athena.niches.robustness_analysis import compute_robustness_analysis
from athena.niches.clustering_metrics import get_metrics
#%%

def k_selection(robustness_analysis_dict: Dict[str,Dict[str, str]], select_k_metric: str ):
    ''' '''
    metrics = pd.Series()
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



def compute_clustering(merged: pd.DataFrame, k:int, seed: int, cluster_filtering_percentage: int, **kmeans_params ):
    if 'cluster_filter' in merged.columns:
        values_df = merged.copy()
        values_df = values_df.drop(columns=['cluster_filter'])
    else:
        values_df = merged.copy()
    
    kmeans = KMeans(n_clusters=k, random_state=seed, **kmeans_params)
    kmeans.fit(values_df.values) 
    labels = kmeans.labels_
    assert len(merged) == len(labels), 'Length of merged DataFrame and labels do not match.'
    merged['labels'] = labels 

    if 'cluster_filter' in merged.columns:
        merged['labels_raw'] = merged['labels']
        filter_df = merged[['cluster_filter', 'labels']]
        filtered_labels = perform_cluster_filtering(merged=filter_df, cluster_filtering_percentage=cluster_filtering_percentage)
        assert filtered_labels.index.equals(merged.index), 'Indices of filtered labels and merged DataFrame do not match.'
        merged['labels'] = filtered_labels
        
    if merged['labels'].hasnans:
        # if there are nan values in the labels => one/ultiple clusters have been filtered out
        # if one/multiple clusters have been filtered out, we need to re-compute the inertia
        inertia = None
        centers = kmeans.cluster_centers_
    else:
        centers = None
        inertia = kmeans.inertia_
    
    return centers, inertia, merged

def k_means_clustering_singlek_one_seed( merged: pd.DataFrame, k: int, seed: int, cluster_filtering_percentage: int = None, **kmeans_params):

    
    centers, inertia, merged = compute_clustering(merged=merged, k=k, seed=seed, cluster_filtering_percentage=cluster_filtering_percentage,**kmeans_params )

    if inertia == None: 
        metrics = get_metrics(merged = merged, centers = centers)
    else:
        metrics = get_metrics(merged = merged, inertia=inertia)

    
    if 'labels_raw' in merged.columns: 
        k_dict = {'labels': merged['labels'], 'labels_raw':merged['labels_raw'], seed: seed, 'k': k, 'metrics': metrics}
    else:
        k_dict = {'labels': merged['labels'], seed: seed, 'k': k, 'metrics': metrics}

    return k_dict


    

def k_means_singlek_multiple_seeds(merged: pd.DataFrame, k:int, seeds: List[int], cluster_filtering_percentage: int = None, **kmeans_params ):
    res_dict = {}
    centers = {}
    inertias = {}
    
    
    for seed in seeds:
        seed_centers, inertia, merged = compute_clustering( merged=merged, k=k, seed=seed, cluster_filtering_percentage=cluster_filtering_percentage,**kmeans_params )

        if inertia == None:
            centers[seed] = seed_centers
        else:
            inertias[seed] = inertia
            
        if 'labels_raw' in merged.columns: 
            seed_dict = {'labels': merged[f'labels'], 'labels_raw':merged['labels_raw'], 'seed': seed, 'k': k}
        else:
            seed_dict = {'labels': merged[f'labels'], 'seed': seed, 'k': k}
        res_dict[seed] = seed_dict    
    
    best_seed_dict = compute_robustness_analysis(res_dict)
    best_seed =  best_seed_dict['seed']
    if best_seed in centers.keys():
        metrics = get_metrics(merged = merged, centers = centers[best_seed])
    elif best_seed in inertias.keys():
        metrics = get_metrics(merged = merged, inertia = inertias[best_seed])
    
    best_seed_dict['metrics']['inertia'] = metrics['inertia']
    best_seed_dict['metrics']['silhouette_score'] = metrics['silhouette_score']
    
    return best_seed_dict
    



def k_means_clustering_multik_one_seed(merged: pd.DataFrame, k:list, seed: int, select_k:bool , select_k_metric: str , cluster_filtering_percentage: int, **kmeans_params ):
    if select_k == True:
        assert select_k_metric in ['silhouette_score', 'inertia'], f'select_k_metric {select_k_metric} not recognized. Use "silhouette_score" or "inertia" when 1 random seed is used.'

    
    ks_dicts = {}
    for i in k:
        merged_copy = merged.copy()
        k_dict = k_means_clustering_singlek_one_seed(merged_copy, i, seed=seed, cluster_filtering_percentage=cluster_filtering_percentage, **kmeans_params)
        ks_dicts[i] = k_dict
    
    
    if select_k:
        ks_dicts = k_selection(ks_dicts, select_k_metric=select_k_metric)

    return ks_dicts


def k_means_clustering_multik_multiple_seeds(merged: pd.DataFrame, k:list, seeds: List[int], select_k:bool , select_k_metric: str , cluster_filtering_percentage: int, **kmeans_params ):
    if select_k == True:
        assert select_k_metric in ['silhouette_score', 'inertia', 'average_ARI'], f'select_k_metric {select_k_metric} not recognized. Use "silhouette_score" or "inertia" or "average_ARI".'
    
    assert len(seeds)>1, 'length of seeds must be bigger than 1'

    ks_dicts = {}
    for i in k:
        merged_copy = merged.copy()
        best_seed_dict = k_means_singlek_multiple_seeds(merged=merged_copy, k=i, seeds=seeds, cluster_filtering_percentage=cluster_filtering_percentage, **kmeans_params)        
        ks_dicts[i] = best_seed_dict
    
    if select_k:
        ks_dicts = k_selection(ks_dicts, select_k_metric=select_k_metric) #adds 'selected' key to ks_dict
    
    return ks_dicts


def obs_add_multi_k(k_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]], clustering_key:str, select_k: bool):

    if select_k:
        obs_add_dict = {}
        for k_value in k_dict.keys():
            if k_dict[k_value]['selected'] == True:
                    obs_add_dict[f'selected_k_{clustering_key}'] = k_dict[k_value]['labels']
                    if 'labels_raw' in  k_dict[k_value].keys():
                        obs_add_dict[f'selected_k_{clustering_key}_raw'] =  k_dict[k_value]['labels_raw'] 
        obs_add = pd.DataFrame(obs_add_dict)

    else:
        obs_add_dict = {}
        for k_value in k_dict.keys(): 
            obs_add_dict[f'k_{k_value}_{clustering_key}'] = k_dict[k_value]['labels']
            if 'labels_raw' in  k_dict[k_value].keys():
                    obs_add_dict[f'k_{k_value}_{clustering_key}_raw'] =  k_dict[k_value]['labels_raw']
        obs_add = pd.DataFrame(obs_add_dict)

    return obs_add

def obs_add_singlek(k_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]], clustering_key:str):
    obs_add_dict = {}
    obs_add_dict[clustering_key] = k_dict['labels']
    if 'labels_raw' in k_dict.keys():
        obs_add_dict[f'{clustering_key}_raw'] = k_dict['labels_raw']
    
    obs_add = pd.DataFrame(obs_add_dict)
    return obs_add


def save_kmeans_multik(ad_dict: Dict[str, AnnData], k_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]], clustering_key:str, select_k: bool = False, inplace:bool = True):
    '''Addition of clustering results to original (inplace) or new(not inplace) AnnData objects in ad_dict.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        k_dict: Dictionary containing clustering results for multiple k values.
        clustering_key: str.
        select_k: Whether to add only the selected k clustering results or all k results.
        inplace: Whether to add the clustering results to the current AnnData instances or to return a new one.
    Return
        no return if inplace
        new ad_dict if not inplace
    '''
    
    obs_add = obs_add_multi_k(k_dict=k_dict, clustering_key=clustering_key, select_k=select_k)
    
    if not inplace:
        ad_dict = ad_dict.copy()

    for sample_id, ad in ad_dict.items():
        obs_add_sample = obs_add.xs(sample_id, level='sample_id')
        not_in_clusters= ad.obs.index.difference(obs_add_sample.index).tolist() # check because of previous filtering steps
        if len(not_in_clusters) > 0:
            complete_index = ad.obs.index.to_list()
            obs_add_sample = obs_add_sample.reindex(complete_index)
            if type(obs_add_sample) == pd.Series:
                assert obs_add_sample.loc[not_in_clusters].isna().all(), 'Reindexed rows that were not in clusters are not all NaN.'
            else:
                assert obs_add_sample.loc[not_in_clusters].isna().all().all(), 'Reindexed rows that were not in clusters are not all NaN.'           
        assert obs_add_sample.index.equals(ad.obs.index), 'Indices of obs_add_sample and ad.obs do not match.'
        
        ad.obs = ad.obs.join(obs_add_sample)
        ad.uns[f'{clustering_key}_metrics'] = k_dict
    if not inplace:
        return ad_dict
    return    
        

def save_kmeans_singlek(ad_dict: Dict[str, AnnData], k_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]], clustering_key:str, inplace:bool=True):
    '''Addition of clustering results to original (inplace) or new(not inplace) AnnData objects in ad_dict.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        k_dict: Dictionary containing clustering results.
        clustering_key: str.
        inplace: Whether to add the clustering results to the current AnnData instances or to return a new one.
    Return
        no return if inplace
        new ad_dict if not inplace
    '''
    obs_add = obs_add_singlek(k_dict=k_dict, clustering_key=clustering_key)

    if not inplace:
        ad_dict = ad_dict.copy()

    for sample_id, ad in ad_dict.items():
        obs_add_sample = obs_add.xs(sample_id, level='sample_id')
        not_in_clusters= ad.obs.index.difference(obs_add_sample.index).tolist()#check because of previous filtering steps
        if len(not_in_clusters) > 0:
            complete_index = ad.obs.index.to_list()
            obs_add_sample = obs_add_sample.reindex(complete_index)
            if type(obs_add_sample) == pd.Series:
                assert obs_add_sample.loc[not_in_clusters].isna().all(), 'Reindexed rows that were not in clusters are not all NaN.'
            else:
                assert obs_add_sample.loc[not_in_clusters].isna().all().all(), 'Reindexed rows that were not in clusters are not all NaN.'        
        assert obs_add_sample.index.equals(ad.obs.index), 'Indices of obs_add_sample and ad.obs do not match.'
        ad.obs = ad.obs.join(obs_add_sample)  
        ad.uns[f'{clustering_key}_metrics'] = k_dict
           
    if not inplace:
        return ad_dict
    
    return   

def k_means_multik(ad_dict: Dict[str, AnnData], merged:pd.DataFrame, k: Union[int, List[int]], random_seeds: int, clustering_key:str, cluster_filtering_percentage:int = None, select_k:bool = False, select_k_metric: str = 'silhouette_score', inplace:bool = True, **kmeans_params):
    # generate seeds
    if random_seeds>1: 
        seeds = np.random.randint(0, 2**32, size=random_seeds, dtype='uint64').tolist()
    else: 
        seed = np.random.randint(0, 2**32, dtype='uint64')

    if random_seeds == 1:
        k_dict = k_means_clustering_multik_one_seed( merged=merged,k= k, seed=seed, select_k=select_k, select_k_metric=select_k_metric, cluster_filtering_percentage=cluster_filtering_percentage, **kmeans_params)
    else:
        k_dict = k_means_clustering_multik_multiple_seeds(merged=merged, k=k, seeds=seeds, select_k=select_k, select_k_metric=select_k_metric, cluster_filtering_percentage=cluster_filtering_percentage, **kmeans_params)
    for k_value in k_dict.keys():
        k_dict[k_value]['clustering_key'] = clustering_key 
    
    if not inplace:
        ad_dict_new = save_kmeans_multik(ad_dict=ad_dict, k_dict=k_dict, clustering_key=clustering_key, select_k=select_k, inplace=inplace)
        return ad_dict_new
    
    save_kmeans_multik(ad_dict=ad_dict, k_dict=k_dict, clustering_key=clustering_key, select_k=select_k, inplace=inplace)
    return

def k_means_singlek(ad_dict: Dict[str, AnnData], merged:pd.DataFrame, k: Union[int, List[int]], random_seeds: int, clustering_key:str, cluster_filtering_percentage:int =None, inplace:bool = True, **kmeans_params):    
    if random_seeds == 1:
        seed = np.random.randint(0, 2**32, dtype='uint64') 
        k_dict = k_means_clustering_singlek_one_seed( merged, k, seed, cluster_filtering_percentage, **kmeans_params)
    else:
        seeds = np.random.randint(0, 2**32, size=random_seeds, dtype='uint64').tolist()
        k_dict = k_means_singlek_multiple_seeds(merged=merged, k=k, seeds=seeds, cluster_filtering_percentage=cluster_filtering_percentage, **kmeans_params)
    k_dict['clustering_key'] = clustering_key

    if not inplace:
        ad_dict_new = save_kmeans_singlek(ad_dict=ad_dict, k_dict=k_dict, clustering_key=clustering_key, inplace=inplace)
        return ad_dict_new
    
    save_kmeans_singlek(ad_dict=ad_dict, k_dict=k_dict, clustering_key=clustering_key, inplace=inplace)
    return
    

def k_means_clustering(ad_dict: Dict[str, AnnData], attr: str, graph_key: str = 'radius_80', mode: str = 'proportions', k: Union[int, List[int]] = 4, random_seeds: int= 1, clustering_key:str = None, select_k:bool = False, select_k_metric: str = 'silhouette_score', cluster_filtering: bool = True, cluster_filtering_percentage: int = 30, cluster_filtering_entity: str = 'sample_id',merged: pd.DataFrame = None, inplace:bool=True, **kmeans_params):
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
        cluster_filtering: whether to filter clusters based on the cluster_filtering_entity.
        cluster_filtering_percentage: percentage of clusters to keep based on the cluster_filtering_entity.
        cluster_filtering_entity: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.
        merged: Precomputed  merged DataFrame of neighborhood representations. It must be filtered from 0/0.0 rows. If None, it will be computed.
        inplace: Whether to add the clustering results to the current AnnData instances or to return a new one.
        **kwargs: Additional arguments for KMeans and silhouette_score. -> if not provided default values will be used.

    Returns: 
        
    """
    assert mode in ['proportion', 'counts'], f'Mode {mode} not recognized. Use "proportion" or "counts".'
    if clustering_key == None:
        clustering_key = f'k_{k}_{graph_key}_{mode}_{random_seeds}seed'
    
    if type(k) == list:
        assert len(k) > 1, 'If k is a list it must contain more than one value.' 
        
    if merged is None:
        merged = retrieve_merged_neighborhood_representations(ad_dict, attr=attr, graph_key=graph_key, mode=mode, filtered=True)
    if mode == 'proportion':
        assert not (merged == 0.0).all(axis=1).any(), 'Merged DataFrame is not filtered. Please check the neighborhood representations.'
    elif mode == 'counts':
        assert not (merged == 0).all(axis=1).any(), 'Merged DataFrame is not filtered. Please check the neighborhood representations.'
    merged_copy = merged.copy()
    
    if cluster_filtering: # add to merged the filtering column 
        merged_copy = cluster_filter_to_merge(ad_dict=ad_dict, merged=merged_copy, cluster_filtering_entity=cluster_filtering_entity)
    
    
    # one k 
    if type(k) == int:
        k_means_singlek(ad_dict=ad_dict, k=k, merged=merged_copy, random_seeds=random_seeds, clustering_key=clustering_key, cluster_filtering_percentage=cluster_filtering_percentage, inplace=inplace, **kmeans_params)
             
    # multiple ks 
    if type(k) == list:
        k_means_multik(ad_dict=ad_dict, k=k, merged=merged_copy, random_seeds=random_seeds, clustering_key=clustering_key, cluster_filtering_percentage=cluster_filtering_percentage, select_k=select_k, select_k_metric=select_k_metric, inplace=inplace, **kmeans_params)

    return 
