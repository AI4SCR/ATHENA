# %%
from random import seed
import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Dict, Union, List
from sklearn.cluster import KMeans
import copy
from athena.niches.neighborhood_representation.neigh_repr import aggregate_n_rep
from athena.niches.clustering.cluster_filtering import cl_filtering, cl_filtering_filt_attr
from athena.niches.clustering.robustness_analysis import cl_robustness
from athena.niches.clustering.clustering_metrics import get_metrics
#%%

def n_slection(res_dict: Dict[str,Dict[str, str]], sel_n_metric: str ):
    '''Selection of best cluster number based on sel_n_metric
    Args:
        res_dict: dictionary with clustering results of each cluster number
        sel_n_metric: metric to choose the best number of clusters
        
    Return
        res_dict with added 'selected' key (True if selected / False if not)'''
    assert sel_n_metric in ['inertia', 'silhouette_score', 'avg_ari' ]
    metrics = pd.Series()
    for n in res_dict.keys():
        res_dict[n]['selected'] = False
        n_dict = res_dict[n]
        metrics[n] = n_dict['metrics'][sel_n_metric]
    if sel_n_metric == 'inertia':
        best_n = metrics.idxmin()
    else:
        best_n = metrics.idxmax()
    
    res_dict[best_n]['selected'] = True
    
    return res_dict


def k_means(n_rep: pd.DataFrame, cl_n:int, seed: int, **cl_params):
    '''k means clustering
    Args:
        n_rep: DataFrame with neighborhood representation to cluster.
        cl_n: number of clusters.
        seed: seed.
        **cl_params = additional parameters for the clustering.
    Return
        - n_rep with 'labels' column added
        - cluster centers
        - cluster intertia
    '''
    
    kmeans = KMeans(n_clusters=cl_n, random_state=seed, **cl_params)
    kmeans.fit(n_rep.values) 
    labels = kmeans.labels_
    assert len(n_rep) == len(labels), 'Length of cl_values DataFrame and labels do not match.'
    n_rep['labels'] = labels

    centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_   

    return n_rep, centers, inertia


def cluster(ad_dict: Dict[str, AnnData], n_rep: pd.DataFrame, cl_n:int, seed: int, cl_algorithm: str, cl_filter:bool, cl_filtering_prop: float , cl_filtering_ent:str,  min_obs: int, filter_attr:str = None,**cl_params ):
    '''clustering and filtering
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        n_rep: DataFrame with neighborhood representation to cluster.
        cl_n: number of clusters.
        seed: seed.
        cl_algorithm = clustering algorithm.
        cl_filter: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_prop: propotion of clusters to keep based on the cluster_filtering_entity.
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.
        min_obs: minimum number of obs of a sample assigned to a cluster to consider the cluster present in the sample when filtering. 
        filter_attr: whether to filter cluster considering entities containing at least 13 of observations with filter_attr
        **cl_params = additional parameters for the clustering
    Return
        - n_rep with new column(s) 
            -'labels' = observations labels (filtered if there has been cluster filtering)
            -'labels_raw' (if there has been cluster filtering) = observations unfiltered labels
            -
        - cluster centers
        - cluster intertia
    '''
    cl_algorithm_map = {
        'kmeans': k_means
    }
    assert cl_algorithm in cl_algorithm_map.keys(), f'cl_algorithm has to be {cl_algorithm_map.keys().tolist()}'
    
    # define clustering function
    func = cl_algorithm_map[cl_algorithm]

    # clustering
    n_rep, centers, inertia = func(n_rep = n_rep, cl_n=cl_n, seed=seed, **cl_params)

    # filter labels if necessary
    if cl_filter:
        if filter_attr: 
            n_rep = cl_filtering_filt_attr(ad_dict=ad_dict, res_df= n_rep, cl_filtering_prop=cl_filtering_prop, cl_filtering_ent=cl_filtering_ent, filt_attr=filter_attr)
        else:
            n_rep = cl_filtering(ad_dict=ad_dict, res_df= n_rep, cl_filtering_prop=cl_filtering_prop, cl_filtering_ent=cl_filtering_ent, min_obs=min_obs)
        n_rep['labels_raw'] = n_rep['labels_raw'].astype('Int64')
    
    n_rep['labels'] = n_rep['labels'].astype('Int64')

    if n_rep['labels'].hasnans:
        # if there are nan values in the labels => one/ultiple clusters have been filtered out
        # if one/multiple clusters have been filtered out, we need to re-compute the inertia
        inertia = None
    else:
        centers = None
    
    return n_rep, centers, inertia

def cluster_singleseed(ad_dict: Dict[str, AnnData], n_rep: pd.DataFrame, cl_n:int, seed: int, cl_algorithm: str, cl_filter:bool, cl_filtering_prop: float , cl_filtering_ent:str, min_obs: int, filter_attr:str = None, **cl_params ):
    '''clustering and metrics one n and one seed
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        n_rep: DataFrame with neighborhood representation to cluster.
        cl_n: number of clusters.
        seed: seed.
        cl_algorithm = clustering algorithm.
        cl_filter: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_prop: propotion of clusters to keep based on the cluster_filtering_entity.
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.
        min_obs: minimum number of obs of a sample assigned to a cluster to consider the cluster present in the sample when filtering. 
        filter_attr: whether to filter cluster considering entities containing at least 13 of observations with filter_attr
        **cl_params = additional parameters for the clustering
    
    Return
        dictionary with 
            - 'labels' = pd.Series with all ad_dict labels (filtered if there has been cluster filtering)
            - 'labels_raw' (if there has been cluster filtering) = pd.Series with all ad_dict unfiltered labels
            - 'seed' = seed
            - 'metrics' = dictionary with
                - 'silhouette_score'
                - 'inertia'
            -'n_clusters' = number of clusters

    '''
    n_rep, centers, inertia = cluster(ad_dict=ad_dict, n_rep=n_rep, cl_n=cl_n, seed=seed, cl_algorithm=cl_algorithm, cl_filter=cl_filter, cl_filtering_ent=cl_filtering_ent, cl_filtering_prop=cl_filtering_prop, filter_attr=filter_attr, min_obs=min_obs **cl_params)

    columns=['labels', 'labels_raw'] if 'labels_raw' in n_rep.columns else ['labels']
    values_df = n_rep.drop(columns=columns)
    metrics = get_metrics(values_df=values_df, labels=n_rep['labels'], centers=centers, inertia=inertia)
    
    if 'labels_raw' in n_rep.columns: 
        res_dict = {'labels': n_rep['labels'], 'labels_raw':n_rep['labels_raw'], 'seed': seed, 'n_clusters': cl_n, 'metrics': metrics}
    else:
        res_dict = {'labels': n_rep['labels'], 'seed': seed, 'n_clusters': cl_n, 'metrics': metrics}

    return res_dict

def multiseed_dicts(ad_dict: Dict[str, AnnData], n_rep: pd.DataFrame, cl_n:int, seeds: List[int], cl_algorithm: str, cl_filter:bool, cl_filtering_prop: float , cl_filtering_ent:str, min_obs: int, filter_attr:str=None, **cl_params ):
    '''clustering and metrics one n and one seed
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        n_rep: DataFrame with neighborhood representation to cluster.
        cl_n: number of clusters.
        seeds: list of seeds.
        cl_algorithm = clustering algorithm.
        cl_filter: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_prop: propotion of clusters to keep based on the cluster_filtering_entity.
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.
        min_obs: minimum number of obs of a sample assigned to a cluster to consider the cluster present in the sample when filtering. 
        filter_attr: whether to filter cluster considering entities containing at least 13 of observations with filter_attr
        **cl_params = additional parameters for the clustering
    
    Return
        res_dict with 
            - seed as keys 
            - dictionary as values with 
                - 'labels' = pd.Series with all ad_dict labels (filtered if there has been cluster filtering)
                - 'labels_raw' (if there has been cluster filtering) = pd.Series with all ad_dict unfiltered labels
                - 'seed' = seed
        centers
            - seed as keys 
            - array with cluster centers or None (depending on cluster filtering) as values
        inertias
            - seed as keys 
            - cluster inertia or None (depending on cluster filtering) as values
        
    '''
    res_dict = {}
    centers = {}
    inertias = {}
    
    for seed in seeds:
        n_rep_copy = n_rep.copy()
        n_rep_copy, seed_centers, seed_inertia = cluster(ad_dict=ad_dict, n_rep=n_rep_copy, cl_n=cl_n, seed=seed, cl_algorithm=cl_algorithm, cl_filter=cl_filter, cl_filtering_ent=cl_filtering_ent, cl_filtering_prop=cl_filtering_prop, min_obs=min_obs, filter_attr=filter_attr, **cl_params)

        centers[seed] = seed_centers
        inertias[seed] = seed_inertia
            
        if 'labels_raw' in n_rep_copy.columns: 
            seed_dict = {'labels': n_rep_copy['labels'], 'labels_raw':n_rep_copy['labels_raw'], 'seed': seed, 'n_clusters': cl_n}
        else:
            seed_dict = {'labels': n_rep_copy['labels'], 'seed': seed, 'n_clusters': cl_n}
        res_dict[seed] = seed_dict 

    return res_dict, centers, inertias   
    

def cluster_multiseed(ad_dict: Dict[str, AnnData], n_rep: pd.DataFrame, cl_n:int, seeds: List[int], cl_algorithm: str, cl_filter:bool, cl_filtering_prop: float, cl_filtering_ent:str,  min_obs: int, filter_attr:str=None, **cl_params ):
    '''clustering and metrics one n and one seed
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        n_rep: DataFrame with neighborhood representation to cluster.
        cl_n: number of clusters.
        seeds: list of seeds.
        cl_algorithm = clustering algorithm.
        cl_filter: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_prop: propotion of clusters to keep based on the cluster_filtering_entity.
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.
        min_obs: minimum number of obs of a sample assigned to a cluster to consider the cluster present in the sample when filtering. 
        filter_attr: whether to filter cluster considering entities containing at least 13 of observations with filter_attr
        **cl_params = additional parameters for the clustering
    
    Return
        dictionary with 
            - 'labels' = pd.Series with all ad_dict labels (filtered if there has been cluster filtering)
            - 'labels_raw' (if there has been cluster filtering) = pd.Series with all ad_dict unfiltered labels
            - 'seed' = seed
            - 'metrics' = dictionary with
                - 'silhouette_score'
                - 'inertia'
                - 'avg_ari' of the selected seed (if multiple random seeds have been used)
            -'n_clusters' = number of clusters
    '''
    res_dict = {}
    centers = {}
    inertias = {}
    
    res_dict, centers, inertias = multiseed_dicts(ad_dict=ad_dict, n_rep=n_rep, cl_n=cl_n, seeds=seeds, cl_algorithm=cl_algorithm, cl_filter=cl_filter, cl_filtering_prop=cl_filtering_prop, cl_filtering_ent=cl_filtering_ent, min_obs=min_obs, filter_attr=filter_attr, **cl_params )

    best_seed_dict = cl_robustness(res_dict)
    best_seed =  best_seed_dict['seed']
      
    metrics = get_metrics(values_df=n_rep, labels=best_seed_dict['labels'], centers = centers[best_seed], inertia = inertias[best_seed])
    
    best_seed_dict['metrics']['inertia'] = metrics['inertia']
    best_seed_dict['metrics']['silhouette_score'] = metrics['silhouette_score']
    
    return best_seed_dict



def obs_add_multin(res_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]], key_added:str, sel_n: Union[bool, None]):
    '''create pd.Dataframe or pd.Series with labels to add to all ad.obs in ad_dict

    Args:
        res_dict = dictionary with clustering results and metrics
        key_added = clustering key
        sel_n: whether the best cluster number has been selected or not. 
        
    Returns
        pd.Dataframe or pd.Series to add to all ad.obs in ad_dict
    '''
    if sel_n == True:
        obs_add_dict = {}
        for k_value in res_dict.keys():
            if res_dict[k_value]['selected'] == True:
                    obs_add_dict[f'selected_k_{key_added}'] = res_dict[k_value]['labels']
                    if 'labels_raw' in  res_dict[k_value].keys():
                        obs_add_dict[f'selected_k_{key_added}_raw'] =  res_dict[k_value]['labels_raw'] 
        obs_add = pd.DataFrame(obs_add_dict)
        

    else:
        obs_add_dict = {}
        for k_value in res_dict.keys(): 
            obs_add_dict[f'k_{k_value}_{key_added}'] = res_dict[k_value]['labels']
            if 'labels_raw' in  res_dict[k_value].keys():
                    obs_add_dict[f'k_{k_value}_{key_added}_raw'] =  res_dict[k_value]['labels_raw']
        obs_add = pd.DataFrame(obs_add_dict)

    return obs_add

def obs_add_singlen(res_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]], key_added:str):
    '''create pd.Dataframe or pd.Series with labels to add to all ad.obs in ad_dict

    Args:
        res_dict = dictionary with clustering results and metrics
        key_added = clustering key
    
    Returns
        pd.Dataframe or pd.Series to add to all ad.obs in ad_dict
    '''
    obs_add_dict = {}
    obs_add_dict[key_added] = res_dict['labels']
    if 'labels_raw' in res_dict.keys():
        obs_add_dict[f'{key_added}_raw'] = res_dict['labels_raw']
    
    obs_add = pd.DataFrame(obs_add_dict)
    return obs_add

def match_index_to_ad_obs(ad: AnnData, obs_add_sample: Union[pd.Series, pd.DataFrame], index_not_in_cluster:list ):
    complete_index = ad.obs.index.to_list()
    obs_add_sample = obs_add_sample.reindex(complete_index)
    if type(obs_add_sample) == pd.Series:
        assert obs_add_sample.loc[index_not_in_cluster].isna().all(), 'Reindexed rows that were not in clusters are not all NaN.'
    else:
        assert obs_add_sample.loc[index_not_in_cluster].isna().all().all(), 'Reindexed rows that were not in clusters are not all NaN.'           

    return obs_add_sample


def clustering_add(ad_dict: Dict[str, AnnData], res_dict: dict, key_added:str, sel_n:bool = None):
    '''Addition of clustering results to original (inplace) or new(not inplace) AnnData objects in ad_dict.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        res_dict: Dictionary containing clustering results.
        clustering_key: str.
        inplace: Whether to add the clustering results to the current AnnData instances or to return a new one.
    Return
        no return if inplace
        new ad_dict if not inplace
    '''
    if type(list(res_dict.keys())[0])==int:
        for n_dict in res_dict.values():
            n_dict['labels'] = n_dict['labels'].dropna().astype(int).map("niche_{}".format).reindex(n_dict['labels'].index).astype('category')
            if 'labels_raw' in n_dict.keys():
                n_dict['labels_raw'] = n_dict['labels_raw'].dropna().astype(int).map("niche_{}".format).reindex(n_dict['labels_raw'].index).astype('category')
        obs_add = obs_add_multin(res_dict=res_dict, key_added=key_added, sel_n=sel_n)    
    else:
        res_dict['labels'] = res_dict['labels'].dropna().astype(int).map("niche_{}".format).reindex(res_dict['labels'].index).astype('category')
        if 'labels_raw' in res_dict.keys():
            res_dict['labels_raw'] = res_dict['labels_raw'].dropna().astype(int).map("niche_{}".format).reindex(res_dict['labels_raw'].index).astype('category')
        obs_add = obs_add_singlen(res_dict=res_dict, key_added=key_added)
    
    for sample_id, ad in ad_dict.items():
        obs_add_sample = obs_add.xs(sample_id, level='sample_id')
        not_in_clusters= ad.obs.index.difference(obs_add_sample.index).tolist()#check because of previous filtering steps
        if len(not_in_clusters) > 0:
            obs_add_sample = match_index_to_ad_obs(ad=ad, obs_add_sample=obs_add_sample, index_not_in_cluster=not_in_clusters)       
        assert obs_add_sample.index.equals(ad.obs.index), 'Indices of obs_add_sample and ad.obs do not match.'
        
        if type(obs_add_sample)==pd.Series:
            if key_added in ad.obs:  # drop previous computation of metric
                ad.obs.drop(key_added, axis=1, inplace=True)
            ad.obs[key_added] = obs_add_sample
        else:
            if obs_add_sample.columns.isin(set(ad.obs.columns)).any():
                ad.obs.drop(columns=set(obs_add_sample.columns).intersection(ad.obs.columns), inplace=True)
            ad.obs = ad.obs.join(obs_add_sample)  
        if f'{key_added}_metrics' in ad.uns.keys():
            del ad.uns[f'{key_added}_metrics'] 
        ad.uns[f'{key_added}_metrics'] = res_dict
    
    return   


def cluster_singlen(ad_dict: Dict[str, AnnData], n_rep: pd.DataFrame, cl_n:int, random_seeds: int, cl_algorithm: str, cl_filter:bool, cl_filtering_prop: float , cl_filtering_ent:str, min_obs: int, random_state: int = 42, filter_attr:str = None, **cl_params ):
    '''clustering and metrics one n and one seed
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        n_rep: DataFrame with neighborhood representation to cluster.
        cl_n: number of clusters.
        random_seeds: number of seeds with which compute the clustering.
        random_state: number with which initialize numpy in order to generate the same seeds for reproducibility of results.
        cl_algorithm = clustering algorithm.
        cl_filtee: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_prop: propotion of clusters to keep based on the cluster_filtering_entity.
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.
        min_obs: minimum number of obs of a sample assigned to a cluster to consider the cluster present in the sample when filtering. 
        filter_attr: whether to filter cluster considering entities containing at least 13 of observations with filter_attr
        **cl_params = additional parameters for the clustering
    
    Return
        dictionary with 
            - keys = cluster numbers 
            - values = res_dict for the cluster number 
                    - 'labels' = pd.Series with all ad_dict labels (filtered if there has been cluster filtering)
                    - 'labels_raw' (if there has been cluster filtering) = pd.Series with all ad_dict unfiltered labels
                    - 'seed' = seed
                    - 'metrics' = dictionary with
                        - 'silhouette_score'
                        - 'inertia'
                        - 'avg_ari' of the selected seed (if multiple random seeds have been used)
                    -'n_clusters' = number of clusters

    '''
    # set random state
    np.random.seed(random_state)

    if random_seeds == 1:
        seed = np.random.randint(0, 2**32, dtype='uint64') 
        res_dict = cluster_singleseed(ad_dict=ad_dict, n_rep=n_rep, cl_n=cl_n, seed=seed, cl_algorithm=cl_algorithm, cl_filter=cl_filter, cl_filtering_prop=cl_filtering_prop, cl_filtering_ent=cl_filtering_ent, min_obs=min_obs, filter_attr=filter_attr, **cl_params )
    
    else:
        seeds = np.random.randint(0, 2**32, size=random_seeds, dtype='uint64').tolist()
        res_dict = cluster_multiseed(ad_dict=ad_dict, n_rep=n_rep, cl_n=cl_n, seeds=seeds, cl_algorithm=cl_algorithm, cl_filter=cl_filter, cl_filtering_prop=cl_filtering_prop, cl_filtering_ent=cl_filtering_ent, min_obs=min_obs, filter_attr=filter_attr, **cl_params )

    return res_dict

def cluster_multin(ad_dict: Dict[str, AnnData], n_rep: pd.DataFrame, cl_n:int, random_seeds: int, cl_algorithm: str, cl_filter:bool, cl_filtering_prop: float , cl_filtering_ent:str, min_obs: int, sel_n: bool, sel_n_metric: str, random_state: int = 42,  filter_attr:str = None,**cl_params):
    '''clustering and metrics one n and one seed
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        n_rep: DataFrame with neighborhood representation to cluster.
        cl_n: number of clusters.
        random_seeds: number of seeds with which compute the clustering.
        random_state: number with which initialize numpy in order to generate the same seeds for reproducibility of results.
        cl_algorithm = clustering algorithm.
        cl_filtee: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_prop: propotion of clusters to keep based on the cluster_filtering_entity.
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.
        min_obs: minimum number of obs of a sample assigned to a cluster to consider the cluster present in the sample when filtering. 
        filter_attr: whether to filter cluster considering entities containing at least 13 of observations with filter_attr
        sel_n: whether to select the best k based on the select_k_metric. 
        sel_n_metric: metric to use for selecting the best k. Options are 'silhouette_score' or 'inertia' or 'average_ARI'.
                    it cannot be average_ARI if random_seeds == 1 is False. 
        **cl_params = additional parameters for the clustering
    
    Return
        dictionary with 
            - keys = cluster numbers 
            - values = res_dict for the cluster number 
                    - 'labels' = pd.Series with all ad_dict labels (filtered if there has been cluster filtering)
                    - 'labels_raw' (if there has been cluster filtering) = pd.Series with all ad_dict unfiltered labels
                    - 'seed' = seed
                    - 'metrics' = dictionary with
                        - 'silhouette_score'
                        - 'inertia'
                        - 'avg_ari' of the selected seed (if multiple random seeds have been used)
                    -'n_clusters' = number of clusters
                    - 'selected' = True or False depending if it was the cluster number selected (if there has been a cluster number selection)

    '''
    # set random state
    np.random.seed(random_state)

    res_dict = {}
    for i in cl_n:
        n_rep_copy = n_rep.copy()
        if random_seeds == 1:
            seed = np.random.randint(0, 2**32, dtype='uint64') 
            n_dict = cluster_singleseed(ad_dict=ad_dict, n_rep=n_rep_copy, cl_n=i, seed=seed, cl_algorithm=cl_algorithm, cl_filter=cl_filter, cl_filtering_prop=cl_filtering_prop, cl_filtering_ent=cl_filtering_ent, min_obs=min_obs, filter_attr=filter_attr, **cl_params )    
        else:
            seeds = np.random.randint(0, 2**32, size=random_seeds, dtype='uint64').tolist()
            n_dict = cluster_multiseed(ad_dict=ad_dict, n_rep=n_rep_copy, cl_n=i, seeds=seeds, cl_algorithm=cl_algorithm, cl_filter=cl_filter, cl_filtering_prop=cl_filtering_prop, cl_filtering_ent=cl_filtering_ent, min_obs=min_obs, filter_attr=filter_attr, **cl_params )
        res_dict[i] = n_dict
    
    if sel_n:
        assert sel_n_metric in ['silhouette_score', 'inertia', 'avg_ari'], f'select_k_metric {sel_n_metric} not recognized. Use "silhouette_score" or "inertia" or "avg_ari".'
        res_dict = n_slection(res_dict=res_dict, sel_n_metric=sel_n_metric)
    
    return res_dict

def clustering(ad_dict: Dict[str,AnnData], n_rep_key: str = None, attr_rep: str = None, mode_rep: str = None, graph_key: str = None, n_filtering: bool = False, min_neigh: int = 0,
                cl_algorithm: str = 'kmeans', cl_n: Union[int, List[int]] = 4, random_seeds: int= 1, random_state: int = 42, key_added: str = None, cl_filter: bool = False, cl_filtering_prop: float = 0.0, cl_filtering_ent: str = 'sample_id', min_obs: int = 0, filter_attr: str = None, sel_n:bool = False, sel_n_metric: str = 'silhouette_score', inplace:bool=True, **cl_params):
    
    """K-means clustering of the merged (all samples together) local neighborhood representation of cells.

    if there are more than one random seed provided -> for each k the clustering with the lowest inertia will be chosen and added to the anndata objects.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        n_rep_key: Specifies the neighborhood representation to use in ad.obsm.
        attr_rep: Categorical feature in ad.obs to use for the neighborhood representation. 
        mode_rep: 'proportion' or 'counts' to specify the type of neighborhood representation to compute.
        graph_key: Specifies the graph representation to use in ad.obsp.
        n_filtering: whether to filter out cells with less than min_neigh neighbors.
        min_neigh: if n_filtering is True -> cells with less than min_neigh neighbors are filtered out of the neighborhood representation
        cl_algorithm = clustering algorithm.
        cl_n: number of clusters or list of number of clusters.
        random_seeds: number of seeds with which compute the clustering.
        random_state: number with which initialize numpy in order to generate the same seeds for reproducibility of results.
        key_added: clustering key 
        cl_filter: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_prop: propotion of clusters to keep based on the cluster_filtering_entity.
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.
        min_obs: minimum number of obs of a sample assigned to a cluster to consider the cluster present in the sample when filtering. 
        filter_attr: whether to filter cluster considering entities containing at least 13 of observations with filter_attr
        sel_n: whether to select the best cluster number based on the sel_n_metric. 
        sel_n_metric: metric to use for selecting the best k. Options are 'silhouette_score' or 'inertia' or 'average_ARI'.
                    it cannot be avg_ari if random_seeds == 1 is False.
        inplace: Whether to add the clustering results to the current AnnData instances or to return a new one.
        **cl_params: Additional arguments for clustering function. -> if not provided default values will be used.

    Returns: 
        
    """

    # generate a copy if necessary
    ad_dict = ad_dict if inplace else copy.deepcopy(ad_dict)
    
    # get aggregated neighbor representation
    n_rep = aggregate_n_rep(ad_dict=ad_dict, n_rep_key=n_rep_key, attr_rep=attr_rep, mode_rep=mode_rep, graph_key=graph_key, n_filtering=n_filtering, min_neigh=min_neigh)

    # one cluster number
    if type(cl_n) == int:
        res_dict = cluster_singlen(ad_dict=ad_dict, n_rep=n_rep, cl_n=cl_n, random_seeds=random_seeds, cl_algorithm=cl_algorithm, cl_filter=cl_filter, cl_filtering_prop=cl_filtering_prop, cl_filtering_ent=cl_filtering_ent, min_obs=min_obs, random_state=random_state, filter_attr=filter_attr, **cl_params)
    
    # multiple cluster numbers 
    elif type(cl_n) == list:
        assert len(cl_n) > 1, 'If k is a list it must contain more than one value.' 
        res_dict = cluster_multin(ad_dict=ad_dict, n_rep=n_rep, cl_n=cl_n, random_seeds=random_seeds, cl_algorithm=cl_algorithm, cl_filter=cl_filter, cl_filtering_prop=cl_filtering_prop, cl_filtering_ent=cl_filtering_ent, min_obs=min_obs, sel_n=sel_n, sel_n_metric=sel_n_metric, random_state=random_state, filter_attr=filter_attr, **cl_params)
    
    # add clusterign results in ad_dict
    clustering_add(ad_dict=ad_dict, res_dict=res_dict, key_added=key_added, sel_n=sel_n)

    return

# %%
