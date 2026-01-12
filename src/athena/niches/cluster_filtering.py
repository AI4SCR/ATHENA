import pandas as pd
from anndata import AnnData
from typing import Dict, Union, List
#%%

def cl_ent(ad_dict: Dict[str, AnnData] , res_df: pd.DataFrame, cl_filtering_ent: str):
    ''' Get pd.Series containing the cluster filtering entity of each observation_id of each sample_id.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        res_df: DataFrame of aggregated neighborhood representations of all the samples and 'labels' column with cluster assignments
        cl_filtering_ent: entity to use for cluster filtering. Options are 'sample_id' or any categorical column in ad.obs.

    Returns:
        pd.Series: Series containing the cluster filtering entity for each observation.
        '''
    
    # if cluster_filtering_ent == sample_id -> get it from the res_dfr MultiIndex
    if cl_filtering_ent == 'sample_id':
        cl_entities = pd.Series(
        res_df.index.get_level_values('sample_id'), 
        index=res_df.index, 
        name='sample_id'
        )
    
    # else, loop the ad_dict and get it from each anndata obs column
    else: 
        cl_ent_list = list()
        for sample_id, ad in ad_dict.items():
            # get the ad.obs column
            ad_ent = ad.obs[cl_filtering_ent].copy()
            # change index to a MultiIndex
            ad_ent.index = pd.MultiIndex.from_product([[sample_id], ad_ent.index], names=['sample_id', 'observation_id'])
            # assert that the observation_id 
            assert ad_ent.index.get_level_values('observation_id').equals(ad.obs_names), f'observation_id level of {sample_id} MultiIndex does not match ad.obs_names.'
            cl_ent_list.append(ad_ent)
        # concatenate the single ad entities in a pd.Series     
        cl_entities = pd.concat(cl_ent_list, axis=0)
    
    return cl_entities



def cl_filter_add(ad_dict: Dict[str, AnnData], res_df: pd.DataFrame, cl_filtering_ent: str):
    ''' Adds cluster filtering column to merged DataFrame.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        res_df: DataFrame of aggregated neighborhood representations of all the samples and 'labels' column with cluster assignments
        cluster_filtering_ent: entity to use for cluster filtering. Options are 'sample_id' or any categorical column in ad.obs.

    Returns:
        merged DataFrame with 'cl-filter' column added.
        '''
    cl_filter = cl_ent(ad_dict=ad_dict,res_df=res_df, cl_filtering_ent=cl_filtering_ent)
    # check if there are some observations that have been filtered before (low number of neighbors)
    only_in_cl_filter = cl_filter.index.difference(res_df.index).tolist() # observations not present in res_df 
    if len(only_in_cl_filter) > 0:
        cl_filter = cl_filter.drop(index=only_in_cl_filter) # if present, drop observations not present in res_df 

    assert res_df.index.equals(cl_filter.index), 'Indices of merged DataFrame and cluster filter column do not match.'      

    res_df['cl_filter'] = cl_filter

    return res_df



def cl_filtering(ad_dict: Dict[str, AnnData], res_df: pd.DataFrame, cl_filtering_perc: int, cl_filtering_ent: str):
    """Filtering of clusters that are present in less than cluster_filtering_perc of the cluster_filtering_ent
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        res_df:  DataFrame of aggregated neighborhood representations of all the samples and 'labels' column with cluster assignments
        cl_filtering: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_perc: percentage of clusters to keep based on the cluster_filtering_entity.
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.

    Returns: 
        modified res_df:
            - 'labels_raw' = copy of original 'labels' column
            - 'labels' = filtered labels, with Na instead of filtered labels
        
    """
    assert (1 <= cl_filtering_perc <= 100), "cluster_filtering_percentage must be between 1 and 100"

    # add column with filtering entity to the aggregated neighborhood representation
    res_df = cl_filter_add(ad_dict=ad_dict, res_df=res_df, cl_filtering_ent=cl_filtering_ent)
    # make a copy of the original labels
    res_df['labels_raw'] = res_df['labels'].copy()

    # calculate the threshold 
    total_filter_entities = res_df['cl_filter'].nunique() # number of unique cluster filtering entities  
    threshold = (cl_filtering_perc / 100) * total_filter_entities # threshold = number of cluster filtering entities representing the cluster filtering percentage
    # count unique entities per label
    cl_filter_counts = res_df.groupby('labels')['cl_filter'].nunique() 
    # identify labels (clusters) that fall below the thereshold = that are present in less than the threshold number of cluster filtering entities
    index_to_filter = cl_filter_counts[cl_filter_counts < threshold].index.tolist()
    # change the labels (cluster) that have to be filtered out to nan values
    res_df.loc[res_df['labels'].isin(index_to_filter),'labels'] = pd.NA
    
    # drop cl_filter because it won't be needed
    res_df = res_df.drop(columns=['cl_filter']
)
    return  res_df


