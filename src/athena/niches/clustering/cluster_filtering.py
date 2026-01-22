import pandas as pd
from anndata import AnnData
from typing import Dict, Union, List
from athena.niches.clustering.cluster_analysis import aggregate_attr, freq_attr
#%%

def cl_ent(ad_dict: Dict[str, AnnData] , res_df: pd.DataFrame, cl_filtering_ent: str, filt_attr: str=None):
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
        if filt_attr:
            cl_filt_attr = aggregate_attr(ad_dict=ad_dict, attr=filt_attr)
            only_cl_filt_attr = cl_filt_attr.index.difference(cl_entities.index).tolist() # observations not present in res_df 
            if len(only_cl_filt_attr) > 0:
                cl_filt_attr = cl_filt_attr.drop(index=only_cl_filt_attr)
            cl_entities = pd.concat({filt_attr: cl_filt_attr, 'cl_filter': cl_entities}, axis=1)

    
    # else, loop the ad_dict and get it from each anndata obs column
    else: 
        if filt_attr is not None:
            cl_entities = aggregate_attr(ad_dict=ad_dict, attr=[cl_filtering_ent,filt_attr])
            cl_entities = cl_entities.rename(columns={cl_filtering_ent: 'cl_filter'})
        else: 
            cl_entities = aggregate_attr(ad_dict=ad_dict, attr=cl_filtering_ent)
        
    
    return cl_entities



def cl_filter_add(ad_dict: Dict[str, AnnData], res_df: pd.DataFrame, cl_filtering_ent: str, filt_attr: str = None):
    ''' Adds cluster filtering column to merged DataFrame.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        res_df: DataFrame of aggregated neighborhood representations of all the samples and 'labels' column with cluster assignments
        cluster_filtering_ent: entity to use for cluster filtering. Options are 'sample_id' or any categorical column in ad.obs.

    Returns:
        merged DataFrame with 'cl-filter' column added.
        '''
    cl_filter = cl_ent(ad_dict=ad_dict,res_df=res_df, cl_filtering_ent=cl_filtering_ent, filt_attr=filt_attr)
    # check if there are some observations that have been filtered before (low number of neighbors)
    only_in_cl_filter = cl_filter.index.difference(res_df.index).tolist() # observations not present in res_df 
    if len(only_in_cl_filter) > 0:
        cl_filter = cl_filter.drop(index=only_in_cl_filter) # if present, drop observations not present in res_df 

    assert res_df.index.equals(cl_filter.index), 'Indices of merged DataFrame and cluster filter column do not match.'      

        
    if filt_attr:
        res_df['cl_filter'] = cl_filter['cl_filter']
        res_df[filt_attr] = cl_filter[filt_attr]
    
    else:
        res_df['cl_filter'] = cl_filter

    return res_df



def cl_filtering(ad_dict: Dict[str, AnnData], res_df: pd.DataFrame, cl_filtering_prop: float , cl_filtering_ent: str, min_obs: int = 0):
    """Filtering of clusters that are present in less than cluster_filtering_perc of the cluster_filtering_ent
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        res_df:  DataFrame of aggregated neighborhood representations of all the samples and 'labels' column with cluster assignments
        cl_filtering: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_prop: propotion of clusters to keep based on the cluster_filtering_entity.
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.
        min_obs: minimum number of obs of a sample assigned to a cluster to consider the cluster present in the sample when filtering.

    Returns: 
        modified res_df:
            - 'labels_raw' = copy of original 'labels' column
            - 'labels' = filtered labels, with Na instead of filtered labels
        
    """
    assert (0.0 <= cl_filtering_prop <= 1), "cluster_filtering_propotion must be between 1 and 100"

    # add column with filtering entity to the aggregated neighborhood representation
    res_df = cl_filter_add(ad_dict=ad_dict, res_df=res_df, cl_filtering_ent=cl_filtering_ent)
    # make a copy of the original labels
    res_df['labels_raw'] = res_df['labels'].copy()

    # calculate the threshold 
    total_filter_entities = res_df['cl_filter'].nunique() # number of unique cluster filtering entities  
    threshold = cl_filtering_prop * total_filter_entities # threshold = number of cluster filtering entities representing the cluster filtering propotion
    # count how many entities per label have more than min_obs observations
    cl_filter_counts = res_df.groupby('labels')['cl_filter'].apply(lambda x: (x.value_counts() >= min_obs).sum())
    # identify labels (clusters) that fall below the thereshold = that are present in less than the threshold number of cluster filtering entities
    index_to_filter = cl_filter_counts[cl_filter_counts < threshold].index.tolist()
    # change the labels (cluster) that have to be filtered out to nan values
    res_df.loc[res_df['labels'].isin(index_to_filter),'labels'] = pd.NA
    
    # drop cl_filter because it won't be needed
    res_df = res_df.drop(columns=['cl_filter']
)
    return  res_df


def cl_filtering_filt_attr(ad_dict: Dict[str, AnnData], filt_attr: str, res_df: pd.DataFrame, cl_filtering_prop: float , cl_filtering_ent: str):
    """Filtering of clusters that are present in less than cluster_filtering_perc of the cluster_filtering_ent
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        res_df:  DataFrame of aggregated neighborhood representations of all the samples and 'labels' column with cluster assignments
        cl_filtering: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_prop: propotion of clusters to keep based on the cluster_filtering_entity.
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.

    Returns: 
        modified res_df:
            - 'labels_raw' = copy of original 'labels' column
            - 'labels' = filtered labels, with Na instead of filtered labels
        
    """
    assert (1 <= cl_filtering_prop <= 100), "cluster_filtering_propotion must be between 1 and 100"

    # add column with filtering entity to the aggregated neighborhood representation
    res_df = cl_filter_add(ad_dict=ad_dict, res_df=res_df, cl_filtering_ent=cl_filtering_ent, filt_attr=filt_attr)
    # make a copy of the original labels
    aggr = res_df[['labels', filt_attr, 'cl_filter']]
    attr_proportions = freq_attr(aggr=aggr, attr=filt_attr, group_key='labels')

    n_top_attr = int(len(res_df[filt_attr].unique())*0.1) # 10% of the attributes
    labels_to_filter = []
    
    for label in res_df['labels'].unique():
        aggr_copy = aggr.copy()
        top_attr = attr_proportions.loc[label].nlargest(n_top_attr).index.tolist()
        # group by the sample_id (level 0 of the MultiIndex)
        # for each sample, check if top_filt is a subset of all its 'filt_attr' values
        
        '''valid_samples_mask = aggr_copy.groupby('cl_filter')[filt_attr].apply(
        lambda x: all(x.value_counts().get(attr, 0) >= 13[] for attr in top_attr)
        )'''
        valid_samples_mask = aggr_copy.groupby('cl_filter')[filt_attr].apply(
        lambda x: all(
        (x == attr).mean() >= attr_proportions.loc['overall'].max()
        for attr in top_attr
        ))

        # list of sample_ids that returned True
        valid_sample_ids = valid_samples_mask[valid_samples_mask].index

        # filter the original dataframe to keep all observations for those samples
        filtered_df = aggr_copy[aggr_copy['cl_filter'].isin(valid_sample_ids)]
        
        # check if the cluster is each cl_filter category of the filtered_df
        cl_pres = filtered_df.groupby('cl_filter')['labels'].apply(lambda x: (x == label).any())

        # propotion of True values (True = 1 and False = 0 => the mean is the propotion)
        propotion = cl_pres.mean()*100
        if propotion < cl_filtering_prop:
            labels_to_filter.append(label)
    # make a copy of the original labels
    res_df['labels_raw'] = res_df['labels'].copy()

    # change the labels (cluster) that have to be filtered out to nan values
    res_df.loc[res_df['labels'].isin(labels_to_filter),'labels'] = pd.NA
    
    # drop cl_filter because it won't be needed
    res_df = res_df.drop(columns=['cl_filter', filt_attr]
    )
    return  res_df