import pandas as pd
from anndata import AnnData
from typing import Dict, Union, List
from athena.niches.clustering.cluster_analysis import aggregate_attr, freq_attr
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
    if cl_filtering_ent != 'sample_id':
        assert ad_dict != None, "if cl_filtering_ent != 'sample_id, ad_dict has to be provided"

    # if cluster_filtering_ent == sample_id -> get it from the res_dfr MultiIndex
    if cl_filtering_ent == 'sample_id':
        cl_entities = pd.Series(
        res_df.index.get_level_values('sample_id'), 
        index=res_df.index, 
        name='sample_id'
        )
        '''if filt_attr:
            cl_filt_attr = aggregate_attr(ad_dict=ad_dict, attr=filt_attr)
            only_cl_filt_attr = cl_filt_attr.index.difference(cl_entities.index).tolist() # observations not present in res_df 
            if len(only_cl_filt_attr) > 0:
                cl_filt_attr = cl_filt_attr.drop(index=only_cl_filt_attr)
            cl_entities = pd.concat({filt_attr: cl_filt_attr, 'cl_filter': cl_entities}, axis=1)'''
    
    # else, loop the ad_dict and get it from each anndata obs column
    else: 
        '''if filt_attr is not None:
            cl_entities = aggregate_attr(ad_dict=ad_dict, attr=[cl_filtering_ent,filt_attr])
            cl_entities = cl_entities.rename(columns={cl_filtering_ent: 'cl_filter'})'''
        cl_entities = aggregate_attr(ad_dict=ad_dict, attr=cl_filtering_ent)
        
    return cl_entities



def cl_filter_add(res_df: pd.DataFrame, cl_filtering_ent: str, ad_dict: Dict[str, AnnData] = None):
    ''' Adds cluster filtering column to merged DataFrame.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        res_df: DataFrame of aggregated neighborhood representations of all the samples and 'labels' column with cluster assignments
        cluster_filtering_ent: entity to use for cluster filtering. Options are 'sample_id' or any categorical column in ad.obs.

    Returns:
        merged DataFrame with 'cl_filter' column added.
        '''
    cl_filter = cl_ent(ad_dict=ad_dict,res_df=res_df, cl_filtering_ent=cl_filtering_ent)
    
    if cl_filtering_ent != 'sample_id':
        assert ad_dict != None, "if cl_filtering_ent != 'sample_id, ad_dict has to be provided"
    
        # check if there are some observations that have been filtered before (low number of neighbors)
        only_in_cl_filter = cl_filter.index.difference(res_df.index).tolist() # observations not present in res_df 
        if len(only_in_cl_filter) > 0:
            cl_filter = cl_filter.drop(index=only_in_cl_filter) # if present, drop observations not present in res_df 

        assert res_df.index.equals(cl_filter.index), 'Indices of merged DataFrame and cluster filter column do not match.'      

    res_df['cl_filter'] = cl_filter

    return res_df



def cl_filtering( res_df: pd.DataFrame, cl_filtering_prop: float = 0.0 , cl_filtering_nent: int = 0, cl_filtering_ent: str = 'sample_id', ad_dict: Dict[str, AnnData]=None,min_obs: int = 0):
    """Filtering of clusters that are present in less than cluster_filtering_perc of the cluster_filtering_ent
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        res_df:  DataFrame of aggregated neighborhood representations of all the samples and 'labels' column with cluster assignments
        cl_filter: whether to filter clusters based on the cluster_filtering_entity.
        cl_filtering_prop: minimum proportion of entities in cl_filtering_ent that a cluster label has to be in to be kept
        cl_filtering_nent: minimum number of entities in cl_filtering_ent that a cluster label has to be in to be kept
        cl_filtering_ent: entity to use for clusters filtering. Options are 'sample_id' or any categorical column in ad.obs.
        min_obs: minimum number of obs of a sample assigned to a cluster to consider the cluster present in the sample when filtering.

    Returns: 
        modified res_df:
            - 'labels_raw' = copy of original 'labels' column
            - 'labels' = filtered labels, with Na instead of filtered labels
        
    """
    #res_df= res_df.copy()
    if cl_filtering_ent != 'sample_id':
        assert ad_dict != None, "if cl_filtering_ent != 'sample_id, ad_dict has to be provided"
    
    assert (cl_filtering_prop>0.0) != (cl_filtering_nent>0), 'provide either minimum proportion (cl_filtering_prop) or number (cl_filtering_nent) of entities that a cluster label has to be in to be kept'
    assert (0.0 <= cl_filtering_prop <= 1), "cluster_filtering_propotion must be between 1 and 100"

    # add column with filtering entity to the aggregated neighborhood representation
    res_df = cl_filter_add(ad_dict=ad_dict, res_df=res_df, cl_filtering_ent=cl_filtering_ent)
    # make a copy of the original labels
    res_df['labels_raw'] = res_df['labels'].copy()

    # calculate the threshold 
    total_filter_entities = res_df['cl_filter'].nunique() # number of unique cluster filtering entities  
    assert (0 <= cl_filtering_nent <= total_filter_entities), 'cl_filtering_nent has to be between 0 and the total number of filter entities'
    
    # threshold = number of cluster filtering entities representing the cluster filtering propotion
    if cl_filtering_prop>0.0:
        threshold = cl_filtering_prop * total_filter_entities
    else:
        threshold = cl_filtering_nent
    # count how many entities per label have more than min_obs observations
    cl_filter_counts = res_df.groupby('labels')['cl_filter'].apply(lambda x: (x.value_counts() >= min_obs).sum())
    # identify labels (clusters) that fall below the thereshold = that are present in less than the threshold number of cluster filtering entities
    index_to_filter = cl_filter_counts[cl_filter_counts < threshold].index.tolist()
    
    if pd.api.types.is_categorical_dtype(res_df['labels']):
        print(index_to_filter)
        res_df['labels'] = res_df['labels'].cat.add_categories(['NaN'])
        res_df.loc[res_df['labels'].isin(index_to_filter),'labels'] = 'NaN'
    else:
        res_df.loc[res_df['labels'].isin(index_to_filter),'labels'] = pd.NA
    # drop cl_filter because it won't be needed
    res_df = res_df.drop(columns=['cl_filter'])
    return  res_df

