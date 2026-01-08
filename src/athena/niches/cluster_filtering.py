import pandas as pd
from anndata import AnnData
#%%

def get_cluster_filter_column(ad_dict, merged, cluster_filtering_entity: str):
    ''' Get cluster filtering column.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        merged: Merged DataFrame of cluster representations.
        cluster_filtering_entity: entity to use for cluster filtering. Options are 'sample_id' or any categorical column in ad.obs.

    Returns:
        pd.Series: Series containing the cluster filtering entity for each cluster in the merged DataFrame.
        '''
    
    if cluster_filtering_entity == 'sample_id':
        final_cluster_filtering_column = pd.Series(
        merged.index.get_level_values('sample_id'), 
        index=merged.index, 
        name='sample_id'
        )
        
    else: 
        cluster_filtering_column = list()
        for sample_id, ad in ad_dict.items():
            cluster_filtering = ad.obs[cluster_filtering_entity].copy()
            cluster_filtering.index = pd.MultiIndex.from_product([[sample_id], cluster_filtering.index], names=['sample_id', 'cell_id'])
            assert cluster_filtering.index.get_level_values('cell_id').equals(ad.obs_names), f'cell_id level of {sample_id} MultiIndex does not match ad.obs_names.'
            cluster_filtering_column.append(cluster_filtering)
        final_cluster_filtering_column = pd.concat(cluster_filtering_column, axis=0)
    return final_cluster_filtering_column

def cluster_filter_to_merge(ad_dict, merged, cluster_filtering_entity: str):
    ''' Adds cluster filtering column to merged DataFrame.
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        merged: Merged DataFrame of cluster representations.
        cluster_filtering_entity: entity to use for cluster filtering. Options are 'sample_id' or any categorical column in ad.obs.

    Returns:
        merged DataFrame with cluster filtering column added.
        '''
    cluster_filtering_column = get_cluster_filter_column(ad_dict, merged, cluster_filtering_entity)
    only_in_cluster_filtering_column = cluster_filtering_column.index.difference(merged.index).tolist() #check because of previous filtering steps
    if len(only_in_cluster_filtering_column) > 0:
        cluster_filtering_column = cluster_filtering_column.drop(index=only_in_cluster_filtering_column)
    assert merged.index.equals(cluster_filtering_column.index), 'Indices of merged DataFrame and cluster filter column do not match.'      
    merged['cluster_filter'] = cluster_filtering_column
    return merged


def perform_cluster_filtering(merged: pd.DataFrame, cluster_filtering_percentage: int):
    assert (1 <= cluster_filtering_percentage <= 100), "cluster_filtering_percentage must be between 1 and 100"

    merged_initial_index = merged.index.copy()
    # calculate the threshold 
    total_filter_entities = merged['cluster_filter'].nunique()
    threshold = (cluster_filtering_percentage / 100) * total_filter_entities
    # count unique entities per label
    cluster_filter_counts = merged.groupby('labels')['cluster_filter'].nunique()
    # identify labels (clusters) that fall below the thereshold
    index_to_filter = cluster_filter_counts[cluster_filter_counts < threshold].index.tolist()
    # change the labels (cluster) that have to be filtered out to nan values
    merged.loc[merged['labels'].isin(index_to_filter),'labels'] = pd.NA


    assert merged_initial_index.equals(merged.index), 'Indices of merged DataFrame have changed after cluster filtering.'
    
    return  merged['labels']