#%%
import numpy as np
import pandas as pd
from anndata import AnnData
from collections import defaultdict, Counter
from typing import Dict, Union, List
from sklearn.preprocessing import StandardScaler

#%%

def merged_info_anndata_dict(ad_dict: Dict[str, AnnData], obs_key: Union[str, List[str]]):
    """Merge the count or proportions of attr for each cell in the AnnData object based on the specified topology.

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



def z_scores(ad_dict, clustering_key = str, attr = str, include_dropped_cluster:  bool = False):
    '''
    Compute z-scores of attr enrichment in each cluster.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        clustering_key (str): Key in AnnData.obs representing the clustering.
        attr (str): Key in AnnData.obs representing the labels to assess enrichment.

    Returns:
        pd.DataFrame: DataFrame of z-scores with clusters as rows and labels as columns.
    '''
    merged = merged_info_anndata_dict(ad_dict, obs_key=[clustering_key, attr])
    if include_dropped_cluster== False:
        merged = merged.dropna()
    clusters = np.unique(merged[clustering_key])
    clusters = [cluster for cluster in clusters if pd.notna(cluster)]
    labels = np.unique(merged[attr])
    df = pd.DataFrame(0, index=clusters, columns=labels)
    for cluster in clusters:
        n = Counter(list(merged[merged[clustering_key]==cluster][attr]))
        for label in n.keys():
            df.loc[cluster, label] = n[label]
    
    # Compute column-wise z-scores
    
    scaler = StandardScaler()
    z_scores = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns )
        
    return z_scores


def attr_proportions(ad_dict: Dict[str, AnnData], attr:str, group: Union[str, List[str]], obs_key: str):
    '''
    Calculate cell type proportions within a specific niche cluster across multiple samples.

    Args:
        ad_dict (Dict[str, AnnData]): Dictionary of AnnData instances with keys as sample names.
        attr (str): Key in AnnData.obs representing the labels to calculate proportions for.
        group (Union[str, List[str]]): 'whole_sample', 'per_group', 'per_group and whole_sample', or specific group value.
        obs_key (str): Key in AnnData.obs representing the grouping.   
    '''
    if group == 'whole_sample':
        label_counts = None
        for sample_id, ad in ad_dict.items():
            label_counts_sample =ad.obs[attr].value_counts()
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
                label_counts_group = ad_group.obs[attr].value_counts()
                if label_counts is None:
                    label_counts = pd.DataFrame(label_counts_group, columns=[group_id])
                else:
                    if group_id in label_counts.columns:
                        label_counts[group_id] = label_counts[group_id].add(label_counts_group, fill_value=0)
                    else:
                        label_counts[group_id] = label_counts_group
            if group == 'per_group and whole_sample':
                label_counts_sample =ad.obs[attr].value_counts()
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
            label_counts_group = ad_group.obs[attr].value_counts()
            if label_counts is None:
                label_counts = label_counts_group
            else:
                label_counts = label_counts.add(label_counts_group, fill_value=0)
        proportions = label_counts / label_counts.sum()
        return proportions