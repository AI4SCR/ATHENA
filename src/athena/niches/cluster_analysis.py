#%%
import numpy as np
import pandas as pd
from anndata import AnnData
from collections import defaultdict, Counter
from typing import Dict, Union, List
from sklearn.preprocessing import StandardScaler

#%%

def aggregate_attr(ad_dict: Dict[str, AnnData], attr: Union[str, List[str]]):
    """merge attr for each observation in each sample

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr_key: .obs column or list of columns to merge in one dataframe

    Returns: 
        dataframe containing the aggr .obs columns of the anndata objs in the ad_dict
        multiindex with sample_id and observation_id
    """
    aggr = None    
    for sample_id in ad_dict.keys():
        ad = ad_dict[sample_id]
        if type(attr)==str: assert attr in ad.obs.columns, 'attr is not in ad.obs.keys()'
        else: assert set(attr).issubset(ad.obs.columns), 'elements of attr are not both in ad.obs.keys()'
        sample_df = ad.obs[attr].copy()
        # Create MultiIndex for the aggr DataFrame with sample_id and cell_id as levels
        sample_ids = [sample_id] * len(sample_df)
        sample_df.index = pd.MultiIndex.from_arrays(
            arrays=[sample_ids, sample_df.index], # [Outer Level (Sample ID), Inner Level (Cell ID)]
            names=['sample_id', 'observation_id']
        )
        if aggr is None:
            aggr = sample_df
        else:
            aggr = pd.concat([aggr, sample_df], axis=0)
    
    return aggr

def mean_std_freq_attr(aggr: pd.DataFrame, attr: str, group_key: str):
    '''compute mean and standard deviation of the frequency of each attribute in each niche and overall
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr: .obs column to compute the z scores
        group_key: .obs_column with cluster labels
    Returns:
        freq_mean = pd.DataFrame with mean frequency of attributes (columns) overall and in each niche (row names) 
        freq_std = pd.DataFrame with standard deviation of frequency of attributes (columns) overall and in each niche (row names) 
    '''
    clusters = np.unique(aggr[group_key])

    counts = aggr.groupby(['sample_id', 'label_name']).size().unstack(fill_value=0)
    # divide counts of each sample by the total number of observations in the sample
    freq = counts.div(counts.sum(axis=1), axis=0)
    # mean the frequency of each attribute across samples 
    freq_mean = freq.mean().to_frame().T
    freq_std = freq.std().to_frame().T

    freq_mean.index = freq_std.index = ['overall']
    for cluster in clusters:
        aggr_cl =  aggr[aggr[group_key]==cluster]
        counts = aggr_cl.groupby(['sample_id', 'label_name']).size().unstack(fill_value=0)
        freq = counts.div(counts.sum(axis=1), axis=0)
        indexes = freq_mean.index.tolist() + [cluster]
        freq_mean = pd.concat([freq_mean, freq.mean().to_frame().T], axis=0).fillna(0)
        freq_std = pd.concat([freq_std, freq.std().to_frame().T], axis=0).fillna(0)
        freq_mean.index = freq_std.index = indexes
    return freq_mean, freq_std


def z_scores(ad_dict: Dict[str, AnnData], attr: str, group_key:str):
    '''compute z scores for each cluster and attribute
            z score (group_i, attr_i)= ((mean frequency of attr_i in group_i)-(mean frequency of attr_i overall))/(standard deviation of frequency of attr_i overall)
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr: .obs column to compute the z scores
        group_key: .obs_column with cluster labels
    Returns:
        pd.DataFrame with z scores for each group
    '''
    if f'{group_key}_raw' in ad_dict[list(ad_dict.keys())[0]].obs.keys():
        aggr = aggregate_attr(ad_dict=ad_dict, attr=[attr, group_key, f'{group_key}_raw'])
        # using the group_key_raw labels
        # drop cells that are filtered out due to low number of neighbors
        aggr = aggr.dropna(subset=[group_key, f'{group_key}_raw'], how='all')
        aggr = aggr.drop(columns=[f'{group_key}_raw']) # don't need it anymore
        # replace the filtered labels <NA> with filtered_labels so that we can analyze their composition too
        aggr = aggr.astype(str).replace('<NA>', 'filtered_labels') 
    else: 
        aggr = aggregate_attr(ad_dict=ad_dict, attr=[attr, group_key])
        # drop cells that are filtered out due to low number of neighbors
        aggr = aggr.dropna()
        aggr = aggr.astype(str)

    # count how many observations for attribute in each sample
    freq_mean, freq_std = mean_std_freq_attr(aggr=aggr, attr=attr, group_key=group_key)

    zscores = pd.DataFrame()
    for cluster in freq_mean.index: 
        if cluster == 'overall': continue
        zscore = ((freq_mean.loc[cluster] - freq_mean.loc['overall'])/freq_std.loc['overall']).to_frame().T
        if zscores.empty: 
            indexes = [cluster]
            zscores = zscore
        else: 
            indexes = zscores.index.tolist() + [cluster] 
            zscores = pd.concat([zscores, zscore], axis=0)
        
        zscores.index = indexes
    
    return zscores

def freq_attr(aggr: pd.DataFrame, attr: str, group_key: str):
    '''compute mean and standard deviation of the frequency of each attribute in each niche and overall
    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr: .obs column to compute the z scores
        group_key: .obs_column with cluster labels
    Returns:
        freq_mean = pd.DataFrame with mean frequency of attributes (columns) overall and in each niche (row names) 
        freq_std = pd.DataFrame with standard deviation of frequency of attributes (columns) overall and in each niche (row names) 
    '''
    clusters = np.unique(aggr[group_key])

    counts = aggr.groupby(['sample_id', 'label_name']).size().unstack(fill_value=0).sum()
    # divide counts of each sample by the total number of observations in the sample
    freqs = (counts/sum(counts)).to_frame().T
    freqs.index = ['overall']

    for cluster in clusters:
        aggr_cl =  aggr[aggr[group_key]==cluster]
        counts = aggr_cl.groupby(['sample_id', 'label_name']).size().unstack(fill_value=0).sum()
        freqs_group = (counts/sum(counts)).to_frame().T
        indexes = freqs.index.tolist() + [cluster]
        freqs = pd.concat([freqs, freqs_group.mean().to_frame().T], axis=0).fillna(0)
        freqs.index =indexes
    return freqs


def attr_proportions(ad_dict: Dict[str, AnnData], attr:str, group_key:str):
    '''
    Calculate attr proportions within a specific niche cluster across multiple samples.

    Args:
        ad_dict: Dictionary of AnnData instances with keys as sample names.
        attr: .obs column to compute the z scores
        group_key: .obs_column with cluster labels
    '''
    if f'{group_key}_raw' in ad_dict[list(ad_dict.keys())[0]].obs.keys():
        aggr = aggregate_attr(ad_dict=ad_dict, attr=[attr, group_key, f'{group_key}_raw'])
        # using the group_key_raw labels
        # drop cells that are filtered out due to low number of neighbors
        aggr = aggr.dropna(subset=[group_key, f'{group_key}_raw'], how='all')
        aggr = aggr.drop(columns=[f'{group_key}_raw']) # don't need it anymore
        # replace the filtered labels <NA> with filtered_labels so that we can analyze their composition too
        aggr = aggr.astype(str).replace('<NA>', 'filtered_labels') 
    else: 
        aggr = aggregate_attr(ad_dict=ad_dict, attr=[attr, group_key])
        # drop cells that are filtered out due to low number of neighbors
        aggr = aggr.dropna()
        aggr = aggr.astype(str)

    props = freq_attr(aggr=aggr, attr=attr, group_key=group_key)
        
    return props


