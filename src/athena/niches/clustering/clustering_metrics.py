#%%
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Union, List

#%%
def compute_inertia(filtered_n_rep: pd.DataFrame, filtered_labels: pd.Series, centers: np.ndarray):
    '''Compute inertia for filtered cluster labels

    Args:
        filtered_n_rep:  DataFrame of neighborhood representations without observations with filtered labels.
        filtered_labels: pd.Series without filtered labels.
        centers: array with cluster centers.

    Return:
        inertia(float)
    '''
    remaining_clusters = np.unique(filtered_labels)
    centers_map = {i: centers[i] for i in remaining_clusters}
    assigned_centers = np.array([centers_map[label] for label in filtered_labels])
    squared_distances = np.sum((filtered_n_rep.values - assigned_centers)**2, axis=1)
    inertia = np.sum(squared_distances)
    return inertia


def get_sampled_silhouette(values_df: pd.DataFrame, labels: pd.Series, sampling_size:int = 10000, save_sil_scores:Union[str, None]=None):
    # stratisfied sampling of the data    
    X = values_df.values
    idx, _ = train_test_split(
        np.arange(len(labels)),
        train_size=min(sampling_size / len(X), 1.0),
        stratify=labels,
        random_state=42
    )
    
    X_sample = X[idx]
    labels_sample = labels.iloc[idx].values
    
    # computing individual silhouette samples
    sample_scores = silhouette_samples(X_sample, labels_sample, metric='euclidean', n_jobs=12)
    if save_sil_scores:
        sil_df = labels.iloc[idx].to_frame(name='label').copy()
        sil_df['silhouette_score'] = sample_scores
        sil_df.to_parquet(save_sil_scores)
    
    sil_score = np.mean(sample_scores)   
    return sil_score



def get_metrics(values_df: pd.DataFrame, labels: pd.Series, centers: Union[np.ndarray, None] = None, inertia: Union[float, None] = None, sampling_size:int = 10000, save_sil_scores:Union[str, None]=None):  
    '''Compute metrics (inertia, silhouette score, CH score, DB score)

    Args:
        values_df: DataFrame of neighborhood representations onto which the clustering has been done.
        labels: Series with clustering labels.
        centers: array with cluster centers (or None if inertia has been given).
        inertia: cluster inertia (or None if the clusters have been filtered).
    
    Return
        dictionary with 'inertia' and 'silhouette_score'
    '''
    assert (type(inertia) == float) or (type(centers) == np.ndarray), "Either inertia or centers must be provided to compute metrics."

    # if there has not have been cluster filtering, inertia does not have to be computed -> onòy compute the silhouette score
    if type(inertia) == float:
        silhouette = get_sampled_silhouette(values_df=values_df, labels=labels, sampling_size=sampling_size, save_sil_scores=save_sil_scores)
        db_score = davies_bouldin_score(values_df, labels)
        ch_score = calinski_harabasz_score(values_df, labels)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette, 'calinski_harabasz_score': ch_score, 'davies_bouldin_score':db_score}
        return metrics
    
    # else, recompute inertia and silhouette score
    elif type(centers) == np.ndarray:
        filtered_labels = labels.dropna() # clusters labels that 'passed' the filtering step
        filtered_labels = filtered_labels.astype(int)
        filtered_n_rep = values_df.loc[filtered_labels.index]
        inertia = compute_inertia(filtered_n_rep, filtered_labels, centers)
        silhouette =get_sampled_silhouette(values_df=filtered_n_rep, labels=filtered_labels, sampling_size=sampling_size, save_sil_scores=save_sil_scores)
        db_score = davies_bouldin_score(filtered_n_rep, filtered_labels)
        ch_score = calinski_harabasz_score(filtered_n_rep, filtered_labels)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette, 'calinski_harabasz_score': ch_score, 'davies_bouldin_score':db_score}
        
        return metrics
    


def stratified_sampling(values_df: pd.DataFrame, labels: pd.Series):
    '''Subsample 10% of total observations for silhouette score taking into account cluster labels frequency. 
    Args:
        values_df: DataFrame of neighborhood representations onto which the clustering has been done.
        labels: Series with clustering labels.
    
    Return
        sampled_values and sampled_labels
    '''
    assert labels.index.equals(values_df.index)
    df = values_df.copy()
    df['labels'] = labels

    total_n = 15
    fraction = total_n / len(df)
    
    #sampled_df = df.groupby('labels', group_keys=False).apply(lambda x: x.sample(frac=fraction, random_state=42))
    sampled_df = df.groupby('labels', group_keys=False).sample(frac=fraction, random_state=42)
    sampled_labels = sampled_df['labels']
    sampled_values = sampled_df.drop(columns=['labels'])
    return sampled_values, sampled_labels