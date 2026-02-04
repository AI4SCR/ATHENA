#%%
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.metrics import silhouette_score
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



def get_metrics(values_df: pd.DataFrame, labels: pd.Series, centers: Union[np.ndarray, None] = None, inertia: Union[float, None] = None):
    '''Compute metrics (inertia and silhouette score)

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
        samples_values, sampled_labels = stratified_sampling(values_df, labels)
        silhouette = silhouette_score(samples_values.values, sampled_labels)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette}
        return metrics
    
    # else, recompute inertia and silhouette score
    elif type(centers) == np.ndarray:
        filtered_labels = labels.dropna() # clusters labels that 'passed' the filtering step
        filtered_labels = filtered_labels.astype(int)
        filtered_n_rep = values_df.loc[filtered_labels.index]
        inertia = compute_inertia(filtered_n_rep, filtered_labels, centers)
        samples_values, sampled_labels = stratified_sampling(filtered_n_rep, filtered_labels)
        silhouette = silhouette_score(samples_values.values, sampled_labels)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette}
    
        return metrics