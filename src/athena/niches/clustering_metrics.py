#%%
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.metrics import silhouette_score

#%%
def filtered_inertia(filtered_merged: pd.DataFrame, filtered_labels: pd.Series, centers: np.ndarray):
    remaining_clusters = np.unique(filtered_labels)
    centers_map = {i: centers[i] for i in remaining_clusters}
    assigned_centers = np.array([centers_map[label] for label in filtered_labels])
    squared_distances = np.sum((filtered_merged.values - assigned_centers)**2, axis=1)
    inertia = np.sum(squared_distances)
    return inertia



def get_metrics(merged: pd.DataFrame,  centers: np.ndarray = None, inertia: float = None):
    assert (type(inertia) == float) or (type(centers) == np.ndarray), "Either inertia or centers must be provided to compute metrics."

    labels = merged['labels']
    if 'cluster_filter' in merged.columns:
        values_df = merged.copy()
        values_df = values_df.drop(columns=['cluster_filter', 'labels_raw','labels'])
    else:
        values_df = merged.copy()
        values_df = values_df.drop(columns=['labels'])

    tot_cells = len(values_df.index)
    if type(inertia) == float:
        silhouette = silhouette_score(values_df.values, labels,sample_size=tot_cells//3)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette}
        return metrics
    elif type(centers) == np.ndarray:
        filtered_labels = labels.dropna() # clusters labels that 'passed' the filtering step
        filtered_labels = filtered_labels.astype(int)
        filtered_merged = values_df.loc[filtered_labels.index]
        inertia = filtered_inertia(filtered_merged, filtered_labels, centers)
        silhouette = silhouette_score(filtered_merged.values, filtered_labels,sample_size=tot_cells//3)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette}
    
        return metrics