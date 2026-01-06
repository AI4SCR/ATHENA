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



def get_metrics(merged: pd.DataFrame,  labels: pd.Series, centers: np.ndarray = None, inertia: float = None):
    assert (type(inertia) == float) or (type(centers) == np.ndarray), "Either inertia or centers must be provided to compute metrics."
    tot_cells = len(merged.index)
    if type(inertia) == float:
        silhouette = silhouette_score(merged.values, labels,sample_size=tot_cells//3)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette}
        return metrics
    elif type(centers) == np.ndarray:
        filtered_labels = labels.dropna()
        filtered_labels = filtered_labels.astype(int)
        filtered_merged = merged.loc[filtered_labels.index]
        inertia = filtered_inertia(filtered_merged, filtered_labels, centers)
        silhouette = silhouette_score(filtered_merged.values, filtered_labels,sample_size=tot_cells//3)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette}
    
        return metrics