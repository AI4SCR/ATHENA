#%%
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.metrics import silhouette_score
from typing import Dict, Union, List

#%%
def filtered_inertia(filtered_n_rep: pd.DataFrame, filtered_labels: pd.Series, centers: np.ndarray):
    remaining_clusters = np.unique(filtered_labels)
    centers_map = {i: centers[i] for i in remaining_clusters}
    assigned_centers = np.array([centers_map[label] for label in filtered_labels])
    squared_distances = np.sum((filtered_n_rep.values - assigned_centers)**2, axis=1)
    inertia = np.sum(squared_distances)
    return inertia



def get_metrics(values_df: pd.DataFrame, labels: pd.Series, centers: Union[np.ndarray, None] = None, inertia: Union[float, None] = None):
    assert (type(inertia) == float) or (type(centers) == np.ndarray), "Either inertia or centers must be provided to compute metrics."


    tot_cells = len(values_df.index)

    if type(inertia) == float:
        silhouette = silhouette_score(values_df.values, labels,sample_size=tot_cells//3)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette}
        return metrics
    elif type(centers) == np.ndarray:
        filtered_labels = labels.dropna() # clusters labels that 'passed' the filtering step
        filtered_labels = filtered_labels.astype(int)
        filtered_n_rep = values_df.loc[filtered_labels.index]
        inertia = filtered_inertia(filtered_n_rep, filtered_labels, centers)
        silhouette = silhouette_score(filtered_n_rep.values, filtered_labels,sample_size=tot_cells//3)
        metrics = {'inertia': inertia, 'silhouette_score': silhouette}
    
        return metrics