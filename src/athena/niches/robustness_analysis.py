# %%
from random import seed
import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Dict, Union, List
from sklearn.metrics import adjusted_rand_score
#%%
def calculate_ari_vs_reference(df, reference_col_name):
    """
    Calculates the Adjusted Rand Index (ARI) for all clustering runs 
    in the DataFrame against a specified reference run.

    Args:
        df (pd.DataFrame): DataFrame where index is cell_ids and columns 
                           are clustering runs (cluster labels).
        reference_col_name (str): The name of the column to use as the 
                                  "ground truth" or reference clustering.

    Returns:
        pd.Series: A Series containing the ARI for each clustering run 
                   against the reference, excluding the reference itself.
    """
    aris = []
    reference_labels = df[reference_col_name]
    for seed in df.columns:
        if seed != reference_col_name:
            ari = adjusted_rand_score(reference_labels, df[seed])
            aris.append(ari)
    return aris


def compute_ARIs(labels_df: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]])-> float:
    '''Compute average ARI per clustering run (seed)

    Arguments
        labels_df = pd.DataFrame with 
            - cell_id and sample_id as row indexes
            - seeds as column names and  clustering run labels assignment as values in each column 
    
    Return
        dictionary with
        - reference seeds as keys
        - dictionary with  ARIs (values) for each other seed (keys)
    '''
    seeds_ARIs = {}

    # change all Nas to 'filtered_labels' to compare the filtered clusters in different runs
    labels_df_filtered = labels_df.astype(str).replace('<NA>', 'filtered_labels') 

    for seed in labels_df.columns:
        aris = calculate_ari_vs_reference(labels_df_filtered, reference_col_name=seed)
        seeds_ARIs[seed] = aris
        
    return seeds_ARIs


def assert_index_series(labels_dict: Dict[int, pd.Series]) -> bool:
    '''Check if all pd.Series in the labels_dict have the same index.

    Arguments
        labels_dict = dict with 
            - seeds as keys 
            - pd.Series with clustering run labels as values and cell_id and sample_id as row indexes
    
    Return
        bool: True if the indexes of the pd.Series are the same
    '''
    indices = [labels_dict[seed].index for seed in labels_dict.keys()]
    first_index = indices[0]
    for idx in indices[1:]:
        if not first_index.equals(idx):
            return False
    return True

def combine_seed_labels(multiple_seeds_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]]):
    labels = {}
    for seed in multiple_seeds_dict.keys():
        labels[seed] = multiple_seeds_dict[seed]['labels']
    
    assert assert_index_series(labels), 'Indices of all label series do not match.'
    
    labels_df = pd.DataFrame(labels)
    return labels_df


def cl_robustness(multiple_seeds_dict: Dict[int, Dict[str, Union[pd.Series, int, Dict[str, float]]]])-> dict:
    '''Select the best seed based on average ARI with other seeds.

    Args:
        multiple_seeds_dict: dict with seeds as keys and seed dictionaries as values
    
    Return
        seed dictionary of the best seed (with highest average ARI)
    '''
    # create df with all labels for each seed
    labels_df = combine_seed_labels(multiple_seeds_dict=multiple_seeds_dict)
    
    # compute ARIs
    aris = compute_ARIs(labels_df)

    # compute avg ari for each seed
    avg_aris = {}
    for seed in aris.keys():
        avg_aris[seed] = np.mean(aris[seed])
    
    # best seed = seed with max avg ari
    best_seed = max(avg_aris, key=avg_aris.get)   

    # add avg ari of the best seed to its dict metrics
    selected_seed_dict = multiple_seeds_dict[best_seed]
    selected_seed_dict['metrics'] = { 'avg_ari': avg_aris[best_seed] }

    return selected_seed_dict
