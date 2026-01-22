
from matplotlib import pyplot as plt
from typing import Dict, Union, List
import pandas as pd
#%%
def get_color_map(labels: List[str]) -> Dict[str, str]:
    cmap = plt.get_cmap('tab20') 

    # 3. Build the dict using a loop (dictionary comprehension)
    color_map = {name: cmap(i) for i, name in enumerate(labels)}
    return color_map

def data_for_circos_plot(color_df: pd.DataFrame, width_df: pd.DataFrame):
    '''
    Create color and width dictionaries for circos plot links and pd.DataFrame for sectors.
    Args:
        color_df (pd.DataFrame): DataFrame with color values for interactions.
        width_df (pd.DataFrame): DataFrame with width values for interactions.
    Returns:
        Dict[str, Dict[Tuple[str, str], float]] and Dict[str, pd.DataFrame]: Dictionary with 'color_dict' and 'width_dict' and 'sectors_df'.
    '''
    #WIDTH
    width_df2 = width_df.reset_index()
    width_df2.columns = ['from', 'to', 'Value']
    width_df2 = width_df2.dropna(subset=['Value'])
    width_dict = {(row['from'], row['to']): row['Value'] for _, row in width_df2.iterrows()}
    #width_dict = {k: v for k, v in width_dict.items()}

    # COLOR
    color_df = color_df.reset_index()
    color_df.columns = ['from', 'to', 'Value']
    color_df = color_df.dropna(subset=['Value'])
    color_dict = {(row['from'], row['to']): row['Value'] for _, row in color_df.iterrows()}

    #SECTORS 
    sectors_df = width_df.copy()
    #sectors_df.columns = ['width']
    sectors_df = width_df.reset_index().pivot(index='attr_1', columns='attr_2', values='score')
    sectors_df = sectors_df.fillna(0)

    return {'color_dict': color_dict, 'width_dict': width_dict, 'sectors_df': sectors_df}