import abc
import networkx as nx
import numpy as np
import pandas as pd
from abc import ABC
from skimage.measure import regionprops_table


class BaseGraphBuilder(ABC):

    def __init__(self, config: dict):
        """Base-Graph Builder constructor

        Args:
            config: Dictionary containing a dict called `builder_params` that provides function call arguments to the build_topology function
            key_added: The key associated with the graph in the so object
        """

        self.config = config
        self.graph = nx.Graph()

    @abc.abstractmethod
    def __call__(self, so, spl):
        """Builds graph topology. Implemented in subclasses.

        Args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError('Implemented in subclasses.')

    def extract_location(self, mask):
        '''Compute centroid from image of labels (mask) and return them as a pandas-compatible table.
        '''
        # The table is a dictionary mapping column names to value arrays.
        ndata = regionprops_table(mask, properties=['label', 'centroid'])

        # The cell_id is the number that identifies a cell in the mask and is set to be the index of ndata
        ndata = pd.DataFrame.from_dict(ndata)
        ndata.columns = ['cell_id', 'y', 'x']  # NOTE: axis 0 is y and axis 1 is x
        ndata.set_index('cell_id', inplace=True)
        ndata.sort_index(axis=0, ascending=True, inplace=True)

        return ndata

    def look_for_miss_specification_error(self, so, spl, filter_col, include_labels):
        ''' Looks for a miss specification error in the config
        '''

        # Raise error if either `filter_col` or `include_labels` is specified but not the other.
        if (filter_col is None) ^ (include_labels is None):
            raise NameError('failed to specify either `filter_col` or `include_labels`')

        # Raise error if `filter_col` is not found in 
        if filter_col not in so.obs[spl].columns:
            raise NameError(f'{filter_col} is not in so.obs[spl].columns')

        # Raise error if `include_labels` is an empty list
        if include_labels == []:
            raise NameError('include_labels variable is empty. You need to give a non-empty list')

        # Raise error if not all `include_labels` have a match in `so.obs[spl][filter_col].cat.categories.values`
        if not np.all(np.isin(include_labels, so.obs[spl][filter_col].values)):
            raise NameError(f'Not all labels are present in column "{filter_col}". Labels: {include_labels}. In {filter_col}: {so.obs[spl][filter_col].values}')
