from sklearn.neighbors import radius_neighbors_graph
import networkx as nx
import pandas as pd
import numpy as np

from ..utils.tools.graph import df2node_attr
from .base_graph_builder import BaseGraphBuilder
from .constants import EDGE_WEIGHT


# %%
class RadiusGraphBuilder(BaseGraphBuilder):
    '''
    Radius graph class for graph building.
    '''

    def __init__(self, config: dict, key_added: str):
        """Build topology using a radius algorithm based on the distance between the centroid of the nodes.

        Args:
            config: dict specifying graph builder params
            key_added: string to use as key for the graph in the spatial omics object

        Default config:
            config = {'builder_params': 
                    {'radius': 36, 
                    'mode':'connectivity', 
                    'metric':'minkowski', 
                    'p':2, 
                    'metric_params':None, 
                    'include_self':False, 
                    'n_jobs':-1},
                'concept_params': 
                    {'filter_col':None,
                    'labels':None},
                'coordinate_keys': ['x', 'y'],
                'mask_key': 'cellmasks'}
        """
        self.builder_type = 'radius'
        super().__init__(config, key_added)

    def __call__(self, so, spl):
        '''
        Build topology using a radius algorithm based on the distance between the centroid of the nodes.
        
        Args:
            so: spatial omic object
            spl: sting identifying the sample in the spatial omics object

        Returns:
            Graph and key to graph in the spatial omics object

        '''

        # Unpack parameters for building
        filter_col = self.config['concept_params']['filter_col']
        labels = self.config['concept_params']['labels']
        mask_key = self.config['mask_key']
        coordinate_keys = self.config['coordinate_keys']
        
        if labels is not None or filter_col is not None:
            self.look_for_miss_specification_error(so, spl, filter_col, labels)
            subset_specified = True
        else:
            subset_specified = False

        # Set key. Depends on the config file. Method defined in the superclass.
        self.add_key(filter_col, labels)

        # If a cell subset is well are specified then simplify the mask
        if mask_key is None:
            # If labels is specified, get rid of coordinates that are in the out-set. # Else get all coordinates.
            if subset_specified:
                # Get coordinates
                ndata = so.obs[spl].query(f'{filter_col} in @labels')[coordinate_keys[0], coordinate_keys[1]]
            else:
                ndata = so.obs[spl][[coordinate_keys[0], coordinate_keys[1]]]

        # Else build graph from masks
        else:
            # Get masks
            mask = so.get_mask(spl, mask_key)

            # If labels are specified then simplify the mask
            if subset_specified:
                # Get cell_ids of the cells that are in `labels`
                cell_ids = so.obs[spl].query(f'{filter_col} in @labels').index.values
                # Simplify masks filling it with 0s for cells that are not in `labels`
                mask = np.where(np.isin(mask, cell_ids), mask, 0)
            
            # Extract location:
            ndata = self.extract_location(mask)

        # compute adjacency matrix, put into df with cell_id for index and columns and ad to graph
        adj = radius_neighbors_graph(ndata.to_numpy(), **self.config['builder_params'])
        df = pd.DataFrame(adj.A, index=ndata.index, columns=ndata.index)
        self.graph = nx.from_pandas_adjacency(df)  # this does not add the nodes in the same sequence as the index, column

        # Puts node attribute (usually just coordinates) into dictionary 
        # and then as node attributes in the graph. Set edge wheight to 1. 
        attrs = df2node_attr(ndata)
        nx.set_node_attributes(self.graph, attrs)
        nx.set_edge_attributes(self.graph, 1, EDGE_WEIGHT)

        return (self.graph, self.key_added)