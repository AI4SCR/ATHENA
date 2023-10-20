from sklearn.neighbors import radius_neighbors_graph
import networkx as nx
import pandas as pd
import numpy as np
from ..utils.tools.graph import df2node_attr
from .base_graph_builder import BaseGraphBuilder
from ..utils.default_configs import EDGE_WEIGHT


class RadiusGraphBuilder(BaseGraphBuilder):
    '''
    Radius graph class for graph building.
    '''

    def __init__(self, config: dict):
        """Build topology using a radius algorithm based on the distance between the centroid of the nodes.

        Args:
            config: dict specifying graph builder params
            key_added: string to use as key for the graph in the spatial omics object
        """
        self.builder_type = 'radius'
        super().__init__(config)

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
        if self.config['build_concept_graph']:
            filter_col = self.config['concept_params']['filter_col']
            include_labels = self.config['concept_params']['include_labels']
            self.look_for_miss_specification_error(so, spl, filter_col, include_labels)

        mask_key = self.config['mask_key']
        coordinate_keys = self.config['coordinate_keys']

        # If a cell subset is well are specified then simplify the mask
        if mask_key is None:
            # If the subset is well specified (no error in `look_for_miss_specification_error`),
            # get rid of coordinates that are in the out-set.
            if self.config['build_concept_graph']:
                # Get coordinates
                ndata = so.obs[spl].query(f'{filter_col} in @include_labels')[[coordinate_keys[0], coordinate_keys[1]]]
            # Else get all coordinates.
            else:
                ndata = so.obs[spl][[coordinate_keys[0], coordinate_keys[1]]]

        # Else build graph from masks
        else:
            # Get masks
            mask = so.get_mask(spl, mask_key)

            # If include_labels are specified then simplify the mask
            if self.config['build_concept_graph']:
                # Get cell_ids of the cells that are in `include_labels`
                cell_ids = so.obs[spl].query(f'{filter_col} in @include_labels').index.values
                # Simplify masks filling it with 0s for cells that are not in `include_labels`
                mask = np.where(np.isin(mask, cell_ids), mask, 0)

            # Extract location:
            ndata = self.extract_location(mask)

        # compute adjacency matrix, put into df with cell_id for index and columns and ad to graph
        ndata.dropna(inplace=True)
        adj = radius_neighbors_graph(ndata.to_numpy(), **self.config['builder_params'])
        df = pd.DataFrame(adj.A, index=ndata.index, columns=ndata.index)
        self.graph = nx.from_pandas_adjacency(df)  # this does not add the nodes in the same sequence as the index, column

        # Puts node attribute (usually just coordinates) into dictionary
        # and then as node attributes in the graph. Set edge weight to 1.
        attrs = df2node_attr(ndata)
        nx.set_node_attributes(self.graph, attrs)
        nx.set_edge_attributes(self.graph, 1, EDGE_WEIGHT)

        return self.graph
