from sklearn.neighbors import kneighbors_graph
import networkx as nx
import pandas as pd
import numpy as np
from ..utils.tools.graph import df2node_attr
from .base_graph_builder import BaseGraphBuilder
from ..utils.default_configs import EDGE_WEIGHT


class KNNGraphBuilder(BaseGraphBuilder):
    '''KNN (K-Nearest Neighbors) class for graph building.
    '''

    def __init__(self, config: dict):
        """KNN-Graph Builder constructor

        Args:
            config: Dictionary containing `builder_params`. Refer to [1] for possible parameters

        Notes:
            [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html

        """
        self.builder_type = 'knn'
        super().__init__(config)

    def __call__(self, so, spl):
        '''Build topology using a kNN algorithm based on the distance between the centroid of the nodes.

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

        # If no masks are provided build graph with centroids.
        if mask_key is None:
            # If labels is specified, get rid of coordinates that are in the out-set. Else get all coordinates.
            if self.config['build_concept_graph']:
                # Get coordinates
                ndata = so.obs[spl].query(f'{filter_col} in @include_labels')[[coordinate_keys[0], coordinate_keys[1]]]
            else:
                ndata = so.obs[spl][[coordinate_keys[0], coordinate_keys[1]]]

        # Else build graph from masks
        else:
            # Get masks
            mask = so.get_mask(spl, mask_key)

            # If a cell subset is well are specified then simplify the mask
            if self.config['build_concept_graph']:
                # Get cell_ids of the cells that are in `include_labels`
                cell_ids = so.obs[spl].query(f'{filter_col} in @include_labels').index.values
                # Simplify masks filling it with 0s for cells that are not in `include_labels`
                mask = np.where(np.isin(mask, cell_ids), mask, 0)

            # Extract location:
            ndata = self.extract_location(mask)

        # compute adjacency matrix
        ndata.dropna(inplace=True)
        adj = kneighbors_graph(ndata.to_numpy(), **self.config['builder_params'])
        df = pd.DataFrame(adj.A, index=ndata.index, columns=ndata.index)
        self.graph = nx.from_pandas_adjacency(df)  # this does not add the nodes in the same sequence as the index, column

        # Puts node attribute (usually just coordinates) into dictionary
        # and then as node attributes in the graph. Set edge weight to 1.
        attrs = df2node_attr(ndata)
        nx.set_node_attributes(self.graph, attrs)
        nx.set_edge_attributes(self.graph, 1, EDGE_WEIGHT)

        return self.graph
