from sklearn.neighbors import kneighbors_graph
import networkx as nx
import numpy as np
import pandas as pd

from ..utils.tools.graph import df2node_attr
from .base_graph_builder import BaseGraphBuilder
from .constants import DILATION_KERNELS, EDGE_WEIGHT


# %%
class KNNGraphBuilder(BaseGraphBuilder):
    '''\
    KNN (K-Nearest Neighbors) class for graph building.
    '''

    def __init__(self, config: dict):
        '''\
        KNN-Graph Builder constructor

        Parameters
        ----------
        config: dict
            Dictionary containing `builder_params`.
            Refer to [1] for possible parameters

        Notes
        _____
        Example `builder_params` :
            config = {'builder_params': {'n_neighbors': 5, 'mode':'connectivity', 'metric':'minkowski', 'p':2, 'n_jobs':-1}, 'cellmask_file': 'path_to_file'}

        See Also
        ________
        .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
        '''
        super().__init__(config)

    def _build_topology(self, **kwargs):
        '''\
        Build topology using a kNN algorithm based on the distance between the centroid of the nodes.
        '''

        # compute adjacency matrix
        adj = kneighbors_graph(self.ndata, **self.config['builder_params'])
        df = pd.DataFrame(adj.A, index=self.ndata.index, columns=self.ndata.index)
        self.graph = nx.from_pandas_adjacency(df)  # this does not add the nodes in the same sequence as the index, column

        attrs = df2node_attr(self.ndata)
        nx.set_node_attributes(self.graph, attrs)
        nx.set_edge_attributes(self.graph, 1, EDGE_WEIGHT)
