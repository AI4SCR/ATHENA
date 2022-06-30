from sklearn.neighbors import radius_neighbors_graph
import networkx as nx
import pandas as pd

from ..utils.tools.graph import df2node_attr
from .base_graph_builder import BaseGraphBuilder
from .constants import EDGE_WEIGHT


# %%
class RadiusGraphBuilder(BaseGraphBuilder):
    '''\
    Radius graph class for graph building.
    '''

    def __init__(self, config: dict):
        """Build topology using a radius algorithm based on the distance between the centroid of the nodes.

        Args:
            config: dict specifying graph builder params

        Examples:
            config = {'builder_params': {'radius': 36, 'mode':'connectivity', 'metric':'minkowski', 'p':2, 'n_jobs':-1}}
        """
        super().__init__(config)

    def _build_topology(self, **kwargs):
        '''\
        Build topology using a radius algorithm based on the distance between the centroid of the nodes.
        '''

        # compute adjacency matrix
        adj = radius_neighbors_graph(self.ndata.to_numpy(), **self.config['builder_params'])
        df = pd.DataFrame(adj.A, index=self.ndata.index, columns=self.ndata.index)
        self.graph = nx.from_pandas_adjacency(df)  # this does not add the nodes in the same sequence as the index, column

        attrs = df2node_attr(self.ndata)
        nx.set_node_attributes(self.graph, attrs)
        nx.set_edge_attributes(self.graph, 1, EDGE_WEIGHT)
