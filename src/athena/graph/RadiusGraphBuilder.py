from sklearn.neighbors import radius_neighbors_graph
import networkx as nx
import pandas as pd
import numpy as np
from .BaseGraphBuilder import BaseGraphBuilder


class RadiusGraphBuilder(BaseGraphBuilder):
    """
    Radius graph class for graph building.
    """

    def build_graph(self, mask: np.ndarray, radius: float, include_self: bool = True, **kwargs):
        ndata = self.object_coordinates(mask)
        if len(ndata) == 0:
            return nx.Graph()

        adj = radius_neighbors_graph(ndata.to_numpy(), radius=radius, include_self=include_self, **kwargs)
        df = pd.DataFrame(adj.toarray(), index=ndata.index, columns=ndata.index)
        self.graph = nx.from_pandas_adjacency(
            df
        )  # this does not add the nodes in the same sequence as the index, column

        return self.graph
