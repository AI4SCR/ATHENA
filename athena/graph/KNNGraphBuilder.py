import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph

from .BaseGraphBuilder import BaseGraphBuilder


class KNNGraphBuilder(BaseGraphBuilder):
    """KNN (K-Nearest Neighbors) class for graph building."""

    def build_graph(self, mask: np.ndarray, n_neighbors: int, include_self: bool, **kwargs):
        ndata = self.object_coordinates(mask)
        if len(ndata) == 0:
            return nx.Graph()

        adj = kneighbors_graph(ndata.to_numpy(), n_neighbors=n_neighbors, include_self=include_self, **kwargs)
        df = pd.DataFrame(adj.toarray(), index=ndata.index, columns=ndata.index)
        self.graph = nx.from_pandas_adjacency(df)

        return self.graph
