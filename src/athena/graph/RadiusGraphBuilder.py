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
        adj = radius_neighbors_graph(ndata.to_numpy(), radius=radius, include_self=include_self, **kwargs)
        return adj, ndata.index.tolist()
