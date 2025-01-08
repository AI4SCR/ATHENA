import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import scipy.spatial
from scipy.sparse import csr_matrix

from .BaseGraphBuilder import BaseGraphBuilder


class DelaunayGraphBuilder(BaseGraphBuilder):
    """Delaunay Graph class for graph building."""

    def build_graph(self, mask: np.ndarray, include_self: bool = True, **kwargs):
        ndata = self.object_coordinates(mask)
        if len(ndata) == 0:
            return nx.Graph()

        ## use scipy implementation of delaunay graph
        


        delaunay = scipy.spatial.Delaunay(ndata, **kwargs)
        edges = set()
        ## Extract edges from returned simplices and store in matrix
        for simplex in delaunay.simplices:
            # Simplex is a triangle (3 points), so extract the edges
            edges.update([(simplex[i], simplex[j]) for i in range(3) for j in range(i + 1, 3)])

        ## Create a sparse adjacency matrix from the edges
        rows, cols = zip(*edges)
        edge_values = np.ones(len(edges), dtype=int)
        adj_sp = csr_matrix((edge_values, (rows, cols)), shape=(ndata.shape[0], ndata.shape[0]))

        ## Create full adjacency matrix and fill self.node connection
        adj =  adj_sp.toarray()
        if include_self:
            np.fill_diagonal(adj,1)

        df = pd.DataFrame(adj, index=ndata.index, columns=ndata.index)
        self.graph = nx.from_pandas_adjacency(df)

        return self.graph
