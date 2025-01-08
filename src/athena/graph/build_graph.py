import networkx as nx
import numpy as np
from networkx import to_scipy_sparse_array

from .ContactGraphBuilder import ContactGraphBuilder
from .KNNGraphBuilder import KNNGraphBuilder
from .RadiusGraphBuilder import RadiusGraphBuilder
from .DelaunayGraphBuilder import DelaunayGraphBuilder

GRAPH_BUILDERS = {
    "knn": KNNGraphBuilder,
    "contact": ContactGraphBuilder,
    "radius": RadiusGraphBuilder,
    "delaunay": DelaunayGraphBuilder,
}

from anndata import AnnData


def build_graph(ad: AnnData, topology: str, mask: np.ndarray = None, graph_key: str = None, copy: bool = False,
                **kwargs) -> nx.Graph:
    """Build graph from mask using specified topology."""

    ad = ad.copy() if copy else ad
    mask = ad.uns["mask"] if mask is None else mask

    if topology not in GRAPH_BUILDERS:
        raise ValueError(
            f"invalid graph topology {topology}. Available topologies are {GRAPH_BUILDERS.keys()}"
        )

    # Instantiate graph builder object
    builder = GRAPH_BUILDERS[topology]()

    # Build graph and get key
    g = builder.build_graph(mask, **kwargs)
    adj = to_scipy_sparse_array(g, nodelist=ad.obs.index.astype(int))

    if graph_key is None:
        graph_key = topology

    ad.obsp[graph_key] = adj
    if copy:
        return ad
