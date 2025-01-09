import networkx as nx
import numpy as np
from networkx import to_scipy_sparse_array

from .ContactGraphBuilder import ContactGraphBuilder
from .KNNGraphBuilder import KNNGraphBuilder
from .RadiusGraphBuilder import RadiusGraphBuilder

GRAPH_BUILDERS = {
    "knn": KNNGraphBuilder,
    "contact": ContactGraphBuilder,
    "radius": RadiusGraphBuilder,
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
    adj, object_ids = builder.build_graph(mask, **kwargs)

    # note: we permute the order of the adj matrix to match the order of the object_ids in anndata
    objs_of_ad = ad.obs.index.astype(int)
    assert set(objs_of_ad) == set(object_ids), "Object IDs do not match, ensure that the ad.obs.index (`str`) is the same as the object_ids in the segmentation mask (`int`)"
    reorder = [object_ids.index(obj) for obj in objs_of_ad]
    adj = adj[reorder][:, reorder]

    if graph_key is None:
        graph_key = topology

    ad.obsp[graph_key] = adj
    if copy:
        return ad
