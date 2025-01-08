########### File to implement n_hop extensions to exisiting "base" graphs

from scipy.sparse import csr_matrix
from anndata import AnnData
import numpy as np

def n_hop_graph(adj_matrix: csr_matrix, n: int) -> csr_matrix:
    """
    Compute the n-hop graph adjacency matrix.
    
    Parameters:
    adj_matrix (csr_matrix): A sparse adjacency matrix representing the graph.
    n (int): The number of hops (n) to consider for neighbors.
    
    Returns:
    csr_matrix: A sparse adjacency matrix of the n-hop graph.
    """
    A_n_hop = adj_matrix.copy()  # Copy the original adjacency matrix
    
    for _ in range(n - 1):
        A_n_hop = A_n_hop.dot(adj_matrix)  # Matrix multiplication
    ### reset values back to 1 (binarize)

    A_n_hop.data[:] = 1 
    return A_n_hop

def build_n_hop_graph(ad: AnnData, base_graph_key: str, n_hops:int, graph_key: str = None, copy: bool = False):

    """
    Build an n-hop graph from a given adjacency matrix stored in an AnnData object.

    Parameters:
    ad (AnnData): The AnnData object containing the graph in `obsp`.
    base_graph_key (str): Key in `ad.obsp` where the base graph is stored.
    n_hops (int): The number of hops for the graph.
    graph_key (str): Key to store the n-hop graph in `ad.obsp`. Defaults to None.
    copy (bool): Whether to return a copy of the AnnData object.

    Returns:
    AnnData (optional): A copy of the updated AnnData object if `copy` is True.
    """

    graph = ad.obsp[base_graph_key]

    ## ensure that it is sparse matrix, if not convert
    if not isinstance(graph, csr_matrix):
        graph = csr_matrix(graph)

    adj = n_hop_graph(graph, n=n_hops)
    if graph_key is None:
        graph_key = f"{base_graph_key}_{n_hops}_hop"

    ad.obsp[graph_key] = adj
    if copy:
        return ad

     