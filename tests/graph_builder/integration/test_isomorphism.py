import networkx as nx
import athena as ath
import copy as cp
from athena.graph_builder.constants import GRAPH_BUILDER_DEFAULT_PARAMS
import pytest

# This tests wheather a sub set graph is sub graph isomorphic of the full graph. 
def test_is_isomorphic():
    # Loead data
    so = ath.dataset.imc()

    # Define sample
    spl = 'slide_49_By2x5'

    # Extrac centroids
    ath.pp.extract_centroids(so, spl, mask_key='cellmasks')

    # Build full graph with radius
    builder_type = 'radius'
    config = cp.deepcopy(GRAPH_BUILDER_DEFAULT_PARAMS[builder_type])
    config['builder_params']['radius'] = 20 # set radius
    ath.graph.build_graph(so, spl, builder_type=builder_type, config=config)

    # Buil concept graph with radius
    # Decide on subset
    labels = ['endothelial']
    filter_col = 'cell_type'

    # Build subset graphs
    # radius graph
    config = cp.deepcopy(GRAPH_BUILDER_DEFAULT_PARAMS[builder_type])
    config['builder_params']['radius'] = 20 # set radius
    config['concept_params']['filter_col'] = filter_col
    config['concept_params']['labels'] = labels
    ath.graph.build_graph(so, spl, builder_type=builder_type, config=config)

    A = so.G[spl][builder_type]  
    B = so.G[spl][f'{builder_type} > {filter_col} > {labels}'] 

    GM = nx.isomorphism.GraphMatcher(A, B)
    assert GM.subgraph_is_isomorphic()