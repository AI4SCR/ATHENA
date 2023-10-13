import networkx as nx
import athena as ath
from athena.utils.default_configs import get_default_config


# This tests weather a sub set graph is sub graph isomorphic of the full graph.
def test_is_isomorphic(so_fixture):
    so, spl = so_fixture
    so.G.clear()

    # Build full graph with radius
    builder_type = 'radius'
    config = get_default_config(
        builder_type=builder_type
    )
    config['builder_params']['radius'] = 20  # set radius
    ath.graph.build_graph(so, spl, config=config)

    # Build concept graph with radius
    # Decide on subset
    labels = ['endothelial']
    filter_col = 'cell_type'

    # Build subset graphs
    # radius graph
    config = get_default_config(
        builder_type=builder_type,
        build_concept_graph=True
    )
    config['builder_params']['radius'] = 20  # set radius
    config['concept_params']['filter_col'] = filter_col
    config['concept_params']['labels'] = labels
    ath.graph.build_graph(so, spl, config=config, key_added='foo')

    A = so.G[spl][builder_type]
    B = so.G[spl]['foo']

    GM = nx.isomorphism.GraphMatcher(A, B)
    assert GM.subgraph_is_isomorphic()
