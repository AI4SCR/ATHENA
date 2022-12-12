import athena as ath
import copy as cp
from athena.utils.default_configs import GRAPH_BUILDER_DEFAULT_PARAMS

def test_build_and_attribute(so_fixture):
    # Unpack data
    so, spl = so_fixture

    # Extrac centroids
    ath.pp.extract_centroids(so, spl, mask_key='cellmasks')

    # Build full graph with radius
    builder_type = 'radius'
    config = cp.deepcopy(GRAPH_BUILDER_DEFAULT_PARAMS[builder_type])
    config['builder_params']['radius'] = 20 # set radius
    config['build_and_attribute'] = True
    config['use_attrs_from'] = 'random'
    ath.graph.build_graph(so, spl, builder_type=builder_type, config=config, key_added='foo')

    assert len(so.G[spl]['foo'].nodes[1]) == config['random']['n_attrs']