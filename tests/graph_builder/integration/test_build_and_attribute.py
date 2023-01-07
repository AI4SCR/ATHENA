import athena as ath
import copy as cp
from athena.utils.default_configs import get_default_config
import json

def test_build_and_attribute(so_fixture):
    # Unpack data
    so, spl = so_fixture

    # Extrac centroids
    ath.pp.extract_centroids(so, spl, mask_key='cellmasks')

    # Build full graph with radius
    builder_type = 'radius'
    config = get_default_config(
        builder_type='radius',
        build_and_attribute=True,
        build_concept_graph=False,
        attrs_type='random'
    )
    print(json.dumps(config, indent=3))
    
    ath.graph.build_graph(so, spl, builder_type=builder_type, config=config, key_added='foo')

    assert len(so.G[spl]['foo'].nodes[1]) == config['attrs_params']['n_attrs']