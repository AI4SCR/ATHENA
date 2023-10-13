import athena as ath
from athena.utils.default_configs import get_default_config
import json


def test_build_and_attribute(so_fixture):
    # Unpack data
    so, spl = so_fixture

    # Build full graph with radius
    config = get_default_config(
        builder_type='radius',
        build_and_attribute=True,
        build_concept_graph=False,
        attrs_type='random'
    )
    print(json.dumps(config, indent=3))

    ath.graph.build_graph(so, spl, config=config, key_added='foo')

    assert len(so.G[spl]['foo'].nodes[1]) == config['attrs_params']['n_attrs']
