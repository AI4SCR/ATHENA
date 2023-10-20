from athena.graph_builder.knn_graph_builder import KNNGraphBuilder
from athena.graph_builder.radius_graph_builder import RadiusGraphBuilder
from athena.graph_builder.contact_graph_builder import ContactGraphBuilder
from athena.utils.default_configs import get_default_config


def test_knn_graph_builder(so_object):
    config = get_default_config(builder_type="knn")
    config['builder_params']['n_neighbors'] = 5
    builder = KNNGraphBuilder(config)
    g = builder(so_object, spl='a')
    assert len(g.nodes) == 5
    assert len(g.edges) == 15


def test_radius_graph_builder(so_object):
    config = get_default_config(builder_type="radius")
    print(config)
    builder = RadiusGraphBuilder(config)
    g = builder(so_object, spl='a')
    assert len(g.nodes) == 5
    assert len(g.edges) == 8


def test_contact_graph_builder(so_object):
    config = get_default_config(builder_type="contact")
    config['builder_params']['radius'] = 15
    builder = ContactGraphBuilder(config)
    g = builder(so_object, spl='a')
    assert len(g.nodes) == 5
    assert len(g.edges) == 8
    assert (1, 2) in g.edges
