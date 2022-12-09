from athena.graph_builder.knn_graph_builder import KNNGraphBuilder
from athena.graph_builder.radius_graph_builder import RadiusGraphBuilder
from athena.graph_builder.contact_graph_builder import ContactGraphBuilder
from athena.utils.default_configs import GRAPH_BUILDER_DEFAULT_PARAMS
import copy as cp
import pytest


def test_knn_graph_builder(so_object):
    config = cp.deepcopy(GRAPH_BUILDER_DEFAULT_PARAMS['knn'])
    config['builder_params']['n_neighbors'] = 5
    builder = KNNGraphBuilder(config)
    g = builder(so_object, spl='a')
    assert len(g.nodes) == 5
    assert len(g.edges) == 15

def test_radius_graph_bulder(so_object):
    config = cp.deepcopy(GRAPH_BUILDER_DEFAULT_PARAMS['radius'])
    print(config)
    builder = RadiusGraphBuilder(config)
    g = builder(so_object, spl='a')
    assert len(g.nodes) == 5
    assert len(g.edges) == 8

def test_contact_graph_builder(so_object):
    config = cp.deepcopy(GRAPH_BUILDER_DEFAULT_PARAMS['contact'])
    config['builder_params']['radius'] = 15
    builder = ContactGraphBuilder(config)
    g = builder(so_object, spl='a')
    assert len(g.nodes) == 5
    assert len(g.edges) == 8
    assert (1,2) in g.edges