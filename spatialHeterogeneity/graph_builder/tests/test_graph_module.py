from spatialHeterogeneity.graph_builder.knn_graph_builder import KNNGraphBuilder
from spatialHeterogeneity.graph_builder.radius_graph_builder import RadiusGraphBuilder
from spatialHeterogeneity.graph_builder.contact_graph_builder import ContactGraphBuilder
from spatialHeterogeneity.graph_builder.constants import GRAPH_BUILDER_DEFAULT_PARAMS
import pandas as pd
import numpy as np
import pytest

@pytest.fixture(scope='module')
def ndata():
    nodes = [1,2,3,4,5]
    coords = [(0,0), (10,10), (-20,20), (-30,-30), (40,-40)]
    return pd.DataFrame.from_records(coords, index=nodes)

@pytest.fixture(scope='module')
def masks():
    s = 5

    masks = np.zeros((50,50), 'int8')
    masks[23:23+s, 23:23+s] = 1
    masks[23-3-s:23-3, 23:23+s] = 2
    masks[23+s+4:23+4+2*s, 23:23+s] = 3
    masks[23:23+s, 23-s-5:23-5] = 4
    masks[23:23+s, 23+s+6:23+6+2*s] = 5
    return masks

def test_knn_graph_builder(ndata):
    config = GRAPH_BUILDER_DEFAULT_PARAMS['knn']
    config['n_neighbors'] = 5
    builder = KNNGraphBuilder(config)
    g = builder(ndata)
    assert len(g.nodes) == len(ndata)
    assert len(g.edges) == 21

def test_radius_graph_bulder(ndata):
    config = GRAPH_BUILDER_DEFAULT_PARAMS['radius']
    builder = RadiusGraphBuilder(config)
    g = builder(ndata)
    assert len(g.nodes) == len(ndata)
    assert len(g.edges) == 3 + len(ndata)


def test_contact_graph_builder(masks):
    nodes = np.unique(masks)
    nodes = nodes[nodes != 0]
    ndata = pd.DataFrame(index=nodes)

    config = GRAPH_BUILDER_DEFAULT_PARAMS['contact']
    builder = ContactGraphBuilder(config)
    g = builder(ndata=ndata, topo_data=dict(mask=masks))
    assert len(g.nodes) == len(ndata)
    assert len(g.edges) == 1
    assert (1,2) in g.edges