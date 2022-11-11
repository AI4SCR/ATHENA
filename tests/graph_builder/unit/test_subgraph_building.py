from athena.graph_builder.knn_graph_builder import KNNGraphBuilder
from athena.graph_builder.radius_graph_builder import RadiusGraphBuilder
from athena.graph_builder.contact_graph_builder import ContactGraphBuilder
from athena.graph_builder.constants import GRAPH_BUILDER_DEFAULT_PARAMS
import pandas as pd
import numpy as np
import pytest

def test_empty_types():
    # TODO: Implement this...
    # 1. Import data and change it
    # 2. Call function
    # 3. Assert 

def test_incongruent_labels_in_types():
    # TODO: Implement this...

def test_col_name_not_in_columns():
    # TODO: Implement this...

def test_col_name_or_types_not_specified():
    # TODO: Implement this...

def test_knn(ndata):
    config = GRAPH_BUILDER_DEFAULT_PARAMS['knn']
    config['builder_params']['n_neighbors'] = 5
    builder = KNNGraphBuilder(config)
    g = builder(ndata)
    assert len(g.nodes) == len(ndata)
    assert len(g.edges) == 15

def test_radius(ndata):
    config = GRAPH_BUILDER_DEFAULT_PARAMS['radius']
    builder = RadiusGraphBuilder(config)
    g = builder(ndata)
    assert len(g.nodes) == len(ndata)
    assert len(g.edges) == 3 + len(ndata)

def test_contact(masks):
    nodes = np.unique(masks)
    nodes = nodes[nodes != 0]
    ndata = pd.DataFrame(index=nodes)

    config = GRAPH_BUILDER_DEFAULT_PARAMS['contact']
    builder = ContactGraphBuilder(config)
    g = builder(ndata=ndata, topo_data=dict(mask=masks))
    assert len(g.nodes) == len(ndata)
    assert len(g.edges) == 1
    assert (1,2) in g.edges
