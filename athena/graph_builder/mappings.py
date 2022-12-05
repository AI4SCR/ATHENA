from .contact_graph_builder import ContactGraphBuilder
from .knn_graph_builder import KNNGraphBuilder
from .radius_graph_builder import RadiusGraphBuilder
from .so_attributer import soAttributer
from .deep_attributer import deepAttributer
from .random_attributer import randomAttributer

GRAPH_BUILDERS = {
    'knn': KNNGraphBuilder,
    'contact': ContactGraphBuilder,
    'radius': RadiusGraphBuilder
}

GRAPH_ATTRIBUTER = {
    'so_feat': soAttributer,
    'deep_feat': deepAttributer,
    'random_feat': randomAttributer
}