from .contact_graph_builder import ContactGraphBuilder
from .knn_graph_builder import KNNGraphBuilder
from .radius_graph_builder import RadiusGraphBuilder

GRAPH_BUILDERS = {
    'knn': KNNGraphBuilder,
    'contact': ContactGraphBuilder,
    'radius': RadiusGraphBuilder
}
