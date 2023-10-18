from .so_attributer import SoAttributer
from .deep_attributer import DeepAttributer
from .random_attributer import RandomAttributer

GRAPH_ATTRIBUTER = {
    'so': SoAttributer,
    'deep': DeepAttributer,
    'random': RandomAttributer
}