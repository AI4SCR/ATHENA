from .so_attributer import soAttributer
from .deep_attributer import deepAttributer
from .random_attributer import randomAttributer

GRAPH_ATTRIBUTER = {
    'so': soAttributer,
    'deep': deepAttributer,
    'random': randomAttributer
}