from .constants import GRAPH_BUILDER_DEFAULT_PARAMS
from .mappings import GRAPH_BUILDERS
import networkx as nx
from pandas import DataFrame

def build_graph(so, spl: str, builder_type = 'knn', mask_key = 'cellmasks', key_added=None, config = None, inplace=True):
    """Build graph representation for a sample

    Args:
        so:
        spl:
        builder_type: graph type to construct {knn, radius, contact}
        mask_key: key in so.masks[spl] to use as segmentation masks
        key_added:
        config: dict containing a dict 'builder_params' that specifies the graph construction parameters
        inplace: whether to return a new SpatialOmics instance

    Returns:
        None or SpatialOmics if inplace = False
    """
    if builder_type not in GRAPH_BUILDERS:
        raise ValueError(f'invalid type {builder_type}. Available types are {GRAPH_BUILDERS.keys()}')
    if config is None:
        config = GRAPH_BUILDER_DEFAULT_PARAMS[builder_type].copy()
    if key_added is None:
        key_added = builder_type

    if mask_key is None:
        ndat = so.obs[spl][['x','y']]
        builder = GRAPH_BUILDERS[builder_type](config)
        g = builder(ndata=ndat)
    else:
        mask = so.get_mask(spl,mask_key)
        g = GRAPH_BUILDERS[builder_type].from_mask(config, mask)

    if 'include_self' in config['builder_params'] and config['builder_params']['include_self'] and builder_type == 'contact':
        edge_list = [(i,i) for i in g.nodes]
        g.add_edges_from(edge_list)

    so = so if inplace else so.copy()
    if spl in so.G:
        so.G[spl].update({key_added: g})
    else:
        so.G[spl] = {key_added:g}