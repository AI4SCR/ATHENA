from .constants import GRAPH_BUILDER_DEFAULT_PARAMS
from .mappings import GRAPH_BUILDERS


def build_graph(so, spl: str, builder_type='knn', mask_key='cellmasks', key_added=None, config=None, inplace=True,
                coordinate_keys=('x', 'y')):
    """Build graph representation for a sample. A graph is constructed based on the provided segmentation masks
    for the sample. For the `knn` and `radius` graph representation the centroid of each mask is used. For the `contact`
    graph representation the segmentation masks are dilation_ is performed. The segmentation masks that overlap after
    dilation are considered to be in physical contact and connected in the `contact` graph.

    Args:
        so: SpatialOmics object
        spl: sample name in so.spl.index
        builder_type: graph type to construct {knn, radius, contact}
        mask_key: key in so.masks[spl] to use as segmentation masks from which the observation coordinates are extracted, if
            `None` `coordinate_keys` from `obs` attribute are used
        key_added: key added in so.G[spl][key_add] to store the graph.
        config: dict containing a dict 'builder_params' that specifies the graph construction parameters
        inplace: whether to return a new SpatialOmics instance
        coordinate_keys: column names of the x and y coordinates of a observation

    Returns:
        None or SpatialOmics if inplace = False

    .. _dilation: https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_dilation
    """

    # Raise error is the builder_type is invalid
    if builder_type not in GRAPH_BUILDERS:
        raise ValueError(f'invalid type {builder_type}. Available types are {GRAPH_BUILDERS.keys()}')

    # Get default bulding parameters if non are specified    
    if config is None:
        config = GRAPH_BUILDER_DEFAULT_PARAMS[builder_type].copy()

    # If no title for the graph is provided use builder_type
    # TODO: make default name also include the cell types involved    
    if key_added is None:
        key_added = builder_type

    # If no masks are provided check that builder_type != 'contact'. Then build
    if mask_key is None:
        # Raise error if no masks were provided and the builder_type is contact
        if builder_type == 'contact':
            raise ValueError(
                'Contact-graph requires segmentations masks. To compute a contact graph please specify `the mask_key` to use in so.masks[spl]')
        
        # Get coordinates for every cell in the sample and build graph
        ndat = so.obs[spl][[coordinate_keys[0], coordinate_keys[1]]]
        builder = GRAPH_BUILDERS[builder_type](config)
        g = builder(ndata=ndat)
    else:
        # Get masks and build graph
        mask = so.get_mask(spl, mask_key)
        g = GRAPH_BUILDERS[builder_type].from_mask(config, mask)

    # If include self is specified and true and the graph built is a contact graph, add self edges in graph. 
    if 'include_self' in config['builder_params'] and config['builder_params'][
        'include_self'] and builder_type == 'contact':
        edge_list = [(i, i) for i in g.nodes]
        g.add_edges_from(edge_list)

    # I would just rewrite: if not inplace: so.copy()
    # TODO: imporve readability here. 
    so = so if inplace else so.copy()

    # If there already is a graph from spl then add or update the so object with graph at key_added
    # TODO: here maybe the if/else is not needed. 
    if spl in so.G:
        so.G[spl].update({key_added: g})
    else:
        so.G[spl] = {key_added: g}
