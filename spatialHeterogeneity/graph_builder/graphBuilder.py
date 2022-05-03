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
    if builder_type not in GRAPH_BUILDERS:
        raise ValueError(f'invalid type {builder_type}. Available types are {GRAPH_BUILDERS.keys()}')
    if config is None:
        config = GRAPH_BUILDER_DEFAULT_PARAMS[builder_type].copy()
    if key_added is None:
        key_added = builder_type

    if mask_key is None:
        if builder_type == 'contact':
            raise ValueError(
                'Contact-graph requires segmentations masks. To compute a contact graph please specify `the mask_key` to use in so.masks[spl]')
        ndat = so.obs[spl][[coordinate_keys[0], coordinate_keys[1]]]
        builder = GRAPH_BUILDERS[builder_type](config)
        g = builder(ndata=ndat)
    else:
        mask = so.get_mask(spl, mask_key)
        g = GRAPH_BUILDERS[builder_type].from_mask(config, mask)

    if 'include_self' in config['builder_params'] and config['builder_params'][
        'include_self'] and builder_type == 'contact':
        edge_list = [(i, i) for i in g.nodes]
        g.add_edges_from(edge_list)

    so = so if inplace else so.copy()
    if spl in so.G:
        so.G[spl].update({key_added: g})
    else:
        so.G[spl] = {key_added: g}
