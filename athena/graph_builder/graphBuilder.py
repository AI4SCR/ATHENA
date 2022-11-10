from .constants import GRAPH_BUILDER_DEFAULT_PARAMS
from .mappings import GRAPH_BUILDERS
import numpy as np

def build_graph(so, 
                spl: str, 
                builder_type: str='knn', 
                mask_key: str='cellmasks', 
                key_added: str=None, 
                config: dict=None, 
                inplace: bool=True,
                coordinate_keys=('x', 'y'),
                col_name: str=None, 
                types: list=None):
    """Build graph representation for a sample. A graph is constructed based on the provided segmentation masks
    for the sample. For the `knn` and `radius` graph representation the centroid of each mask is used. For the `contact`
    graph representation the segmentation masks are dilation_ is performed. The segmentation masks that overlap after
    dilation are considered to be in physical contact and connected in the `contact` graph.

    Args:
        so: SpatialOmics object
        spl: sample name in so.spl.index
        builder_type: graph type to construct {knn, radius, contact}
        mask_key: key in so.masks[spl] to use as segmentation masks from which the observation
                coordinates are extracted, if `None` `coordinate_keys` from `obs` attribute are used
        key_added: key added in so.G[spl][key_add] to store the graph. If not specified it defaluts to `builder_type`.
                If the graph is being built on a subset of the nodes (e.g `col_name` and `types` are not None) 
                then the key is `f'{builder_type} > {col_name} > {types}'`
        config: dict containing a dict 'builder_params' that specifies the graph construction parameters
        inplace: whether to return a new SpatialOmics instance
        coordinate_keys: column names of the x and y coordinates of a observation
        col_name: string of the column in so.obs[spl][col_name] which has the labels on which you want 
                to subset the cells.
        types: list of stirngs which identify the labels in so.obs[spl][col_name] that should be included in the grapgh. 
                If no list is provided the graph is built using all the cells/cell types. 

    Returns:
        None or SpatialOmics if inplace = False

    .. _dilation: https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_dilation
    """

    # Raise error is the builder_type is invalid
    if builder_type not in GRAPH_BUILDERS:
        raise ValueError(f'invalid type {builder_type}. Available types are {GRAPH_BUILDERS.keys()}')

    # Raise error if either `col_name` `types` is specified but not the other.
    if (col_name is None) ^ (types is None):
        raise NameError(f'failed to specify either `col_name` or `types`')

    # Get default bulding parameters if non are specified    
    if config is None:
        config = GRAPH_BUILDER_DEFAULT_PARAMS[builder_type].copy()

    # If no title for the graph is provided use builder_type
    # TODO: make default name also include the cell types involved    
    if key_added is None:
        if types is not None:
            key_added = f'{builder_type} > {col_name} > {types}'
        else:
            key_added = builder_type

    # If no masks are provided build graph with centroids.
    if mask_key is None:
        # Raise error if no masks were provided and the builder_type is contact
        if builder_type == 'contact':
            raise ValueError(
                'Contact-graph requires segmentations masks. To compute a contact graph please specify `the mask_key` to use in so.masks[spl]')
        
        # If types is specified, get rid of coordinates that are in the out-set. 
        if types is not None:
            # Raise error if `col_name` is not found in 
            if col_name not in so.obs[spl].columns:
                raise NameError(f'{col_name} is not in so.obs[spl].columns')
            
            # Raise error if `types` is an empty list
            if types == []:
                raise NameError(f'types varaibel is empty. You need to give a non-empty list')

            # Raise error if not all `types` have a match in `so.obs[spl][col_name].cat.categories.values`
            if not np.all(np.isin(types, so.obs[spl][col_name].cat.categories.values)):
                raise NameError(f'Not all elements provided in variable types are in so.obs[spl][col_name]')

            # Finally, get coordinates
            ndat = so.obs[spl].query(f'{col_name} in @types')[coordinate_keys[0], coordinate_keys[1]]
        # Else get all coordinates.
        else:
            ndat = so.obs[spl][[coordinate_keys[0], coordinate_keys[1]]]

        # Build graph
        builder = GRAPH_BUILDERS[builder_type](config)
        g = builder(ndata=ndat)
    else:
        # Get masks
        mask = so.get_mask(spl, mask_key)

        # If types are specified then simplify the mask
        if types is not None:
            # Get cell_ids of the cells that are in `types`
            cell_ids = so.obs[spl].query(f'{col_name} in @types').index.values
            # Simplify masks filling it with 0s for cellsa that are not in `types`
            mask = np.where(np.isin(mask, cell_ids), mask, 0)

        # Build graph
        g = GRAPH_BUILDERS[builder_type].from_mask(config, mask)

    # If include self is specified and true and the graph built is a contact graph, add self edges in graph. 
    if 'include_self' in config['builder_params'] and config['builder_params'][
        'include_self'] and builder_type == 'contact':
        edge_list = [(i, i) for i in g.nodes]
        g.add_edges_from(edge_list)

    # Copys `so` if inplace == Ture.
    so = so if inplace else so.copy()

    # If there already is a graph from spl then add or update the so object with graph at key_added
    if spl in so.G:
        so.G[spl].update({key_added: g})
    else:
        # otherwise initialize new dictionary object at key `spl`
        so.G[spl] = {key_added: g}
