from ..utils.default_configs import GRAPH_BUILDER_DEFAULT_PARAMS
from .mappings import GRAPH_BUILDERS
from ..attributer.node_features import add_node_features
import copy as cp

def build_graph(so, 
                spl: str, 
                builder_type: str='knn',  
                key_added: str=None, 
                config: dict=None, 
                inplace: bool=True):
    """Build graph representation for a sample. A graph is constructed based on the provided segmentation masks
    for the sample. For the `knn` and `radius` graph representation the centroid of each mask is used. For the `contact`
    graph representation the segmentation masks are dilation_ is performed. The segmentation masks that overlap after
    dilation are considered to be in physical contact and connected in the `contact` graph.

    Args:
        `so`: SpatialOmics object
        `spl`: sample name in so.spl.index
        `builder_type`: graph type to construct {knn, radius, contact}
        `config`: dict containing a dict 'builder_params' that specifies the graph construction parameters.
        Also includes other parameters. See other prameters in config section below. 
        `inplace`: whether to return a new SpatialOmics instance

    Other parameters in `config`:
        `mask_key`: key in so.masks[spl] to use as segmentation masks from which the observation
                coordinates are extracted, if `None` `coordinate_keys` from `obs` attribute are used
        `key_added`: key added in so.G[spl][key_add] to store the graph. If not specified it defaluts to `builder_type`.
                If the graph is being built on a subset of the nodes (e.g `filter_col` and `labels` are not None) 
                then the key is `f'{builder_type} > {filter_col} > {labels}'`
        `coordinate_keys`: column names of the x and y coordinates of a observation
        `filter_col`: string of the column in so.obs[spl][filter_col] which has the labels on which you want 
                to subset the cells.
        `labels`: list of stirngs which identify the labels in so.obs[spl][filter_col] that should be included in the grapgh. 
                If no list is provided the graph is built using all the cells/cell labels.
        `build_and_attribute: bool indicating whether to call the attributer functionality. 

    Returns:
        None or SpatialOmics if inplace = False

    .. _dilation: https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_dilation
    """

    # Raise error is the builder_type is invalid
    if builder_type not in GRAPH_BUILDERS:
        raise ValueError(f'invalid type {builder_type}. Available types are {GRAPH_BUILDERS.keys()}')

    # Get default bulding parameters if non are specified    
    if config is None:
        config = cp.deepcopy(GRAPH_BUILDER_DEFAULT_PARAMS[builder_type])

    # Instantiate graph builder object
    builder = GRAPH_BUILDERS[builder_type](config)

    # Build graph and get key
    g = builder(so, spl)

    # If no graph key is provided then use builder_type
    if key_added is None:
        key_added = builder_type

    # Copys `so` if inplace == Ture.
    so = so if inplace else so.copy()

    # If there already is a graph from spl then add or update the so object with graph at key_added
    if spl in so.G:
        so.G[spl].update({key_added: g})
    else:
        # otherwise initialize new dictionary object at key `spl`
        so.G[spl] = {key_added: g}

    # If in config build_and_attribute == True then attribute graph.
    if config['build_and_attribute']:
        features_type = config['use_attrs_from']

        # Check that the feature type is valid
        assert features_type in ['so', 'deep', 'random'], 'features_type (in the config) must be one of the following: "so", "deep", "random"'

        # Extract config and attribute graph
        attrs_config = config[features_type]
        add_node_features(so=so, spl=spl, graph_key=key_added, features_type=features_type, config=attrs_config)