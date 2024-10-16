from skimage.morphology import square, diamond, disk
import copy as cp

DILATION_KERNELS = {
    'disk': disk,
    'square': square,
    'diamond': diamond
}

EDGE_WEIGHT = 'weight'

CONCEPT_DEFAULT_PARAMS = {
    'concept_params': {
        'filter_col' : None,
        'include_labels' : None
    },
}

GRAPH_ATTRIBUTER_DEFAULT_PARAMS = {
    'so': {
        'attrs_type': 'so',
        'attrs_params': {
            'from_obs': True,
            'obs_cols': [
                'meta_id',
                'cell_type_id',
                'phenograph_cluster',
                'y',
                'x'
            ],
            'from_X': True,
            'X_cols': 'all'
        }
    },
    'deep': {
        'attrs_type': 'deep',
        'attrs_params': {}
    },
    'random' : {
        'attrs_type': 'random',
        'attrs_params': {'n_attrs': 3}
    }
}

OTHER_PARAMETERS = {
    'coordinate_keys': ['x', 'y'],
    'mask_key': None,
    'build_and_attribute': False,
    'build_concept_graph': False
}

GRAPH_BUILDER_DEFAULT_PARAMS = {
    'knn': {
        'builder_type': 'knn',
        'builder_params': {
            'n_neighbors': 6,
            'mode': 'connectivity',
            'metric': 'minkowski',
            'p': 2,
            'metric_params': None,
            'include_self': True,
            'n_jobs': -1
        }
    },
    'contact': {
        'builder_type': 'contact',
        'builder_params': {
            'dilation_kernel': 'disk',
            'radius': 4,
            'include_self': True
        }
    },
    'radius': {
        'builder_type': 'radius',
        'builder_params': {
            'radius': 36,
            'mode': 'connectivity',
            'metric': 'minkowski',
            'p': 2,
            'metric_params': None,
            'include_self': True,
            'n_jobs': -1
        }
    }
}


def get_default_config(builder_type: str,
                       build_concept_graph: bool = False,
                       build_and_attribute: bool = False,
                       attrs_type : str = None) -> dict:
    """
    Gets a default configuration of parameters for graph building depending on the desired graph construction.

    Args:
        builder_type: string indicating the type of graph to build, namely 'knn', 'contact', or 'radius'.
        build_concept_graph: indicates whether to build a concept graph (True) or a graph using all the cells (False)
        build_and_attribute: whether to assign attributes to the nodes of the graph.
        attrs_type: string indicating which type of attributes to assign.

    Returns:
        A dictionary with the default configuration.
    """

    config = cp.deepcopy(GRAPH_BUILDER_DEFAULT_PARAMS[builder_type])
    other_params = cp.deepcopy(OTHER_PARAMETERS)
    config = {**config, **other_params}

    if builder_type == "contact":
        config["mask_key"] = "cellmasks"
        config["coordinate_keys"] = None

    if build_concept_graph:
        concept_config = cp.deepcopy(CONCEPT_DEFAULT_PARAMS)
        config = {**config, **concept_config}
        config['build_concept_graph'] = True

    if build_and_attribute:
        assert attrs_type is not None, 'If `build_and_attribute = True`, `attrs_type` must be specified.'
        attrs_config = cp.deepcopy(GRAPH_ATTRIBUTER_DEFAULT_PARAMS[attrs_type])
        config = {**config, **attrs_config}
        config['build_and_attribute'] = True

    return config
