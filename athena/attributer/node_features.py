import copy as cp
from ..utils.default_configs import GRAPH_ATTRIBUTER_DEFAULT_PARAMS
from .mappings import GRAPH_ATTRIBUTER


def add_node_features(so,
                      spl: str,
                      graph_key: str,
                      features_type: str,
                      config: dict) -> None:
    """
    Maps parameters in `config` to downstream function which
    adds node features to graph `so.G[spl][graph_key]`.

    Args:
        - `so`: Spatial omics object which contains the graph to be
        attributed.
        - `spl`: String identifying the sample in `so`.
        - `graph_key`: String identifying the graph in `so` to attribute.
        - `features_type`: String specifying the type of parameters to assign.
        At the moment 3 possibilities.
            - 'so_feat'. Features from `so[spl].X` and/or `so[spl].obs`
            - 'deep_feat'. TODO: To be implemented.
            - 'random_feat' Random uniform [0, 1) features.
        - `config`: Parameters of the attribute method to be used downstream.
        If none are specified the defaults are used. Default values for each
        feature type can be seen in GRAPH_ATTRIBUTER_DEFAULT_PARAMS[features_type]
        dictionary in `athena.utils.default_configs`.

    Returns:
        - `None`: The changes are saved to the `so.G[spl][graph_key]`
    """

    # Check for Args miss specification.
    try:
        so.G[spl][graph_key]
    except KeyError:
        raise KeyError(f'Either spl: "{spl}" or graph_key: "{graph_key}" is invalid.')

    # If no config is specified, use default config.
    if config is None:
        config = cp.deepcopy(GRAPH_ATTRIBUTER_DEFAULT_PARAMS[features_type])

    # Instantiate attributer class and call it, thereby attributing `so.G[spl][graph_key]`.
    attributer = GRAPH_ATTRIBUTER[features_type](so, spl, graph_key, config)
    attributer()
