import networkx.algorithms.community as nx_comm
from spatialOmics import SpatialOmics


def modularity(so: SpatialOmics, spl: str, community_id: str,
               graph_key: str = 'knn', resolution: float = 1,
               key_added=None, inplace=True) -> None:
    """Computes the modularity of the sample graph.

    Args:
        so: SpatialOmics instance
        spl: str Spl for which to compute the metric
        community_id: str column that specifies the community membership of each observation. Must be categorical.
        graph_key: str Specifies the graph representation to use in so.G[spl]
        resolution: float
        key_added: str Key added to spl
        inplace: bool Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Returns:
        SpatialOmics if inplace=True, else nothing.

    References: networkx_

    .. _networkx: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html
    """

    so = so if inplace else so.deepcopy()

    if key_added is None:
        key_added = f'modularity_{community_id}_{resolution}'

    if community_id not in so.obs[spl]:
        raise ValueError(f'{community_id} not in so.obs[spl]')
    elif so.obs[spl][community_id].dtype != 'category':
        raise TypeError(f'expected dtype `category` but got {so.obs[spl].meta_id.dtype}')

    # get communities
    communities = []
    for _, obs in so.obs[spl].groupby(community_id):
        communities.append(set(obs.index))

    res = nx_comm.modularity(so.G[spl][graph_key], communities)

    so.spl.loc[spl, key_added] = res

    if not inplace:
        return so
