import networkx.algorithms.community as nx_comm
from anndata import AnnData

from ...utils.general import get_nx_graph_from_anndata


def modularity(ad: AnnData, community_id: str,
               graph_key: str = 'knn', resolution: float = 1,
               key_added=None, inplace=True) -> None:
    """Computes the modularity of the sample graph.

    Args:
        ad: AnnData
        community_id: str column that specifies the community membership of each observation. Must be categorical.
        graph_key: str Specifies the graph representation to use in so.G[spl]
        resolution: float
        key_added: str Key added to spl
        inplace: bool Whether to add the metric to the current AnnData instance or to return a new one.

    Returns:
        AnnData if inplace=True, else nothing.

    References: networkx_

    .. _networkx: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html
    """

    ad = ad if inplace else ad.deepcopy()

    if key_added is None:
        key_added = f'modularity_{community_id}_res{resolution}'

    if community_id not in ad.obs:
        raise ValueError(f'{community_id} not in ad.obs')
    elif ad.obs[community_id].dtype != 'category':
        raise TypeError(f'expected dtype `category` but got {ad.obs[community_id].dtype}')

    # get communities
    communities = []
    for _, obs in ad.obs.groupby(community_id):
        communities.append(set(obs.index))

    g = get_nx_graph_from_anndata(ad=ad, key=graph_key)
    res = nx_comm.modularity(g, communities)

    ad.uns[key_added] = res

    if not inplace:
        return ad
