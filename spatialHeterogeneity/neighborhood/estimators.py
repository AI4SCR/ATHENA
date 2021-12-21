# %%
import pandas as pd
import numpy as np

from .base_estimators import Interactions, _infiltration, RipleysK

from .utils import get_node_interactions
from ..utils.general import is_categorical
from tqdm import tqdm
# %%
def interactions(so, spl: str, attr: str, mode: str ='classic', prediction_type: str ='observation', *, n_permutations: int =100,
                 random_seed=None, alpha: float =.01, try_load: bool =True, key_added: str =None, graph_key: str ='knn',
                 inplace: bool =True) -> None:
    """Compute interaction strength between species.

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        mode: One of {classic, histoCAT, proportion}, see notes
        n_permutations: Number of permutations to compute p-values and the interactions strength score (mode diff)
        random_seed: Random seed for permutations
        alpha: Threshold for significance
        prediction_type: One of {observation, pvalue, diff}, see Notes
        try_load: load pre-computed permutation results if available
        key_added: Key added to SpatialOmics.uns[spl][metric][key_added]
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Returns:

    """
    so = so if inplace else so.copy()

    # NOTE: uns_path = f'{spl}/interactions/'
    if key_added is None:
        key_added = f'{attr}_{mode}_{prediction_type}_{graph_key}'

    if random_seed is None:
        random_seed = so.random_seed

    estimator = Interactions(so=so, spl=spl, attr=attr, mode=mode, n_permutations=n_permutations,
                             random_seed=random_seed, alpha=alpha, graph_key=graph_key)

    estimator.fit(prediction_type=prediction_type, try_load=try_load)
    res = estimator.predict()

    # add result to uns attribute
    add2uns(so, res, spl, 'interactions', key_added)

    if not inplace:
        return so


def infiltration(so, spl: str, attr: str, *, interaction1=('tumor', 'immune'), interaction2=('immune', 'immune'),
                 add_key='infiltration', inplace=True, graph_key='knn', local=False) -> None:
    """Compute infiltration score.

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        interaction1: labels of enumerator interaction
        interaction2: labels of denominator interaction
        key_added: Key added to SpatialOmics.uns[spl][metric][key_added]
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.

    Returns:

    """
    so = so if inplace else so.copy()

    data = so.obs[spl][attr]
    if isinstance(data, pd.DataFrame):
        raise ValueError(f'multidimensional attr ({data.shape}) is not supported.')

    if not is_categorical(data):
        raise TypeError('`attr` needs to be categorical')

    if not np.in1d(np.array(interaction1 + interaction2), data.unique()).all():
        mask = np.in1d(np.array(interaction1 + interaction2), data.unique())
        missing = np.array(interaction1 + interaction2)[~mask]
        raise ValueError(f'specified interaction categories are not all in `attr`. Missing {missing}')

    G = so.G[spl][graph_key]
    if local:
        cont = []
        for node in tqdm(G.nodes):
            neigh = G[node]
            g = G.subgraph(neigh)
            nint = get_node_interactions(g, data)
            res = _infiltration(node_interactions=nint, interaction1=interaction1, interaction2=interaction2)
            cont.append(res)

        res = pd.DataFrame(cont, index=G.nodes, columns=[add_key])
        if add_key in so.obs[spl]:
            so.obs[spl] = so.obs[spl].drop(columns=[add_key])

        so.obs[spl] = pd.concat((so.obs[spl], res), 1)

    else:
        nint = get_node_interactions(G, data)

        res = _infiltration(node_interactions=nint, interaction1=interaction1, interaction2=interaction2)

        so.spl.loc[spl, add_key] = res

    if not inplace:
        return so


def ripleysK(so, spl: str, attr: str, id, *, mode='K', radii=None, correction='ripley', inplace=True, key_added=None):
    """Compute Ripley's K

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        id: The category in the categorical feature `attr`, for which Ripley's K should be computed
        mode: {K, csr-deviation}. If `K`, Ripley's K is estimated, with `csr-deviation` the deviation from a poission process is computed.
        radii: List of radiis for which Ripley's K is computed
        correction: Correction method to use to correct for boarder effects, see [1].
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.
        key_added: Key added to SpatialOmics.uns[spl][metric][key_added]

    Returns:
        Ripley's K estimates

    Notes:
        .. [1] https://docs.astropy.org/en/stable/stats/ripley.html

    """
    so = so if inplace else so.copy()

    # NOTE: uns_path = f'{spl}/clustering/'
    if key_added is None:
        key_added = f'{id}_{attr}_{mode}_{correction}'

    estimator = RipleysK(so=so, spl=spl, id=id, attr=attr)
    res = estimator.predict(radii=radii, correction=correction, mode=mode)

    # add result to uns attribute
    add2uns(so, res, spl, 'ripleysK', key_added)

    if not inplace:
        return so


def add2uns(so, res, spl: str, parent_key, key_added):
    if spl in so.uns:
        if parent_key in so.uns[spl]:
            so.uns[spl][parent_key][key_added] = res
        else:
            so.uns[spl].update({parent_key: {key_added: res}})
    else:
        so.uns.update({spl: {parent_key: {key_added: res}}})
