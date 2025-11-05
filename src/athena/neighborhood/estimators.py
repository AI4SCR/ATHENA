# %%
import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from .base_estimators import Interactions, _infiltration, RipleysK, Distance
from .utils import get_node_interactions
from ..utils.general import is_categorical


# %%
def distance(ad: AnnData, *, attr: str, linkage: str = 'min', top_k: int | None = None, ascending: bool = True,
             coordinate_keys: list = ['x', 'y'], key_added: str = None, inplace: bool = True) -> None | AnnData:

    ad = ad if inplace else ad.copy()

    if key_added is None:
        key_added = f'distance_{attr}_{linkage}_{top_k}_{ascending}'

    estimator = Distance(attr=attr, linkage=linkage, coordinate_keys = coordinate_keys,
                         top_k=top_k, ascending=ascending)
    res = estimator.predict(ad=ad)

    ad.uns[key_added] = res

    if not inplace:
        return ad


# %%
def interactions(ad: AnnData, *, attr: str,
                 mode: str = 'classic', prediction_type: str = 'observation',
                 n_permutations: int = 100,
                 random_seed=42, alpha: float = .01, key_added: str = None,
                 graph_key: str = 'knn',
                 inplace: bool = True) -> None | AnnData:
    """Compute interaction strength between species. This is done by counting the number of interactions (edges in the graph)
        between pair-wise observation types as encoded by `attr`. See notes for more information or the
        `methodology <https://ai4scr.github.io/ATHENA/source/methodology.html>`_ section in the docs.

    Args:
        ad: AnnData instance

        attr: Categorical feature in ad.obs to use for the grouping
        mode: One of {classic, histoCAT, proportion}, see notes
        n_permutations: Number of permutations to compute p-values and the interactions strength score (mode diff)
        random_seed: Random seed for permutations
        alpha: Threshold for significance
        prediction_type: One of {observation, pvalue, diff}, see Notes
        key_added: Key added to SpatialOmics.uns[spl][metric][key_added]
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Notes:
        `classic` and `histoCAT` are python implementations of the corresponding methods pubished by the Bodenmiller lab at UZH.
        The `proportion` method is similar to the `classic` method but normalises the score by the number of edges and is thus bound [0,1].

    Returns:

    """
    ad = ad if inplace else ad.copy()

    # NOTE: uns_path = f'{spl}/interactions/'
    if key_added is None:
        key_added = f'interaction_{attr}_{mode}_{prediction_type}_{graph_key}'

    if random_seed is None:
        random_seed = 42

    estimator = Interactions(ad=ad, attr=attr, mode=mode, n_permutations=n_permutations,
                             random_seed=random_seed, alpha=alpha, graph_key=graph_key)

    estimator.fit(prediction_type=prediction_type)
    res = estimator.predict()

    # add result to uns attribute
    ad.uns[key_added] = res
    # add2uns(ad, res, 'interactions', key_added)

    if not inplace:
        return ad


from ..utils.general import get_nx_graph_from_anndata


def infiltration(ad: AnnData, attr: str, *, interaction1=('tumor', 'immune'), interaction2=('immune', 'immune'),
                 add_key='infiltration', inplace=True, graph_key=None, local=False) -> None | AnnData:
    """Compute infiltration score. Generalises the infiltration score presented in
    `A Structured Tumor-Immune Microenvironment in Triple Negative Breast Cancer Revealed by Multiplexed Ion Beam Imaging <https://pubmed.ncbi.nlm.nih.gov/30193111/>`_
    The score comptes a ratio between the number of interactions observed between the observation types specified in `interactions1`
    and `interaction2` as :math:`\\frac{\\texttt{number of interactions 1}}{\\texttt{number of interactions 2}}`. This ratio can
    be undefined. See notes for more information.

    Args:
        ad: AnnData instance

        attr: Categorical feature in ad.obs to use for the grouping
        interaction1: labels in `attr` of enumerator interaction
        interaction2: labels in `attr` of denominator interaction
        key_added: Key added to SpatialOmics.uns[spl][metric][key_added]
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.

    Returns:

    Notes:
        The default arguments are replicating the `immune infiltration score <infiltrationScore_>`_. However, you
        can compute any kind of "infiltration" between observation types. The `attr` argument specifies the column
        in the `obs` dataframe which encodes different observation types. `interaction{1,2}` argument defines between
        which types the score should be computed.

    .. _infiltrationScore: https://pubmed.ncbi.nlm.nih.gov/30193111/

    """
    ad = ad if inplace else ad.copy()

    data = ad.obs[attr]
    if isinstance(data, pd.DataFrame):
        raise ValueError(f'multidimensional attr ({data.shape}) is not supported.')

    if not is_categorical(data):
        raise TypeError('`attr` needs to be categorical')

    missing = set(interaction1 + interaction2) - set(data.unique())
    if missing:
        raise ValueError(f'specified interaction categories are not all in `attr`. Missing {missing}')

    G = get_nx_graph_from_anndata(ad=ad, key=graph_key)
    if local:
        cont = []
        for node in tqdm(G.nodes):
            neigh = G[node]
            g = G.subgraph(neigh)
            nint = get_node_interactions(g, data)
            res = _infiltration(node_interactions=nint, interaction1=interaction1, interaction2=interaction2)
            cont.append(res)

        res = pd.DataFrame(cont, index=G.nodes, columns=[add_key])
        if add_key in ad.obs:
            ad.obs = ad.obs.drop(columns=[add_key])

        ad.obs = pd.concat((ad.obs, res), axis=1)

    else:
        nint = get_node_interactions(G, data)

        res = _infiltration(node_interactions=nint, interaction1=interaction1, interaction2=interaction2)

        ad.uns[add_key] = res

    if not inplace:
        return ad


def ripleysK(ad: AnnData, attr: str, id, *, mode='K', radii=None, correction='ripley', inplace=True, key_added=None):
    """Compute Ripley's K as implemented by `[1]`_.

    Args:
        ad: AnnData instance

        attr: Categorical feature in ad.obs to use for the grouping
        id: The category in the categorical feature `attr`, for which Ripley's K should be computed
        mode: {K, csr-deviation}. If `K`, Ripley's K is estimated, with `csr-deviation` the deviation from a poission process is computed.
        radii: List of radiis for which Ripley's K is computed
        correction: Correction method to use to correct for boarder effects, see [1].
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.
        key_added: Key added to SpatialOmics.uns[spl][metric][key_added]

    Returns:
        Ripley's K estimates

    References:
        .. _[1]: https://docs.astropy.org/en/stable/stats/ripley.html

    """
    ad = ad if inplace else ad.copy()

    # NOTE: uns_path = f'{spl}/clustering/'
    if key_added is None:
        key_added = f'ripleysK_{attr}_{id}_{mode}_{correction}'

    estimator = RipleysK(ad, id=id, attr=attr)
    res = estimator.predict(radii=radii, correction=correction, mode=mode)

    # add result to uns attribute
    # add2uns(ad, res, 'ripleysK', key_added)
    ad.uns[key_added] = res

    if not inplace:
        return ad


def add2uns(ad, res, parent_key, key_added):
    if parent_key in ad.uns:
        ad.uns[parent_key][key_added] = res
    else:
        ad.uns.update({parent_key: {key_added: res}})
