# %%
from collections import Counter

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.preprocessing import StandardScaler

from .base_metrics import _shannon, _richness, _simpson, _hill_number, \
    _renyi, _abundance, _quadratic_entropy
from ...utils.general import is_categorical
from ...utils.general import get_nx_graph_from_anndata

# %%

def richness(ad: AnnData, attr: str, *, local=True, key_added=None, graph_key='knn', inplace=True) -> None:
    """Computes the richness on the observation or the sample level

    Args:
        ad: AnnData instance
        attr: Categorical feature in ad.obs to use for the grouping
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Examples:

        .. code-block:: python

            sh.metrics.richness(ad, attr='meta_id', local=False)
            sh.metrics.richness(ad, attr='meta_id', local=True)
    """

    if key_added is None:
        key_added = 'richness'
        key_added = f'{key_added}_{attr}'
        if local:
            key_added += f'_{graph_key}'

    metric = _richness
    kwargs_metric = {}

    return _compute_metric(ad=ad, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def shannon(ad: AnnData, attr: str, *, local=True, key_added=None, graph_key='knn', base=2, inplace=True) -> None:
    """Computes the Shannon Index on the observation or the sample level

    Args:
        ad: AnnData instance
        attr: Categorical feature in ad.obs to use for the grouping
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Examples:

        .. code-block:: python

            sh.metrics.shannon(ad=ad, attr='meta_id', local=False)
            sh.metrics.shannon(ad=ad, attr='meta_id', local=True)

    """
    if key_added is None:
        key_added = 'shannon'
        key_added = f'{key_added}_{attr}'
        if local:
            key_added += f'_{graph_key}'

    metric = _shannon
    kwargs_metric = {'base': base}

    return _compute_metric(ad=ad, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def simpson(ad: AnnData, attr: str, *, local=True, key_added=None, graph_key='knn', inplace=True) -> None:
    """Computes the Simpson Index on the observation or the sample level

    Args:
        ad: AnnData instance
        attr: Categorical feature in ad.obs to use for the grouping
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Examples:

        .. code-block:: python

            sh.metrics.simpson(ad=ad, attr='meta_id', local=False)
            sh.metrics.simpson(ad=ad, attr='meta_id', local=True)

    """
    if key_added is None:
        key_added = 'simpson'
        key_added = f'{key_added}_{attr}'
        if local:
            key_added += f'_{graph_key}'

    metric = _simpson
    kwargs_metric = {}

    return _compute_metric(ad=ad, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def hill_number(ad: AnnData, attr: str, q: float, *, local=True, key_added=None, graph_key='knn', inplace=True):
    """Computes the Hill Numbers on the observation or the sample level

    Args:
        ad: AnnData instance

        attr: Categorical feature in ad.obs to use for the grouping
        q: The hill coefficient as defined here_.
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Examples:

        .. code-block:: python

            sh.metrics.hill_number(ad=ad, attr='meta_id', q=2, local=False)
            sh.metrics.hill_number(ad=ad, attr='meta_id', q=2, local=True)

    """
    if key_added is None:
        key_added = 'hill_number'
        key_added = f'{key_added}_{attr}_q{q}'
        if local:
            key_added += f'_{graph_key}'

    metric = _hill_number
    kwargs_metric = {'q': q}

    return _compute_metric(ad=ad, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def renyi_entropy(ad: AnnData, attr: str, q: float, *, local=True, key_added=None, graph_key='knn', base=2,
                  inplace=True):
    """Computes the Renyi-Entropy.

    Args:
        ad: AnnData instance

        attr: Categorical feature in ad.obs to use for the grouping
        q: The renyi coefficient as defined here_
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Examples:

        .. code-block:: python

            sh.metrics.renyi_entropy(ad=ad, attr='meta_id', q=2, local=False)
            sh.metrics.renyi_entropy(ad=ad, attr='meta_id', q=2, local=True)

    .. _here: https://ai4scr.github.io/ATHENA/source/methodology.html
    """
    if key_added is None:
        key_added = 'renyi'
        key_added = f'{key_added}_{attr}_q{q}'
        if local:
            key_added += f'_{graph_key}'

    metric = _renyi
    kwargs_metric = {'q': q,
                     'base': base}

    return _compute_metric(ad=ad, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def quadratic_entropy(ad: AnnData, attr: str, *, metric='minkowski', metric_kwargs={}, scale: bool = True,
                      local=True, key_added=None, graph_key='knn', inplace=True):
    """Computes the quadratic entropy, taking relative abundance and similarity between observations into account.

    Args:
        ad: AnnData instance

        attr: Categorical feature in ad.obs to use for the grouping
        metric: metric used to compute distance of observations in the features space ad.X
        metric_kwargs: key word arguments for metric
        scale: whether to scale features of observations to unit variance and 0 mean
        local: whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Notes:
        The implementation computes an average feature vector for each group in attr based on all observations in the
        sample. Thus, if staining biases across samples exists this will directly distort this metric.

    Examples:

        .. code-block:: python

            sh.metrics.quadratic_entropy(ad=ad, attr='meta_id', local=False)
            sh.metrics.quadratic_entropy(ad=ad, attr='meta_id', local=True)

    """
    if key_added is None:
        key_added = 'quadratic'
        key_added = f'{key_added}_{attr}'
        if local:
            key_added += f'_{graph_key}'

    # collect feature vectors of all observations and add attr grouping
    features: pd.DataFrame = pd.DataFrame(ad.X, index=ad.obs.index, columns=ad.var.index)
    features = pd.concat((features, ad.obs[[attr]]), axis=1)
    assert len(features) == len(ad.X), 'inner merge resulted in dropped index ids'

    # compute average feature vector for each attr group and standardise
    features = features.groupby(attr, observed=False).mean()
    if scale:
        tmp = StandardScaler().fit_transform(features)
        features = pd.DataFrame(tmp, index=features.index, columns=features.columns)

    base_metric = _quadratic_entropy
    kwargs_metric = {'features': features,
                     'metric': metric,
                     'metric_kwargs': metric_kwargs,
                     'scale': False}  # we scaled already

    return _compute_metric(ad=ad, attr=attr, key_added=key_added, graph_key=graph_key, metric=base_metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def abundance(ad: AnnData, attr: str, *, mode='proportion', key_added: str = None, graph_key='knn',
              local=False, inplace: bool = True):
    """Computes the abundance of species on the observation or the sample level.

    Args:
        ad: AnnData instance

        attr: Categorical feature in ad.obs to use for the grouping
        mode: Either `proportion` or `counts`. If `proportion` we compute the frequency of the species, else the absolute counts.
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either uns[spl] or obs depending on the choice of `local`
        graph_key: Specifies the graph representation to use in ad.obsp if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Examples:

        .. code-block:: python

            sh.metrics.abundance(ad, attr='meta_id', local=False)
            sh.metrics.abundance(ad, attr='meta_id', local=True)

    """

    if key_added is None:
        key_added = f'abundance_{attr}_{mode}'
        if local:
            key_added += f'_{graph_key}'

    event_space = ad.obs[attr]
    if is_categorical(event_space):
        event_space = event_space.dtypes.categories
    else:
        raise TypeError(f'{attr} is not categorical')

    metric = _abundance
    kwargs_metric = {'event_space': event_space,
                     'mode': mode}

    return _compute_metric(ad=ad, attr=attr, key_added=key_added, metric=metric, graph_key=graph_key,
                           kwargs_metric=kwargs_metric, local=local, inplace=inplace)


def _compute_metric(ad, attr, key_added, graph_key, metric, kwargs_metric, local, inplace=True):
    """Computes the given metric for each observation or the sample
    """

    # generate a copy if necessary
    ad = ad if inplace else ad.copy()

    # extract relevant categorisation
    data = ad.obs[attr]
    if not is_categorical(data):
        raise TypeError('`attr` needs to be categorical')

    if local:
        # get graph
        g = get_nx_graph_from_anndata(ad, graph_key)

        # compute metric for each observation
        res = []
        # TODO: handle the `str` index of AnnData
        observation_ids = ad.obs.index
        for observation_id in observation_ids:
            n = list(g.neighbors(observation_id))
            if len(n) == 0:
                res.append(0)
                continue
            counts = Counter(data.loc[n].values)
            res.append(metric(counts, **kwargs_metric))

        if np.ndim(res[0]) > 0:
            res = pd.DataFrame(res, index=observation_ids)
            ad.obsm[key_added] = res
        else:
            res = pd.DataFrame({key_added: res}, index=observation_ids)
            if key_added in ad.obs:  # drop previous computation of metric
                ad.obs.drop(key_added, axis=1, inplace=True)
            ad.obs = pd.concat((ad.obs, res), axis=1)
    else:
        res = metric(Counter(data), **kwargs_metric)
        ad.uns[key_added] = res

    if not inplace:
        return ad
