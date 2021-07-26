# %%
from .base_metrics import _shannon, _richness, _simpson, _shannon_evenness, _hill_number, \
    _simpson_evenness, _gini_simpson, _renyi, _abundance

from ...utils.general import is_categorical, make_iterable, is_numeric

import numpy as np
import pandas as pd
from collections import Counter


# %%

def richness(so, spl: str, attr: str, *, local=True, key_added=None, graph_key='knn', inplace=True) -> None:
    """Computes the richness on the observation or the sample level

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Returns:

    """

    if key_added is None:
        key_added = 'richness'
        key_added = f'{key_added}_{attr}'
        if local:
            key_added += f'_{graph_key}'

    metric = _richness
    kwargs_metric = {}

    return _compute_metric(so=so, spl=spl, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def shannon(so, spl: str, attr: str, *, local=True, key_added=None, graph_key='knn', base=2, inplace=True) -> None:
    """Computes the Shannon Index on the observation or the sample level

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Returns:

    """
    if key_added is None:
        key_added = 'shannon'
        key_added = f'{key_added}_{attr}'
        if local:
            key_added += f'_{graph_key}'

    metric = _shannon
    kwargs_metric = {'base': base}

    return _compute_metric(so=so, spl=spl, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def simpson(so, spl: str, attr: str, *, local=True, key_added=None, graph_key='knn', inplace=True) -> None:
    """Computes the Simpson Index on the observation or the sample level

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Returns:

    """
    if key_added is None:
        key_added = 'simpson'
        key_added = f'{key_added}_{attr}'
        if local:
            key_added += f'_{graph_key}'

    metric = _simpson
    kwargs_metric = {}

    return _compute_metric(so=so, spl=spl, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def hill_number(so, spl: str, attr: str, q: float, *, local=True, key_added=None, graph_key='knn', inplace=True):
    """Computes the Hill Numbers on the observation or the sample level

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Returns:

    """
    if key_added is None:
        key_added = 'hill_number'
        key_added = f'{key_added}_{attr}_q{q}'
        if local:
            key_added += f'_{graph_key}'

    metric = _hill_number
    kwargs_metric = {'q': q}

    return _compute_metric(so=so, spl=spl, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def renyi_entropy(so, spl: str, attr: str, q: float, *, local=True, key_added=None, graph_key='knn', base=2,
                  inplace=True):
    """Computes the Renyi-Entropy

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Returns:

    """
    if key_added is None:
        key_added = 'renyi'
        key_added = f'{key_added}_{attr}_q{q}'
        if local:
            key_added += f'_{graph_key}'

    metric = _renyi
    kwargs_metric = {'q': q,
                     'base': base}

    return _compute_metric(so=so, spl=spl, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def abundance(so, spl: str, attr: str, *, mode='proportion', key_added: str = None, graph_key='knn',
              local=False, inplace: bool = True):
    """Computes the abundance of species on the observation or the sample level

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Returns:

    """

    if key_added is None:
        key_added = f'{mode}'
        if local:
            key_added += f'_{graph_key}'

    event_space = so.obs[spl][attr]
    if is_categorical(event_space):
        event_space = event_space.dtypes.categories
    else:
        raise TypeError(f'{attr} is not categorical')

    metric = _abundance
    kwargs_metric = {'event_space': event_space,
                     'mode': mode}

    return _compute_metric(so=so, spl=spl, attr=attr, key_added=key_added, metric=metric, graph_key=graph_key,
                           kwargs_metric=kwargs_metric, local=local, inplace=inplace)


def _compute_metric(so, spl: str, attr, key_added, graph_key, metric, kwargs_metric, local, inplace=True):
    """Computes the given metric for each observation or the sample
    """

    # generate a copy if necessary
    so = so if inplace else so.copy()

    # extract relevant categorisation
    data = so.obs[spl][attr]
    if not is_categorical(data):
        raise TypeError('`attr` needs to be categorical')

    if local:
        # get graph
        g = so.G[spl][graph_key]

        # compute metric for each observation
        res = []
        observation_ids = so.obs[spl].index
        for observation_id in observation_ids:
            n = list(g.neighbors(observation_id))
            if len(n) == 0:
                res.append(0)
                continue
            counts = Counter(data.loc[n].values)
            res.append(metric(counts, **kwargs_metric))

        if np.ndim(res[0]) > 0:
            res = pd.DataFrame(res, index=observation_ids)
            if spl not in so.obsm:
                so.obsm[spl] = {}
            so.obsm[spl][key_added] = res
        else:
            res = pd.DataFrame({key_added: res}, index=observation_ids)
            so.obs[spl] = pd.concat((so.obs[spl], res), axis=1)
    else:
        res = metric(Counter(data), **kwargs_metric)

        if np.ndim(res) > 0:
            if spl not in so.splm:
                so.splm[spl] = {}
            so.splm[spl][key_added] = res
        else:
            so.spl.loc[spl, key_added] = res

    if not inplace:
        return so
