# %%
from .base_metrics import _shannon, _richness, _simpson, _shannon_evenness, _hill_number, \
    _simpson_evenness, _gini_simpson, _renyi, _abundance, _quadratic_entropy

from ...utils.general import is_categorical, make_iterable, is_numeric

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler


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

    Examples:

        .. code-block:: python

            so = sh.dataset.imc()
            spl = so.spl.index[0]

            sh.metrics.richness(so, spl, 'meta_id', local=False)
            sh.metrics.richness(so, spl, 'meta_id', local=True)
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

    Examples:

        .. code-block:: python

            so = sh.dataset.imc()
            spl = so.spl.index[0]

            sh.metrics.shannon(so, spl, 'meta_id', local=False)
            sh.metrics.shannon(so, spl, 'meta_id', local=True)

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

    Examples:

        .. code-block:: python

            so = sh.dataset.imc()
            spl = so.spl.index[0]

            sh.metrics.simpson(so, spl, 'meta_id', local=False)
            sh.metrics.simpson(so, spl, 'meta_id', local=True)

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
        q: The hill coefficient as defined here_.
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Examples:

        .. code-block:: python

            so = sh.dataset.imc()
            spl = so.spl.index[0]

            sh.metrics.hill_number(so, spl, 'meta_id', q=2, local=False)
            sh.metrics.hill_number(so, spl, 'meta_id', q=2, local=True)

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
    """Computes the Renyi-Entropy.

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        q: The renyi coefficient as defined here_
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Examples:

        .. code-block:: python

            so = sh.dataset.imc()
            spl = so.spl.index[0]

            sh.metrics.renyi_entropy(so, spl, 'meta_id', q=2, local=False)
            sh.metrics.renyi_entropy(so, spl, 'meta_id', q=2, local=True)

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

    return _compute_metric(so=so, spl=spl, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def quadratic_entropy(so, spl: str, attr: str, *, metric='minkowski', metric_kwargs={}, scale: bool = True,
                      local=True, key_added=None, graph_key='knn', inplace=True):
    """Computes the quadratic entropy, taking relative abundance and similarity between observations into account.

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        metric: metric used to compute distance of observations in the features space so.X[spl]
        metric_kwargs: key word arguments for metric
        scale: whether to scale features of observations to unit variance and 0 mean
        local: whether to compute the metric on the observation or the sample level
        key_added: Key added to either obs or spl depending on the choice of `local`
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Notes:
        The implementation computes an average feature vector for each group in attr based on all observations in the
        sample. Thus, if staining biases across samples exists this will directly distort this metric.

    Examples:

        .. code-block:: python

            so = sh.dataset.imc()
            spl = so.spl.index[0]

            sh.metrics.quadratic_entropy(so, spl, 'meta_id', local=False)
            sh.metrics.quadratic_entropy(so, spl, 'meta_id', local=True)

    """
    if key_added is None:
        key_added = 'quadratic'
        key_added = f'{key_added}_{attr}'
        if local:
            key_added += f'_{graph_key}'

    # collect feature vectors of all observations and add attr grouping
    features: pd.DataFrame = so.X[spl]
    features = features.merge(so.obs[spl][attr], right_index=True, left_index=True)
    assert len(features) == len(so.X[spl]), 'inner merge resulted in dropped index ids'

    # compute average feature vector for each attr group and standardise
    features = features.groupby(attr).mean()
    if scale:
        tmp = StandardScaler().fit_transform(features)
        features = pd.DataFrame(tmp, index=features.index, columns=features.columns)

    base_metric = _quadratic_entropy
    kwargs_metric = {'features': features,
                     'metric': metric,
                     'metric_kwargs': metric_kwargs,
                     'scale': False}  # we scaled already

    return _compute_metric(so=so, spl=spl, attr=attr, key_added=key_added, graph_key=graph_key, metric=base_metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)


def abundance(so, spl: str, attr: str, *, mode='proportion', key_added: str = None, graph_key='knn',
              local=False, inplace: bool = True):
    """Computes the abundance of species on the observation or the sample level.

    Args:
        so: SpatialOmics instance
        spl: Spl for which to compute the metric
        attr: Categorical feature in SpatialOmics.obs to use for the grouping
        local: Whether to compute the metric on the observation or the sample level
        key_added: Key added to either uns[spl] or obs depending on the choice of `local`
        graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Examples:

        .. code-block:: python

            so = sh.dataset.imc()
            spl = so.spl.index[0]

            sh.metrics.abundance(so, spl, 'meta_id', local=False)
            sh.metrics.abundance(so, spl, 'meta_id', local=True)

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
            if key_added in so.obs[spl]:  # drop previous computation of metric
                so.obs[spl].drop(key_added, 1, inplace=True)
            so.obs[spl] = pd.concat((so.obs[spl], res), axis=1)
    else:
        res = metric(Counter(data), **kwargs_metric)

        if np.ndim(res) > 0:
            if spl not in so.uns:
                so.uns[spl] = {}
            so.uns[spl][key_added] = res
        else:
            so.spl.loc[spl, key_added] = res

    if not inplace:
        return so
