# %%
from .base_metrics import _shannon, _richness, _simpson, _shannon_evenness, _hill_number, \
    _simpson_evenness, _gini_simpson, _renyi, _quadratic_entropy, _abundance, _diveristy_profile

from ...utils.general import is_categorical, make_iterable, is_numeric

import numpy as np
import pandas as pd
from collections import Counter

# configure logging
import logging

logger = logging.getLogger(__name__)


# %%

def richness(so, spl: str, attr: str, *, local=True, key_added=None, graph_key='knn', inplace=True):
    """Computes the richness for each cell or the spl

    Parameters
    ----------
    so
    spl
    attr: str
        attribute for which to compute the richness
    local: bool
        if richness is computed for each cell or on the spl level
    key_added:
        key where result is stored
    observation_ids:
        observation_ids for which to compute the richness

    Notes
    -----
    If local is true the result is stored in obs, else in spl.

    Returns
    -------

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


def shannon(so, spl: str, attr: str, *, local=True, key_added=None, graph_key='knn', base=2, inplace=True):
    """Computes the shannon entropy for each cell or the spl

    Parameters
    ----------
    so
    spl
    attr: str
        attribute for which to compute the entropy
    local: bool
        if entropies are computed for each cell or on the spl level
    key_added:
        key where result is stored
    base:
        basis of logarithm in entropy

    Notes
    -----
    If local is true the result is stored in obs, else in cores.

    Returns
    -------

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


def simpson(so, spl: str, attr: str, *, local=True, key_added=None, graph_key='knn', inplace=True):
    """Computes the simpson index for each cell or the spl

    Parameters
    ----------
    so
    spl
    attr: str
        attribute for which to compute the simpson index
    local: bool
        if simpson index is computed for each cell or on the spl level
    key_added:
        key where result is stored
    base:
        basis of logarithm in simpson index

    Notes
    -----
    If local is true the result is stored in obs, else in cores.

    Returns
    -------

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
    """Computes the gini simpson index for each cell or the spl

    Parameters
    ----------
    so
    spl
    attr: str
        attribute for which to compute the gini simpson index index
    local: bool
        if gini simpson index index is computed for each cell or on the spl level
    key_added:
        key where result is stored
    base:
        basis of logarithm in gini simpson index index

    Notes
    -----
    If local is true the result is stored in obs, else in cores.

    Returns
    -------

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
    """Computes the renyi entropy for each cell or the spl

    Parameters
    ----------
    so
    spl
    attr: str
        attribute for which to compute the renyi entropy
    local: bool
        if renyi entropy is computed for each cell or on the spl level
    key_added:
        key where result is stored
    base:
        basis of logarithm in renyi entropy

    Notes
    -----
    If local is true the result is stored in obs, else in cores.

    Returns
    -------

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
    """Computes the proportion of cells with a given attribute (like phenotype) given by the categories of attr.

        Parameters
        ----------
        so: SpatialOmics instance
        spl: Sample for which to compute the metric
        attr: str
            attribute indicating the classification of an observation.
            Needs to be dtype categorical.
        mode: str
            either `proportion` to compute the relative attr frequency or `count` for number of observations.
        key_added:
            prefix-key where result is stored in splm (local=False) or obsm (local=True).
        local: bool
            if metric is computed for each observation (local=True) or on the sample level (local=False)
        inplace: bool
            whether to return a copy of so with the result

        Notes
        -----

        Returns
        -------
        so if inplace is False, else nothing
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
