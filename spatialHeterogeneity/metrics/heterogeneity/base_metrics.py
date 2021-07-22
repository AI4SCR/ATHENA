import numpy as np
from ..utils import _process_input, _validate_counts_vector
from ...utils.general import make_iterable

from scipy.spatial.distance import pdist
import pandas as pd
from collections import Counter

import warnings


# ----- ENTROPYC MEASURES -----#

def _richness(counts) -> int:
    '''

    Parameters
    ----------
    counts : Counter or Iterable
        Either counts or observations
    Returns
    -------
    Richeness of sample.

    '''

    props = _process_input(counts)

    if np.isnan(props).any():
        # this would correspond to richness == 0
        return 0

    return len(props)


def _shannon(counts, base: float = 2) -> float:
    '''
    Compute the shannon entropy of the counts.

    Parameters
    ----------
    counts : Counter, Iterable with integers or float
        If a Counter object or an interable with intergers is provided it is assumed that those are counts of the different species.
        If the iterable contains dtype `float` it is interpreded as probabilities of the different classes
    base : float
        Base of logarithm. Defaults to 2

    Returns
    -------
    Shannon entropy of observations.

    '''

    props = _process_input(counts)

    nonzero_props = props[props.nonzero()]  # by definition log(0) is 0 for entropy
    return -(nonzero_props * np.log(nonzero_props)).sum() / np.log(base)


def _shannon_evenness(counts,
                      base: float = 2) -> np.float:
    '''

    Parameters
    ----------
    counts : Counter or Iterable
        Counts of species
    base : float
        Base of logarithm. Defaults to 2.

    Returns
    -------
    A numpy array with the evenness.

    Notes
    _____
    The value is 1 in case that all species have the same relative abundances.

    References
    __________
    [1] https://anadat-r.davidzeleny.net/doku.php/en:div-ind#fnt__2

    '''
    H = _shannon(counts, base)
    H_max = np.log(_richness(counts)) / np.log(base)
    if H == 0 and H_max == 0:
        return 0
    return H / H_max


def _simpson(counts) -> np.float:
    '''

    Parameters
    ----------
    counts : Counter or Iterable
        Counts of species

    Returns
    -------
    Simpson index as defined in [2]

    Notes
    _____
    Implementation according to [2].
    Simpson's index is also considering both richness and evenness,
    but compared to Shannon it is more influenced by evenness than richness. It represents the probability that two randomly selected individuals will be of the same species. Since this probability decreases with increasing species richness, the Simpson index also decreases with richness, which is not too intuitive. For that reason, more meaningful is to use Gini-Simpson index, which is simply 1-Simpson index, and which with increasing richness increases [2]

    Simpson's index is heavily weighted towards the most abundant species in the sample, while being less sensitive to species richness.[1]

    #If approximate is true :math:`p_i^2` is approximated as :math:`(n_i / N)^2`
    #otherwise as :math:`n*(n_i-1) / (N (N-1))`

    References
    __________
    .. [1]: http://www.pisces-conservation.com/sdrhelp/index.html

    .. [2]: https://anadat-r.davidzeleny.net/doku.php/en:div-ind#fnt__2

    '''

    props = _process_input(counts)
    return (props ** 2).sum()


def _simpson_evenness(counts) -> np.float:
    '''

    Parameters
    ----------
    counts : Counter or Iterable
        Counts of species

    Returns
    -------

    Notes
    _____
    Also called equitability.
    Is calculated from Simpson’s effective number of species divided by observed number of species. Effective number of species (ENS) is the number of equally abundant species which would need to be in a community so as it has the same Simpson’s index as the one really calculated

    '''

    D = _simpson(counts)
    S = _richness(counts)
    return (1 / D) / S


def _gini_simpson(counts) -> np.float:
    '''Computes the Gini Simpson index

    Parameters
    ----------
    counts : Counter or Iterable
        Counts of species

    Returns
    -------
    float

    '''
    return 1 - _simpson(counts)


def _hill_number(counts, q: float) -> float:
    """Compute the hill number.

    Parameters
    ----------
    counts
    q

    Returns
    -------

    .. [1]: https://anadat-r.davidzeleny.net/doku.php/en:div-ind#fnt__2
    """

    if q < 0:
        warnings.warn('q is generally limited to non-negative values')

    if q == 1:
        # exponential of shannon entropy with base e
        return np.exp(_shannon(counts, np.exp(1)))

    props = _process_input(counts)

    if q == np.inf:
        return 1 / np.max(props)

    nonzero_props = props[props.nonzero()]
    return np.power(np.sum(nonzero_props ** q), (1 / (1 - q)))


def _renyi(counts,
           q: float,
           base: float = 2) -> np.float:
    '''Computes the Gini Simpson index

    Parameters
    ----------
    counts : Counter or Iterable
        Counts of species
    q: float
        order of the renyi entropy
    base:
        base of logarithm


    Returns
    -------
    float

    '''

    if q < 0:
        warnings.warn('q is generally limited to non-negative values')

    props = _process_input(counts)

    if q == 1:
        # special case q == 1: shannon entropy
        return _shannon(counts, base)

    if q == np.inf:
        return - np.log(np.max(props)) / np.log(base)

    return 1 / (1 - q) * np.log(np.sum(props ** q)) / np.log(base)


def _quadratic_entropy(counts, obs_data=None, reducer=np.mean, metric: str = 'cosine',
                       kwargs_metric=None, kwargs_reducer=None,
                       normalize=True, precomputed_dists=None) -> np.float:
    """The average difference between two individuals drawn at random from a population.

    .. [1]: https://ecolres.hu/sites/default/files/rao.pdf
    .. [2]: https://digitalcommons.odu.edu/cgi/viewcontent.cgi?article=1074&context=mathstat_etds
    """

    # NOTE: weights contributions by the inverse of the covariance
    # NOTE: It is highly recommended to use a counter instance to match the counts with obs_data

    if kwargs_metric is None:
        kwargs_metric = {}
    if kwargs_reducer is None:
        kwargs_reducer = {}

    if obs_data is not None and precomputed_dists is not None:
        warnings.warn('provide both, `obs_data` and `precomputed_dists`. Precomputed distances will be used.')
    if obs_data is None and precomputed_dists is None:
        warnings.warn('neither `obs_data` nor `precomputed_dists` provided.')

    if isinstance(counts, Counter) and isinstance(obs_data, pd.DataFrame):
        obs_uniq = obs_data.index.unique()

        # check if all observations in counts are in obs and vice versa
        if not np.in1d(list(counts.keys()), obs_uniq).all():
            raise ValueError(
                'some entities in `counts` are not in observations. Please provide all observations used to count.')
        if not np.in1d(obs_uniq, list(counts.keys())).all():
            raise ValueError('some entities in `obs_data` missing in `counts`')

        # compute observation features, group by index name
        obs_data = obs_data.groupby(obs_data.index.name).agg(reducer, **kwargs_reducer)

        # NOTE: this obsolete in the new implementation since we perform the obs_data aggregation in in _quadratic_entropy itself
        # subset features
        obs_data = obs_data[obs_data.index.isin(counts.keys())]

        # order features according to appearance in Counter
        obs_data = obs_data.loc[list(counts.keys())].values

    # compute distances between entities
    if precomputed_dists:
        dists = precomputed_dists
    elif callable(metric):
        dists = metric(obs_data, **kwargs_metric)
    elif isinstance(metric, str):
        dists = pdist(obs_data, metric=metric, **kwargs_metric)
    else:
        raise TypeError(f'Expected string or callable as metric, got {type(metric)}')

    # normalize by number of features
    if normalize:
        dists /= len(obs_data)

    props = _process_input(counts)
    n = len(props)

    # construct probability vectors corresponding to dists
    pi = np.zeros_like(dists)
    pj = np.zeros_like(dists)
    for i in range(len(props) - 1):
        js = np.array(range(i + 1, n))
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
        idx = n * i + js - ((i + 2) * (i + 1)) // 2
        pj[idx] = props[js]
        pi[idx] = props[i]

    return (dists * pi * pj).sum()


# ----- MULTIDIMENSIONAL MEASURES -----#

def _abundance(counts, mode='proportion', event_space=None) -> pd.Series:
    VALID_MODES = ['proportion', 'count']
    # if a counter is provided we will add zero counts for all the events in the event space
    # if an list-like object is provided we compute the counts first
    if event_space is None:
        raise ValueError('No event_space provided.')

    if not isinstance(counts, Counter):
        counts = Counter(counts)

    c0 = Counter({i: 0 for i in event_space})
    counts.update(c0)
    index = counts.keys()

    if mode == 'proportion':
        vals = _process_input(counts)
        dtype = np.float
    elif mode == 'count':
        vals = np.asarray(list(counts.values()))
        _validate_counts_vector(vals)
        dtype = np.int
    else:
        raise ValueError(f'{mode} is not a valid mode, available are {VALID_MODES}')

    return pd.Series(vals, index=index, dtype=dtype)


def _diveristy_profile(counts, metric, parameters, parameter_name='q', kwargs_metric=None) -> np.ndarray:
    if kwargs_metric is None:
        kwargs_metric = {}

    parameters = make_iterable(parameters)
    parameters = np.asarray(parameters)

    profile = np.zeros(len(parameters))
    # TODO: use np.vectorize instead? Probably only minor speed up.
    for i, p in enumerate(parameters):
        kwargs_metric[parameter_name] = p
        profile[i] = metric(counts, **kwargs_metric)

    return pd.Series(profile, index=[parameter_name + str(p) for p in parameters])
