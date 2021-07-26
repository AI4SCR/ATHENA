import numpy as np
from ..utils import _process_input, _validate_counts_vector
# from ...utils.general import make_iterable

# from scipy.spatial.distance import pdist
import pandas as pd
from collections import Counter
from typing import Counter as ct, Iterable, Union

import warnings


# ----- ENTROPYC MEASURES -----#

def _richness(counts: Union[ct, Iterable]) -> int:
    """Compute richness.

    Args:
        counts: Either counts or observations

    Returns:
        Richness of sample.

    """

    props = _process_input(counts)

    if np.isnan(props).any():
        # this would correspond to richness == 0
        return 0

    return len(props)


def _shannon(counts: Union[ct, Iterable], base: float = 2) -> float:
    """Compute the shannon entropy of the counts.

    Args:
        counts: If a Counter object or an interable with intergers is provided it is assumed that those are counts of the different species.
                If the iterable contains dtype `float` it is interpreted as probabilities of the different classes
        base: Base of logarithm. Defaults to 2

    Returns:
        Shannon entropy of observations.

    """

    props = _process_input(counts)

    nonzero_props = props[props.nonzero()]  # by definition log(0) is 0 for entropy
    return -(nonzero_props * np.log(nonzero_props)).sum() / np.log(base)


def _shannon_evenness(counts: Union[ct, Iterable],
                      base: float = 2) -> np.float:
    """Compute shannon evenness. This is a normalised form of the shannon index.

    Args:
        counts: If a Counter object or an interable with intergers is provided it is assumed that those are counts of the different species.
                If the iterable contains dtype `float` it is interpreted as probabilities of the different classes
        base: Base of logarithm. Defaults to 2

    Returns:
        A numpy array with the evenness.

    Notes:
        The value is 1 in case that all species have the same relative abundances.
        .. [1] https://anadat-r.davidzeleny.net/doku.php/en:div-ind#fnt__2

    """

    H = _shannon(counts, base)
    H_max = np.log(_richness(counts)) / np.log(base)
    if H == 0 and H_max == 0:
        return 0
    return H / H_max


def _simpson(counts: Union[ct, Iterable]) -> np.float:
    """Compute Simpson Index.

    Args:
        counts: If a Counter object or an interable with intergers is provided it is assumed that those are counts of the different species.
                If the iterable contains dtype `float` it is interpreted as probabilities of the different classes

    Returns:
        Simpson index.

    Notes:
        Implementation according to [2].
        Simpson's index is also considering both richness and evenness,
        but compared to Shannon it is more influenced by evenness than richness. It represents the probability that two randomly selected individuals will be of the same species. Since this probability decreases with increasing species richness, the Simpson index also decreases with richness, which is not too intuitive. For that reason, more meaningful is to use Gini-Simpson index, which is simply 1-Simpson index, and which with increasing richness increases [2]

        Simpson's index is heavily weighted towards the most abundant species in the sample, while being less sensitive to species richness.[1]

    References:
        .. [1]: http://www.pisces-conservation.com/sdrhelp/index.html
        .. [2]: https://anadat-r.davidzeleny.net/doku.php/en:div-ind#fnt__2
    """

    props = _process_input(counts)
    return (props ** 2).sum()


def _simpson_evenness(counts: Union[ct, Iterable]) -> np.float:
    """Compute Simpson Evenness.

    Args:
        counts: If a Counter object or an interable with intergers is provided it is assumed that those are counts of the different species.
                If the iterable contains dtype `float` it is interpreted as probabilities of the different classes

    Returns:
        Simpson evenness

    Notes:
        Also called equitability. Is calculated from Simpson’s effective number of species divided by observed number of species. Effective number of species (ENS) is the number of equally abundant species which would need to be in a community so as it has the same Simpson’s index as the one really calculated


    """

    D = _simpson(counts)
    S = _richness(counts)
    return (1 / D) / S


def _gini_simpson(counts: Union[ct, Iterable]) -> np.float:
    """Computes the Gini Simpson index

    Args:
        counts: If a Counter object or an interable with intergers is provided it is assumed that those are counts of the different species.
                If the iterable contains dtype `float` it is interpreted as probabilities of the different classes

    Returns:
        Gini-Simpson index.
    """

    return 1 - _simpson(counts)


def _hill_number(counts: Union[ct, Iterable], q: float) -> float:
    """Compute the hill number.

    Notes:
        [1]: https://anadat-r.davidzeleny.net/doku.php/en:div-ind#fnt__2

    Args:
        counts: If a Counter object or an interable with intergers is provided it is assumed that those are counts of the different species.
                If the iterable contains dtype `float` it is interpreted as probabilities of the different classes
        q: Order of hill number.

    Returns:
        Hill number.
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


def _renyi(counts: Union[ct, Iterable],
           q: float,
           base: float = 2) -> np.float:
    """Computes the Renyi-Entropy of order q.

    Args:
         counts: If a Counter object or an interable with intergers is provided it is assumed that those are counts of the different species.
                If the iterable contains dtype `float` it is interpreted as probabilities of the different classes
        q: Order of Renyi-Entropy.
        base: Base of logarithm.

    Returns:
        Renyi-Entropy of order q.
    """

    if q < 0:
        warnings.warn('q is generally limited to non-negative values')

    props = _process_input(counts)

    if q == 1:
        # special case q == 1: shannon entropy
        return _shannon(counts, base)

    if q == np.inf:
        return - np.log(np.max(props)) / np.log(base)

    return 1 / (1 - q) * np.log(np.sum(props ** q)) / np.log(base)


# ----- MULTIDIMENSIONAL MEASURES -----#

def _abundance(counts: Union[ct, Iterable], mode='proportion', event_space=None) -> pd.Series:
    """Compute abundance of different species in counts

    Args:
        counts: If a Counter object or an interable with intergers is provided it is assumed that those are counts of the different species.
                If the iterable contains dtype `float` it is interpreted as probabilities of the different classes
        mode: Either `proportion` or `counts`. If `proportion` we compute the frequency of the species, else the absolute counts.
        event_space: If provided, computes the abundance of all species in the event space. Useful to compute results including all species even of those not present in the current counts.

    Returns:
        Abundance of species, either as frequency (proportion) or absolute count.
    """
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
