import numpy as np
from collections import Counter
from pandas.api.types import is_list_like
import warnings

def _process_input(counts) -> np.ndarray:
    # check that input is not empty
    if len(counts) == 0:
        warnings.warn('counts is an empty object.')
        return np.array([np.nan])

    # convert to numpy array
    if isinstance(counts, Counter):
        c = np.fromiter(counts.values(), dtype=int)
    elif is_list_like(counts):
        c = np.asarray(counts)
    else:
        raise TypeError(f'counts is neither a counter nor list-like')

    # validate input
    if c.dtype == int:
        c = _counts_to_props(c)
    elif c.dtype == float:
        _validate_props_vector(c)
    else:
        raise TypeError(
            f'counts has an invalid type {type(c.dtype)}. Must be `int` for counts and `float` for probabilities.')

    return c


def _validate_counts_vector(counts):
    '''Validate counts vector.

    Parameters
    ----------
    counts: numpy 1d array of type int
    '''

    if not isinstance(counts, np.ndarray):
        raise TypeError(f'counts vector is not an numpy.ndarray but {type(counts)}')
    if np.isnan(counts).any():
        raise ValueError("counts vector contains nan values.")
    if counts.dtype != int:
        raise TypeError(f'counts should have type `int`, found invalid type {counts.dtype}')
    if counts.ndim != 1:
        raise ValueError("Only 1-D vectors are supported.")
    if (counts < 0).any():
        raise ValueError("Counts vector cannot contain negative values.")


def _counts_to_props(counts):
    """Validates and converts counts to probabilities"""
    _validate_counts_vector(counts)
    props = counts / counts.sum()
    _validate_props_vector(props)
    return props


def _validate_props_vector(props):
    '''Validate props vector.

    Parameters
    ----------
    props: numpy 1d array of type float
    '''

    if not isinstance(props, np.ndarray):
        raise TypeError(f'Probabilities vector is not an numpy.ndarray but {type(props)}')
    if props.dtype != float:
        raise TypeError(f'Probabilities should have type `float`, found invalid type {props.dtype}')
    if props.ndim != 1:
        raise ValueError("Only 1-D vectors are supported.")
    if (props < 0).any():
        raise ValueError("Probabilities vector cannot contain negative values.")
    if (props > 1).any():
        raise ValueError("Probabilities vector cannot contain values larger than 1.")
    if np.isnan(props).any():
        raise ValueError("Probabilities vector contains nan values.")
    if not np.isclose(props.sum(), 1):
        raise ValueError(f'Probabilities do not sum to 1, props.sum = {props.sum()}')