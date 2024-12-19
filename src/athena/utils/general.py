from collections.abc import Iterable
from typing import Any

import numpy as np
from anndata import AnnData
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from pandas.api.types import is_dtype_equal, CategoricalDtype
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted


def get_nx_graph_from_anndata(ad: AnnData, key: str):
    import networkx as nx
    adj = ad.obsp[key]
    g = nx.from_scipy_sparse_array(adj)
    # mapping = {k: v for v, k in zip(ad.obs.index, range(len(ad.obs.index)))}
    mapping = {i: v for i, v in enumerate(ad.obs.index)}
    g = nx.relabel_nodes(g, mapping)
    return g


# import pandas as pd
def is_numeric(*args, **kwargs):
    return is_numeric_dtype(*args, **kwargs)


def is_categorical(series):
    return isinstance(series.dtype, CategoricalDtype)


def make_iterable(obj: Any):
    '''

    Parameters
    ----------
    obj : Any
        Any object that you want to make iterable

    Returns
    -------
    Packed object, possible to iterate overs
    '''

    # if object is iterable and its not a string return object as is
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return obj
    else:
        return (obj,)


def is_seq(x, step=1):
    """Checks if the elements in a list-like object are increasing by step

    Parameters
    ----------
    x: list-like
    step

    Returns
    -------
    True if elements increase by step, else false and the index at which the condition is violated.

    """
    for i in range(1, len(x)):
        if not x[i] == (x[i - 1] + step):
            print('Not seq at: ', i)
            return False
    return True


def order(x, transform=None, decreasing=False):
    """\
    Returns the indices of the ordered elements in x.

    Parameters
    ----------
    x: list_like
        A list_like object
    transform: function
        a function applied to the elements of x to compute a value based on which the array x should be sorted
    decreasing: bool
        indicating if the sorting should be in decreasing order

    Returns
    -------
    The indices of the sorted array

    Examples
    ________
    a = [1,3,2]
    order(a)
    # [0,2,1]

    a = np.linspace(0,np.pi,5)
    order(a, transform = lambda x: np.cos(x))
    # [4,3,2,1,0]

    a = np.linspace(0,np.pi,5)
    order(a, transform = lambda x: np.cos(x)**2)
    # [2,3,1,0,4]
    """
    if transform is None:
        transform = lambda a: a

    sign = 1
    if decreasing: sign = 1

    _x = [(i, transform(x[i])) for i in range(len(x))]
    _x = sorted(_x, key=lambda x: sign * x[1])
    return np.array([_x[i][0] for i in range(len(_x))])


def _check_is_fitted(estimator):
    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))
    if hasattr(estimator, 'fitted'):
        if not is_fitted(estimator):
            raise ValueError('this instance is not fitted.')
    else:
        sklearn_check_is_fitted(estimator)


def is_fitted(estimator):
    if hasattr(estimator, 'fitted'):
        return estimator.fitted
    else:
        try:
            sklearn_check_is_fitted(estimator)
            return True
        except:
            return False
