# imports
import numpy as np
import pandas as pd
from typing import Callable, Optional, Union, List, Tuple, Iterable, Counter as type_Counter
from pandas.api.types import is_list_like

from ..general import make_iterable

_callList = Iterable[Callable[..., np.ndarray]]

def compute_metrics(df: Union[pd.DataFrame, pd.Series],
                    metrics: Union[Callable[...,np.ndarray], _callList],
                    key: Optional[str] = None,
                    metrics_kwargs: dict = {},
                    name = None) -> pd.DataFrame:

    """

    Parameters
    ----------
    df :  pd.DataFrame or pd.Series
        DataFrame or Series with the counts of the species in a column defined by key (if DataFrame).
    metrics : callable or list of
        A list of functions that take counts and output a numpy.ndarray.
    key : str or None
        If df is a DataFrame the key indicates which column the counts are.
    Returns
    -------

    """

    if isinstance(df, pd.Series):
        key = df.name
        df = df.to_frame()
    elif key is None:
        raise TypeError('No look up key specified')

    if not is_list_like(metrics):
        metrics = make_iterable(metrics)

    if is_list_like(metrics):
        for metric in metrics:
            df = df.assign(metric_name = df[key].apply(lambda x: metric(x, **metrics_kwargs)))
            if name is not None:
                df = df.rename(columns={'metric_name': name})
            elif (metrics_kwargs is None) or ('metric' not in metrics_kwargs):
                df = df.rename(columns={'metric_name': metric.__name__})
            else:
                df = df.rename(columns={'metric_name': f'{metric.__name__}.{metrics_kwargs["metric"]}'})

    return df

def get_group_feature(expr: np.ndarray,
                      groups: dict,
                      reducer: Callable[[Union[int, float, Iterable]], np.ndarray],
                      axis = 0,
                      **kwargs):

    # cast to int
    # np.fromiter(val dtype=int)
    grps = {key:np.array(val, dtype=int) for key,val in groups.items()}

    out = np.zeros((len(grps), expr.shape[1]))
    for idx, grp in enumerate(grps.values()):
        out[idx,:] = reducer(expr[grp, :], axis, **kwargs)

    return pd.DataFrame(out, index=groups.keys())

def extract_metric_results(so):
    # TODO
    '''Extract all the computed results from the spatialOmics instance.

    Args:
        so:

    Returns:

    '''
    pass