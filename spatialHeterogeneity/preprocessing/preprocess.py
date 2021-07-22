# %%
from ._preprocessors import CensorData, Arcsinh, ReduceLocal
from ..utils.general import make_iterable, _check_is_fitted, is_fitted

import pandas as pd
import numpy as np
import warnings
from skimage.measure import regionprops, regionprops_table

from sklearn.preprocessing import StandardScaler

# %%

def extract_centroids(so, spl, mask_key='cellmasks', inplace=True):
    so = so if inplace else so.copy()

    mask = so.get_mask(spl, mask_key)

    ndata = regionprops_table(mask, properties=['label', 'centroid'])
    ndata = pd.DataFrame.from_dict(ndata)
    ndata.columns = ['cell_id', 'y', 'x']  # NOTE: axis 0 is y and axis 1 is x
    ndata.set_index('cell_id', inplace=True)
    ndata.sort_index(axis=0, ascending=True, inplace=True)

    if spl in so.obs:
        so.obs[spl] = pd.concat((so.obs[spl], ndata), axis=1)
    else:
        so.obs[spl] = ndata

    if not inplace:
        return so

def extract_image_properties(so, spl, inplace=True):
    so = so if inplace else so.copy()

    img = so.get_image(spl)
    data = list(img.shape[1:])
    data.append(data[0]*data[1])

    if not np.all([i in so.spl.columns for i in ['height', 'width', 'area']]):
        so.spl = pd.concat((so.spl, pd.DataFrame(columns = ['height', 'width', 'area'])), axis=1)

    so.spl.loc[spl, ['height', 'width','area']] = data

    if not inplace:
        return so

def reduce_local(so, cores, attr, ref=None, mode='reduce', reducer = np.mean, metric='mse', disp_fn = np.std,
                 inplace=True, key_added=None, fillna=0):
    so = so if inplace else so.copy()
    cores = make_iterable(cores)

    if key_added is None:
        key_added = f'{attr}_local'

    preprocessor = ReduceLocal(ref=ref,mode=mode, reducer=reducer,metric=metric, disp_fn=disp_fn, fillna=fillna)
    res = []
    for spl in cores:
        obs = so.obs[spl][attr]
        res.append(preprocessor.transform(obs))

    df = pd.DataFrame({key_added:res}, index=cores)
    so.obsc = pd.concat((so.obsc, df), axis=1)

    if not inplace:
        return so


def censor_data(so, cores: str = None, *, fit_global: bool = True,
                quant=.99, symmetric=False, axis=0,
                preprocessor=None, inplace=True):
    """Censore X attribute of cores.

    Parameters
    ----------
    so
    cores
    fit_global:
        if data from all cores should be used to fit the preprocessor
    quant:
        quantile to censor data
    symmetric:
        whether to censor data on both sides of the range
    axis:
        axis along which the observations are
    preprocessor:
        pre-fitted preprocessor
    inplace:
        If an IMCData is passed, determines whether a copy is returned.

    Returns
    -------
    IMCData with preprocessed X attribute for the given cores
    """
    if preprocessor is None:
        preprocessor = CensorData(quant=quant, symmetric=symmetric, axis=axis)
    else:
        _check_preprocessor(preprocessor, CensorData)

    return _apply_preprocessor(so=so, cores=cores, preprocessor=preprocessor, fit_global=fit_global, inplace=inplace)


def standard_scaler(so, cores: str = None, *, fit_global: bool = True,
                    zero_center=True, with_std=True,
                    preprocessor=None, inplace=True):
    if preprocessor is None:
        preprocessor = StandardScaler(with_mean=zero_center, with_std=with_std)
    else:
        _check_preprocessor(preprocessor, StandardScaler)

    return _apply_preprocessor(so=so, cores=cores, preprocessor=preprocessor, fit_global=fit_global, inplace=inplace)


def arcsinh(so, cores: str = None, cofactor=5, *,
            fit_global=None, preprocessor=None, inplace=True):
    if fit_global is not None:
        warnings.warn('fit_global parameter has no effect.')

    if preprocessor is None:
        preprocessor = Arcsinh(cofactor=cofactor)
    else:
        _check_preprocessor(preprocessor, Arcsinh)

    return _apply_preprocessor(so=so, cores=cores, preprocessor=preprocessor, fit_global=fit_global, inplace=inplace)


def _apply_preprocessor(so, cores, preprocessor, fit_global, inplace):
    so = so if inplace else so.copy()

    # if an array is passed apply preprocessing to it and return result
    if isinstance(so, np.ndarray):
        return preprocessor.fit_transform(so)

    if cores is None:
        cores = so.cores.index
    else:
        cores = make_iterable(cores)

    # fit preprocessor
    if not is_fitted(preprocessor):

        # get data `X` to fit the preprocessor
        if fit_global:
            X = pd.concat([so.X[i] for i in so.X], axis=0)
        else:
            X = pd.concat([so.X[i] for i in cores], axis=0)

        # fit preprocessor
        preprocessor.fit(X)

    # apply preprocessing to cores
    for spl in cores:
        transformed = preprocessor.transform(so.X[spl].values)
        # TODO: is there a better way than transforming back and forth between numpy/pandas?
        so.X[spl] = pd.DataFrame(transformed, index=so.X[spl].index, columns=so.X[spl].columns)

    if not inplace:
        return so


def _check_preprocessor(preprocessor, processor_type):
    if not isinstance(preprocessor, processor_type):
        raise TypeError(f'`preprocessor` is not of type CensorData but {type(preprocessor)}')

    _check_is_fitted(preprocessor)

# %%
