# %%
# from ._preprocessors import CensorData, Arcsinh, ReduceLocal
# from ..utils.general import make_iterable, _check_is_fitted, is_fitted

import pandas as pd
import numpy as np
import warnings
from skimage.measure import regionprops, regionprops_table

# from sklearn.preprocessing import StandardScaler

# %%

def extract_centroids(so, spl, mask_key='cellmasks', inplace=True):
    """Extract centroids from segementation masks.

    Args:
        so: SpatialOmics instance
        spl: sample for which to extract centroids
        mask_key: segmentation masks to use
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Returns:

    """
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
    """

    Args:
        so: SpatialOmics instance
        spl: sample for which to extract centroids
        inplace: Whether to add the metric to the current SpatialOmics instance or to return a new one.

    Returns:

    """
    so = so if inplace else so.copy()

    img = so.get_image(spl)
    data = list(img.shape[1:])
    data.append(data[0]*data[1])

    if not np.all([i in so.spl.columns for i in ['height', 'width', 'area']]):
        so.spl = pd.concat((so.spl, pd.DataFrame(columns = ['height', 'width', 'area'])), axis=1)

    so.spl.loc[spl, ['height', 'width','area']] = data

    if not inplace:
        return so