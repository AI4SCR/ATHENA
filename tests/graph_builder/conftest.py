import numpy as np
import pandas as pd
import pytest
from spatialOmics import SpatialOmics


@pytest.fixture(scope="module")
def so_object():
    # create empty instance
    so = SpatialOmics()

    # populate a sample
    data = {'sample': ['a'],
            'anno': [None]}
    so.spl = pd.DataFrame(data)

    # Populate so.obs
    data = {'cell_id': [1, 2, 3, 4, 5],
            'y': [8, 13, 10, 80, 50], 
            'x': [8, 8, 15, 80, 60],
            'cell_type': ['tumor', 'tumor', 'tumor', 'epithilial', 'stromal']}
    ndata = pd.DataFrame(data)
    ndata.set_index('cell_id', inplace=True)
    ndata.sort_index(axis=0, ascending=True, inplace=True)
    so.obs['a'] = ndata

    # Populate so.masks
    n = 100
    r = 2
    array = np.zeros((n, n))
    y, x = so.obs['a'].loc[1]
    array[make_mask(y, x, r)] = 1
    y, x = so.obs['a'].loc[2]
    array[make_mask(y, x, r)] = 2
    y, x = so.obs['a'].loc[3]
    array[make_mask(y, x, r)] = 3
    y, x = so.obs['a'].loc[4]
    array[make_mask(y, x, r)] = 4
    y, x = so.obs['a'].loc[5]
    array[make_mask(y, x, r)] = 5
    so.masks['a'] = {'cell_masks' : array}

def make_mask(a, b, r):
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    return mask