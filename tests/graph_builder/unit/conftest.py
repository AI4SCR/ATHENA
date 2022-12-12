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

    for i in range(1, 6):
        y, x = so.obs['a'].loc[i][['y', 'x']]
        array[make_mask(y, x, r, n)] = i

    so.masks['a'] = {'cellmasks' : array.astype(int)}

    return so

def make_mask(a, b, r, n):
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    return mask