# %%
import pandas as pd
import numpy as np

from .base_estimators import Interactions, _infiltration, RipleysK

from .utils import get_node_interactions
from ..utils.general import is_categorical


# %%
def interactions(so, spl, attr, mode='classic', prediction_type='observation', *, n_permutations=100,
                 random_seed=None, alpha=.01, try_load=True, key_added=None, graph_key='knn',
                 inplace=True):
    so = so if inplace else so.copy()

    # NOTE: uns_path = f'{spl}/interactions/'
    if key_added is None:
        key_added = f'{attr}_{mode}_{prediction_type}'

    if random_seed is None:
        random_seed = so.random_seed

    estimator = Interactions(so=so, spl=spl, attr=attr, mode=mode, n_permutations=n_permutations,
                             random_seed=random_seed, alpha=alpha, graph_key=graph_key)

    estimator.fit(prediction_type=prediction_type, try_load=try_load)
    res = estimator.predict()

    # add result to uns attribute
    add2uns(so, res, spl, 'interactions', key_added)

    if not inplace:
        return so


def infiltration(so, spl, attr, *, interaction1=('tumor', 'immune'), interaction2=('immune', 'immune'),
                 add_key='infiltration', inplace=True, graph_key='knn'):
    so = so if inplace else so.copy()

    data = so.obs[spl][attr]
    if isinstance(data, pd.DataFrame):
        raise ValueError(f'multidimensional attr ({data.shape}) is not supported.')

    if not is_categorical(data):
        raise TypeError('`attr` needs to be categorical')

    if not np.in1d(np.array(interaction1 + interaction2), data.unique()).all():
        mask = np.in1d(np.array(interaction1 + interaction2), data.unique())
        missing = np.array(interaction1 + interaction2)[~mask]
        raise ValueError(f'specified interaction categories are not all in `attr`. Missing {missing}')

    nint = get_node_interactions(so.G[spl][graph_key], data)

    res = _infiltration(node_interactions=nint, interaction1=interaction1, interaction2=interaction2)

    so.spl.loc[spl, add_key] = res

    if not inplace:
        return so


def ripleysK(so, spl, attr, id, *, mode='K', radii=None, correction='ripley', inplace=True, key_added=None,
             graph_key='knn'):
    so = so if inplace else so.copy()

    # NOTE: uns_path = f'{spl}/clustering/'
    if key_added is None:
        uns_path = f'{attr}_{id}_{mode}_{correction}'

    estimator = RipleysK(so=so, spl=spl, id=id, attr=attr)
    res = estimator.predict(radii=radii, correction=correction, mode=mode)

    # add result to uns attribute
    add2uns(so, res, spl, 'ripleysK', key_added)

    if not inplace:
        return so


def add2uns(so, res, spl, parent_key, key_added):
    if spl in so.uns:
        if parent_key in so.uns[spl]:
            so.uns[spl][parent_key][key_added] = res
        else:
            so.uns[spl].update({parent_key: {key_added: res}})
    else:
        so.uns.update({spl: {parent_key: {key_added: res}}})
