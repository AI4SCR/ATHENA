# %%
import networkx as nx
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from networkx import Graph
import os
import time
from astropy.stats import RipleysKEstimator
import logging
logging.basicConfig(level=logging.INFO)

from ..utils.general import make_iterable
from .utils import get_node_interactions, get_interaction_score, permute_labels
# %%
def _infiltration(node_interactions, interaction1=('tumor', 'immune'), interaction2=('immune', 'immune')):
    nint = node_interactions  # verbos

    (a1, a2), (b1, b2) = interaction1, interaction2
    num = nint[(nint.source_label == a1) & (nint.target_label == a2)].shape[0]
    denom = nint[(nint.source_label == b1) & (nint.target_label == b2)].shape[0]

    return num / denom if denom > 0 else np.nan  # TODO: np.inf or np.nan


class Interactions:
    VALID_MODES = ['classic', 'histoCAT', 'proportion']
    VALID_PREDICTION_TYPES = ['pvalue', 'observation', 'diff']

    def __init__(self, so, spl, attr='meta_id', mode='classic', n_permutations=500, random_seed=None, alpha=.01, graph_key='knn'):
        self.so = so
        self.spl: str = spl
        self.graph_key = graph_key
        self.g: Graph = so.G[spl][graph_key]
        self.attr: str = attr
        self.data: pd.Series = so.obs[spl][attr]
        self.mode: str = mode
        self.n_perm: int = int(n_permutations)
        self.random_seed = random_seed if random_seed else so.random_seed
        self.rng = np.random.default_rng(random_seed)
        self.alpha: float = alpha
        self.fitted: bool = False

        # set dtype categories of data to attributes that are in the data
        self.data = self.data.astype(CategoricalDtype(categories=self.data.unique(), ordered=False))

        # path where h0 models would be
        self.path = os.path.expanduser(f'~/.spatial-heterogeneity/h0-models/')
        self.h0_file = f'{spl}_{attr}_{graph_key}_{mode}.pkl'
        self.h0 = None

    def fit(self, prediction_type='observation', try_load=True):
        if prediction_type not in self.VALID_PREDICTION_TYPES:
            raise ValueError(
                f'invalid `prediction_type` {prediction_type}. Available modes are {self.VALID_PREDICTION_TYPES}')

        self.prediction_type = prediction_type

        # extract observed interactions
        if self.mode == 'classic':
            relative_freq, observed = False, False
        elif self.mode == 'histoCAT':
            relative_freq, observed = False, True
        elif self.mode == 'proportion':
            relative_freq, observed = True, False
        else:
            raise ValueError(f'invalid mode {self.mode}. Available modes are {self.VALID_MODES}')

        node_interactions = get_node_interactions(self.g, self.data)
        obs_interaction = get_interaction_score(node_interactions, relative_freq=relative_freq, observed=observed)
        self.obs_interaction = obs_interaction.set_index(['source_label', 'target_label'])

        if not prediction_type == 'observation':
            if try_load:
                if os.path.isdir(self.path) and self.h0_file in os.listdir(self.path):
                    logging.info(
                        f'loading h0 for {self.spl}, graph type {self.graph_key} and mode {self.mode}')
                    self.h0 = pd.read_pickle(os.path.join(self.path, self.h0_file))
            # if try_load was not successful
            if self.h0 is None:
                logging.info(
                    f'generate h0 for {self.spl}, graph type {self.graph_key} and mode {self.mode} and attribute {self.attr}')
                self.generate_h0(relative_freq=relative_freq, observed=observed, save=True)

        self.fitted = True

    def predict(self):
        if self.prediction_type == 'observation':
            return self.obs_interaction
        elif self.prediction_type == 'pvalue':
            # TODO: Check p-value computation
            data_perm = pd.concat((self.obs_interaction, self.h0), axis=1)
            data_perm.fillna(0, inplace=True)
            data_pval = pd.DataFrame(index=data_perm.index)

            # see h0_models_analysis.py for alterantive p-value computation
            data_pval['score'] = self.obs_interaction.score
            data_pval['perm_mean'] = data_perm.apply(lambda x: np.mean(x[1:]), axis=1, raw=True)
            data_pval['perm_std'] = data_perm.apply(lambda x: np.std(x[1:]), axis=1, raw=True)
            data_pval['perm_median'] = data_perm.apply(lambda x: np.median(x[1:]), axis=1, raw=True)

            data_pval['p_gt'] = data_perm.apply(lambda x: np.sum(x[1:] >= x[0]) / self.n_perm, axis=1, raw=True)
            data_pval['p_lt'] = data_perm.apply(lambda x: np.sum(x[1:] <= x[0]) / self.n_perm, axis=1, raw=True)
            data_pval['perm_n'] = data_perm.apply(lambda x: self.n_perm, axis=1, raw=True)

            data_pval['p'] = data_pval.apply(lambda x: x.p_gt if x.p_gt <= x.p_lt else x.p_lt, axis=1)
            data_pval['sig'] = data_pval.apply(lambda x: x.p < self.alpha, axis=1)
            data_pval['attraction'] = data_pval.apply(lambda x: x.p_gt <= x.p_lt, axis=1)
            data_pval['sigval'] = data_pval.apply(lambda x: np.sign((x.attraction - .5) * x.sig), axis=1)
            return data_pval
        elif self.prediction_type == 'diff':
            data_perm = pd.concat((self.obs_interaction, self.h0), axis=1)
            data_perm.fillna(0, inplace=True)
            data_pval = pd.DataFrame(index=data_perm.index)

            # see h0_models_analysis.py for alterantive p-value computation
            data_pval['score'] = self.obs_interaction.score
            data_pval['perm_mean'] = data_perm.apply(lambda x: np.mean(x[1:]), axis=1, raw=True)
            data_pval['perm_std'] = data_perm.apply(lambda x: np.std(x[1:]), axis=1, raw=True)
            data_pval['perm_median'] = data_perm.apply(lambda x: np.median(x[1:]), axis=1, raw=True)

            data_pval['diff'] = (data_pval['score'] - data_pval['perm_mean'])
            return data_pval

        else:
            raise ValueError(
                f'invalid `prediction_type` {self.prediction_type}. Available modes are {self.VALID_PREDICTION_TYPES}')

    def generate_h0(self, relative_freq, observed, save=True):
        connectivity = get_node_interactions(self.g).reset_index(drop=True)

        res_perm, durations = [], []
        for i in range(self.n_perm):
            tic = time.time()

            data = permute_labels(self.data, self.rng)
            source_label = data.loc[connectivity.source].values.ravel()
            target_label = data.loc[connectivity.target].values.ravel()

            # create pd.Series and node_interaction pd.DataFrame
            source_label = pd.Series(source_label, name='source_label', dtype=self.data.dtype)
            target_label = pd.Series(target_label, name='target_label', dtype=self.data.dtype)
            df = pd.concat((connectivity, source_label, target_label), axis=1)

            # get interaction count
            perm = get_interaction_score(df, relative_freq=relative_freq, observed=observed)
            perm['permutation_id'] = i

            # save result
            res_perm.append(perm)

            # stats
            toc = time.time()
            durations.append(toc - tic)

            if (i + 1) % 10 == 0:
                print(f'{time.asctime()}: {i + 1}/{self.n_perm}, duration: {np.mean(durations):.2f}) sec')
        print(
            f'{time.asctime()}: Finished, duration: {np.sum(durations) / 60:.2f} min ({np.mean(durations):.2f}sec/it)')

        h0 = pd.concat(res_perm)
        self.h0 = pd.pivot(h0, index=['source_label', 'target_label'], columns='permutation_id', values='score')

        # create folders
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.h0.to_pickle(os.path.join(self.path, self.h0_file))


class RipleysK():
    def __init__(self, so, spl, id, attr='cell_type_id', graph_key='knn'):
        self.g = so.G[spl][graph_key]
        self.id = id
        self.area = so.spl.loc[spl].area
        self.width = so.spl.loc[spl].width
        self.height = so.meta.loc[spl].height
        self.rkE = RipleysKEstimator(area=float(self.area),
                                     # we need to cast since the implementation checks for type int/float and does not recognise np.int64
                                     x_max=float(self.width), x_min=0,
                                     y_max=float(self.height), y_min=0)
        df = so.obs[spl][['x', 'y', attr]]
        self.df = df[df[attr] == id]

    def fit(self):
        pass

    def predict(self, radii, correction='ripley', mode='K'):

        if radii is None:
            radii = np.linspace(0, min(self.height, self.width) / 2, 10)
        radii = make_iterable(radii)

        # if we have no observations of the given id, K is zero
        if len(self.df) > 0:
            K = self.rkE(data=self.df[['x', 'y']], radii=radii, mode=correction)
        else:
            K = np.zeros_like(radii)

        if mode == 'K':
            res = K
        elif mode == 'csr-deviation':
            L = np.sqrt(K / np.pi)  # transform, to stabilise variance
            res = L - radii

        res = pd.Series(res, index=radii)
        return res

    def csr_deviation(self, radii, correction='ripley'):
        # http://doi.wiley.com/10.1002/9781118445112.stat07751
        radii = make_iterable(radii)
        K = self.rkE(data=self.df[['x', 'y']], radii=radii, mode=correction)
        L = np.sqrt(K / np.pi)  # transform, to stabilise variance
        dev = L - radii

        return dev
