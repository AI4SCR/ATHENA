# %%
import logging
import time

import numpy as np
import pandas as pd
from astropy.stats import RipleysKEstimator
from networkx import Graph
from pandas.api.types import CategoricalDtype

logging.basicConfig(level=logging.INFO)

from ..utils.general import make_iterable
from .utils import get_node_interactions, get_interaction_score, permute_labels
from anndata import AnnData


# %%

def _infiltration(node_interactions: pd.DataFrame, interaction1=('tumor', 'immune'),
                  interaction2=('immune', 'immune')) -> float:
    """
    Compute infiltration score between two species.

    Args:
        node_interactions: Dataframe with columns `source_label` and `target_label` that specifies interactions.
        interaction1: labels of enumerator interaction
        interaction2: labels of denominator interaction

    Notes:
        The infiltration score is computed as #interactions1 / #interactions2.

    Returns:
        Interaction score

    """
    nint = node_interactions  # verbose

    (a1, a2), (b1, b2) = interaction1, interaction2
    num = nint[(nint.source_label == a1) & (nint.target_label == a2)].shape[0]
    denom = nint[(nint.source_label == b1) & (nint.target_label == b2)].shape[0]

    return num / denom if denom > 0 else np.nan  # TODO: np.inf or np.nan


from ..utils.general import get_nx_graph_from_anndata


class Interactions:
    """
    Estimator to quantify interaction strength between different species in the sample.
    """

    VALID_MODES = ['classic', 'histoCAT', 'proportion']
    VALID_PREDICTION_TYPES = ['pvalue', 'observation', 'diff']

    def __init__(self, ad: AnnData, attr: str = 'meta_id', mode: str = 'classic', aggregation: str = 'mean',
                 n_permutations: int = 500,
                 random_seed=42, alpha: float = .01, graph_key: str = 'knn'):
        """Estimator to quantify interaction strength between different species in the sample.

        Args:
            so: SpatialOmics
            spl: Sample for which to compute the interaction strength
            attr: Categorical feature in ad.obs to use for the grouping
            mode: One of {classic, proportion}, see notes
            aggregation: How to aggregate the observed interactions for a given source node
            n_permutations: Number of permutations to compute p-values and the interactions strength score (mode diff)
            random_seed: Random seed for permutations
            alpha: Threshold for significance
            graph_key: Specifies the graph representation to use in so.G[spl] if `local=True`.

        Notes:
            `classic` counts for each pair-wise interaction the number of edges between the two species.
            The proportion method is similar to the classic method but normalises the score by the number of edges and is thus bound [0,1].
        """

        self.ad = ad
        self.graph_key = graph_key
        self.g: Graph = get_nx_graph_from_anndata(ad=ad, key=graph_key)
        self.attr: str = attr
        self.data: pd.Series = ad.obs[attr]
        self.mode: str = mode
        self.aggregation: str = aggregation
        self.n_perm: int = int(n_permutations)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.alpha: float = alpha
        self.fitted: bool = False

        # set dtype categories of data to attributes that are in the data
        self.data = self.data.astype(CategoricalDtype(categories=self.data.unique(), ordered=False))
        self.h0 = None

    def fit(self, prediction_type: str = 'observation') -> None:
        """Compute the interactions scores for the sample.

        Args:
            prediction_type: One of {observation, pvalue, diff}, see Notes
            try_load: load pre-computed permutation results if available

        Returns:

        Notes:
            `observation`: computes the observed interaction strength in the sample

            `pvalue`: computes the P-value of a two-sided t-test for the interactions strength based on the random permutations

            `diff`: computes the difference between observed and average interaction strength (across permutations)
        """
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
        obs_interaction = get_interaction_score(node_interactions, aggregation=self.aggregation,
                                                relative_freq=relative_freq, observed=observed)
        self.obs_interaction = obs_interaction.set_index(['source_label', 'target_label'])

        if not prediction_type == 'observation':
            if self.h0 is None:
                logging.info(
                    f'generate h0 for graph type {self.graph_key} and mode {self.mode} and attribute {self.attr}')
                self.generate_h0(relative_freq=relative_freq, observed=observed, save=True)

        self.fitted = True

    def predict(self) -> pd.DataFrame:
        """Predict interaction strengths of observations.

        Returns: A dataframe with the interaction results.

        """
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


class RipleysK():
    def __init__(self, ad: AnnData, id, attr: str, obsm_key: str = 'spatial'):
        """Compute Ripley's K for a given sample and group.

        Args:
            ad: AnnData
            id: The category in the categorical feature `attr`, for which Ripley's K should be computed
            attr: Categorical feature in ad.obs to use for the grouping
            obsm_key: Key in ad.obsm that contain the spatial coordinates in 'yx' format

        """

        self.id = id
        self.width = ad.uns['width']
        self.height = ad.uns['height']
        self.area = self.width * self.height
        self.rkE = RipleysKEstimator(area=float(self.area),
                                     # we need to cast since the implementation checks for type int/float and does not recognise np.int64
                                     x_max=float(self.width), x_min=0,
                                     y_max=float(self.height), y_min=0)

        df = ad.obs[[attr, 'y', 'x']]
        self.df = df[df[attr] == id]

    def fit(self):
        pass

    def predict(self, radii: list, correction: str = 'ripley', mode: str = 'K'):
        """Estimate Ripley's K

        Args:
            radii: List of radiis for which Ripley's K is computed
            correction: Correction method to use to correct for boarder effects, see [1].
            mode: {K, csr-deviation}. If `K`, Ripley's K is estimated, with `csr-deviation` the deviation from a poission process is computed.

        Returns:
            Ripley's K estimates

        Notes:
            .. [1] https://docs.astropy.org/en/stable/stats/ripley.html
        """

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

    def csr_deviation(self, radii, correction='ripley') -> np.ndarray:
        """
        Compute deviation from random poisson process.

        Args:
            radii: List of radiis for which Ripley's K is computed
            correction: Correction method to use to correct for boarder effects, see [1].

        Returns:

        """
        # http://doi.wiley.com/10.1002/9781118445112.stat07751
        radii = make_iterable(radii)
        K = self.rkE(data=self.df[['x', 'y']], radii=radii, mode=correction)
        L = np.sqrt(K / np.pi)  # transform, to stabilise variance
        dev = L - radii

        return dev
