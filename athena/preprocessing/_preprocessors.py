# %%
import numpy as np
from scipy.spatial.distance import euclidean, minkowski, mahalanobis, cosine
from sklearn.metrics import mean_squared_error, mean_absolute_error


# %%

# censore data
# NOTE: This function is the python implementation (extended to 2D ndarrays) from the R bbRtools
class CensorData:
    def __init__(self, quant=.99, symmetric=False, axis=0):
        self.quant = quant
        self.symmetric = symmetric
        self.axis = axis
        self.lower_quant = None
        self.upper_val = None
        self.fitted = False

    def fit(self, X):

        if self.symmetric:
            self.lower_quant = (1 - self.quant) / 2
            self.upper_quant = self.lower_quant + self.quant

            self.lower_val = np.quantile(X, self.lower_quant, axis=self.axis)
        else:
            self.upper_quant = self.quant

        q = np.quantile(X, self.upper_quant, axis=self.axis)
        self.upper_val = q
        self.fitted = True

    def transform(self, X):
        x = X.copy()

        if np.ndim(x) > 1:
            for i in range(x.shape[1]):
                x[x[:, i] > self.upper_val[i], i] = self.upper_val[i]
                if self.symmetric:
                    x[x[:, i] < self.lower_val[i], i] = self.lower_val[i]
        else:
            x[x > self.upper_val] = self.upper_val

            if self.symmetric:
                x[x < self.lower_val] = self.lower_val

        return x

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Arcsinh:
    """
    .. [1]: https://support.cytobank.org/hc/en-us/articles/206148057-About-the-Arcsinh-transform
    """

    def __init__(self, cofactor=5):
        self.cofactor = cofactor
        self.fitted = True

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return np.arcsinh(X / self.cofactor)

    def fit_transform(self, X, y=None):
        return self.transform(X)


class ReduceLocal:
    METRICS = dict(euclidean=euclidean, minkowski=minkowski, mahalanobis=mahalanobis, cosine=cosine,
                   mse=mean_squared_error, mae=mean_absolute_error)

    def __init__(self, ref=None, mode='reduce', reducer=np.mean, metric='mse', disp_fn=np.std, groupby=None, fillna=0,
                 kwargs=None):
        self.fitted = True
        self.groupby = groupby
        self.fillna = fillna
        self.ref = ref
        self.mode = mode
        self.reducer = reducer
        self.disp_fn = disp_fn

        if callable(metric):
            self.metric = metric
        elif isinstance(metric, str):
            self.metric = self.METRICS[metric]
        else:
            raise ValueError(f'Invalid metric {metric}. Valid {self.METRICS}')

        self.kwargs = kwargs if kwargs is not None else {}

    def fit(self):
        pass

    def transform(self, obs):
        if self.mode == 'distance':
            if self.ref is None:
                self.ref = np.zeros_like(obs)
                self.kwargs.update({'ref': self.ref})
                # raise ValueError('Please provide a reference to which to compute the distance.')
            if self.groupby is None:
                res = self.metric(obs, **self.kwargs)
            else:
                res = obs.groupby(self.groupby).agg(self.metric, **self.kwargs)
        elif self.mode == 'dispersion':
            if self.groupby is None:
                res = self.disp_fn(obs, **self.kwargs)
            else:
                res = obs.groupby(self.groupby).agg(self.disp_fn, **self.kwargs)
        elif self.mode == 'reduce':
            if self.groupby is None:
                res = self.reducer(obs, **self.kwargs)
            else:
                res = obs.groupby(self.groupby).agg(self.reducer, **self.kwargs)
        else:
            raise ValueError(f'Invalid mode {self.mode}. Select either [distance, dispersion, reduce]')

        res = res.fillna(self.fillna)

        return res

    def fit_transform(self, obs):
        return self.transform(obs)
