import pandas as pd
import pytest
import numpy as np
from spatialHeterogeneity.metrics.heterogeneity.base_metrics import _shannon, _shannon_evenness, _simpson,\
    _simpson_evenness, _gini_simpson, _richness, _hill_number, _renyi, _abundance, _quadratic_entropy
from collections import Counter

obs = [[1], [1,1], [1,2]]
_counts = [Counter(i) for i in obs]

_expected = [0, 0, 1]
@pytest.mark.parametrize('input, res', [(c,e) for c,e in zip(_counts, _expected)])
def test_shannon(input, res):
    assert np.isclose(_shannon(input),res)

_expected = [0, 0, 1]
@pytest.mark.parametrize('input, res', [(c,e) for c,e in zip(_counts, _expected)])
def test_shannon_evenness(input, res):
    assert np.isclose(_shannon_evenness(input), res)

_expected = [1, 1, 0.5]
@pytest.mark.parametrize('input, res', [(c,e) for c,e in zip(_counts, _expected)])
def test_simpson(input, res):
    assert np.isclose(_simpson(input), res)

_expected = [1, 1, 1]
@pytest.mark.parametrize('input, res', [(c,e) for c,e in zip(_counts, _expected)])
def test_simpson_evenness(input, res):
    assert np.isclose(_simpson_evenness(input), res)

_expected = [0, 0, 0.5]
@pytest.mark.parametrize('input, res', [(c,e) for c,e in zip(_counts, _expected)])
def test_gini_simpson(input, res):
    assert np.isclose(_gini_simpson(input), res)

_expected = [1, 1, 2]
@pytest.mark.parametrize('input, res', [(c,e) for c,e in zip(_counts, _expected)])
def test_richness(input, res):
    assert np.isclose(_richness(input),res)

_expected = [1, 1, np.sqrt(0.5)]
@pytest.mark.parametrize('input, res', [(c,e) for c,e in zip(_counts, _expected)])
def test_hill_number(input, res, q=2):
    assert _hill_number(input, q=q) == res

_expected = [0, 0, -1/1*np.log2(0.5)]
@pytest.mark.parametrize('input, res', [(c,e) for c,e in zip(_counts, _expected)])
def test_renyi(input, res, q=2):
    assert np.isclose(_renyi(input, q=q),res)

_expected = [pd.Series([1], index=[1]), pd.Series([1], index=[1]), pd.Series([0.5,0.5], [1,2])]
@pytest.mark.parametrize('input, res', [(c,e) for c,e in zip(_counts, _expected)])
def test_abundance(input, res):
    a = _abundance(input)
    assert np.all(a.eq(res))

# 1.) No difference between groups, i.e. entropy should be 0
# 2.) Difference is always 1 between all pair-wise groups and all groups are equally abundant
# entropy should be (1/N)**2
from string import ascii_lowercase
n = 10
_features = [pd.DataFrame(np.ones((n,5)), index=[i for i in ascii_lowercase[:n]]), pd.DataFrame(np.diag(np.repeat(0.5,n)), index=[i for i in ascii_lowercase[:n]])]
_counts = [Counter({key:1 for key in ascii_lowercase[:n]}), Counter({key:1 for key in ascii_lowercase[:n]})]
_input = [(f,c) for f,c in zip(_features, _counts)]
_expected = [0, 1/n ** 2 * 1 * (n ** 2 - n)]
@pytest.mark.parametrize('input, res', [(c,e) for c,e in zip(_input, _expected)])
def test_quadratic_entropy(input, res):
    feat, counts = input
    return np.isclose(_quadratic_entropy(counts, feat, metric_kwargs={'p': 1}, scale=False), res)