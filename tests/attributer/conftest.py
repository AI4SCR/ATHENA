import pytest
import athena as ath
import copy as cp
from athena.utils.default_configs import GRAPH_ATTRIBUTER_DEFAULT_PARAMS

@pytest.fixture(scope="module")
def so_fixture():
    # Loead data
    so = ath.dataset.imc()
    print('I was called')

    # Define sample
    spl = 'slide_49_By2x5'

    # Extrac centroids
    ath.pp.extract_centroids(so, spl, mask_key='cellmasks')

    return (so, spl)

@pytest.fixture(scope="module")
def default_params():
    return GRAPH_ATTRIBUTER_DEFAULT_PARAMS