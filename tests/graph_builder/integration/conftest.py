import pytest
import athena as ath

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
