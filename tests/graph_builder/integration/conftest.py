import pytest
import athena as ath


@pytest.fixture(scope="module")
def so_fixture():
    # %%
    # Load data
    so = ath.dataset.imc_sample()

    # Define sample
    spl = 'slide_7_Cy2x4'

    # Set right index
    so.obs[spl].set_index('CellId', inplace=True)

    # Extract centroids
    ath.pp.extract_centroids(so, spl, mask_key='cellmasks')

    return (so, spl)
