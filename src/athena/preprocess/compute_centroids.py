import pandas as pd
from skimage.measure import regionprops_table


def compute_centroids(ad, mask_key='mask', coord_keys=('y', 'x'), copy=False):
    """Compute centroids from mask and add them to AnnData object.

    Args:
        ad (AnnData): AnnData object.
        mask_key (str): Key in `ad.uns` where the mask is stored.
        coord_keys (tuple): Tuple of strings specifying the column names for the centroid coordinates.

    Returns:
        AnnData: AnnData object with centroids added to `ad.obs`.

    """
    ad = ad.copy() if copy else ad

    centroids = pd.DataFrame(regionprops_table(ad.uns[mask_key], properties=('label', 'centroid')))
    centroids.columns = ['object_id', *coord_keys]
    centroids = centroids.set_index('object_id')
    centroids.index = centroids.index.astype(str)  # we need to convert to `str` to match the index of `ad.obs`
    ad.obs = pd.concat((ad.obs, centroids), axis=1)

    if copy:
        return ad
