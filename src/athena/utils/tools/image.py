# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:50:24 2020

@author: art
"""
# %%
import numpy as np
from skimage import io
from skimage.measure import regionprops_table
import os
import pandas as pd

import warnings


# %%

def get_shape_outline(mask):
    '''

    Parameters
    ----------
    mask: 2d array
        Array in which objects are marked with labels > 0. Background is 0.

    Returns
    -------
    The same as mask but with only the border of the masks, i.e. outline.

    '''

    mc = mask.copy()
    for x in range(1, mask.shape[0] - 1):
        for y in range(1, mask.shape[1] - 1):
            if mask[x, y] != 0:
                if mask[x + 1, y] == mask[x, y] and mask[x - 1, y] == mask[x, y] and mask[x, y + 1] == mask[x, y] and \
                        mask[x, y - 1] == mask[
                    x, y]:
                    mc[x, y] = 0
    return mc


def extract_mask_expr_feat(mask_file: str, img_file: str, norm=None, channels=None, reducer=np.mean, mask_bg: int = 0):
    """Extracts the pixel values of the mask objects in the mask_file given the pixel intensities in the image for each channel.

    Parameters
    ----------
    mask_file: str
        path to cell mask file
    img_file: str
        path to image file, usually this would be the tiff stack
    norm:
        a instance that normalises the pixel values
    channels: list-like
        indicates for which channels in the image the pixel values of the mask should be extracted, defaults to all channels
    reducer:
        function to summarise the pixel values of a object in the mask, default np.mean
    mask_bg:
        value which indicates background in the mask_file, defaults to 0

    Returns
    -------
    dataframe of shape n_objects x n_channels.
    """

    # load image, image mask
    img = io.imread(img_file)
    img_mask = io.imread(mask_file)

    if img.shape[1:] != img_mask.shape:
        raise (f'Image dimensions {img.shape[1:]} and image mask shape {img_mask.shape} do not match')

    if not channels:
        channels = list(range(len(img)))

    if not norm:
        norm = lambda x: x

    # extract objects fom mask
    objs = np.unique(img_mask)
    objs = objs[objs != mask_bg]  # remove background label
    # print(f'...{len(objs)} cells')

    # initialise expression data frame
    expr = pd.DataFrame(np.zeros((len(objs), len(channels))))
    expr = expr.set_index(objs)
    expr.index.name = 'cell_id'
    expr.columns = channels
    expr.columns.name = 'channel'

    for obj in objs:
        vals = img[:, img_mask == obj]
        # TODO: normalise image pixel intensities, currently identity func
        vals = norm(vals)  # normalise intensities
        vals = reducer(vals, axis=1)  # compute single cell intensity
        expr.loc[obj, :] = vals

    return expr


def extract_mask_morph_feat(mask_file, properties=['label', 'area', 'extent', 'eccentricity'], as_dict=True):
    """Extracts morphological features form cell mask files.

    Parameters
    ----------
    mask_file: str
        path to cell mask file
    properties: list of properties, see [1]
    as_dict

    Returns
    -------

    References
    __________
    .. [1]: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    """

    # load image mask
    img_mask = io.imread(mask_file)

    res = regionprops_table(img_mask, properties=properties)
    res['cell_id'] = res['label']
    del res['label']

    if as_dict:
        return res
    else:
        return pd.DataFrame.from_dict(res).set_index('cell_id')
