#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:43:48 2020

@author: art
"""
import pandas as pd
from ..general import make_iterable
#%%

def df2node_attr(df):
    """Convert dataframe to dict keyed by index which can be used to set node attributes with networkx"""
    # NOTE: df.index has to be the nodes
    return df.T.to_dict()

def node_attrs2df(g, attrs=None):
    """Convert networkx graph node attributes to a dataframe index by node"""
    df = pd.DataFrame.from_dict(dict(g.nodes.data()), orient='index')
    if attrs is not None:
        attrs = list(make_iterable(attrs))
        return df[attrs]
    else:
        return df
