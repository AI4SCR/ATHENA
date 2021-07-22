
from spatialOmics.spatialOmics import SpatialOmics
import spatialHeterogeneity as sh
import pickle as pk
import os

f = '/Users/art/Documents/spatial-omics/spatialOmics.hdf5'
f = '/Users/art/Documents/spatial-omics/spatialOmics.pkl'
# so = SpatialOmics.form_h5py(f)
with open(f, 'rb') as f:
    so = pk.load(f)

so.spl_keys = list(so.X.keys())
spl = so.spl_keys[1]

sh.neigh.infiltration(so, spl, 'cell_type')

mode = 'classic'
sh.neigh.interactions(so, spl, 'meta_id', mode=mode, prediction_type='observation')
sh.neigh.interactions(so, spl, 'meta_id', mode=mode, prediction_type='pvalue')
sh.neigh.interactions(so, spl, 'meta_id', mode=mode, prediction_type='diff')

mode = 'histoCAT'
sh.neigh.interactions(so, spl, 'meta_id', mode=mode, prediction_type='observation')
sh.neigh.interactions(so, spl, 'meta_id', mode=mode, prediction_type='pvalue')
sh.neigh.interactions(so, spl, 'meta_id', mode=mode, prediction_type='diff')

mode = 'proportion'
sh.neigh.interactions(so, spl, 'meta_id', mode=mode, prediction_type='observation')
sh.neigh.interactions(so, spl, 'meta_id', mode=mode, prediction_type='pvalue')
sh.neigh.interactions(so, spl, 'meta_id', mode=mode, prediction_type='diff')

sh.neigh.ripleysK(so, spl, 'cell_type_id', id=25)




