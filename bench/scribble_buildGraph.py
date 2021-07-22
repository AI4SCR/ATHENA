import spatialHeterogeneity as sh
import pickle as pk
import os

f = '/Users/art/Documents/spatial-omics/spatialOmics.hdf5'
f = '/Users/art/Documents/spatial-omics/spatialOmics.pkl'
# so = SpatialOmics.form_h5py(f)
with open(f, 'rb') as f:
    so = pk.load(f)

so.spl_keys = list(so.X.keys())
spl = so.spl_keys[0]

# so.G = {}
print('knn')
sh.graph.build_graph(so, spl, builder_type='knn')
print('radius')
sh.graph.build_graph(so, spl, builder_type='radius')
print('contact')
sh.graph.build_graph(so, spl, builder_type='contact')