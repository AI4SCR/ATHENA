from skimage.morphology import square, diamond, disk

DILATION_KERNELS = {
    'disk': disk,
    'square': square,
    'diamon': diamond
}

EDGE_WEIGHT = 'weight'

GRAPH_BUILDER_DEFAULT_PARAMS = {
    'knn': {'builder_params': {'n_neighbors': 6, 'mode':'connectivity', 'metric':'minkowski', 'p':2, 'metric_params':None, 'include_self':True, 'n_jobs':-1}},
    'contact': {'builder_params': {'dilation_kernel': 'disk', 'radius': 4, 'include_self':True}},
    'radius': {'builder_params': {'radius': 36, 'mode':'connectivity', 'metric':'minkowski', 'p':2, 'metric_params':None, 'include_self':True, 'n_jobs':-1}}
}