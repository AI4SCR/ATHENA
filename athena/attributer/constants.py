GRAPH_ATTRIBUTER_DEFAULT_PARAMS = {
    'so': {'from_obs': True,
           'obs_cols': ['meta_id', 
                        'cell_type_id',
                        'phenograph_cluster', 
                        'y', 
                        'x'],
           'from_X': True,
           'X_cols': 'all'},
    'deep': {},
    'random': {'n_attrs': 3}
}