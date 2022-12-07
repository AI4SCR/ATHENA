from athena.attributer import add_node_features
import copy as cp
import pytest

# Test that error are bieng handleled correctly:
def test_raise_both_false(so_fixture, default_params):
    so, spl = so_fixture
    config = cp.deepcopy(default_params['so'])
    config['from_obs'] = False
    config['from_X'] = False

    with pytest.raises(NameError):
        add_node_features(so=so, spl=spl, graph_key='knn', features_type='so', config=config)

# Test empty list in obs_cols.
def test_raise_empty_list(so_fixture, default_params):
    so, spl = so_fixture
    config = cp.deepcopy(default_params['so'])
    config['obs_cols'] = []

    # Test empty list on obs
    with pytest.raises(NameError):
        add_node_features(so=so, spl=spl, graph_key='knn', features_type='so', config=config)

    # Test empty list on X
    config['obs_cols'] = ['x', 'y']
    config['X_cols'] = []
    with pytest.raises(NameError):
        add_node_features(so=so, spl=spl, graph_key='knn', features_type='so', config=config)

# Test. Not all elements provided in list config["obs_cols"] are in so.obs[spl].columns.
def test_invalid_colnames(so_fixture, default_params):
    so, spl = so_fixture
    config = cp.deepcopy(default_params['so'])
    config['obs_cols'] = ['x', 'not_a_col_name']

    # Test wrong col name on obs
    with pytest.raises(NameError):
        add_node_features(so=so, spl=spl, graph_key='knn', features_type='so', config=config)

    config['obs_cols'] = ['x', 'y']
    config['X_cols'] = ['Cytokeratin5', 'not_a_col_name']

    # Test wrong col name on X
    with pytest.raises(NameError):
        add_node_features(so=so, spl=spl, graph_key='knn', features_type='so', config=config)

# Test expected behaviour, namely 'so' attributes are in graph. 
def test_features_from_so(so_fixture, default_params):
    so, spl = so_fixture
    config = cp.deepcopy(default_params['so'])
    config['obs_cols'] = ['x']
    config['from_X'] = False

    # By modifying obs
    add_node_features(so=so, spl=spl, graph_key='knn', features_type='so', config=config)

    # Test that 'y' attribute is left out
    with pytest.raises(KeyError):
        so.G[spl]['knn'].nodes[1]['y']

    # Check that 'x' is in
    assert 'x' in so.G[spl]['knn'].nodes[1].keys(), 'Attribute not found.'

    # By modifying X
    config['from_obs'] = False
    config['from_X'] = True
    config['X_cols'] = ['Cytokeratin5']

    add_node_features(so=so, spl=spl, graph_key='knn', features_type='so', config=config)

    # Check that 'Cytokeratin5' is in
    assert 'Cytokeratin5' in so.G[spl]['knn'].nodes[1].keys(), 'Attribute not found.'

# Test that random attributes are in graph. 
def test_random_features(so_fixture, default_params):
    so, spl = so_fixture
    config = cp.deepcopy(default_params['random'])
    config['n_attrs'] = 5

    add_node_features(so=so, spl=spl, graph_key='knn', features_type='random', config=config)

    print(so.G[spl]['knn'].nodes[1])

    # This assertion indirectly tests wheather the errrase functionality also works properyl. 
    assert len(so.G[spl]['knn'].nodes[1]) == config['n_attrs']


