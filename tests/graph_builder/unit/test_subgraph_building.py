from athena.utils.default_configs import get_default_config
from athena.graph_builder import build_graph
import copy as cp
import pytest

def test_empty_labels(so_object):
    # This tests whether passing the labels varaible as an empty list returns and error
    config = get_default_config(
        builder_type='knn',
        build_concept_graph=True
    )
    config['builder_params']['n_neighbors'] = 2 # set parameter k
    config['concept_params']['filter_col'] = 'cell_type'
    config['concept_params']['labels'] = []
    with pytest.raises(NameError):
        build_graph(so_object, 
                    spl='a', 
                    builder_type='knn',  
                    config=config)

def test_incongruent_labels_in_labels(so_object):
    # This tests whether passing labes that are not 'filter_col' returns and error
    config = get_default_config(
        builder_type='knn',
        build_concept_graph=True
    )
    config['builder_params']['n_neighbors'] = 2 # set parameter k
    config['concept_params']['filter_col'] = 'cell_type'
    config['concept_params']['labels'] = ['tumor', 'stromal', 'not_a_type']
    with pytest.raises(NameError):
        build_graph(so_object, 
                    spl='a', 
                    builder_type='knn',  
                    config=config)

def test_filter_col_not_in_columns(so_object):
    # This tests whether passing an invalid 'filter_col' returns and error
    config = get_default_config(
        builder_type='knn',
        build_concept_graph=True
    )
    config['builder_params']['n_neighbors'] = 2 # set parameter k
    config['concept_params']['filter_col'] = 'not_a_filter_col'
    config['concept_params']['labels'] = ['tumor', 'stromal']
    with pytest.raises(NameError):
        build_graph(so_object, 
                    spl='a', 
                    builder_type='knn',  
                    config=config)
  
def test_filter_col_or_labels_not_specified(so_object):
    # This tests whether filter_col was passed but not labels or the other why around
    config = get_default_config(
        builder_type='knn',
        build_concept_graph=True
    )
    config['builder_params']['n_neighbors'] = 2 # set parameter k
    config['concept_params']['filter_col'] = 'cell_type'
    print(config['concept_params'])
    with pytest.raises(NameError):
        build_graph(so_object, 
                    spl='a', 
                    builder_type='knn',  
                    config=config)

def test_knn(so_object):
    config = get_default_config(
        builder_type='knn',
        build_concept_graph=True
    )
    config['builder_params']['n_neighbors'] = 3
    config['concept_params']['filter_col'] = 'cell_type'
    config['concept_params']['labels'] = ['tumor']
    build_graph(so_object, 
                spl='a', 
                builder_type='knn',  
                config=config,
                key_added='knn')
    assert len(so_object.G['a']['knn'].nodes) == 3
    assert len(so_object.G['a']['knn'].edges) == 6

def test_radius(so_object):
    config = get_default_config(
        builder_type='radius',
        build_concept_graph=True
    )
    config['concept_params']['filter_col'] = 'cell_type'
    config['concept_params']['labels'] = ['tumor']
    build_graph(so_object, 
                spl='a', 
                builder_type='radius',  
                config=config,
                key_added='radius')
    assert len(so_object.G['a']['radius'].nodes) == 3
    assert len(so_object.G['a']['radius'].edges) == 6

def test_contact(so_object):
    config = get_default_config(
        builder_type='contact',
        build_concept_graph=True
    )
    config['concept_params']['filter_col'] = 'cell_type'
    config['concept_params']['labels'] = ['tumor']
    build_graph(so_object, 
                spl='a', 
                builder_type='contact', 
                config=config,
                key_added='contact')
    assert len(so_object.G['a']['contact'].nodes) == 3
    assert len(so_object.G['a']['contact'].edges) == 5

def test_name(so_object):
    # this testes whther the name gets assigned 
    config = get_default_config(
        builder_type='contact',
        build_concept_graph=True
    )
    config['concept_params']['filter_col'] = 'cell_type'
    config['concept_params']['labels'] = ['tumor']
    build_graph(so_object, 
                spl='a', 
                builder_type='contact', 
                config=config,
                key_added='foo')
    assert "foo" in so_object.G['a']