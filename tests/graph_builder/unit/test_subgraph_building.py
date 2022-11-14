from athena.graph_builder.constants import GRAPH_BUILDER_DEFAULT_PARAMS
from athena.graph_builder import build_graph
import pytest

def test_empty_types(so_object):
    # This tests whether passing the types varaible as an empty list returns and error
    config = GRAPH_BUILDER_DEFAULT_PARAMS['knn']
    config['builder_params']['n_neighbors'] = 2 # set parameter k
    with pytest.raises(NameError):
        build_graph(so_object, 
                    spl='a', 
                    builder_type='knn', 
                    mask_key='cellmasks', 
                    config=config,
                    inplace=True,
                    coordinate_keys=('x', 'y'),
                    col_name='cell_type', 
                    types=[])

def test_incongruent_labels_in_types(so_object):
    # This tests whether passing labes that are not 'col_name' returns and error
    config = GRAPH_BUILDER_DEFAULT_PARAMS['knn']
    config['builder_params']['n_neighbors'] = 2 # set parameter k
    with pytest.raises(NameError):
        build_graph(so_object, 
                    spl='a', 
                    builder_type='knn', 
                    mask_key='cellmasks', 
                    config=config,
                    inplace=True,
                    coordinate_keys=('x', 'y'),
                    col_name='cell_type', 
                    types=['tumor', 'stromal', 'not_a_type'])

def test_col_name_not_in_columns(so_object):
    # This tests whether passing an invalid 'col_name' returns and error
    config = GRAPH_BUILDER_DEFAULT_PARAMS['knn']
    config['builder_params']['n_neighbors'] = 2 # set parameter k
    with pytest.raises(NameError):
        build_graph(so_object, 
                    spl='a', 
                    builder_type='knn', 
                    mask_key='cellmasks', 
                    config=config,
                    inplace=True,
                    coordinate_keys=('x', 'y'),
                    col_name='not_a_col_name', 
                    types=['tumor', 'stromal'])

def test_col_name_or_types_not_specified(so_object):
    # This tests whether col_name was passed but not types or the other why around
    config = GRAPH_BUILDER_DEFAULT_PARAMS['knn']
    config['builder_params']['n_neighbors'] = 2 # set parameter k
    with pytest.raises(NameError):
        build_graph(so_object, 
                    spl='a', 
                    builder_type='knn', 
                    mask_key='cellmasks', 
                    config=config,
                    inplace=True,
                    coordinate_keys=('x', 'y'),
                    col_name='cell_types')

def test_knn(so_object):
    config = GRAPH_BUILDER_DEFAULT_PARAMS['knn']
    config['builder_params']['n_neighbors'] = 3
    build_graph(so_object, 
                spl='a', 
                builder_type='knn', 
                mask_key='cellmasks',
                key_added='knn',
                config=config,
                inplace=True,
                coordinate_keys=('x', 'y'),
                col_name='cell_type', 
                types=['tumor'])
    assert len(so_object.G['a']['knn'].nodes) == 3
    assert len(so_object.G['a']['knn'].edges) == 6

def test_radius(so_object):
    config = GRAPH_BUILDER_DEFAULT_PARAMS['radius']
    build_graph(so_object, 
                spl='a', 
                builder_type='radius', 
                mask_key='cellmasks', 
                key_added='radius',
                config=config,
                inplace=True,
                coordinate_keys=('x', 'y'),
                col_name='cell_type', 
                types=['tumor'])
    assert len(so_object.G['a']['radius'].nodes) == 3
    assert len(so_object.G['a']['radius'].edges) == 6

def test_contact(so_object):
    config = GRAPH_BUILDER_DEFAULT_PARAMS['contact']
    build_graph(so_object, 
                spl='a', 
                builder_type='contact', 
                mask_key='cellmasks', 
                key_added='contact',
                config=config,
                inplace=True,
                coordinate_keys=('x', 'y'),
                col_name='cell_type', 
                types=['tumor'])
    assert len(so_object.G['a']['contact'].nodes) == 3
    assert len(so_object.G['a']['contact'].edges) == 5

def test_name(so_object):
    # this testes whther the name gets assigned 
    config = GRAPH_BUILDER_DEFAULT_PARAMS['contact']
    build_graph(so_object, 
                spl='a', 
                builder_type='contact', 
                mask_key='cellmasks',
                config=config,
                inplace=True,
                coordinate_keys=('x', 'y'),
                col_name='cell_type', 
                types=['tumor'])
    assert "contact > cell_type > ['tumor']" in so_object.G['a']