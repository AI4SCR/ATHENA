# %%
import networkx as nx
import numpy as np
import pandas as pd

# %%
def get_edge_interactions(g: nx.Graph, data: pd.Series):
    # IMPORTANT: Be aware of the symmetry issues when only looking at edges. This is, two cells A,B that share an edge
    # are only represented once, either as A B or B A

    # probably the fasted way to solve this is would be by multidimensional indexing into a numpy array.
    # data[edges.T]
    # however the api is based on cell_ids that are not sequential, thus we have to index into pd.Series which is fast
    # or we convert the non-sequential cell_ids into sequential ones

    # NOTE: The data pd.Series is categorical with globally all categories

    edges = np.array(g.edges)
    edge_interactions = pd.DataFrame({'source': edges[:, 0], 'source_label': data.loc[edges[:, 0]].values,
                                      'target': edges[:, 1], 'target_label': data.loc[edges[:, 1]].values})
    return edge_interactions


def get_node_interactions(g: nx.Graph, data: pd.Series = None):
    # NOTE: The data pd.Series is categorical with globally all categories

    source, neighs = [], []
    for i in g.nodes:
        if len(g[i]) > 0:  # some nodes might have no neighbors
            source.append(i)
            neighs.append(list(g[i]))

    node_interactions = pd.DataFrame({'source': source, 'target': neighs}).explode('target')
    if data is not None:
        node_interactions['source_label'] = data.loc[node_interactions.source].values
        node_interactions['target_label'] = data.loc[node_interactions.target].values

    return node_interactions


def get_interaction_score(interactions, relative_freq=False, observed=False):
    # NOTE: this is not necessarily len(source_labels) == len(g) since only source nodes with neighbors are included
    source_label = interactions[['source', 'source_label']].drop_duplicates().set_index('source')
    source_label = source_label.squeeze()

    source2target_label = interactions.groupby(['source', 'target_label'], observed=observed,
                                               as_index=False).size().rename({'size': 'counts'}, axis=1)
    source2target_label.loc[:, 'source_label'] = source_label[source2target_label.source].values

    if relative_freq:
        tots = source2target_label.groupby('source')['counts'].agg('sum')
        source2target_label['n_neigh'] = tots.loc[source2target_label.source].values
        source2target_label['relative_freq'] = source2target_label['counts'] / source2target_label['n_neigh']
        label2label = source2target_label\
            .groupby(['source_label', 'target_label'], observed=observed)['relative_freq'] \
            .agg('mean') \
            .rename('score') \
            .fillna(0) \
            .reset_index()
    else:
        label2label = source2target_label \
            .groupby(['source_label', 'target_label'], observed=observed)['counts'] \
            .agg('mean') \
            .rename('score') \
            .fillna(0) \
            .reset_index()

    return label2label


# why is this so slow???
def permute_labels_deprecate(data, rng: np.random.Generator):
    attr_copy = data.copy()
    attr_copy[:] = rng.permutation(attr_copy)
    return attr_copy


def permute_labels(data, rng: np.random.Generator):
    return pd.Series(rng.permutation(data), index=data.index)
