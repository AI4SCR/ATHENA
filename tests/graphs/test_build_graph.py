import athena as ath


def test_knn():
    ad = ath.dataset.imc()['slide_7_Cy2x3']
    ath.graph.build_graph(ad=ad, topology='knn', n_neighbors=5, include_self=False, graph_key='knn')
    assert (ad.obsp['knn'].sum(axis=1) == 5).all()

    ath.graph.build_graph(ad=ad, topology='knn', n_neighbors=10, include_self=False, graph_key='knn')
    assert (ad.obsp['knn'].sum(axis=1) == 10).all()

    import numpy as np
    idc = np.random.choice(range(len(ad)), len(ad), replace=False)
    ad.obs.index = ad.obs.index[idc]

    ath.graph.build_graph(ad=ad, topology='knn', n_neighbors=10, include_self=False, graph_key='knn')
    assert (ad.obsp['knn'].sum(axis=1) == 10).all()

    ath.pp.compute_centroids(ad)
    ath.pl.spatial(ad, attr='meta_id', edges=True)

def test_radius():
    ad = ath.dataset.imc()['slide_7_Cy2x3']
    ath.graph.build_graph(ad=ad, topology='radius', radius=16, include_self=False, graph_key='knn')
    ath.graph.build_graph(ad=ad, topology='radius', radius=32, include_self=False, graph_key='knn')

    ath.pp.compute_centroids(ad)
    ath.pl.spatial(ad, attr='meta_id', edges=True, graph_key='radius')


def test_contact():
    ad = ath.dataset.imc()['slide_7_Cy2x3']
    ath.graph.build_graph(ad=ad, topology='contact', include_self=False, graph_key='knn')
    ath.graph.build_graph(ad=ad, topology='contact', include_self=False, graph_key='knn')

    ath.pp.compute_centroids(ad)
    ath.pl.spatial(ad, attr='meta_id', edges=True, graph_key='contact')
