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
