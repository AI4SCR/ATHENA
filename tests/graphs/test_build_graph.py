import athena as ath


def test_knn():
    ad = ath.dataset.imc()['slide_7_Cy2x3']
    ath.graph.build_graph(ad=ad, topology='knn', n_neighbors=5, include_self=False, graph_key='knn')
    assert (ad.obsp['knn'].sum(axis=1) == 5).all()

    ath.graph.build_graph(ad=ad, topology='knn', n_neighbors=10, include_self=False, graph_key='knn')
    assert (ad.obsp['knn'].sum(axis=1) == 10).all()
