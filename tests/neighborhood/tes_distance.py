
def tes_distance():
    import athena as ath
    from pathlib import Path
    import pickle

    path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/PCa/anndatas/raw/240222_ivb_x9y8_66_14.pkl')
    path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/PCa/anndatas/with_athena_scores/231204_ibl_x2y4_29_11.pkl')
    with open(path, 'rb') as f:
        ad = pickle.load(f)

    ath.neigh.distance(ad, attr='label', linkage='min')
    assert 'distance_label_min_None_True' in ad.uns
tes_distance()