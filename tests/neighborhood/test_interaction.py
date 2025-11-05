import athena as ath
import anndata
anndata.settings.allow_write_nullable_strings = True
import pickle

with open('/work/FAC/FBM/DBC/mrapsoma/prometex/data/PCa/anndatas/raw_bc2/231204_ibl_x2y4_29_11.pkl', 'rb') as f:
    ad = pickle.load(file=f)

ath.neigh.interactions(ad=ad,  attr='label', graph_key='radius_32',
                       mode='classic', prediction_type='observation', aggregation='sum')

ath.neigh.interactions(ad=ad,  attr='label', graph_key='radius_32',
                       mode='classic', prediction_type='observation', aggregation='mean')
