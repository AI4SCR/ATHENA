# %%
import pickle
with open('/Users/adrianomartinelli/Downloads/adata.h5ad', 'rb') as f:
    ad = pickle.load(file=f)

# %%
import athena as ath
ath.metrics.richness(ad, attr='label0')
