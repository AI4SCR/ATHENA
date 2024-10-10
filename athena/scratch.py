# %%
import athena as ath
so = ath.dataset.imc() 

from anndata import AnnData
# spl = 'slide_10_By10x4'

sov2 = dict()
for spl in so.X.keys():
  x = so.X[spl]
  obs = so.obs[spl]

  assert x.index.equals(obs.index)
  # x, obs = x.align(obs, axis=0)

  var = so.var[spl]
  var = var.set_index('target')

    # save masks in Anndata
    
    # save graphs as csr in AnnData, probably in `obsp`

  sov2[spl] = AnnData(X=x, obs=obs, var=var)
  
#%%
import athena as ath
spl = 'slide_10_By10x4'
ath.metrics.shannon(so=sov2, spl=spl, attr='cell_type')
# %%
