# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import spatialHeterogeneity as sh

# %%
so = sh.dataset.imc()
spl = list(so.X.keys())[0]
# %%
sh.pp.extract_centroids(so, spl)
sh.neigh.ripleysK(so, spl, 'meta_id', 1)
so.uns[spl]['ripleysK']

# %% generate artifical data

# evenly spaced
x, y = np.linspace(0, 100, 80), np.linspace(0, 100, 80)
xv, yv = np.meshgrid(x, y)
xv, yv = xv.reshape(-1), yv.reshape(-1)
even = pd.DataFrame([xv, yv]).T
even.columns = ['x', 'y']
even['meta_id'] = 0

# random
random = even.copy()
random['meta_id'] = np.random.randint(0,2,len(even))

# clustering 1
squares = even.copy()
squares.loc[ (squares.x < 50) & (squares.y < 50), 'meta_id' ] = 0
squares.loc[ (squares.x > 50) & (squares.y < 50), 'meta_id' ] = 1
squares.loc[ (squares.x > 50) & (squares.y > 50), 'meta_id' ] = 2
squares.loc[ (squares.x < 50) & (squares.y < 50), 'meta_id' ] = 3

# clustering 2
x, y = 90, 90
meta_id = np.zeros((x,y), int)

for i in range(x):
    for j in range(y):
        if (i//6 + j//6) % 2:
            meta_id[i,j] = 0
        else:
            meta_id[i, j] = 1

xv, yv = np.meshgrid(range(x),range(y))
xv, yv = xv.reshape(-1), yv.reshape(-1)
clu2 = pd.DataFrame([xv, yv]).T
clu2.columns = ['x', 'y']
clu2['meta_id'] = meta_id.reshape(-1)

# %% plot data sets
datasets = [even, random, squares, clu2]
names = ['even', 'random', 'squares', 'clu2']

fig, axs = plt.subplots(2,2)
for dat,ax,title in zip(datasets, axs.flat, names):
    sns.scatterplot(data=dat, x='x', y='y', hue='meta_id', ax=ax)
    ax.set_title(title)
plt.tight_layout()
fig.show()

# %% compute Ripley's K

# populate SpatialOmics
for dat, spl in zip(datasets, names):
    so.obs.update({spl:dat})
    so.spl.loc[spl, 'area'] = dat.x.max() * dat.y.max()
    so.spl.loc[spl, 'width'] = dat.x.max()
    so.spl.loc[spl, 'height'] = dat.y.max()

for spl in names:
    for id in np.unique(so.obs[spl].meta_id):
        if spl == 'clu2':
            sh.neigh.ripleysK(so, spl, 'meta_id', id, mode='csr-deviation', radii=np.arange(90)+0.01)
        else:
            sh.neigh.ripleysK(so, spl, 'meta_id', id, mode='csr-deviation')

# %% plot results

res = []
for spl in names:
    keys = list(so.uns[spl]['ripleysK'].keys())
    keys = [i for i in keys if 'csr' in i]
    ids =[i.split('_')[2] for i in keys]
    cont = [so.uns[spl]['ripleysK'][i] for i in keys]
    df = pd.concat(cont, 1)
    df = df.reset_index()
    df.columns = ['radii'] + ids
    df = df.melt(id_vars='radii')
    res.append(df.copy())

fig, axs = plt.subplots(2,2)
for dat,ax,spl in zip(res, axs.flat, names):
    sns.lineplot(data=dat, hue='variable', x='radii', y='value', ax=ax)
    ax.plot(dat.radii, np.repeat(0, len(dat.radii)), color='black', linestyle='dotted')
    ax.set_title(spl)
fig.tight_layout()
fig.show()

dat = res[-1][dat.radii < 20]
sns.lineplot(data=dat, hue='variable', x='radii', y='value')
plt.plot(dat.radii.values, np.repeat(0, len(dat.radii)), color='black', linestyle='dotted')
plt.show()

# %%
from astropy.stats import RipleysKEstimator
res_even = so.uns['even']['ripleysK']['meta_id_0_K_ripley']

area = even.x.max() * even.y.max()
radii = res_even.index
Kest = RipleysKEstimator(area=area,
                        x_max=even.x.max(),
                        x_min=0,
                        y_max=even.y.max(),
                        y_min=0)
plt.plot(radii, Kest(data=even[['x','y']], radii=radii, mode='ripley'),
         label=r'$K_{ripley}$')
plt.show()