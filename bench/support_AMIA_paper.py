# %%
import spatialHeterogeneity as sh
from spatialOmics import SpatialOmics
import matplotlib.pyplot as plt
from pathlib import Path

# %%
import pickle
with open(Path('~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/imc_full.pkl').expanduser(), 'rb') as f:
    imc = pickle.load(f)

f_mibi = Path('~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/mibi-final-no-images.h5py').expanduser()
mibi = SpatialOmics.from_h5py(f_mibi)
mibi.images = {}

# %% md
# However, real data examples convincingly showing the utility of these metrics are lacking. In particular, considering
# the 'immune infiltration' score implemented (Supp Fig 7), a minimum addition would be to quantitatively show how the
# software can be used to separate immune infiltrated and immune excluded samples.

# Infiltration
# Patients with less than 250 immune cells (N = 6) were defined as cold. Patients with a mixing score < 0.22 (N = 15)
# were defined as compartmentalized and the rest of the patients (N = 20) were defined as mixed

# selected cores shown in publication
spls = [4, 12, 24]  # selected examples from Keren paper
fig, axs = plt.subplots(1,3, figsize=(12,4))
for i,ax in zip(spls, axs.flat):
    sh.pl.spatial(mibi, str(i), 'tumorYN', ax=ax)
    ax.invert_yaxis()
fig.tight_layout()
fig.show()

for i in spls:
    mibi.spl.loc[str(i), 'numberOfImmuneCells'] = (mibi.obs[str(i)].tumorYN == 0).sum()
    sh.neigh.infiltration(mibi, str(i), 'tumorYN', interaction1=(1,0), interaction2=(0,0), graph_key='contact')
    sh.neigh.infiltration(mibi, str(i), 'tumorYN', interaction1=(1, 0), interaction2=(0, 0), graph_key='radius', local=True)

fig, axs = plt.subplots(2,3, figsize=(12,8))
for i,ax in zip(spls, axs.T):
    i = str(i)
    sh.pl.spatial(mibi, i, 'tumorYN', ax=ax[0])
    g = ax[1].hist(mibi.obs[i].infiltration, bins=100)
    ax[0].set_title(f'{i} ({mibi.spl.loc[i].numberOfImmuneCells:.0f}), infiltration: {mibi.spl.loc[i].infiltration:.3f}')
fig.tight_layout()
fig.show()

# plot all cores with infiltration
fig, axs = plt.subplots(8,5, figsize=(5*4, 8*4))
for i,ax in zip(mibi.spl.index, axs.flat):
    sh.neigh.infiltration(mibi, str(i), 'tumorYN', interaction1=(1, 0), interaction2=(0, 0), graph_key='contact')
    sh.pl.spatial(mibi, i, 'tumorYN', ax=ax, cbar=False)

    n = mibi.spl.loc[str(i), 'numberOfImmuneCells'] = (mibi.obs[str(i)].tumorYN == 0).sum()
    inf = mibi.spl.loc[i, 'infiltration']
    t_class = None
    if n < 250:
        t_class = 'cold'
    elif inf < 0.22:
        t_class = 'compartmentalized'
    elif inf >= 0.22:
        t_class = 'mixed'
    ax.set_title(f"{i}, {t_class}, {inf:.3f}")
fig.tight_layout()
fig.savefig(Path('~/Downloads/mibi.pdf').expanduser())

# mixed cores
spls = [39, 33, 41, 29]
for i in spls:
    sh.neigh.infiltration(mibi, str(i), 'tumorYN', interaction1=(1, 0), interaction2=(0, 0), graph_key='radius', local=True)

fig, axs = plt.subplots(2,4, figsize=(4*4,2*4))
for i,ax in zip(spls, axs.T):
    i = str(i)
    sh.pl.spatial(mibi, i, 'tumorYN', ax=ax[0])
    g = ax[1].hist(mibi.obs[i].infiltration, bins=100)
    ax[0].set_title(f'{i} ({mibi.spl.loc[i].numberOfImmuneCells:.0f}), infiltration: {mibi.spl.loc[i].infiltration:.3f}')
fig.tight_layout()
fig.show()

# compartmentalized cores
spls = [10, 13, 5, 3, 16]
for i in spls:
    sh.neigh.infiltration(mibi, str(i), 'tumorYN', interaction1=(1, 0), interaction2=(0, 0), graph_key='radius', local=True)

fig, axs = plt.subplots(2,5, figsize=(4*5,2*4))
for i,ax in zip(spls, axs.T):
    i = str(i)
    sh.pl.spatial(mibi, i, 'tumorYN', ax=ax[0])
    g = ax[1].hist(mibi.obs[i].infiltration, bins=100)
    ax[0].set_title(f'{i} ({mibi.spl.loc[i].numberOfImmuneCells:.0f}), infiltration: {mibi.spl.loc[i].infiltration:.3f}')
fig.tight_layout()
fig.show()


# %% imc

isImmune = list(range(7))
mapping = {i:0 for i in isImmune}
isTumor = list(range(14,27))
mapping.update({i:1 for i in isTumor})

for i in imc.spl.index:
    imc.obs[i]['tumorYN'] = imc.obs[i].meta_id.map(mapping).fillna(3).astype(int).astype('category')

for i in imc.spl.index:
    imc.spl.loc[i, 'nCells'] = len(imc.obs[i])
    n = imc.spl.loc[i, 'nImmuneCells'] = (imc.obs[i].tumorYN == 0).sum()
    imc.spl.loc[i, 'nTumorCells'] = (imc.obs[i].tumorYN == 1).sum()

    if n == 0 or (imc.obs[i].tumorYN == 1).sum() == 0:
        continue
    sh.neigh.infiltration(imc, str(i), 'tumorYN', interaction1=(1, 0), interaction2=(0, 0), graph_key='contact')

# FIND SUITABLE CORES TO SHOW
spl = imc.spl.copy()
spl['frac_immune'] = spl.nImmuneCells / spl.nCells
spl['frac_tumor'] = spl.nTumorCells / spl.nCells
spl = spl[~spl.infiltration.isna()]

sel = spl[(spl.nCells > 1500) & (spl.frac_immune > .33) & (spl.frac_tumor > 0.33)]
sel = sel[sel.numberOfImmuneCells > 250*1500/2000]

sel = sel.sort_values(['infiltration'])
sel = sel[['infiltration', 'frac_immune', 'frac_tumor', 'nCells']]

nrow,ncol = 4,5
fig, axs = plt.subplots(nrow,ncol, figsize=(4*ncol, nrow*4))
for i,ax in zip(sel.index, axs.flat):
    sh.pl.spatial(imc, i, 'tumorYN', ax=ax, cbar=True)
    ax.set_title(f'{i}, infiltration: {imc.spl.infiltration.loc[i]:.3f}')
fig.tight_layout()
fig.show()

# %%
# 1. The functions of framework are described, but with no validation. My questions is, how to tell the analysis is
# biologically valid? The authors should explain the presented results in more details and clarify this point specifically.

import seaborn as sns
sel = spl[(spl.nCells > 2000) & (spl.frac_tumor > 0.33)]
sel = sel[['grade', 'patient_status', 'HR_status', 'ER_status', 'PR_status',
       'HER2_status', 'os_month', 'nCells', 'nImmuneCells', 'nTumorCells',
       'frac_immune', 'frac_tumor', 'infiltration']]
sel = sel[~sel.HR_status.isna()]
sel = sel[sel.nImmuneCells > 250]
sel['tumor_class'] = 'mixed'
sel['tumor_class'][sel.infiltration < 0.22] = 'compartmentalized'

fig, axs = plt.subplots(2,3,figsize=(12,8))
feat = ['grade', 'HR_status', 'ER_status', 'PR_status', 'HER2_status']
for ax, x in zip(axs.flat, feat):
    sns.boxplot(data=sel, x=x, y='infiltration', ax=ax)
sns.regplot(data=sel[sel.tumor_class == 'mixed'], x='os_month', y='infiltration', ax=axs[-1,-1])
sns.regplot(data=sel[sel.tumor_class != 'mixed'], x='os_month', y='infiltration', ax=axs[-1,-1])
fig.tight_layout()
fig.show()

# %%
sel = spl[(spl.nCells > 1500) & (spl.frac_immune > .15) & (spl.frac_tumor > 0.15)]
sel = sel[sel.nImmuneCells > 250*1500/2000]

sel = sel.sort_values(['infiltration'])
sel = sel[['infiltration', 'frac_immune', 'frac_tumor', 'nCells']]



