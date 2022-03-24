# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import spatialHeterogeneity as sh

# %%
f_imc = Path('~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/imc_full.pkl').expanduser()
with open(f_imc, 'rb') as f:
    imc = pickle.load(f)

# f_mibi = Path('~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/mibi-final-no-images.h5py').expanduser()
# mibi = SpatialOmics.from_h5py(f_mibi)

# %% filter cores
spl = imc.spl.copy()
sel = spl[(spl.nCells > 1500) & (spl.frac_immune > .15) & (spl.frac_tumor > 0.15)]
sel = sel[sel.nImmuneCells > 250 * 1500 / 2000]  # exclude are cold tumors (i.e. only few immune cells)


# %%
# from matplotlib.colors import ListedColormap
# cmap = plt.get_cmap('Paired')
# newCmap = []
# newCmap.extend([cmap(i) for i in (2,4,0)])
# newCmap = ListedColormap(newCmap)
# imc.uns['cmaps']['tumorYN'] = newCmap
# imc.uns['cmap_labels']['tumorYN'] = {0:'immune', 1: 'tumor', 2: 'other'}

# %%
def plot_infiltration(selected):
    metric = 'infiltration'
    root = Path(
        f'~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/review_support_{metric}/').expanduser()
    root.mkdir(parents=True, exist_ok=True)

    import seaborn as sns
    for spl in tqdm(selected):
        fig, axs = plt.subplots(1, 2)
        sh.pl.spatial(imc, spl, 'tumorYN', ax=axs[0], node_size=1)

        data = imc.obs[spl][[metric]]
        sns.histplot(data=data, x=metric, ax=axs[1], bins=100)

        inf = imc.spl.loc[spl][metric]
        yMax = axs[1].get_ylim()[1]
        axs[1].plot([inf, inf], [0, yMax], 'r-')

        axs[1].set_title(f'{spl}, global: {inf:.4f}')
        fig.tight_layout()
        fig.show()
        break
        fig.savefig(root / (spl + '.pdf'))
        plt.close(fig)


def plot_entropy(selected, entropy):
    metric_obs = entropy
    metric_spl = '_'.join(entropy.split('_')[:-1])
    root = Path(
        f'~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/review_support_{metric_obs}/').expanduser()
    root.mkdir(parents=True, exist_ok=True)

    import seaborn as sns
    i = 0
    for spl in tqdm(selected):
        fig, axs = plt.subplots(1, 2)
        sh.pl.spatial(imc, spl, 'tumorYN', ax=axs[0], node_size=1, background_color='black')

        data = imc.obs[spl][[metric_obs]]
        sns.histplot(data=data, x=metric_obs, ax=axs[1], bins=100)

        inf = imc.spl.loc[spl][metric_spl]
        yMax = axs[1].get_ylim()[1]
        axs[1].plot([inf, inf], [0, yMax], 'r-')

        axs[1].set_title(f'{spl}, global: {inf:.4f}')
        fig.tight_layout()
        # fig.show()
        # break
        fig.savefig(root / (str(i) + '_' + spl + '.pdf'))
        plt.close(fig)
        i = i + 1


def plot_entropy_meta_id(selected, entropy):
    metric_obs = entropy
    metric_spl = '_'.join(entropy.split('_')[:-1])
    root = Path(
        f'~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/review_support_{metric_obs}/').expanduser()
    root.mkdir(parents=True, exist_ok=True)

    import seaborn as sns
    i = 0
    for spl in tqdm(selected):
        fig, axs = plt.subplots(1, 2)
        sh.pl.spatial(imc, spl, 'meta_id', ax=axs[0], node_size=1, background_color='black')

        data = imc.obs[spl][[metric_obs]]
        sns.histplot(data=data, x=metric_obs, ax=axs[1], bins=50, kde=True)

        inf = imc.spl.loc[spl][metric_spl]
        yMax = axs[1].get_ylim()[1]
        axs[1].plot([inf, inf], [0, yMax], 'r-')

        axs[1].set_title(f'{spl}, global: {inf:.4f}')
        fig.tight_layout()
        # fig.show()
        # break
        fig.savefig(root / (str(i) + '_' + spl + '.pdf'))
        plt.close(fig)
        i = i + 1

def plot_entropy_tissue_type(selected, entropy):
    metric_obs = entropy
    metric_spl = '_'.join(entropy.split('_')[:-1])
    root = Path(
        f'~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/review_support_{metric_obs}/').expanduser()
    root.mkdir(parents=True, exist_ok=True)

    import seaborn as sns
    i = 0
    for spl in tqdm(selected):
        fig, axs = plt.subplots(1, 2)
        sh.pl.spatial(imc, spl, 'tissue_type', ax=axs[0], node_size=1, background_color='black')

        data = imc.obs[spl][[metric_obs]]
        sns.histplot(data=data, x=metric_obs, ax=axs[1], bins=50, kde=True)

        inf = imc.spl.loc[spl][metric_spl]
        yMax = axs[1].get_ylim()[1]
        axs[1].plot([inf, inf], [0, yMax], 'r-')

        axs[1].set_title(f'{spl}, global: {inf:.4f}')
        fig.tight_layout()
        # fig.show()
        # break
        fig.savefig(root / (str(i) + '_' + spl + '.pdf'))
        plt.close(fig)
        i = i + 1


# %% plot infiltration
plot_infiltration(sel.index)

# %% plot shannon
# sel = sel.sort_values('shannon_tumorYN')
# plot_entropy(sel.index, 'shannon_tumorYN_contact')

# %% plot shannon meta_id

sel = sel.sort_values('shannon_meta_id')
plot_entropy_meta_id(sel.index, 'shannon_meta_id_contact')
plot_entropy_meta_id(sel.index, 'shannon_meta_id_radius')

# %% plot shannon tissue_type

sel = sel.sort_values('shannon_tissue_type')
plot_entropy_tissue_type(sel.index, 'shannon_tissue_type_contact')
plot_entropy_tissue_type(sel.index, 'shannon_tissue_type_radius')

# %% modularity
root = Path(f'~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/review_support_modularity/').expanduser()
root.mkdir(parents=True, exist_ok=True)

data = imc.spl.copy()
feat = ['grade', 'ER_status', 'PR_status',
        'HER2_status', 'os_month', 'HR_status', 'HER2', 'ER', 'PR', 'modularity_tumorYN_res1']
data = data.loc[sel, feat]


def map_HER2(x):
    if type(x.HER2_status) == float:
        if x.HER2 == '+':
            return 'positive'
        elif x.HER2 == '-':
            return 'negative'
        else:
            return x.HER2
    return x.HER2_status


data.HER2_status = data.apply(map_HER2, axis=1)
data.grade = data.grade.astype('str')

fig, axs = plt.subplots(3, 2, figsize=(2 * 4, 3 * 4))
for ax, x in zip(axs.flat, ['grade', 'ER_status', 'PR_status',
                            'HER2_status', 'HR_status']):
    sns.boxplot(data=data, y='modularity_tumorYN_res1', ax=ax, x=x)
    sns.swarmplot(data=data, y='modularity_tumorYN_res1', ax=ax, x=x)
fig.tight_layout()
fig.show()
fig.savefig(root / 'meta_data.pdf')

# %% selected cores
sel2 = ['slide_27_Cy10x8',
        'SP43_144_X15Y1',
        'SP43_79_X7Y4',
        'SP43_270_X6Y8',
        'SP43_193_X9Y7',
        'SP43_144_X15Y1',
        'SP43_116_X3Y4',
        'SP42_179_X13Y8',
        'SP42_70_X5Y3',
        'SP41_239_X11Y3_165',
        'SP41_191_X15Y7',
        'SP41_186_X5Y4',
        'SP43_193_X9Y7',
        'slide_59_Cy8x2',
        'slide_59_Cy7x8']

# %% infiltration

root = Path(
    f'~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/final_plots/').expanduser()
root.mkdir(parents=True, exist_ok=True)

# inter = ['SP41_191_X15Y7', 'SP41_186_X5Y4']
inter = ['SP41_191_X15Y7']
loc_inter = ['SP41_220_X10Y5']
comp = ['SP43_116_X3Y4']

import copy

cmap = copy.copy(plt.get_cmap("plasma"))
naColor = 'lightgrey'
cmap.set_bad(naColor)
imc.uns['cmaps']['infiltration_norm'] = cmap
imc.uns['cmaps']['infiltration'] = cmap

fig, axs = plt.subplots(3, 3, figsize=(3 * 4, 3 * 4))
for axc, spl in zip(axs.T, comp + loc_inter + inter):
    sh.pl.spatial(imc, spl, 'tumorYN', mode='mask', ax=axc[0], background_color='black')

    x = imc.obs[spl].infiltration
    imc.obs[spl]['infiltration_norm'] = x / x.max()

    sh.pl.spatial(imc, spl, 'infiltration_norm', mode='mask', ax=axc[1], background_color='black')
    x = imc.obs[spl].infiltration_norm
    sns.histplot(x=x, bins=50, ax=axc[2], kde=True, fill=False)
    # sns.kdeplot(x=x, ax=axc[2], cut=0, c='orange')

fig.tight_layout()
fig.show()
fig.savefig(root / f'infiltration_naIs{naColor}.pdf')

# %% interactions
root = Path(
    f'~/Library/CloudStorage/Box-Box/AI4SCR_group/Papers/ATHENA/Review/final_plots/').expanduser()
root.mkdir(parents=True, exist_ok=True)

fig, axs = plt.subplots(2, 3, figsize=(3 * 4 * 1.25, 2 * 4 * 1.25))
for axc, spl in zip(axs.T, comp + loc_inter + inter):
    sh.pl.spatial(imc, spl, 'meta_id', mode='mask', ax=axc[0], background_color='black')

    sh.pl.interactions(imc, spl, 'meta_id', mode='proportion', prediction_type='observation', graph_key='contact',
                       ax=axc[1], cbar=False)

fig.tight_layout()
fig.show()
fig.savefig(root / f'interactions.pdf')

# %% shannon
import numpy as np

# compute difference of global and local entropy
for spl in sel.index:
    ent = imc.obs[spl].shannon_meta_id_contact
    ent = ent[~np.isnan(ent)]
    ent = ent[ent!=0]
    imc.spl.loc[spl, 'shannon_meta_id_local_median'] = np.median(ent)

sel = imc.spl.loc[sel.index].copy()
sel['diff'] = sel.shannon_meta_id - sel.shannon_meta_id_local_median
res = sel.sort_values('shannon_meta_id')[['shannon_meta_id', 'diff', 'shannon_meta_id_local_median']]

# %%
