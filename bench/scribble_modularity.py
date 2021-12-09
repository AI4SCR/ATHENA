import networkx as nx
import networkx.algorithms.community as nx_comm
import spatialHeterogeneity as sh

# %% networkx example
G = nx.barbell_graph(3, 0)
nx_comm.modularity(G, [{0, 1, 2}, {3, 4, 5}])
nx_comm.modularity(G, nx_comm.label_propagation_communities(G))

# %%
so = sh.dataset.imc()
spl = list(so.obs.keys())[0]

# %% manual
communities=[]
for _,obs in so.obs[spl].groupby('meta_id'):
    communities.append(set(obs.index))

nx_comm.modularity(so.G[spl]['contact'], communities)

# %% framework
sh.metrics.modularity(so, spl, 'meta_id', graph_key='contact')
so.spl.loc[spl]

