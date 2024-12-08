# Migration to AnnData Notes
def richness(so, spl: str, attr: str, *, local=True, key_added=None, graph_key='knn', inplace=True) -> None:
    if key_added is None:
        key_added = 'richness'
        key_added = f'{key_added}_{attr}'
        if local:
            key_added += f'_{graph_key}'

    metric = _richness
    kwargs_metric = {}

    return _compute_metric(so=so, spl=spl, attr=attr, key_added=key_added, graph_key=graph_key, metric=metric,
                           kwargs_metric=kwargs_metric,
                           local=local, inplace=inplace)

def _compute_metric(so, spl: str, attr, key_added, graph_key, metric, kwargs_metric, local, inplace=True):
    """Computes the given metric for each observation or the sample
    """

    # generate a copy if necessary
    so = so if inplace else so.copy()

    # extract relevant categorisation
    data = get_data_from_anndata(ad, spl, attr)
    # data = so.obs[spl][attr]
    if not is_categorical(data):
        raise TypeError('`attr` needs to be categorical')

    def save_obs_results_to_anndata():
        if np.ndim(res[0]) > 0:
            res = pd.DataFrame(res, index=observation_ids)
            if spl not in so.obsm:
                so.obsm[spl] = {}
            so.obsm[spl][key_added] = res
        else:
            res = pd.DataFrame({key_added: res}, index=observation_ids)
            if key_added in ad.obs:  # drop previous computation of metric
                ad.obs.drop(key_added, axis=1, inplace=True)
            so.obs[spl] = pd.concat((ad.obs, res), axis=1)
    
    def save_spl_results_to_anndata():
        if np.ndim(res) > 0:
            if spl not in so.uns:
                so.uns[spl] = {}
            ad.uns[spl][key_added] = res
        else:
            ad.spl.loc[spl, key_added] = res

    if local:
        # get graph
        g = get_graph_from_anndata(ad, spl, graph_key)
        # g = so.G[spl][graph_key]

        # compute metric for each observation
        res = []
        observation_ids = get_obs_ids(ad, spl)
        #observation_ids = so.obs[spl].index
        for observation_id in observation_ids:
            n = list(g.neighbors(observation_id))
            if len(n) == 0:
                res.append(0)
                continue
            counts = Counter(data.loc[n].values)
            res.append(metric(counts, **kwargs_metric))
        
        
    else:
        res = metric(Counter(data), **kwargs_metric)

    if not inplace:
        return so