from .base_attributer import BaseAttributer
import numpy as np
import pandas as pd
import networkx as nx

class soAttributer(BaseAttributer):

    def __init__(self,
                 so,
                 spl: str,
                 graph_key: str,
                 config: dict) -> None:
        """
        Attributer class constructor. Gets attributes from `so`.
        """
        self.features_type = 'so'
        super().__init__(so, spl, graph_key, config)

    def __call__(self) -> None:
        # TODO: assign attributes to graph.
        # 0. Check config (that it is well defined). 
        # If no error is raised, slice and return data. 
        obs_df, X_df = self.check_config()

        # 1. Join data and transform into dictionary. 
        attrs = obs_df.merge(X_df, left_index=True, right_index=True, how='inner').to_dic('index')

        # 2. Add to node attributes of so.G[spl][graph_key]
        nx.set_node_attributes(self.so.G[self.spl][self.graph_key], attrs)

    def check_config(self) -> tuple:
        """
        Checks wehther config is well defined. If no error is raised then the data is sliced 
        according to the config. 

        Returns:
            - `obs_df`: sliced so.obs[spl] or empty df with index same as so.obs[spl]
            - `X_df`: sliced so.X[spl] or empty df with index same as so.X[spl]
        """

        # At least one of the config options must be set to true. Raise error otw. 
        if self.config['from_obs'] and self.config['from_X']:
            raise NameError('At least one should be true (config["from_obs"] or config["from_X"])')

        # Check self.config['from_obs'] if its set to true. 
        if self.config['from_obs']:
            # Raise error if list is empty
            if len(self.config['obs_cols']) == 0:
                raise NameError('self.config["obs_cols"] is empty. Please provide list of columns to be included.')

            # Raise error is not all column names given are in so.obs[self.spl].columns
            if not np.all(np.isin(self.config['obs_cols'], self.so.obs[self.spl].columns)):
                raise NameError(f'Not all elements provided in list config["obs_cols"] are in so.obs[spl].columns')

            # Subset obs[spl]
            obs_df = self.so.obs[self.spl][self.config['obs_cols']]
        else:
            obs_df = pd.DataFrame(index = self.so.obs[self.spl].index)

        # Check self.config['from_X'] if its set to true. 
        if self.config['from_X']:
            if self.config['X_cols'] != 'all':
                if not np.all(np.isin(self.config['X_cols'], self.so.X[self.spl].columns)):
                    raise NameError(f'Not all elements provided in list config["X_cols"] are in so.X[spl].columns')

            # Subset X[spl]
            X_df = self.so.X[self.spl][self.config['obs_cols']]
        else:
            X_df = pd.DataFrame(index = self.so.X[self.spl].index)

        return (obs_df, X_df)
