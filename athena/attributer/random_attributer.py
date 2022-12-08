from .base_attributer import BaseAttributer
import numpy as np
import pandas as pd
import networkx as nx

class randomAttributer(BaseAttributer):

    def __init__(self,
                 so,
                 spl: str,
                 graph_key: str,
                 config: dict) -> None:
        """
        Attributer class constructor. Assigns uniform [0, 1) random values to each attribute. 
        `config` must a dict with the following structure;

        `config = {'n_attrs': n_attrs}`

        where `n_attrs` is the number of random attributes to generate. 
        """
        self.features_type = 'random'
        super().__init__(so, spl, graph_key, config)

    def __call__(self) -> None:
        """
        Generates random features and atributes them to nodes in so.G[spl][graph_key]

        Returns:
            None. Attributes are saved to `so`.
        """
        
        # Chek config. If no assertions are raised, return number of attrs. 
        n_attrs = self.check_config()

        # Get index values form so
        index = self.so.obs[self.spl].index.values

        # Sample and put values into dict
        attrs = pd.DataFrame(np.random.rand(len(index), n_attrs), index=index).to_dict('index')

        # Check if the nodes already have attributes. If yes, clear them.
        self.clear_node_attrs()

        # Assign attrs to graph
        nx.set_node_attributes(self.so.G[self.spl][self.graph_key], attrs)

    def check_config(self) -> int:
        """
        Check config integrity.

        Returns:
            Number of attributes to be generated. 
        """
        assert type(self.config['n_attrs']) is int, "Number of attributes must be an integer."
        assert self.config['n_attrs'] > 0, "Number of attributes connot be zero."
        return self.config['n_attrs']