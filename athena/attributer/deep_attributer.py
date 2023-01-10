from .base_attributer import BaseAttributer
import numpy as np
import pandas as pd
import networkx as nx

class deepAttributer(BaseAttributer):

    def __init__(self,
                 so,
                 spl: str,
                 graph_key: str,
                 config: dict) -> None:
        """
        Attributer class constructor. TODO: Compleate description, specify config structure. 
        """
        super().__init__(so, spl, graph_key, config)

    def __call__(self) -> None:
        """
        Generates deep features and atributes them to nodes in so.G[spl][graph_key]. 

        Returns:
            None. Attributes are saved to `so`.
        """
        
        # TODO: implement this
        raise NotImplementedError('Not implemented yet.')

    def check_config(self) -> int:
        """
        Check config integrity. TODO: Copleate description. 
        """

        # TODO: implement this
        raise NotImplementedError('Not implemented yet.')