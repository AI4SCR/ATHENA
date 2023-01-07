from abc import ABC
import abc

class BaseAttributer(ABC):
    
    def __init__(self, 
                 so,
                 spl: str,
                 graph_key: str,
                 config: dict) -> None:
        """Base attributer constructor
        """
        
        self.so = so
        self.spl = spl
        self.graph_key = graph_key
        self.config = config['attrs_params']

    @abc.abstractmethod
    def __call__(self):
        """Attributes graph. Implemented in subclasses.
        """
        raise NotImplementedError('Implemented in subclasses.')

    def clear_node_attrs(self) -> None:
        # Get key from the first node
        any_node = list(self.so.G[self.spl][self.graph_key].nodes)[0]

        # If the any_node ditionary (attrs) is not empty then clear all nodes. 
        if len(self.so.G[self.spl][self.graph_key].nodes[any_node]) > 0:
            for node in self.so.G[self.spl][self.graph_key].nodes:
                self.so.G[self.spl][self.graph_key].nodes[node].clear()