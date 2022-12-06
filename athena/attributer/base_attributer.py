from abc import ABC

class BaseAttributer:
    
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
        self.config = config

    @abc.abstractmethod
    def __call__(self):
        """Attributes graph. Implemented in subclasses.
        """
        raise NotImplementedError('Implemented in subclasses.')