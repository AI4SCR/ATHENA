import abc
import networkx as nx
import pandas as pd
from abc import ABC
from skimage.measure import regionprops_table


class BaseGraphBuilder(ABC):
    def __init__(self):
        """Base-Graph Builder constructor"""

        self.EDGE_WEIGHT = 1
        self.graph = nx.Graph()

    @abc.abstractmethod
    def build_graph(self, *args, **kwargs):
        """Builds graph topology. Implemented in subclasses.

        Args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError("Implemented in subclasses.")

    @staticmethod
    def object_coordinates(mask) -> pd.DataFrame:
        """Compute centroid from image of labels (mask) and return them as a pandas-compatible table."""
        # The table is a dictionary mapping column names to value arrays.
        ndata = regionprops_table(mask, properties=["label", "centroid"])

        ndata = pd.DataFrame.from_dict(ndata)
        ndata.columns = ["object_id", "y", "x"]  # NOTE: axis 0 is y and axis 1 is x
        ndata.set_index("object_id", inplace=True)
        ndata.sort_index(axis=0, ascending=True, inplace=True)

        return ndata
