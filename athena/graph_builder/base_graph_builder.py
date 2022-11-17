import abc

import networkx as nx
import numpy as np
import pandas as pd
from ..utils.tools.graph import df2node_attr
from abc import ABC

class BaseGraphBuilder(ABC):

    def __init__(self, config: dict):
        """Base-Graph Builder constructor

        Args:
            config: Dictionary containing a dict called `builder_params` that provides function call arguments to the build_topology function
        """

        self.config = config
        self.ndata = None
        self.edata = None
        self.graph = nx.Graph()

    def __call__(self, ndata: pd.DataFrame, edata: pd.DataFrame = None, topo_data: dict = None) -> nx.Graph:
        """Builds graph

        Args:
            ndata: dataframe with node data, index is the node
            edata: dataframe with edge data, index specifies edges, i.e. (node1, node2)
            topo_data: dict with additional data for graph construction (necessary for contact graph)

        Returns:
            nx.Graph
        """
        
        # Write input parameters to instance varaibles
        self.ndata = ndata
        self.edata = edata

        # Add nodes and node attributes to graph
        self._add_nodes()
        self._add_nodes_attr()

        # If `edata` is given add edges and edge attributes to graph. Else build graph.
        if edata is None:
            self._build_topology(topo_data=topo_data)
        else:
            self._add_edges()
            self._add_edges_attr()

        return self.graph

    def _add_nodes(self):
        """Adds nodes in ndata to graph

        Returns:

        """
        self.graph.add_nodes_from(self.ndata.index)

    def _add_nodes_attr(self) -> None:
        """Adds node attributes in ndata to graph

        Returns:

        """
        attr = df2node_attr(self.ndata)
        nx.set_node_attributes(self.graph, attr)

    def _add_edges(self) -> None:
        """Adds edges in edata to graph

        Returns:

        """
        self.graph.add_edges_from(self.edata.index)

    def _add_edges_attr(self) -> None:
        """Adds edge attributes in edata to graph

        Returns:

        """
        attr = df2node_attr(self.edata)
        nx.set_edge_attributes(self.graph, attr)

    @abc.abstractmethod
    def _build_topology(self, **kwargs) -> None:
        """Builds graph topology. Implemented in subclasses.

        Args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError('Implemented in subclasses.')

    # Convenient method to build graph from cellmask
    @classmethod
    def from_mask(cls, config: dict, mask: np.ndarray) -> nx.Graph:
        """Construct graph topology from segmentation masks.

        Args:
            config: config: Dictionary containing a dict called `builder_params` that provides function call arguments to the build_topology function
            mask: image file that provides the image segmentation

        Returns:
            nx.Graph
        """

        # load required dependencies
        try:
            import numpy as np
            from skimage.io import imread
            from skimage.measure import regionprops_table
        except ImportError:
            raise ImportError(
                'Please install the skimage: `conda install -c anaconda scikit-image`.')

        # This creates a new instance of the `BaseGraphBuilder` class
        instance = cls(config)

        # Extract location:
        # Compute centroid from image of labels (mask) and return them as a pandas-compatible table.
        # The table is a dictionary mapping column names to value arrays.
        ndata = regionprops_table(mask, properties=['label', 'centroid'])

        # The cell_id is the number that identifies a cell in the mask and is set to be the index of ndata
        ndata = pd.DataFrame.from_dict(ndata)
        ndata.columns = ['cell_id', 'y', 'x']  # NOTE: axis 0 is y and axis 1 is x
        ndata.set_index('cell_id', inplace=True)
        ndata.sort_index(axis=0, ascending=True, inplace=True)

        # Here we use the `__call__` method on our new instance with the ndata included
        return instance(ndata, topo_data={'mask': mask})
