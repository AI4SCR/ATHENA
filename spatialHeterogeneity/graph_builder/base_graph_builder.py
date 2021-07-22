import networkx as nx
import numpy as np
import pandas as pd
from ..utils.tools.graph import df2node_attr

class BaseGraphBuilder:

    def __init__(self, config: dict):
        '''\
        Contact-Graph Builder constructor

        Parameters
        ----------
        config: dict
            Dictionary containing `builder_params` and optionally `cellmask_file` if graph should be constructed from cellmask file alone.
        '''
        self.config = config
        self.ndata = None
        self.edata = None
        self.graph = nx.Graph()

    def __call__(self, ndata: pd.DataFrame, edata: pd.DataFrame = None, topo_data = None):
        self.ndata = ndata
        self.edata = edata

        self._add_nodes()
        self._add_nodes_attr()

        if edata is None:
            self._build_topology(topo_data = topo_data)
        else:
            self._add_edges()
            self._add_edges_attr()

        return self.graph

    def _add_nodes(self):
        self.graph.add_nodes_from(self.ndata.index)

    def _add_nodes_attr(self):
        attr = df2node_attr(self.ndata)
        nx.set_node_attributes(self.graph, attr)

    def _add_edges(self):
        self.graph.add_edges_from(self.edata.index)

    def _add_edges_attr(self):
        attr = df2node_attr(self.edata)
        nx.set_edge_attributes(self.graph, attr)

    def _build_topology(self, **kwargs):
        raise NotImplementedError('Implemented in subclasses.')

    # Convenient method to build graph from cellmask
    @classmethod
    def from_mask(cls, config: dict, mask: np.ndarray):

        # load required dependencies
        try:
            import numpy as np
            from skimage.io import imread
            # from skimage.measure import regionprops
            from skimage.measure import regionprops_table
        except ImportError:
            raise ImportError(
                'Please install the skimage: `conda install -c anaconda scikit-image`.')

        instance = cls(config)

        # extract location
        ndata = regionprops_table(mask, properties=['label', 'centroid'])

        ndata = pd.DataFrame.from_dict(ndata)
        ndata.columns = ['cell_id', 'y', 'x'] # NOTE: axis 0 is y and axis 1 is x
        ndata.set_index('cell_id', inplace=True)
        ndata.sort_index(axis=0, ascending=True,inplace=True)

        return instance(ndata, topo_data={'mask': mask})

