import networkx as nx
import numpy as np

from skimage.morphology import binary_dilation

from .base_graph_builder import BaseGraphBuilder
from .constants import EDGE_WEIGHT
from .constants import DILATION_KERNELS

from tqdm import tqdm

# %%
def dilation(args):
    mask, obj, kernel = args
    dilated_img = binary_dilation(mask == obj, kernel)
    cells = np.unique(mask[dilated_img])
    cells = cells[cells != obj]  # remove object itself
    cells = cells[cells != 0]  # remove background
    return [(obj, cell, {EDGE_WEIGHT: 1}) for cell in cells]


# %%
class ContactGraphBuilder(BaseGraphBuilder):
    '''Contact-Graph class.

    Build contact graph based on pixel expansion of cell masks.
    '''

    def __init__(self, config: dict):
        '''Contact-Graph Builder constructor

        Parameters
        ----------
        config: dict
            Dictionary containing dict `builder_params` that specifies the dilation_kernel, radius and `cellmask_file`.
        '''
        super().__init__(config)

    def _build_topology(self, topo_data, **kwargs):
        '''Build topology using pixel expansion of cell masks. Cells which cell masks overlap after expansion are connected in the graph.
        '''

        # type hints
        self.graph: nx.Graph

        params = self.config['builder_params']

        mask = topo_data['mask']

        if params['dilation_kernel'] in DILATION_KERNELS:
            kernel = DILATION_KERNELS[params['dilation_kernel']](params['radius'])
        else:
            raise ValueError(
                f'Specified dilate kernel not available. Please use one of {{{", ".join(DILATION_KERNELS)}}}.')

        # get object ids, 0 is background.
        objs = np.unique(mask)
        objs = objs[objs != 0]

        # compute neighbours
        edges = []
        for obj in tqdm(objs):
            dilated_img = binary_dilation(mask == obj, kernel)
            cells = np.unique(mask[dilated_img])
            cells = cells[cells != obj]  # remove object itself
            cells = cells[cells != 0]  # remove background
            edges.extend([(obj, cell, {EDGE_WEIGHT: 1}) for cell in cells])

        self.graph.add_edges_from(edges)
