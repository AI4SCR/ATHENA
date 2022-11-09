import networkx as nx
import numpy as np

from skimage.morphology import binary_dilation

from .base_graph_builder import BaseGraphBuilder
from .constants import EDGE_WEIGHT
from .constants import DILATION_KERNELS

from tqdm import tqdm


# %%
def dilation(args) -> list:
    """Compute dilation of a given object in a segmentation mask

    Args:
        args: masks, obj and dilation kernel

    Returns:

    """
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
        """Base-Graph Builder constructor

        Args:
            config: Dictionary containing a dict called `builder_params` that provides function call arguments to the build_topology function
        """
        super().__init__(config)

    def _build_topology(self, topo_data: dict, **kwargs) -> None:
        """Build topology using pixel expansion of segmentation masks provided by topo_data['mask']. Masks that overlap after expansion are connected in the graph.

        Args:
            topo_data: dict providing the segmentation mask under key 'mask'

        Returns:

        """

        # type hints
        self.graph: nx.Graph
        
        # Instance variable to in function variable
        params = self.config['builder_params']

        # Function param to function variable
        mask = topo_data['mask']

        # TODO: change masks such that only the masks from the desired cell types persist in the local variable mask

        # If dilation_kernel instantiate kernel object, else raise error
        if params['dilation_kernel'] in DILATION_KERNELS:
            kernel = DILATION_KERNELS[params['dilation_kernel']](params['radius'])
        else:
            raise ValueError(
                f'Specified dilate kernel not available. Please use one of {{{", ".join(DILATION_KERNELS)}}}.')

        # Context: Each pixel that belongs to cell i, was value i in the mask. 
        # get object ids, 0 is background.
        objs = np.unique(mask)
        objs = objs[objs != 0]

        # compute neighbours (object = the mask of a single cell)
        edges = []
        for obj in tqdm(objs):
            # This creates the augemnted object mask in a bool array from
            dilated_img = binary_dilation(mask == obj, kernel)

            cells = np.unique(mask[dilated_img]) # This identifies the intersecting objects
            cells = cells[cells != obj]  # remove object itself
            cells = cells[cells != 0]  # remove background

            # Appends a list of the edges found at this iteration
            edges.extend([(obj, cell, {EDGE_WEIGHT: 1}) for cell in cells])

        # Adds edges to instance variable graph object
        self.graph.add_edges_from(edges)

# %%
