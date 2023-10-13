import numpy as np
from skimage.morphology import binary_dilation
from .base_graph_builder import BaseGraphBuilder
from ..utils.default_configs import EDGE_WEIGHT
from ..utils.default_configs import DILATION_KERNELS
from tqdm import tqdm


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


class ContactGraphBuilder(BaseGraphBuilder):
    """Contact-Graph class.

    Build contact graph based on pixel expansion of cell masks.
    """

    def __init__(self, config: dict):
        """Base-Graph Builder constructor

        Args:
            config: Dictionary containing a dict called `builder_params` that provides function call arguments to the build_topology function

        Default config:
            config = {'builder_params':
                    {'dilation_kernel': 'disk',
                     'radius': 4,
                     'include_self':False},
                'concept_params':
                    {'filter_col':None,
                    'labels':None},
                'coordinate_keys': ['x', 'y'],
                'mask_key': 'cellmasks'}
        """
        self.builder_type = "contact"
        super().__init__(config)

    def __call__(self, so, spl):
        """Build topology using pixel expansion of segmentation masks provided by topo_data['mask']. Masks that overlap after expansion are connected in the graph.

        Args:
            so: spatial omic object
            spl: sting identifying the sample in the spatial omics object

        Returns:
            Graph and key to graph in the spatial omics object

        """

        # Unpack parameters for building
        if self.config["build_concept_graph"]:
            filter_col = self.config["concept_params"]["filter_col"]
            labels = self.config["concept_params"]["labels"]
            self.look_for_miss_specification_error(so, spl, filter_col, labels)

        mask_key = self.config["mask_key"]
        params = self.config["builder_params"]

        # Raise error if mask is not provided
        if mask_key is None:
            raise ValueError(
                "Contact-graph requires segmentation masks. To compute a contact graph please specify `the mask_key` to use in so.masks[spl]"
            )

        # Get masks
        mask = so.get_mask(spl, mask_key)

        # If a cell subset is well are specified then simplify the mask
        if self.config["build_concept_graph"]:
            # Get cell_ids of the cells that are in `labels`
            cell_ids = so.obs[spl].query(f"{filter_col} in @labels").index.values
            # Simplify masks filling it with 0s for cells that are not in `labels`
            mask = np.where(np.isin(mask, cell_ids), mask, 0)

        # If dilation_kernel instantiate kernel object, else raise error
        if params["dilation_kernel"] in DILATION_KERNELS:
            kernel = DILATION_KERNELS[params["dilation_kernel"]](params["radius"])
        else:
            raise ValueError(
                f'Specified dilate kernel not available. Please use one of {{{", ".join(DILATION_KERNELS)}}}.'
            )

        # Context: Each pixel that belongs to cell i, was value i in the mask.
        # get object ids, 0 is background.
        objs = np.unique(mask)
        objs = objs[objs != 0]

        # Add nodes to graph
        self.graph.add_nodes_from(objs)

        # compute neighbors (object = the mask of a single cell)
        edges = []
        for obj in tqdm(objs):
            # This creates the augmented object mask in a bool array from
            dilated_img = binary_dilation(mask == obj, kernel)

            cells = np.unique(mask[dilated_img])  # This identifies the intersecting objects
            cells = cells[cells != obj]  # remove object itself
            cells = cells[cells != 0]  # remove background

            # Appends a list of the edges found at this iteration
            edges.extend([(obj, cell, {EDGE_WEIGHT: 1}) for cell in cells])

        # Adds edges to instance variable graph object
        self.graph.add_edges_from(edges)

        # Include self edges if desired
        if (
            "include_self" in self.config["builder_params"]
            and self.config["builder_params"]["include_self"]
            and self.builder_type == "contact"
        ):
            edge_list = [(i, i) for i in self.graph.nodes]
            self.graph.add_edges_from(edge_list)

        return self.graph
