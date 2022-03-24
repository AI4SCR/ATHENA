import pandas as pd
import numpy as np
from skimage import io
import time
import os
import copy

from tqdm import tqdm
import pickle

GRAPH_BUILDERS = None
GRAPH_BUILDER_DEFAULT_PARAMS = None

make_iterable = None
get_shape_outline, extract_mask_morph_feat = None, None

# %%
class IMCData():

    def __init__(self):

        self.graph_engine = 'networkx'
        self.random_seed = 42  # for reproducibility
        self.root = None  # data repo

        self.G = {}  # graphs
        self.meta = None  # container for cores meta data
        self.obs = {}  # container for cell level features of each core
        self.obsc = None  # container for core level features
        self.var = {}  # container with variable descriptions of X
        self.X = {}  # container for cell level expression data of each core
        self.patients = None  # container for patient level data
        self.uns = {}  # unstructured container
        self.cores = None  # list of cores in self.obsc.index

        self.tiffstacks = {}
        self.cellmasks = {}
        self.cellmask_outlines = {}
        self.tumor_stroma_mask = {}

    def add_tiffstacks(self, cores=None):
        """Add the tiffstacks of the selected cores"""
        cores = self._get_cores_subset(cores, self.tiffstacks.keys())

        # load images into dict
        print('\t...reading in tiffstacks')
        for idx, row in tqdm(cores.iterrows()):
            if not row.file_fullstack is None:
                self.tiffstacks.update({row.core: io.imread(row.file_fullstack)})
            else:
                print(f'No tiff_stack file specified for core {row.core}')

    def get_tiffstack(self, core):
        if core in self.tiffstacks:
            return self.tiffstacks[core]
        else:
            self.add_tiffstacks(core)
            return self.tiffstacks[core]

    def add_cellmasks(self, cores=None):
        """Add the cellmasks and cellmask outline of the selected cores"""
        cores = self._get_cores_subset(cores)

        # load images into dict
        print(f'\t...reading in cellmasks')
        for i, (core, data) in enumerate(cores.iterrows()):
            print(f'\t\t{core} ({i + 1}/{len(cores)}), {time.asctime()}')
            if data.file_cellmask:
                im = io.imread(data.file_cellmask).astype(np.uint32)
                self.cellmasks.update({core: im})
                self.cellmask_outlines.update({core: get_shape_outline(self.cellmasks[core])})
            else:
                print(f'{core}: cellmask file missing.')

    def get_cellmasks(self, core):
        if core in self.cellmasks:
            return self.cellmasks[core]
        else:
            self.add_cellmasks(core)
            return self.cellmasks[core]

    def add_tumor_stroma_mask(self, cores=None):
        """Add the tumor and stroma mask of the selected cores"""
        cores = self._get_cores_subset(cores)

        # load images into dict
        print('\t...reading in tumor stroma masks')
        for idx, row in tqdm(cores.iterrows()):
            if not row.file_tumor_stroma_mask is None:
                self.tumor_stroma_mask.update({row.core: io.imread(row.file_tumor_stroma_mask)})

    def add_graphs(self, cores=None, config=None, builder_type: str = None):

        cores = self._get_cores_subset(cores)

        # build graphs with own graph constructor class
        if builder_type not in GRAPH_BUILDERS:
            raise ValueError(f'Invalid graph builder type specified. Available options are {GRAPH_BUILDERS.keys()}')
        if config is None:
            # default config if not provided
            config = GRAPH_BUILDER_DEFAULT_PARAMS[builder_type]
            print('Building graph with default configuration:')
            print(config)

        for idx, row in cores.iterrows():
            print(f'\t...adding graph for core {row.core}, ({int(idx) + 1}/{len(cores)})')

            config['cellmask_file'] = row.file_cellmask
            graph_builder = GRAPH_BUILDERS[builder_type]
            g = graph_builder.from_cellmask(config)
            self.G.update({row.core: g})

    def _get_cores_subset(self, cores):
        """Returns a dataframe containing the selected cores only"""
        if cores is None:
            cores = self.meta
        else:
            cores = make_iterable(cores)
            assert all([c in self.meta.index.values for c in cores]), 'Not all cores in ad.meta of instance'
            cores = self.meta[self.meta.index.isin(cores)]

        # filter out cores that are already loaded
        is_loaded = self.cellmasks.keys()
        cores = cores[~cores.index.isin(is_loaded)]
        return cores

    def data_integrity_check(self):
        print('\t...check data integrity.')
        for core, data in tqdm(self.meta.iterrows()):
            mask_obj = np.unique(self.get_cellmasks(core))
            mask_obj = set(mask_obj[mask_obj != 0])

            graph_obj = set(self.G[core].nodes)

            # check that all objects are in the graphs
            if not mask_obj == graph_obj:
                print(f'mask objects and graph nodes not the same for {core}')

            # check that we have the same number of cells in the graph and in X
            if not len(self.X[core]) == len(self.G[core]):
                print(f'Number of cells in X and graph are not the same for {core}')

    def __str__(self):
        # TODO: Add number of cells, measures proteins, memory size
        s = f'''IMCData object
{len(self.G)} graphs, {len(self.meta)} cores, {len(self.patients)} patients
{self.obsc.shape[1]} features extracted from cores
{len(self.tiffstacks)} tiffstacks, {len(self.cellmasks)} cellmask, {len(self.tumor_stroma_mask)} stroma masks'''
        return s

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.meta)

    def to_pickle(self, fname='IMC.pkl'):
        fname = os.path.join(self.root, fname)
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
        print(f'File `{os.path.basename(fname)}` save at {os.path.dirname(fname)}')
        print(f'File size: {os.path.getsize(fname) / (1024 * 1024):.2f} MB')

    @classmethod
    def from_pickle(cls, file=None, cohort='basel', root=None):

        if root is None:
            root = os.path.join(os.path.expanduser('~/Documents/thesis/data/SingleCellPathologyLandscapeOfBreastCancer'))

        fname = os.path.join(root, cohort, file)
        with open(fname, 'rb') as f:
            ad = pickle.load(f)

        return ad

    def copy(self):
        """copy IMCData without copying graphs, masks and tiffstacks"""
        c = copy.copy(self)
        c.meta = copy.deepcopy(self.meta)
        c.obs = copy.deepcopy(self.obs)
        c.obsc = copy.deepcopy(self.obsc)
        c.var = copy.deepcopy(self.var)
        c.X = copy.deepcopy(self.X)
        c.patients = copy.deepcopy(self.patients)
        c.uns = copy.deepcopy(self.uns)
        c.cores = copy.deepcopy(self.cores)
        return c

    def deepcopy(self):
        return copy.deepcopy(self)

    def get_uns(self, metric=None, key=None, uns_path=None, cores=None):
        if cores is None:
            cores = self.cores

        cores = make_iterable(cores)

        if uns_path:
            path = uns_path.split('/')
        else:
            path = [metric, key]

        data = []
        for core in cores:
            cur = self.uns[core]
            for i in range(len(path) - 1):
                cur = cur[path[i]]

            data.append(cur[path[-1]])

        return pd.DataFrame(data, index=cores)

    def get_uns_keys(self, path=None, cores=None):
        # TODO later, with a stack
        if path is None:
            cur = self.uns
        else:
            path = path.split('/')
            for i in range(len(path) - 1): cur = cur[path[i]]

        keys = []
        for core in self.cores:
            pass
