API Overview
============
Quick overview of the :ref:`methods` and :ref:`datasets` available in ATHENA.

.. _methods:

Methods
-------
Depending on the underlying mathematical foundations, the heterogeneity
scores included in ATHENA can be classified into the following categories: (i) spatial statistics scores that
quantify the degree of clustering or dispersion of each phenotype individually, (ii) graph-theoretic scores that
examine the topology of the tumor graph, (iii) information-theoretic scores that quantify how diverse the
tumor is with respect to different phenotypes present and their relative proportions, and (iv) cell interaction
scores that assess the pairwise connections between different phenotypes in the tumor ecosystem. The interested reader
is advised to read the *Methodology* section.

Pre-processing
^^^^^^^^^^^^^^
Collection of common pre-processing functionalities.

.. autosummary::
    ~spatialHeterogeneity.preprocessing.preprocess.extract_centroids
    ~spatialHeterogeneity.preprocessing.preprocess.arcsinh


Graph building
^^^^^^^^^^^^^^
The :mod:`~.graph` submodule of SpatialHeterogeneity constructs a graph representation of the tissue using the
cell masks extracted from the high-dimensional images. The graph construction module implements three
different graph flavors that capture different kinds of cell-cell communication:

    - *contact*-graph: juxtacrine signaling, where cells exchange information via membrane receptors, junctions or extracellular matrix glycoproteins
    - *radius*-graph: representation mimics paracrine signaling, where signaling molecules that are secreted into the extracellular environment interact with membrane receptors of neighboring cells and induce changes in their cellular state.
    - *knn*-graph: common graph topology, successfully used in digital pathology

.. autosummary::
    ~spatialHeterogeneity.graph_builder.graphBuilder.build_graph

Visualisation
^^^^^^^^^^^^^
The plotting module (:mod:`~.plotting`) enables the user to visualise the data and provides out-of-the-box plots for some
of the metrics.

.. autosummary::
    ~spatialHeterogeneity.plotting.visualization.spatial
    ~spatialHeterogeneity.plotting.visualization.napari_viewer
    ~spatialHeterogeneity.plotting.visualization.interactions
    ~spatialHeterogeneity.plotting.visualization.ripleysK
    ~spatialHeterogeneity.plotting.visualization.infiltration


Entropic metrics
^^^^^^^^^^^^^^^^^^
ATHENA brings together a number of established as well as novel scores that enable the quantification of
tumor heterogeneity in a spatially-aware manner, borrowing ideas from ecology, information theory, spatial
statistics, and network analysis.

.. autosummary::
    ~spatialHeterogeneity.metrics.heterogeneity.metrics.richness
    ~spatialHeterogeneity.metrics.heterogeneity.metrics.abundance
    ~spatialHeterogeneity.metrics.heterogeneity.metrics.shannon
    ~spatialHeterogeneity.metrics.heterogeneity.metrics.simpson
    ~spatialHeterogeneity.metrics.heterogeneity.metrics.renyi_entropy
    ~spatialHeterogeneity.metrics.heterogeneity.metrics.hill_number
    ~spatialHeterogeneity.metrics.heterogeneity.metrics.quadratic_entropy

Graph metrics
^^^^^^^^^^^^^^^^^^
Currently, this module only implements modularity which captures the structure of a graph by quantifying the degree at which it can
be divided into communities of the same label. In the context of tumor heterogeneity, modularity can be
thought of as the degree of self-organization of the cells with the same phenotype into spatially distinct
communities.

.. autosummary::
    ~spatialHeterogeneity.metrics.graph.graph.modularity

Cell-cell interaction metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
More sophisticated heterogeneity scores additionally consider cell-cell interactions by exploiting the cell-cell graph,
where nodes encode cells, edges encode interactions, and each node is associated with a label
that encodes the cell’s phenotype. The cell interaction scores implemented in ATHENA’s :mod:`~neighborhood` submodule
include:

.. autosummary::
    ~spatialHeterogeneity.neighborhood.estimators.interactions
    ~spatialHeterogeneity.neighborhood.estimators.infiltration
    ~spatialHeterogeneity.neighborhood.estimators.ripleysK

.. _datasets:

Datasets
--------
ATHENA provides two datasets that enables users to explore the implemented functionalities and analytical tools:

    - An image mass cytometry dataset [Jackson]_
    - An multiplexed ion beam imaging dataset [Keren]_

.. autosummary::
    ~spatialHeterogeneity.dataset.datasets.imc
    ~spatialHeterogeneity.dataset.datasets.mibi

References
^^^^^^^^^^
.. [Jackson] Jackson, H. W. et al. The single-cell pathology landscape of breast cancer.
    `Nature. <https://www.nature.com/articles/s41586-019-1876-x>`_

.. [Keren] Keren, L. et al. A Structured Tumor-Immune Microenvironment in Triple Negative Breast Cancer Revealed by
    Multiplexed Ion Beam Imaging. Cell 174, 1373-1387.e19 (2018). `Cell. <https://doi.org/10.1016/j.cell.2018.08.039>`_