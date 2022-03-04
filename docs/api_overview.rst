API Overview
============
Quick overview of the :ref:`methods` and :ref:`datasets` available in ATHENA.

.. _methods:

Methods
-------

Pre-processing
^^^^^^^^^^^^^^
.. autosummary::
    ~spatialHeterogeneity.preprocessing.preprocess.extract_centroids


Graph building
^^^^^^^^^^^^^^
.. autosummary::
    ~spatialHeterogeneity.graph_builder.graphBuilder.build_graph

Visualisation
^^^^^^^^^^^^^
.. autosummary::
    ~spatialHeterogeneity.plotting.visualization.spatial
    ~spatialHeterogeneity.plotting.visualization.napari_viewer
    ~spatialHeterogeneity.plotting.visualization.interactions
    ~spatialHeterogeneity.plotting.visualization.ripleysK
    ~spatialHeterogeneity.plotting.visualization.infiltration


Entropic metrics
^^^^^^^^^^^^^^^^^^
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
.. autosummary::
    ~spatialHeterogeneity.metrics.graph.graph.modularity

Cell-cell interaction metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    ~spatialHeterogeneity.neighborhood.estimators.interactions
    ~spatialHeterogeneity.neighborhood.estimators.infiltration
    ~spatialHeterogeneity.neighborhood.estimators.ripleysK

.. _datasets:

Datasets
--------
.. autosummary::
    ~spatialHeterogeneity.dataset.datasets.imc
    ~spatialHeterogeneity.dataset.datasets.mibi
