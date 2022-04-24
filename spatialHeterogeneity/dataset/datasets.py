from ._utils import DataSet


def imc(force_download=False):
    """Pre-processed zurich cohort IMC dataset from *Jackson, H.W., Fischer, J.R., Zanotelli, V.R.T. et al. The single-cell pathology landscape of breast cancer.* `Nature <https://doi.org/10.1038/s41586-019-1876-x>`_
    """
    print('warning: to get the latest version of this dataset use `so = sh.dataset.imc(force_download=True)`')
    so = DataSet(
        name='imc',
        url='https://figshare.com/ndownloader/files/34877643',
        doc_header='Pre-processed subset IMC dataset from `Jackson et al '
    )
    return so(force_download=force_download)

# def imc_basel():
#     """Pre-processed subset IMC dataset from *Jackson, H.W., Fischer, J.R., Zanotelli, V.R.T. et al. The single-cell pathology landscape of breast cancer.* `Nature <https://doi.org/10.1038/s41586-019-1876-x>`_
#     """
#     so = DataSet(
#         name='imc',
#         url='https://figshare.com/ndownloader/files/31750769',
#         doc_header='Pre-processed subset IMC dataset from `Jackson et al '
#     )
#     return so()


def mibi():
    """
    Processed data from *A Structured Tumor-Immune Microenvironment in Triple Negative Breast Cancer Revealed by Multiplexed Ion Beam Imaging.* `Cell <https://doi.org/10.1016/j.cell.2018.08.039>`_.
    Normalised expression values of segmented cells and cell masks from `here <https://www.angelolab.com/mibi-data>`_ and tiff stacks from `here <https://mibi-share.ionpath.com/tracker/imageset/>`_

    """
    so = DataSet(
        name='mibi',
        url='https://figshare.com/ndownloader/files/34148859',
        doc_header='Processed data from https://www.angelolab.com/mibi-data and https://mibi-share.ionpath.com/tracker/imageset'
    )
    return so()


# mibi_pop = DataSet(
#     name='mibi',
#     url=None,
#     doc_header='Processed and populated data (graphs, metrics) from https://www.angelolab.com/mibi-data and https://mibi-share.ionpath.com/tracker/imageset'
# )
