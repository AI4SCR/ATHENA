from ._utils import DataSet

imc = DataSet(
    name='imc',
    url='https://figshare.com/ndownloader/files/31750769',
    doc_header='Pre-processed subset IMC dataset from `Jackson et al '
)

mibi = DataSet(
    name='mibi',
    url='https://figshare.com/ndownloader/files/31750769',
    doc_header='Processed data from https://www.angelolab.com/mibi-data and https://mibi-share.ionpath.com/tracker/imageset'
)

mibi_pop = DataSet(
    name='mibi',
    url='https://figshare.com/ndownloader/files/31750769',
    doc_header='Processed and populated data (graphs, metrics) from https://www.angelolab.com/mibi-data and https://mibi-share.ionpath.com/tracker/imageset'
)