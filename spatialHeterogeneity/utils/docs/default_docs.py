def ddoc(fun):
    key_added = 'DEFAULT'

    doc = fun.__doc__

    # replace so
    doc = doc.replace('so:', 'so: SpatialOmics instance')

    # replace spl
    doc = doc.replace('spl:', 'spl: sample for which to apply the function')

    # replace key_added
    doc = doc.replace('key_added', f'key_added: key added to {key_added}')

    # replace inplace
    doc = doc.replace('inplace:', 'inplace: whether to return a new SpatialOmics instance')

    # replace doc
    fun.__doc__ = doc

    return fun

from docrep import DocstringProcessor
docstrings = DocstringProcessor()

@docstrings.get_sections
def default_doc(so, inplace):
    """

    Args:
        so: SpatialOmics
        inplace: Apply function in place

    Returns:

    """
    pass