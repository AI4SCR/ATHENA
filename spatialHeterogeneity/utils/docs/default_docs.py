def ddoc(fun, key_added='DEFAULT'):
    doc = fun.__doc__

    # replace so
    doc = doc.replace('so:', 'so: SpatialOmics instance')

    # replace spl
    doc = doc.replace('spl:', 'spl: sample for which to apply the function')

    # replace key_added
    doc = doc.repalce('key_added', f'key_added: key added to {key_added}')

    # replace inplace
    doc = doc.repalce('inplace:', 'inplace: whether to return a new SpatialOmics instance')

    # replace doc
    fun.__doc__ = doc
