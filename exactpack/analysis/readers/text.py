import numpy


class TextReader(object):
    r"""Read data from a text file.

    A :class:`TextReader` is used to read data for analysis from an
    ASCII text file.  It is built on the :func:`numpy.loadtxt`
    function, and is designed to support text files of various
    different formats.
    """

    def __init__(self, names, **kwargs):
        """Set up a :class:`TextReader`.

        The :class:`TextReader` constructor takes a list of names
        which are used as the column names for the data being read.
        The names should reflect the ExactPack naming convention, not
        the names used in the code that generated the data.

        Additional arguments can be used to set the column delimiter,
        comment syntaxt, etc., and are passed directly to
        :func:`numpy.loadtxt` (check the Numpy documentation for
        details).
        """
        self.names = names
        self.loadtxt_args = kwargs
        if 'x_position' in self.names:
            self.dims = 1
        elif 'r_position' in self.names:
            self.dims = 2
        elif 'radius' in self.names:
            self.dims = 3

    def __call__(self, filename):
        r"""Read data from a file.

        Read the column data from *filename*, and return the results
        as a Numpy :class:`numpy.recarray`.  The only ExactPack
        specific attribute which is set is the number of dimensions,
        *dims*, which is computed based on which coordinates are
        defined in the *names* list passed to the constructor.
        """
        try:
            data = numpy.loadtxt(filename, **self.loadtxt_args)
        except IOError:
            return None

        dump=data.view(type=numpy.recarray,
                       dtype=list(zip(self.names, len(self.names)*[data.dtype])))[:,0]
        dump.dims = self.dims
        
        return dump
