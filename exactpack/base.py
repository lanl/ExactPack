import csv
import re
import os
from contextlib import redirect_stdout
from textwrap import dedent
from warnings import warn

import numpy


_whitespace_only_re = re.compile('^[ \t]+$', re.MULTILINE)
_leading_whitespace_re = re.compile('(^[ \t]*)(?:[^ \t\n])', re.MULTILINE)


def print_when_verbose(method):
    """Capture all stdout and redirect to null unless verbose = True"""
    def wrapper(cls, *args, **kwargs):
        if cls.verbose:
            result = method(cls, *args, **kwargs)
        else:
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                result = method(cls, *args, **kwargs)
        return result

    return wrapper


def _get_margin(text):
    """Get the common leading whitespace of text.

    The code for this function is taken out of
    :func:`textwrap.dedent`, except it simply returns the leading
    whitespace as a string, rather than removes it.  It also ignores
    the first line, so is compatible with docstrings.
    """

    margin = None
    text = _whitespace_only_re.sub('', text)
    indents = _leading_whitespace_re.findall(text)
    for indent in indents[1:]:
        if margin is None:
            margin = indent

        # Current line more deeply indented than previous winner:
        # no change (previous winner is still on top).
        elif indent.startswith(margin):
            pass

        # Current line consistent with and no deeper than previous winner:
        # it's the new winner.
        elif margin.startswith(indent):
            margin = indent

        # Current line and previous winner have no common whitespace:
        # there is no margin.
        else:
            margin = ""
            break

    return margin


class _AddParametersToDocstring(type):
    """A metaclass to add information from the parameter attribute to the doctstring.
    """

    def __new__(meta, name, bases, dct):

        doc = dedent(dct.get('__doc__', ""))

        margin = _get_margin(doc) or ""

        if doc and dct['parameters']:
            doc += "\n"

        for key, value in dct['parameters'].items():
            doc += "{}:param {}: {}\n".format(margin, key, value)

        dct['__doc__'] = doc

        return super(_AddParametersToDocstring, meta).__new__(meta, name, bases, dct)


class Jump(object):
    """A class to hold values at jump points.

    A :class:`Jump` is a container for left and right states used by
    :class:`JumpCondition` to hold the two limiting values at
    discontinuous points in an exact solution.
    """

    def __init__(self, left, right=None):
        """
        :param left: the left state
        :param right: the right state

        There are three ways of constructing a :class:`Jump` object.

        1. If two arguments are provided (*left* and *right*), these
           are used for the left and right states.
        2. Alternately, one tuple or another sequence argument can be
           provided, and the first two elements will be used for the
           left and right states.
        3. Finally, providing a single argument of type :class:`Jump`
           will act as a copy constructor, creating a new object that
           is a copy of the original.
        """

        if not right is None:
            #: The left state, or the limiting value at the point from below 
            #: [ :math:`\lim_{x \rightarrow a^-} f(x)` ].
            self.left = left

            #: The right state, or the limiting value at the point from above
            #: [ :math:`\lim_{x \rightarrow a^+} f(x)` ].
            self.right = right
        elif isinstance(left, Jump):
            self.left = left.left
            self.right = left.right
        else:
            self.left = left[0]
            self.right = left[1]

    def __repr__(self):

        return "Jump(left={}, right={})".format(self.left, self.right)


# class JumpCondition(object):
#     """A class for jump conditions.
#
#     By definition, weak solutions of differential equations may have discontinuities.
#     These are points at which the solution, considered as a function,
#     has no value, but for which the left and right limits on the
#     function value are different.  Numerically computed discrete
#     solutions cannot directly capture discontinuities.  The
#     :class:`JumpCondition` class is used to provide a numerical
#     description of the mathematical properties of the jump.
#
#     Each solution discontinuity has a location and a set of variables
#     for which the left and right states are provided.  In addition to
#     the attributes described below, a :class:`JumpCondition` object
#     will have an attribute for each problem variable, with a value of
#     type :class:`Jump` giving the left and right states.
#
#     For example, the jump in the Heaviside step function could be
#     described by the following::
#
#        JumpCondition(location=0,
#                      description="Mathematical Discontinuity",
#                      H=(0, 1))
#
#     For more information see :ref:`jump-conditions`.
#     """
#
#     def __init__(self, location, description="", **kwargs):
#         """
#         :param Number location: the location of the jump point
#         :param str description: a short description of the type of
#           discontinuity (e.g., 'Shock', 'Material Interface')
#         :param kwargs: the remaining keywords arguments set the jump
#           conditions: the keywords are the variable names, and the
#           values are either of type :class:`Jump`, or a 2-tuple to be
#           converted
#         """
#
#         #: The location of the jump point
#         self.location = location
#
#         #: A short description of the type of discontinuity
#         self.description = description
#
#         self._vars = {}
#         for key, val in kwargs.items():
#             self._vars[key] = Jump(val)
#
#     def __getattr__(self, name):
#
#         try:
#             return self._vars[name]
#         except KeyError:
#             raise AttributeError("JumpCondition has no attribute '{}'".format(name))
#
#     def __repr__(self):
#
#
#         vars = [ "{}=({},{})".format(key, val.left, val.right)
#                  for key, val in self._vars.items() ]
#
#         return "JumpCondition(location={},{})".format(self.location,
#                                                       ",".join(vars))
 

def JumpCondition(location, *args, **kwargs):
    """This dummy function essentially removes most of the functionality of the
    JumpCondition class, leaving only the location.

    Args:
        location (float): Location of the Jump

    Returns:
        float: The location of the Jump
    """
    return location


class ExactSolver(object, metaclass=_AddParametersToDocstring):
    """A virtual base class for ExactPack solvers.

    Solvers are Python callable objects which can be used to generate
    solutions at specific points in time and space.  A solver instance
    can be invoked as a function with two arguments.  The first is a
    list of points at which to generate the exact solution, and the
    second is the time at which the solution is required.  The
    points list can be a :mod:`numpy` array or a Python sequence (list or
    tuple) of points.  Depending on the solver, each point may be 1-, 2-,
    or 3-dimensional, and may be in rectangular, cylindrical, or spherical
    coordinates.  For 1-dimensional problems, the points are given by a
    rank-1 :mod:`numpy` array, or a list of scalar values.  For 2- or
    3-dimensional problems, the points are given by a :mod:`numpy` array of
    shape ``(N,2)`` or ``(N,3)``, or by a list of 2- or 3-tuples.  Check
    the documentation for a particular solver for details.

    For an example of how to write an ExactSolver child class, see
    :ref:`adding-a-solver`.
    """

    #: A list of parameters which can be used as keyword arguments to
    #: the solver's constructor.
    parameters = {}

    def __init__(self, verbose=False, **params):

        # Check that all params are in the self.parameters list
        if not params.keys() <= set(self.parameters):
            raise ValueError("Unknown parameters: "
                             +",".join(params.keys() - set(self.parameters)))

        self.__dict__.update(params)
        self.verbose = verbose

        for param in self.parameters:
            # Check that all parameters have been set
            if not hasattr(self, param):
                raise ValueError("Missing parameter: {}".format(param))

    def __call__(self, r, t):

        return self._run(numpy.asarray(r), t)


class ExactSolution(numpy.recarray):
    r"""A class for solutions returned by ExactPack solvers.

    :param data: a sequence of :class:`numpy.ndarray`\s (including a
      rank-2 :class:`numpy.ndarray`) to use as the fields of the
      :class:`ExactSolution`.

    :mod:`exactpack.base.ExactSolution` is derived from the
    :class:`numpy.ndarray`, specifically a Numpy record array
    (:ref:`numpy:structured_arrays`).  There are some special things
    that need to be done to take care of the way a
    :class:`numpy.ndarray` is created.  The code used here is based on
    the example in :ref:`numpy:basics.subclassing`.

    For compatibility, the field names must follow a standardized
    naming convention.  For any variable that has an entry in the
    following table, the listed field name will be used.  Other
    problem specific variables, which are not listed in the table,
    should be fully described in the documentation for the specific
    solver.

    .. _table-variable-names:

    .. table:: Standardized Variable Names

       ==========================      ===============================================
       Field Name                      Description
       ==========================      ===============================================
       density                         Density
       pressure                        Pressure
       specific_internal_energy        Specific internal energy
       velocity                        Velocity along the problem direction
       position                        Generic position variable r or x along the problem direction
       position_x                      Cartesian position variable for x in 1D, 2D or 3D
       position_y                      Cartesian position variable for y in 2D or 3D
       position_z                      Cartesian position variable for z in 3D
       ==========================      ===============================================

    Internally, :func:`numpy.core.records.fromarrays` is used to map
    the *data* to a structured array.
    """

    #: A list of :class:`JumpCondition`\s.  Note the following important
    #: distinction: An empty list means a continuous solution (the
    #: solver is reporting there are no jumps), whereas a value of
    #: ``None`` means the solver is not reporting any information
    #: about jumps (that is, there may or may not be jumps in the
    #: analytic solution).
    jumps = None

    def __new__(cls, data, names, jumps=None):

        # Currently, this does a copy even if data is already an array.
        obj = numpy.rec.fromarrays(data, names=names).view(cls)
        obj.jumps = jumps

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.jumps = getattr(obj, 'jumps', None)

    def plot(self, name, **kwargs):
        """Plot one solution variable using matplotlib.

        *name* is the name of the variable to plot.

        The optional argument *scale* gives a scaling factor to be
        applied to the data before plotting.  If ``scale='auto'``, then
        :meth:`plot` will determine a scale factor so that the plotted
        data will be of order one.

        If a *label* argument is given, it is used with no changes.
        If no *label* is given (or ``label=None``), then the label
        defaults to *name* (with the optional *scale* indicated in the
        label if one is used).

        All other keywords are passed directly to the
        :func:`matplotlib.pyplot.plot` command.

        .. note::

           ExactPack doesn't load matplotlib until the first time
           :meth:`plot` is called.  Scripts that need to select a
           different matplotlib backend can do this any time before
           :meth:`plot` is first called.
        """

        # Plotting routines are imported here rather than the top of
        # the file, as is correct Python style, for three reasons:
        # 1. Since this is where matplotlib.pyplot is loaded, this
        # defers choice of backend, so that scripts can choose an
        # alternate backend.
        # 2. Since matplotlib is only loaded if plotting occurs,
        # exactpack can be imported on systems without matplotlib if
        # no plotting is done during the script.
        # 3. The performance hit should be minimal
        from .plotting import plot

        plot(self, name, **kwargs)

    def plot_all(self):
        """Plot all variables.

        Plot all the variables against radial distance, using auto
        scaling.
        """

        for name in self.dtype.names[1:]:
            self.plot(name, scale='auto')

    def dump(self, filename):
        """Dump the solution variables to a CSV file.
        """

        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(self.dtype.names)
            writer.writerows(self)

