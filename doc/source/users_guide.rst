.. _users-guide:

************ 
User's Guide
************

The User's Guide is intended to explain how to use ExactPack to build
Python scripts for verification.  It describes the basic layout of the
libraries and the :term:`API`.

Available Solvers
=================

Importing :mod:`exactpack` loads some base code and utility functions,
but no actual solvers.  Individual solvers should be imported as
needed.  A solver sub-package is given by the name of the problem in
lower case, preceded by its path. So the default Noh problem solver is
imported using [#]_::

    import exactpack.solvers.noh

If there are multiple solvers available for a given problem, they will
be in sub-sub-packages.  The base package will load a default choice
of the solver, which is the solver recommended for general use by the
maintainers of ExactPack.

For example, the default Noh solver is really
:mod:`exactpack.solvers.noh.noh1`, and this is what you get if you import
:mod:`exactpack.solvers.noh`.  If you want an interface to the Fortran
implementation of Frank Timmes, then you can explicitly import
:mod:`exactpack.solvers.noh.timmes`.

The solver itself is a Python class.  Multiple solver variants may be
available within a given package, but these will all use the same
underlying solver.  For example, the general Noh solver
:class:`exactpack.solvers.noh.Noh` can be used for the Noh problem in
1-, 2-, or 3-dimensions, or the wrapper classes
:class:`exactpack.solvers.noh.PlanarNoh`,
:class:`exactpack.solvers.noh.CylindricalNoh`, and
:class:`exactpack.solvers.noh.SphericalNoh` can be used.

For readability, it is often preferable to use a ``from ... import
...`` construct to load just the desired solvers, for example::

    from exactpack.solvers.noh import SphericalNoh

To obtain a list of all available solvers, use the
:func:`exactpack.utils.discover_solvers` function.  The example
scripts are also a good place to learn about importing solvers.
      
.. [#] Due to its simplicity, the Noh solver will be used as the
   example and reference implementation throughout this document.

Basic Usage
===========
   
The Solver Class
----------------

.. currentmodule:: exactpack.base

All solvers present a uniform :term:`API`.  The API needs to be
sufficiently flexible to accommodate solvers for different problems,
which may differ in input parameters and output variables, but uniform
enough to make it simple to create re-usable scripts for analysis and
plotting.  This is accomplished by having all solvers derive from the
:class:`ExactSolver` base class.

The :class:`ExactSolver` interface is quite simple.  The constructor
takes a series of keyword parameters which define the parameters and
initial conditions for the problem, and returns a solver object.  The
attribute :attr:`ExactSolver.parameters` is a dictionary with keywords
which are the available parameters, and values which are help strings
for each parameter.  If there are uninitialized parameters, or the
constructor is passed parameters that it does not know, it will raise
an exception.  Relying on default parameters will generate a warning,
to make sure the user knows exactly what parameters are being used.

For example, the :class:`exactpack.solvers.noh.noh1.Noh` takes four
parameters, the geometry, the gas constant, and the reference velocity
and density.  To instantiate a solver for the spherical Noh problem,
with the value :math:`\gamma=5/3` as in the original paper by Noh[#]_,
we can use the following code

.. testcode::

    from exactpack.solvers.noh import Noh

    solver = Noh(geometry=3, gamma=5.0/3.0)

The solver is a function-like object that can be called with two
arguments.  The first is a sequence listing the spatial points at
which the solution is returned, and the second is the time.  The
points list can be a :mod:`numpy` array or a Python sequence (list or
tuple) of points.  Depending on the solver, each point may be 1-, 2-,
or 3-dimensional, and may be in rectangular, cylindrical, or spherical
coordinates.  For 1-dimensional problems, the points are given by a
rank-1 :mod:`numpy` array, or a list of scalar values.  For 2- or
3-dimensional problems, the points are given by a :mod:`numpy` array of
shape ``(N,2)`` or ``(N,3)``, or by a list of 2- or 3-tuples.  Check
the documentation for a particular solver for details.

Attributes of a solver, such as ``geometry`` and ``gamma``, should be
treated as read-only once the solver has been instantiated.

Using the solver object defined above, the following call will return
the solution at time :math:`t=0.6` in the interval :math:`x\in[0,1]`:

.. testcode::

    from numpy import linspace

    solution = solver(linspace(0, 1), 0.6)

(The :func:`numpy.linspace` function is a convenient way to generate
an evenly spaced set of points.  It has three arguments: the start point,
the stop point, and an optional number of points to use.)

.. [#] Be careful not to use integer division to represent rationals;
   in Python ``5/3`` is ``1``.

A Note on Dimensions
^^^^^^^^^^^^^^^^^^^^

In general, the equations which are solved are dimensionally
consistent, which means that as long as all the inputs are provided in
consistent units (or consistently non-dimensionalized), the output will
also be consistent.  In any case, where a solver requires input in
particular units, this will be specified in the solver documentation;
if no units are specified, any consistent set may be employed.

The Solution Class
------------------

When a solver object is called to generate a solution, the return
value is an :class:`ExactSolution` instance.  This is a structured
array in which the fields (columns) can be accessed by variable name
and the spatial points by indices.  The first field is always the
solution locations passed as the first argument to the solver.  So,
for example, using the solution computed in the previous section, one
can do the following:

.. doctest::

   >>> solution[1]
   (0.02040816326530612, 64.0, 21.333333333333332, 0.5, 0.0)

   >>> solution.density[:8]
   array([ 64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.])

   >>> solution['specific_internal_energy'][0]
   0.5

The available variable names are standardized, and are listed in
:ref:`table-variable-names`.  (For more details, on possible syntax,
see the documentation for Numpy :ref:`numpy:structured_arrays`, of which 
:class:`ExactSolution` is a subclass.)

Plotting
^^^^^^^^

The methods :meth:`ExactSolution.plot` and
:meth:`ExactSolution.plot_all` are convenient wrappers for matplotlib
plotting.  The syntax should be familiar to anyone familiar with that
package.

To get a quick, auto-scaled, plot of all solution variables::

   solution.plot_all()

.. note::

   :meth:`ExactSolution.plot` and :meth:`ExactSolution.plot_all` do
   not issue a :func:`matplotlib.pyplot.show` internally, so it may be
   necessary to do this manually, depending on which backend you use.
   If you want to work with plots interactively from the command-line
   running IPython with the :mod:`matplotlib.pylab` environment::

      ipython --pylab

   will automatically take care of updating plots, as well as setting up
   your environment to include the most common Numpy and matplotlib
   functions.

For more control over your plots, the :meth:`ExactSolution.plot`
function takes one required argument, the name of the variable to
plot, and also accepts all the keyword options understood by the
:func:`matplotlib.pyplot.plot` function.  So, for a solution object
`solution`::

    solution.plot('density', 'r--', marker='o')

will plot the density with a red dashed line and circles.

In addition to all the keywords supported by
:func:`matplotlib.pyplot.plot`, :meth:`ExactSolution.plot` understands
one additional keyword, *scale*, which is an arbitrary scaling factor
to apply to the data before plotting.  For ``scale='auto'``,
:meth:`ExactSolution.plot` will pick a scaling that results in a
plotted value of order one.

If no *label* keyword is specified, :meth:`ExactSolution.plot` will
generate a label.

Other Functions
^^^^^^^^^^^^^^^

To dump the solution to a CSV file use :meth:`ExactSolution.dump`::

    solution.dump("filename.csv")
       
Jump Conditions
---------------

The :class:`ExactSolution` object can optionally report the exact
locations and left and right states for points at which the solution
is discontinuous.  These values are stored in the attribute
:attr:`ExactSolution.jumps`.  If this attribute is set to ``None``,
this means that the solver does not report discontinuities.  Otherwise
this will be set to a list, each element of which will be of type
:class:`JumpCondition`.  An empty list means that the solution is
continuous (i.e., the list of jumps has no elements).

A :class:`JumpCondition` has a location, and left and right values for
each variable:

.. doctest::

   >>> solution.jumps
   [JumpCondition(location=0.2,velocity=(0,-1),density=(64.0,16.0), 
     specific_internal_energy=(0.5,0),pressure=(21.3333333333,0))]
   >>> solution.jumps[0].location
   0.19999999999999998
   >>> solution.jumps[0].density
   Jump(left=64.0, right=16.0)
   >>> solution.jumps[0].density.left
   64.0
   >>> solution.jumps[0].density.right
   16.0
   
.. _convergence-analysis:
  
Convergence Analysis
====================

ExactPack is primarily designed as a tool for code verification.  A
comprehensive introduction to code verification is beyond the scope of
this guide.  In brief, the idea is to compare the output of a
numerical solution produced by a PDE solver to an exact analytic or
semi-analytic solution, such as those provided by ExactPack.  The
magnitude of the error in the solution is computed for a seriers of
grid or timestep refinements, and one confirms the error converges to
zero.  Ideally, the rate of convergence is also compared to the rate
predicted by theory, if one is available.

Although ExactPack can provide solutions for any user script for
analyzing convergence, there are also some built-in tools for
convergence analysis in :mod:`exactpack.analysis`.  The basic object
for doing verification analysis is
:class:`~exactpack.analysis.CodeVerificationStudy`.  To perform a code
verification study, instantiate a study object by passing it a list of
data files, an ExactPack solver, and a data reader::

    from exactpack.analysis import CodeVerificationStudy
    from exactpack.analysis.readers import TextReader
    from exactpack.solvers.riemann import Sod

    study = CodeVerificationStudy(['coarse.dat', 'medium.dat', 'fine.dat'],
                                  Sod(),
				  reader=TextReader)
    
The study object has some built in plotting routines.  For example,
to plot the density profiles for all the cases against the exact
solution::

    study.plot('density')

The :meth:`~exactpack.analysis.CodeVerificationStudy.convergence`
method produces a :class:`~exactpack.analysis.CodeVerificationResult`
object, which also has built-in plotting::

    study.convergence('density', norm=PointNorm()).plot()

This will generate a convergence plot showing the density error
computed using a pointwise norm and the best fit convergence rate.

