*****************
Developer's Guide
*****************

The Developer's Guide is intended to explain the internal architecture
of ExactPack and how to add new solvers.  It assumes you have already
read and understood the :ref:`users-guide`.  In the description that
follows, extensive reference will be made to the implementation of
:mod:`exactpack.noh`.  This should be considered the reference
implementation, and a good template for new solvers.  Potential
developers are advised to carefully look through the source code for
:mod:`exactpack.noh`, particularly :class:`exactpack.noh.noh1.Noh` and
:class:`exactpack.noh.timmes.Noh`, in conjunction with reading this
chapter.  (The source is directly linked to from the documentation in
the HTML version of this manual.)

Coding Style
============

General Guidelines
------------------

The following are some recommended guidelines for project coding
style.

* Be pythonic.  When in doubt about the best way to implement
  something, ask whether this is the most natural way to do it in
  Python.

* When in doubt, use Python standards for formatting code.  General
  coding style is covered in :pep:`8`.  For docstrings, see
  :pep:`257`, except use Sphinx compatible RST markup wherever
  possible, since the docstrings are used to generate this
  documentation.  (A brief description of RST can be found in
  :ref:`sphinx:rst-primer`, and Sphinx specific markup in
  :ref:`sphinx:sphinxmarkup` and :ref:`sphinx:domains`.)

* Use only libraries in the `Enthough Python Distribution
  <https://www.enthought.com/products/canopy/package-index/>`_.
  Installing packages with complex dependencies can be a pain, we can
  make it easier by at least assuring that ExactPack will always
  install under EPD. 

* Good documentation takes time.  Do it anyway, the time will be amply
  repaid in the eventual service life of the code.

Requirements for New Solvers
----------------------------

Before releasing a new solver, please make sure the following
requirements have been met.

* The solver supports the ExactPack API as described in this chapter.

* The solver is fully documented including,

  * A complete description of the physics and geometry of the problem
    and the equations being solved.

  * Either a description of the solution method, or a pointer to an
    archived publication with such a description.

  * A description of the all the input and output parameters,
    including the reason for the choice of any default values.  (See
    the notes in :ref:`doc-params` for the correct format to do this.)

* Unittests for the solver are provided.

  * Unittests should be documented to indicate what they do and why
    these tests are selected.

  * At a minimum, unittests should demonstrate that the solver
    reproduces the correct results in limiting cases and
    well established, previously published, data.

  * Additional tests, such as solution verification (checking that the
    output of the solver converges), are always good.  The more tests,
    the more confidence we can have in the output.

A Tour of the Package Source
============================

The main ExactPack package consists of several sub-packages, some of
which provide solvers, and others containing supporting code.  The
layout of the package directory looks like::

    ExactPack/
        doc/
        exactpack/
	    __init__.py
	    analysis/
   	    base.py
	    noh/
	        __init__.py
		noh1.py
		timmes.py
		_timmes.so
	    testing.py
	    tests/
	        __init__.py
		test_noh.py
	    utils.py
        setup.py
        src/
	    timmes/
                noh/
                    noh.f

The ``doc`` directory contains the Sphinx documentation.  The ``src``
directory contains Fortran or other non-Python source files for exact
solution codes, organized by author or contributor, and then by
problem.  ``setup.py`` is the packaging script.

The ``exactpack`` directory has several ``.py`` files providing
package level support: ``base.py`` defines the base classes,
``utils.py`` defines support functions, and ``testing.py`` provides
support to the :mod:`unittest` testing infrastructure.  The
``analysis`` directory contains tools for plotting and convergence
testing.  The ``tests`` directory contains unittests for the library
(about which, more later).

Finally, the ``noh`` directory is a sub-package providing an exact
solution code for the Noh problem.  This includes two different
implementations of the Noh problem, ``noh1.py`` is written in pure
Python code using Numpy, and ``timmes.py`` is a wrapper for the
Fortran implementation whose source code is in
``src/timmes/noh/noh.f``.  The dynamically loadable shared object
library ``_timmes.so`` is not actually distributed with the source, it
is automatically compiled when the package is installed using the
``setup.py`` script, and could have a different name on some
architectures.  It is a Python importable object, but is not meant to
be accessed directly, but rather, via the ``timmes.py`` wrapper (which
is why it is named with a leading underscore).

This description does not include every file that you will see if you
browse the source tree.  There will be many more sub-package
directories and corresponding source directories for the different
solvers.  Each problem has its own sub-package, with each solver a
separate module within that package.  That allows the library to
provide multiple solvers for each problem, which is useful for cases
when different solvers work in different regimes, or provide different
functionality.  Each module contains one or more solver classes.
Although the classes in a module will all use the same solver,
different classes may be used as interfaces for specific cases.  For
example, :class:`exactpack.noh.noh1.Noh` is a general Noh solver,
whereas :class:`exactpack.noh.noh1.SphericalNoh` is a derived
convenience class with default settings for a solution in a spherical
geometry.

.. _adding-a-solver:

Adding a New Solver
===================

Let's take a closer look at the :mod:`exactpack.noh` package, and see
how we would go about adding a new solver.

First we need to create a sub-directory to hold the solver, in this
case this is the ``noh`` directory.  Next, we add an ``__init__.py``
file, to tell Python that this directory is a package.  At the head of
the ``__init__.py`` we put a docstring describing the Noh problem.
This docstring can be quite long, since it will be the complete
documentation of the problem which will appear in the API section of
this manual.

A Native Python Solver
----------------------

.. currentmodule:: exactpack.base

Since the Noh problem has a closed form solution, it is relatively
simple to implement it directly in Python using Numpy.  An example of
such an implementation is in the file ``noh1.py``.

The main part of the implementation is a class,
:class:`exactpack.noh.noh1.Noh` (class names are to be capitalized by
convention).  It derives from the :class:`ExactSolver`
base class:

.. literalinclude:: ../../exactpack/solvers/noh/noh1.py
   :pyobject: Noh
   :end-before: def __init__

In order to insure a uniform interface, initialization of solver
attributes is handled by :class:`ExactSolver`.  The
solver class itself should not try to set any parameters.  Instead the
class definition should set a ``parameter`` attribute.  ``parameter``
should be a dictionary, with each key, value pair consisting of a
parameter name and a corresponding help string.  This allows the base
class to provide uniform error checking, documentation, and other
functionality.  This attribute is also used by other function to
determine the valid parameters for the solver class.  For example, the
GUI uses this to build a dialog box that includes the appropriate
parameters for a given solver class.

Providing a constructor is optional.  The default constructor for
:class:`ExactSolver` takes an arbitrary list of keyword arguments, and
adds them to the instance's dictionary.  It also does error checking
for missing or unknown parameters.  In most cases, the solver class
does not need to provide a constructor.  If a constructor is provided,
it *must* use the :func:`super` function to call the base class
constructor, which will take care of the parameter initialization.

For the Noh problem, we want to raise an error if the ``geometry``
parameter is not a valid value.  The constructor looks like:

.. literalinclude:: ../../exactpack/solvers/noh/noh1.py
   :pyobject: Noh.__init__

The actual computation of the exact solution is done by the ``_run``
method.  The ``_run`` method takes two arguments: a Numpy array of
containing the spatial points at which the solution should be
computed, and a scalar time.  The return value must be an
:class:`ExactSolution`.

The Noh problem has a solution which can be expressed simply as a
piecewise analytic function (e. g. [Gehmeyr]_), which is easily
expressed in Numpy:

.. literalinclude:: ../../exactpack/solvers/noh/noh1.py
   :pyobject: Noh._run

The :class:`ExactSolution` constructor takes two required arguments.
The first is an sequence of arrays, the first of which is the solution
locations (this is the same as the first argument passed to ``_run``),
and the rest are solution values for each variable at the
corresponding locations.  Note that a rank-2 Numpy array, with first
index corresponding to the solution variable, and second index to the
spatial position, will work.  The second argument is a sequence of
strings giving the names of each of the corresponding solution
variable.  For variable names, the standard names found in table
:ref:`table-variable-names` should be used, if one is listed.

.. _doc-params:

Documenting the Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ExactSolver` class uses some Python magic (in particular,
meta-classes, for details see the source code) to simplify documenting
the parameters.  We want the parameters to be consistently documented
in the manuals, :func:`help` function, and have the help information
available to tools such as the GUI.  This is done in an automated
fashion via the :attr:`ExactSolver.parameters` attribute.  As
described above, this attribute consists of a :class:`dict` with all
parameters of the solver class as keywords, and a short, one-line,
description as the associated value.  At class creation time, this
help information is automatically added to the docstring.

The parameter list is also used to check the parameters used to
instantiate a solver, and issue errors or warnings as necessary.

If a longer description of any of the parameters is needed, that
should be incorporated into the documentation for the class.  By
default, any documentation of the ``__init__`` method (if there is
one) will be appended to the documentation for the class in generating
the manuals.

.. _jump-conditions:

More About Jump Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^

There is an optional third argument, ``jumps``, which can be used to
indicate the location and values of the solution at discontinuities.
If this is omitted, the value of ``jumps`` defaults to ``None``, which
means that the solver does not provide information about jumpt
conditions.  Otherwise, this is a list of :class:`JumpCondition`\s,
one for each discontinuity in the solution.  To indicate a continuous
solution use and empty list, ``[]``.

.. note:: ``jump==None`` indicates that the solver does not provide
   information regarding jumps.  ``jump==[]`` means there are no
   jumps.  Since analysis tools may use the jumps for comparison to
   computed solution, it is important distinguish between these
   settings.

The first argument to :class:`JumpCondition` is the spatial location
of the discontinuity.  The second is a short string describing the
type of discontinuity.  The remaining arguments are keyword arguments,
the keyword giving the variable name (which should be the same as used
in the ``names`` argument of the :class:`ExactSolution` constructor),
and the values at the left and right state.  These are converted to
:class:`Jump` objects before being stored.  In the Noh example, above,
the values are given as 2-tuples, which are interpreted as ``(left,
right)`` states.

Setting the Default Solver
^^^^^^^^^^^^^^^^^^^^^^^^^^

For the :mod:`exactpack.noh` sub-package, there are two available
implementations of the solver.  The ``__init__.py`` file loads the
default solver, in this case::

   from .noh1 import *

Even in cases with a single solver, it is preferable to put it in a
separate file, and import it in ``__init__.py``.  This is consistent
with the Python convention of avoiding code in ``__init__.py`` and
also keeps the package names from changing if an additional solver is
added in the future.

Convenience Wrappers
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: exactpack.noh.noh1

In some cases, it may be useful to have multiple interface classes for
the same solver.  For example, the general Noh solver, :class:`Noh`
can be used to set up the Noh problem in planar, cylindrical, or
spherical geometries, but it is convenient to have specialized classes
for each of these three cases.

We can derive a :class:`SphericalNoh` from the general solver
described above:

.. literalinclude:: ../../exactpack/solvers/noh/noh1.py
   :pyobject: SphericalNoh

In this case, the class definition does not include any methods; they
are all defined in the parent class.  The ``geometry`` attribute
defaults to 3, corresponding to a spherical geometry.  There is no
``geometry`` attribute listed in the ``parameters`` list, since we
don't want users to change it: it will neither be a legal parameter to
the constructor, nor will it show up as a paramater in the GUI.

Setting a value of ``gamma`` in the class definition is the correct
way a providing a default.  In this case we chose not to provide any
default values in the general solution class :class:`Noh`, but give
the :class:`SphericalNoh` class the same default as used in Noh's
original paper [Noh]_.  The results in the following behavior:
:class:`Noh` requires exactly two parameters (``geometry`` and
``gamma``) when invoking its constructor, and will fail, raising an
exception, otherwise.  :class:`SphericalNoh` has one optional
parameter (``geometry``), and will issue a warning and continue
running if you rely on the default.  (This behavior is designed to
decrease the likelihood of user's relying on default values of which
they are unaware.)  There is no supported method of changing the
``geometry`` of :class:`SphericalNoh`.  (There is nothing stopping
someone from typing ``SphericalNoh.geometry = 2``; it should be
obvious that this is a very bad idea.)

A Fortran Solver
----------------

Often we already have a solver in another language, usually some
variant of Fortran, and we wish to incorporate it into the ExactPack
framework.  Converting the solver to pure Python may not be advisable:
there may be to high a risk of introducing new bugs, it may run too
slow, or the time involved may just be too much to justify.  In this
case we can use the `F2PY <http://cens.ioc.ee/projects/f2py2e>`_
package to wrap the existing Fortran code for easy access from Python.

The first step is to instrument the original Fortran code with F2PY
directives.  These are specially formatted Fortran comments that are
used by the F2PY package to determine the calling interface.

For our Noh problem example, we can start with the code from `Frank
Timmes verification website
<http://cococubed.asu.edu/code_pages/noh.shtml>`_.  Some modifications
that are needed in this case are removing the main driver (this will
be provided on the Python side) and unused support routines, removing
the input and output (this will also by handled in Python), and moving
the loop over spatial points into the main computation routine (for
efficiency).  Once this is done, the computation routine is
instrumented with F2PY directives:

.. literalinclude:: ../../src/timmes/noh/noh.f
   :language: fortran
   :end-before: solves the standard case

The ``cf2py`` prefix identifies an F2PY directive.  ``intent(out)`` is
used to identify which variable will be passed to Python as return
values.  ``intent(hide)`` is used to hide an argument so it doesn't
appear explicitly in the Python interface.  In this case the variable
in question, ``nstep`` does not need to be explicitly passed from
Python because it can be determined by the wrapper based on the length
of ``xpos``.  The type declarations are used by the Python wrapper to
do type checking and conversion on the arguments.

With the addition of these directives, this Fortran source can now be
compiled into a shared library which is loadable by Python.  The
signature for this function is:

.. currentmodule:: exactpack.noh._timmes

.. function:: noh_1d(time, xpos, rho1, u1, gamma, xgeom) -> (den, ener, pres, vel, jumps)

It takes six positional or keyword arguments, which map to the
arguments of the Fortran function with neither ``intent(out)`` or the
``intent(hide)``, and it returns a tuple of Numpy arrays, which are
the Fortran arrays with ``intent(out)``, in order.  F2PY takes care of
creating arrays which will persist after the Fortran subroutine exits.

Compilation can be done from the command line using the ``f2py``
utility, but we automate it through ExactPack's ``setup.py`` file.
The allows us to make a source distribution and compile automatically
when installing using Python's packaging tools.  This is done by
adding an :class:`numpy.distutils.core.Extension` object to the
``ext_modules`` list of the ``setup`` class:

.. literalinclude:: ../../setup.py
   :start-after: requires
   :end-before: sedov

``name`` is the name that will be used for the compiled package.  On
most systems, for this example, building ExactPack will generate a
file ``exactpack/noh/_timmmes.so``, which is the loadable library.  We
choose a name with a leading underscore to indicate that this module
is not meant to be used directly (as we shall see).  ``sources`` is a
list of source files to compile.  By convention, ExactPack groups
source files in problem specific directories, organized first by
source code author, then by problem.

Sometimes a source file includes multiple routines, only some of which
need to be exposed in the Python interface.  In that case, a
``f2py_options`` argument should be passed to
:class:`numpy.distutils.core.Extension`'s constructor, with an
``only:`` flag.  The syntax would be::

   f2py_options = ['only:'] + [ 'noh_1d' ] + [':']

In our Noh example, this is not necessary, since the source file
contains only one routine.

At this point we are only half done.  We now have a Fortran library
that can be accessed by importing::

   import exactpack.noh._timmes
   
but the interface is hard to use and does not conform to the ExactPack
API.  We next must create an solver class to wrap the Fortran
library.  This we put in the file ``exactpack/noh/timmes.py``.  The
class header looks identical to the one for the Python solver:

.. literalinclude:: ../../exactpack/solvers/noh/timmes.py
   :pyobject: Noh
   :end-before: def __init__

The constructor, should we choose to provide one, in this case also
looks identical:

.. literalinclude:: ../../exactpack/solvers/noh/timmes.py
   :pyobject: Noh.__init__

For more complex problems, we might want to call some Fortran code
immediately in the constructor to compute intermediate values that
will not change between calls to the ``_run`` method.

The ``_run`` method now is simply a wrapper to the Fortran function:

.. literalinclude:: ../../exactpack/solvers/noh/timmes.py
   :pyobject: Noh._run

Note that we do not need to create the arrays ``density``, etc., since
that is taken care of by the F2PY wrapper.  They are then passed to
the :class:`exactpack.base.ExactSolution` constructor.  The original
Fortran source was modified to return an additional array, ``jumps``,
into which is packed the location and pre- and post-shock states.
These could equally well have been returned as nine separate scalars.
The result would have been a more complex call signature, but would
have eliminated the need to make sure the array is packed and unpacked
consistently on the Fortran and Python sides, respectively.

For more information about F2PY and using it to interface with Numpy
arrays, see the `F2PY User's Guide and Reference Manual
<http://cens.ioc.ee/projects/f2py2e/usersguide>`_ and `Using Python as Glue
<http://docs.scipy.org/doc/numpy/user/c-info.python-as-glue.html>`_.

Unit Tests
----------

Strictly speaking, real code verification for exact solution codes is
difficult, since there is often nothing to compare them to. On the
other hand, if users are to have confidence in the results, there
should be some testing.  Depending on the problem, this could consist
of solution verification (convergence here being in some internal
quadrature tolerance, not grid points), comparison to limiting cases
with known analytic solutions, or comparison to well reviewed
published results.  At a minimum, any self tests provided by the
original stand-alone solver should be integrated into the ExactPack
testing framework.

Tests for solvers are found in the ``tests`` directory, with file
names ``test_<solver>.py``.  For examples of testing, see the files in
that directory.  The tests are built using the standard Python
:mod:`unittest` framework.
