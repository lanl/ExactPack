.. _quickstart:

**************** 
Quickstart Guide
****************

ExactPack is intended to be used as a Python module, particularly for use in
building more complete code verification tools. 

Installation
============

Installing ExactPack is simple.  Just unpack the tar file and run the
setup script::

    tar xzf ExactPack-1.0.tar.gz
    cd ExactPack-1.0
    pip install ./

This will install to the system Python library directories.  If you
do not have write permission for these directories, you can usually
install to a user specific location by using the ``--user`` flag.  All
the other standard setup options should also be available (run
``pip install --help`` for more information).

.. _quickstart-library:

Using ExactPack as a Python Library
===================================

ExactPack is designed to be used to be used as a Python library in
postprocessing scripts for code verification.

By default, importing ExactPack does not load in any specific
solutions.  Instead you must load in the particular problem you want.
So, for example, you can access the default solver for the Noh problem
by::

   >>> from exactpack.solvers.noh import Noh

The following is an example of a script that computes a spherical Noh
solution, with a specific-heat ratio of 1.4, and then plots it::

   import numpy
   from exactpack.solvers.noh import Noh
   
   solver = Noh(gamma=1.4, geometry=3)
   soln = solver(linspace(0, 1.0), 0.3)
   soln.plot_all()

Note that there may be several solvers for a particular problem.  For example,
by default :mod:`exactpack.solvers.noh2` loads a pure Python implementation,
:mod:`exactpack.solvers.noh2.noh2`.  If you want a version of the Noh solver based on the :mod:`exactpack.solvers.cog.cog1` solver then you need to explicitly import :mod:`exactpack.solvers.noh2.noh2_cog.Noh2Cog`.
     
For documentation on specific solvers, including information on what
parameters they accept and what solution methods are employed, see the
API documentation (:mod:`exactpack`).