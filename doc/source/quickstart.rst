.. _quickstart:

**************** 
Quickstart Guide
****************

There are three ways to use ExactPack: in Python as a module
(:ref:`quickstart-library`), using the GUI (:ref:`quickstart-gui`), or
using the command line utility (:ref:`quickstart-cmd-line`).  The
first is intended as the primary interface, particularly for use in
building more complete code verification tools.  The other two are 
easy ways to quickly visualize exact solutions, or interactively
explore data sets.

Installation
============

Installing ExactPack is simple.  Just unpack the tar file and run the
setup script::

    tar xzf ExactPack-1.0.tar.gz
    cd ExactPack-1.0
    python setup.py install

This will install to the system Python library directories.  If you
do not have write permission for these directories, you can usually
install to a user specific location by using the ``--user`` flag.  All
the other standard setup options should also be available (run
``python setup.py --help`` for more information).

The setup script also installs the GUI and command-line scripts
described below.

.. _quickstart-gui:

The ExactPack Graphical User Interface
======================================

.. warning::

   As of this writing, the GUI is undergoing active, early-stage
   development, so the interface may have major changes in subsequent
   releases.

ExactPack has a simple GUI, which can be started by running from the
command line::

   epgui

The interface is relatively straightforward.  Choose a problem from
the menu, input values for any parameters for which the defaults are
not satisfactory, and click *Plot*.  Multiple plots can be combined,
or you can clear the canvas before plotting using *Clear and Plot*.
*Save* writes a CSV data file containing the data from the last item
plotted.  To save the plot itself, use the controls in the plot
window.

.. _quickstart-cmd-line:

Using ExactPack from the Command Line
=====================================

.. warning::

   As of this writing, the command-line interface is undergoing
   active, early-stage development, so the interface may have major
   changes in subsequent releases.  Users are advised not to develop
   any scripts or code that depend on the current behavior of the
   command-line tool.

Complete and current help information can be found by running::

   exactpack --help

Also, running::

   exactpack --doc

will open this manual (in either HTML or PDF format) in an appropriate
viewer.

.. _quickstart-library:

Using ExactPack as a Python Library
===================================

ExactPack is designed to be used to be used as a Python library in
postprocessing scripts for code verification.  It provides a number
of tools to make it easy to compare exact solutions against the
output of other codes, and for convergence analysis.

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

To get a complete list of the available solvers use
:func:`exactpack.utils.discover_solvers`.  Note that there may be
several solvers for a particular problem.  For example, by default
:mod:`exactpack.solvers.noh` loads a pure Python implementation,
:mod:`exactpack.solvers.noh.noh1`.  If you want the Fortran implementation by
Frank Timmes, you need to explicitly import
:mod:`exactpack.solvers.noh.timmes`.
     
For documentation on specific solvers, including information on what
parameters they accept and what solution methods are employed, see the
API documentation (:mod:`exactpack`).  For information about utility
functions and analysis tools, see the APIs for those specific packages
(:mod:`exactpack.utils` and :mod:`exactpack.analysis`).
