.. ExactPack documentation master file, created by
   sphinx-quickstart on Thu Dec 12 11:26:25 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#######################
ExactPack Documentation
#######################

ExactPack is a collection of exact hydrodynamics solutions packaged as
an easy to use Python library with a uniform interface.  In this
context, an exact solution is not necessarily an analytic expression.
Most solutions provided by the library actually require the solution
of an ordinary differential equation or the numerical approximation of
an integral.

The primary purpose of ExactPack is to provide reference solutions for
verification analysis of hydrodynamics codes.  A graphics user
interface and command-line utility are both also provided, which makes
it easy to use ExactPack as an interactive exploration tool, as well.

.. toctree::
   :maxdepth: 2
   :numbered:

   quickstart
   users_guide
   developers_guide
   api
   solvers
   testing

Indices and tables
==================

.. toctree::

   credits
   todos
   glossary

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
