# ExactPack: An Open-Source Software Package for Code Verification

ExactPack is an open-source software package that has been developed for the verification & validation community. The package has two major capabilities:

1. The generation of exact solutions for common benchmark problems in computational physics.
2. The analysis of computational physics code output to assess accuracy and convergence rates.

ExactPack is designed to be an open project, readily expandable to include new test problems for new physical models. At present, it contains the following modules and solutions (see reference list for details of these test problems):

* Hydrodynamics
  * Noh problem 
  * Sedov problem 
  * Riemann problem
* Reactive flows
  * Escape of high-explosive products
  * Programmed burn timing
  * Steady-detonation reaction-zone
  * Blake problem

The analysis capabilities of ExactPack include the import of data from computational physics codes, calculation of error norms between exact solutions and numerical results, and the calculation of spatial convergence rates. These tools follow the standards defined by the American Society of Mechanical Engineers (ASME).

Dependencies
------------

Python dependencies:
* Python 2.7
* setuptools
* numpy
* sphinx
* matplotlib
* scipy

Other dependencies:
* C compiler
* Fortran compiler
* VTK

*ExactPack* is written in Python and interfaces readily with compiled code written in other languages such as Fortran and C. It is designed to be object-oriented



Installation
------------

ExactPack is a Python library providing a set of exact solution
solvers for use in verifying numerical codes, along with some error
and convergence analysis tools.  It can be installed in the standard
Python manner by running the setup.py script.  If you are not familiar
with how this is done, you may want to ask your system adminstrator to
help you.

Complete documentation, including a quickstart guide, can be found in
the :file:`doc/` directory.  The documentation can be built by doing::

  cd doc
  make html

The HTML documentation can then be found in the file
:file:`build/html/index.html`.

Note on MathJax Behind a Firewall
---------------------------------

The ExactPack documentation includes many mathematical formulae.  By
default, these are rendered in MathJax.  In order for the HTML
documentation to load, the system must be able to find the MathJax
scripts.

If you are operating behind a firewall, the public distribution of
MathJax (http://www.mathjax.org) may not be accessible.  The easiest
solution is to download a local copy of MathJax and point to it with
the :envvar:`MATHJAX_PATH` environment variable when building the
HTML files.  (Note that the path is hard-coded into the resulting
HTML, the :envvar:`MATHJAX_PATH` environment variable will have no
effect when loading the HTML files.)

It is also possible to modify the :file:`conf.py` file to use another
math renderer, or to point to a local copy of MathJax.

References
----------
## The Noh Problem
W. F. Noh, *Errors for Calculations of Strong Shocks Using an Artificial Viscosity and an Artificial Heat Flux*, Journal of Computational Physics, **72** (1978), pp 78-120.

Francis X. Timmes, Galen Gisler, George Hrbek, *Automated analyses of the Tri-Lab verification test suite on uniform and adaptive grids for Code Project A*, Los Alamos report, LA-UR-05-6865 (2005).

M. Gehmeyr, B. Cheng, D. Mihalas, *Noh's Constant-Velocity Shock Problem Revisited*, Shock Waves, **7** 255 (1997).

## The Riemann Problem
E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Springer, 2009.

## Escape of HE Products

W. Fickett and C. Rivard, *Test Problems for Hydrocodes*, LASL Report, LA-5479 (1974, Rev 1981).

P. Dykema, S. Brandon, J. Bolstad, T. Woods, R. Klein, *Level 1 V&V, Test Problem 10: Escape of High Explosive Products, LLNL Report, UCRL-ID-150418 (2002).

S. Doebling, *The Escape of High Explosive Products: An Exact-Solution Problem for Verification of Hydrodynamics Codes*, LA-UR-15-22547

* Escape of high-explosive products
  * Programmed burn timing
  * Steady-detonation reaction-zone
  * Blake problem


