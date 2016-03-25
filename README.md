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
