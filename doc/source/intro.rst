Introduction
============

For code verification, one compares the code output against known
exact solutions. There are many standard test problems used in this
capacity, such as the Noh and Sedov problems.
ExactPack is a python package that integrates many of these exact
solution codes into a common API (application program interface). ExactPack
consists of python driver scripts that access a library of exact
solutions written in Fortran or Python. The spatial profiles of the
relevant physical quantities, such as the density, fluid velocity, sound
speed, or internal energy, are returned at a time specified by the
user.  We have documented the physics
of each problem in the solution library, and provided complete
documentation on how to extend the library to include additional exact
solutions.  ExactPack's code architecture makes it easy to extend the
solution-code library to include additional exact solutions in a
robust, reliable, and maintainable manner.

How To Get Started
------------------

To get up and running with ExactPack quickly, you should start by
reading the :ref:`quickstart` and trying out some of the examples.
There are examples available for all of the solvers, as well as for
the convergence analysis tools.  Examples are located in the
distribution directory under :file:`exactpack/examples`, and can be
run as standalone Python scripts, or as Python modules::

  python -m exactpack.<example name>

If you want to use ExactPack for code verification or physics
exploration, the :ref:`users-guide` gives a complete explanation of
how to use ExactPack.  You will then want to look up the details of
the specific solver you are working with in :ref:`solvers`.

For those who want to add new solvers to ExactPack, read the
:ref:`developers-guide`, with particular attention to
:ref:`coding-style` and :ref:`adding-a-solver`.  The
:ref:`reference-guide` is mostly intended for those wishing a more
thorough understanding of ExactPack's internals, such as developers
adding additional functionality for the code verification tools, etc.

Finally, the :ref:`testing` section documents what internal self-tests
ExactPack uses to verify the solutions it provides.


How To Get Help
---------------

The first place to look is in this manual, particularly the API
documentation, as well as looking at the source code itself.  If you
need additional assistance, or wish to report a bug, please e-mail
exactpack-support@lanl.gov.  This software is provided free of charge,
and we make no guarantee of support, but will be happy to look at your
issue if we can.
