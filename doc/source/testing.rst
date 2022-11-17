.. _testing:

*******
Testing
*******

This chapter documents the internal self-tests of the library.  The
tests are all implemented as Python :mod:`pytest` unit tests, and can be run
using the standard :mod:`pytest` framework.  The easiest way to
test the library is through the ``pytest`` command::

   pytest

To test a specific solver just give the path to the test module::

   pytest exactpack/tests/test_noh.py

.. toctree::
   :glob:

   tests/*

.. automodule:: exactpack.tests
   :members:

