"""Unit testing for ExactPack.

Although (or, perhaps, because) one important use case for ExactPack
is for generating solutions to verify a general hydro code, ExactPack
itself needs to be tested.  The modules in :mod:`exactpack.test` contain
python :py:mod:`pytest` test cases designed to self-test the library.
To run a specific test, say for the Noh solver, type "pytest
exactpack/tests/test_noh.py".
"""
