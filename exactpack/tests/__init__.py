"""Unit testing for ExactPack.

Although (or, perhaps, because) one important use case for ExactPack
is for generating solutions to verify a general hydro code, ExactPack
itself needs to be tested.  The modules in :mod:`exactpack.test` contain
python :py:mod:`unittest` test cases designed to self-test the library.
To run a specific test, say for the Noh solver, type "python -m unittest
exactpack.tests.test_noh".
"""
