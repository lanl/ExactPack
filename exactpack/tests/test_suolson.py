"""Unittests for the Su-Olson solver.
"""

import unittest

import numpy

from exactpack.solvers.suolson.timmes import SuOlson


class TestSuOlsonTimmes(unittest.TestCase):
    r"""Regresstion tests for :class:`exactpack.solvers.suolson.timmes.SuOlson`.

    Comparisions are made at :math:`r=0.1` and :math:`t=10^{-9}`.
    """
    solver = SuOlson(trad_bc_ev=1.0e3, opac=1.0)
    data = numpy.linspace(0, 20.0, 4)
    soln = solver(data, 1.e-9)

    def test_mat_temperature(self):
        """SuOlson problem: mat temperature"""
        expected = [955.9875670217054, 388.14357235758376,
                    72.16540048091636, 7.158933356184126]
        result = self.soln.Tmaterial
        numpy.testing.assert_allclose(result, expected)

    def test_radiation_temperature(self):
        """SuOlson problem: radiation temperature"""
        expected = [956.7434855255531, 396.89673904429804,
                    76.45509373805481, 7.802002997462728]
        result = self.soln.Tradiation
        numpy.testing.assert_allclose(result, expected)
