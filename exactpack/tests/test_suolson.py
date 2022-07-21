"""Tests for the Su-Olson problem.
"""

import unittest

import numpy

# This will fail to import because the suolsen solver won't build
# from exactpack.solvers.suolson.timmes import SuOlson

class TestSuOlsonTimmes(unittest.TestCase):
    r"""Test for :class:`exactpack.suolson.timmes.SuOlson`. 

    Comparisions are made at :math:`r=0.1` and :math:`t=10^{-9}`.
    """
    @unittest.expectedFailure
    def test_radiation_temperature(self):
        """SuOlson problem: radiation temperature"""

        self.assertAlmostEqual(SuOlson(trad_bc_ev=1.0e3,opac=1.0)(numpy.array([0.1]),1.e-9).Tradiation[0], 949.7484176757317)

    @unittest.expectedFailure
    def test_matter_temperature(self):
        """SuOlson problem: matter temperature"""

        self.assertAlmostEqual(SuOlson(trad_bc_ev=1.0e3,opac=1.0)(numpy.array([0.1]),1.e-9).Tmaterial[0], 948.8608008525508)


        
