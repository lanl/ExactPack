"""Unittests for the Noh2 solver.
"""

import unittest

import numpy as np

from exactpack.solvers.noh2 import Noh2, PlanarNoh2, CylindricalNoh2, SphericalNoh2
from exactpack.solvers.noh2.noh2_cog import Noh2Cog


class TestNoh2(unittest.TestCase):
    """Tests for the Noh2 problem :class:`exactpack.solvers.noh2.noh2.Noh2`."""

    def test_noh2(self):
        """Regression test for Noh2 """

        t = 0.1
        r = np.linspace(0.0, 1.2, 10)
        solver = Noh2()
        soln = solver(r, t)
        ri = 2
        self.assertAlmostEqual(soln.density[ri], 1.371742112482853)
        self.assertAlmostEqual(soln.pressure[ri], 1.1290058538953525)
        self.assertAlmostEqual(soln.specific_internal_energy[ri], 1.2345679012345678)
        self.assertAlmostEqual(soln.velocity[ri], -0.2962962962962963)

    def test_illegal_value_t_noh2(self):
        """Confirm that illegal parameter values raise an error"""
        with self.assertRaises(ValueError):
            solver = Noh2()
            solver(r=[], t=2.0)

    def regression_test_planar_noh2(self):
        """Regression test for Planar Noh2 """

        t = 0.1
        r = np.linspace(0.0, 1.2, 10)
        solver = PlanarNoh2()
        soln = solver(r, t)
        ri = 2
        self.assertAlmostEqual(soln.density[ri], 1.1111111111111112)
        self.assertAlmostEqual(soln.pressure[ri], 0.794641468811218)
        self.assertAlmostEqual(soln.specific_internal_energy[ri], 1.0727659829)
        self.assertAlmostEqual(soln.velocity[ri], -0.296296296296)

    def regression_test_cylindrical_noh2(self):
        """Regression test for Cylindrical Noh2 """

        t = 0.1
        r = np.linspace(0.0, 1.2, 10)
        solver = CylindricalNoh2()
        soln = solver(r, t)
        ri = 2
        self.assertAlmostEqual(soln.density[ri], 1.23456790123)
        self.assertAlmostEqual(soln.pressure[ri], 0.947182595932)
        self.assertAlmostEqual(soln.specific_internal_energy[ri], 1.15082685406)
        self.assertAlmostEqual(soln.velocity[ri], -0.296296296296)

    def regression_test_spherical_noh2(self):
        """Regression test Spherical Noh2 """

        t = 0.1
        r = np.linspace(0.0, 1.2, 10)
        solver = CylindricalNoh2()
        soln = solver(r, t)
        ri = 2
        self.assertAlmostEqual(soln.density[ri], 1.371742112482853)
        self.assertAlmostEqual(soln.pressure[ri], 1.1290058538953525)
        self.assertAlmostEqual(soln.specific_internal_energy[ri], 1.2345679012345678)
        self.assertAlmostEqual(soln.velocity[ri], -0.2962962962962963)


class TestNoh2Cog(unittest.TestCase):
    """Regression test for Noh2Cog solver.

    Noh2 is a special case of Cog1. The solver Noh2Cog is implemented in
    :class:`exactpack.solvers.noh2.noh2_cog.Noh2Cog`, which inherits from Cog1.
    """

    def test_noh2_cog(self):
        """Regression test for Noh2Cog."""
        t = 0.1
        r = np.linspace(0.0, 1.2, 10)
        solver = Noh2Cog()
        soln = solver(r, t)
        ri = 2
        self.assertAlmostEqual(soln.density[ri], 1.371742112482853)
        self.assertAlmostEqual(soln.pressure[ri], 1.1290058538953525)
        self.assertAlmostEqual(soln.specific_internal_energy[ri], 1.2345679012345678)
        self.assertAlmostEqual(soln.velocity[ri], -0.2962962962962963)

    def test_illegal_value_t_noh2cog(self):
        """Confirm that illegal parameter values raise an error"""
        with self.assertRaises(ValueError):
            solver = Noh2Cog()
            solver(r=[], t=2.0)

    def test_noh2_cog_cf(self):
        """Compares Noh2 with Noh2Cog."""
        t = 0.1
        r = np.linspace(0.0, 1.2, 10)
        solver1 = Noh2Cog()
        soln1 = solver1(r, t)
        solver2 = Noh2()
        soln2 = solver2(r, t)
        ri = 2
        np.testing.assert_array_almost_equal(soln1.density[ri], soln2.density[ri])
        np.testing.assert_array_almost_equal(soln1.pressure[ri], soln2.pressure[ri])
        np.testing.assert_array_almost_equal(soln1.specific_internal_energy[ri], soln2.specific_internal_energy[ri])
        np.testing.assert_array_almost_equal(soln1.velocity[ri], soln2.velocity[ri])
