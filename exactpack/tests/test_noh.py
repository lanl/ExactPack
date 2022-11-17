"""Unittests for the Noh solver.

Since the solution is an analytic expression, the tests essentially
consist of checks for typographical errors.
"""

import unittest

import numpy

import exactpack.solvers.noh.noh1 as noh1


class TestNoh1(unittest.TestCase):
    r"""Tests for :class:`exactpack.solvers.noh.noh1.Noh`.

    The tests consist of comparing the values at two points, one in
    front of the shock (:math:`r=0.3`) and one behind the shock
    (:math:`r=0.1`), to the analytic solutions at a fixed time
    (:math:`t=0.6`) and :math:`\gamma=5/3`.
    """

    @classmethod
    def setUpClass(self):

        self.soln = noh1.Noh(geometry=3, gamma=5.0/3.0)(numpy.array([0.1, 0.3]), 0.6)

    def test_velocity_error(self):
        """Noh Problem: Test for valid value of velocity"""

        self.assertRaises(ValueError, noh1.Noh, u0=+1)

    def test_preshock_density(self):
        """Noh problem: Pre-shock density"""

        self.assertEqual(self.soln.density[1], 9.0)

    def test_preshock_energy(self):
        """Noh problem: Pre-shock internal energy"""

        self.assertEqual(self.soln.sie[1], 0.0)

    def test_preshock_velocity(self):
        """Noh problem: Pre-shock velocity"""

        self.assertEqual(self.soln.velocity[1], -1.0)

    def test_preshock_pressure(self):
        """Noh problem: Pre-shock pressure"""

        self.assertEqual(self.soln.pressure[1], 0.0)

    def test_postshock_density(self):
        """Noh problem: Post-shock density"""

        self.assertEqual(self.soln.density[0], 64.0)

    def test_postshock_energy(self):
        """Noh problem: Post-shock internal energy"""

        self.assertEqual(self.soln.sie[0], 0.5)

    def test_postshock_velocity(self):
        """Noh problem: Post-shock velocity"""

        self.assertEqual(self.soln.velocity[0], 0.0)

    def test_postshock_pressure(self):
        """Noh problem: Post-shock pressure"""

        self.assertAlmostEqual(self.soln.pressure[0], 64.0/3.0)

    def test_geometry_error(self):
        """Noh Problem: Test for valid value of geometry"""

        self.assertRaises(ValueError, noh1.Noh, geometry=-1)


class TestNohWrappers(unittest.TestCase):
    """Test wrappers for Noh in specific geometries.

    Test the wrapper functions for specific geometries from
    :mod:`exactpack.solvers.noh.noh1`, by comparing the results computed via
    the wrappers to those from the general solver.
    """

    def test_planar(self):
        """Planar Noh wrapper"""

        numpy.testing.assert_array_equal(
            noh1.Noh(geometry=1, gamma=1.4)
            (numpy.linspace(0.1, 1), 0.6),
            noh1.PlanarNoh(gamma=1.4)
            (numpy.linspace(0.1, 1), 0.6))

    def test_cylindrical(self):
        """Cylindrical Noh wrapper"""

        numpy.testing.assert_array_equal(
            noh1.Noh(geometry=2, gamma=1.4)
            (numpy.linspace(0.1, 1), 0.6),
            noh1.CylindricalNoh(gamma=1.4)
            (numpy.linspace(0.1, 1), 0.6))

    def test_spherical(self):
        """Spherical Noh wrapper"""

        numpy.testing.assert_array_equal(
            noh1.Noh(geometry=3, gamma=1.4)
            (numpy.linspace(0.1, 1), 0.6),
            noh1.SphericalNoh(gamma=1.4)
            (numpy.linspace(0.1, 1), 0.6))
