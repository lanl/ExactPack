"""Unit tests for the Noh solver.

Since the solution is an analytic expression, the tests essentially
consist of checks for typographical errors.
"""

import pytest

import numpy as np

import exactpack.solvers.noh.noh1 as noh1


class TestNoh1():
    r"""Tests for :class:`exactpack.solvers.noh.noh1.Noh`.

    The tests consist of comparing the values at two points, one in
    front of the shock (:math:`r=0.3`) and one behind the shock
    (:math:`r=0.1`), to the analytic solutions at a fixed time
    (:math:`t=0.6`) and :math:`\gamma=5/3`.
    """

    @classmethod
    def setup_method(self):

        self.soln = noh1.Noh(geometry=3, gamma=5.0/3.0)(np.array([0.1, 0.3]), 0.6)

    def test_velocity_error(self):
        """Noh Problem: Test for valid value of velocity"""

        with pytest.raises(ValueError):
            noh1.Noh(u0=+1)

    def test_preshock_density(self):
        """Noh problem: Pre-shock density"""

        np.testing.assert_array_equal(self.soln.density[1], 9.0)

    def test_preshock_energy(self):
        """Noh problem: Pre-shock internal energy"""

        np.testing.assert_array_equal(self.soln.specific_internal_energy[1], 0.0)

    def test_preshock_velocity(self):
        """Noh problem: Pre-shock velocity"""

        np.testing.assert_array_equal(self.soln.velocity[1], -1.0)

    def test_preshock_pressure(self):
        """Noh problem: Pre-shock pressure"""

        np.testing.assert_array_equal(self.soln.pressure[1], 0.0)

    def test_postshock_density(self):
        """Noh problem: Post-shock density"""

        np.testing.assert_array_equal(self.soln.density[0], 64.0)

    def test_postshock_energy(self):
        """Noh problem: Post-shock internal energy"""

        np.testing.assert_array_equal(self.soln.specific_internal_energy[0], 0.5)

    def test_postshock_velocity(self):
        """Noh problem: Post-shock velocity"""

        np.testing.assert_array_equal(self.soln.velocity[0], 0.0)

    def test_postshock_pressure(self):
        """Noh problem: Post-shock pressure"""

        np.testing.assert_allclose(self.soln.pressure[0], 64.0/3.0)

    def test_geometry_error(self):
        """Noh Problem: Test for valid value of geometry"""

        with pytest.raises(ValueError):
            noh1.Noh(geometry=-1)


class TestNohWrappers():
    """Test wrappers for Noh in specific geometries.

    Test the wrapper functions for specific geometries from
    :mod:`exactpack.solvers.noh.noh1`, by comparing the results computed via
    the wrappers to those from the general solver.
    """

    def test_planar(self):
        """Planar Noh wrapper"""

        np.testing.assert_array_equal(
            noh1.Noh(geometry=1, gamma=1.4)
            (np.linspace(0.1, 1), 0.6),
            noh1.PlanarNoh(gamma=1.4)
            (np.linspace(0.1, 1), 0.6))

    def test_cylindrical(self):
        """Cylindrical Noh wrapper"""

        np.testing.assert_array_equal(
            noh1.Noh(geometry=2, gamma=1.4)
            (np.linspace(0.1, 1), 0.6),
            noh1.CylindricalNoh(gamma=1.4)
            (np.linspace(0.1, 1), 0.6))

    def test_spherical(self):
        """Spherical Noh wrapper"""

        np.testing.assert_array_equal(
            noh1.Noh(geometry=3, gamma=1.4)
            (np.linspace(0.1, 1), 0.6),
            noh1.SphericalNoh(gamma=1.4)
            (np.linspace(0.1, 1), 0.6))
