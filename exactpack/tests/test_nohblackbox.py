"""Unit tests for the NohBlackBox solver.
"""

import pytest

import numpy as np

from exactpack.solvers.nohblackboxeos import NohBlackBoxEos, PlanarNohBlackBox, CylindricalNohBlackBox, SphericalNohBlackBox


class ideal_gas_eos: 
    def __init__(self, gamma = 5/3): 
        self.gamma = gamma
        assert(gamma != 1), "Gamma cannot be equal to 1"

    def e(self, rho, P): 
        if(rho == 0.0):
            raise ValueError("Error: rho = 0. Please do not break math and divide by zero.")
        return P/(rho*(self.gamma -1.))     
    
    def de_dP(self, rho, P): 
        if(rho == 0.0):
            raise ValueError("Error: rho = 0. Please do not break math and divide by zero.")
        return 1./(rho*(self.gamma-1))
    
    def de_drho(self, rho,P): 
        if(rho == 0.0):
            raise ValueError("Error: rho = 0. Please do not break math and divide by zero.")
        return -P/(rho**2*(self.gamma-1))


class TestNohBlackBoc():
    r"""Tests for :class:`exactpack.solvers.noh.noh1.Noh`.

    The tests consist of comparing the values at two points, one in
    front of the shock (:math:`r=0.3`) and one behind the shock
    (:math:`r=0.1`), to the analytic solutions at a fixed time
    (:math:`t=0.6`) and :math:`\gamma=5/3`.
    """

    @classmethod
    def setup_method(self):
        self.ideal_gas = ideal_gas_eos()
        self.solver = NohBlackBoxEos(self.ideal_gas)
        self.solver.solve_jump_conditions()
        self.soln = self.solver(np.array([0.1, 0.3]), 0.6)

    def test_velocity_error(self):
        """Noh Problem: Test for valid value of velocity"""

        with pytest.raises(ValueError):
            NohBlackBoxEos(self.ideal_gas, u0=+1)

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

        np.testing.assert_array_equal(self.soln.density[0], pytest.approx(64.0))

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
            NohBlackBoxEos(self.ideal_gas, geometry=-1)


class TestNohWrappers():
    """Test wrappers for Noh in specific geometries.

    Test the wrapper functions for specific geometries from
    :mod:`exactpack.solvers.noh.noh1`, by comparing the results computed via
    the wrappers to those from the general solver.
    """
    @classmethod
    def setup_method(self):
        self.ideal_gas = ideal_gas_eos(gamma=1.4)

    def test_planar(self):
        """Planar Noh wrapper"""
        base_solver = NohBlackBoxEos(self.ideal_gas, geometry=1)
        base_solver.solve_jump_conditions()
        planar_solver = PlanarNohBlackBox(self.ideal_gas)
        planar_solver.solve_jump_conditions()

        np.testing.assert_array_equal(
            base_solver(np.linspace(0.1, 1), 0.6),
            planar_solver(np.linspace(0.1, 1), 0.6))

    def test_cylindrical(self):
        """Cylindrical Noh wrapper"""
        base_solver = NohBlackBoxEos(self.ideal_gas, geometry=2)
        base_solver.solve_jump_conditions()
        cylin_solver = CylindricalNohBlackBox(self.ideal_gas)
        cylin_solver.solve_jump_conditions()

        np.testing.assert_array_equal(
            base_solver(np.linspace(0.1, 1), 0.6),
            cylin_solver(np.linspace(0.1, 1), 0.6))

    def test_spherical(self):
        """Spherical Noh wrapper"""
        base_solver = NohBlackBoxEos(self.ideal_gas, geometry=3)
        base_solver.solve_jump_conditions()
        sph_solver = SphericalNohBlackBox(self.ideal_gas)
        sph_solver.solve_jump_conditions()

        np.testing.assert_array_equal(
            base_solver(np.linspace(0.1, 1), 0.6),
            sph_solver(np.linspace(0.1, 1), 0.6))
