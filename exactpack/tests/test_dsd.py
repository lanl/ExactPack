r"""Tests for the DSD verification problems exact solution solvers.

Test problems consist of comparison of the calculated burntime for a 2D
point with the known analytic solution. Solver parameter inputs are
tested individually, both for validity and for new solutions.

A time value is passed to the ExactPack solver in order to maintain
a consistent format of input with the other solvers. Because the
burn time solution is not time dependent, this value is ignored by
the solver.
"""

import unittest

import numpy as np

# from exactpack.solvers.dsd.ratestick import RateStick
from exactpack.solvers.dsd.cylexpansion import CylindricalExpansion
# from exactpack.solvers.dsd.explosivearc import ExplosiveArc


# class TestRateStick(unittest.TestCase):
#     r"""Tests for
#     :class:`exactpack.solvers.dsd.ratestick.RateStick`.
#
#     Solution tests consist of comparing the calculated burn time to the
#     analytic solution at a fixed point. Input tests check that invalid
#     input raises the appropriate error expression.
#     """
#
#     def test_burntime_2d_base(self):
#         """Tests burntime solution at 2D point.
#
#         Uses default parameter values for all parameters.
#         """
#
#         soln = RateStick(geometry=2)(np.array([[3.0, 4.0]]), 0.6)
#
#         self.assertEqual(soln.burntime, 5.0)
#
#     def test_burntime_2d_detvel(self):
#         """Tests burntime solution at 2D point.
#
#         Uses default parameter values for detonator location and detonation
#         time. HE detonation velocity is :math:`D=2.0`.
#         """
#
#         soln = RateStick(geometry=2, D=2.0)(np.array([[3.0, 4.0]]), 0.6)
#
#         self.assertEqual(soln.burntime, 2.5)
#
#     def test_burntime_2d_dettime(self):
#         """Tests burntime solution at 2D point.
#
#         Uses default parameter values for HE detonation velocity and
#         detonator location. Detonation time is :math:`t_d=-2.0`.
#         """
#
#         soln = RateStick(geometry=2, t_d=-2.0)(np.array([[3.0, 4.0]]), 0.6)
#
#         self.assertEqual(soln.burntime, 3.0)
#
#     def test_burntime_2d_detloc(self):
#         """Tests burntime solution at 2D point.
#
#         Uses default parameter values for HE detonation velocity and
#         detonation time. Detonator location is :math:`x_d=(1.0, 1.0)`.
#         """
#
#         soln = RateStick(geometry=2,
#                          x_d=(1.0, 1.0))(np.array([[2.5, 3.0]]), 0.6)
#
#         self.assertEqual(soln.burntime, 2.5)
#
#     def test_geometry_error(self):
#         """Test for valid value of geometry."""
#
#         self.assertRaisesRegexp(ValueError, "geometry must be 1 or 2",
#                                 RateStick, geometry=5)
#
#     def test_D_CJ_neg_error(self):
#         """Test for valid value of HE detonation velocity, :math:`D`."""
#
#         self.assertRaisesRegexp(ValueError, "Det velocity must be > 0",
#                                 RateStick, D_CJ=-1.0)
#
#     def test_D_CJ_zero_error(self):
#         """Test for valid value of HE detonation velocity, :math:`D`."""
#
#         self.assertRaisesRegexp(ValueError, "Det velocity must be > 0",
#                                 RateStick, D_CJ=0.0)
#
#     def test_detspec_2d_error(self):
#         """Test for valid geometry of detonator, :math:`x_d`."""
#
#         self.assertRaisesRegexp(ValueError,
#                                 "Detonator location and geometry " +
#                                 "dimensions must be compatible",
#                                 RateStick, x_d=(0.0, 0.0, 0.0))
#

class TestCylindricalExpansion(unittest.TestCase):
    r"""Tests for
    :class:`exactpack.solvers.dsd.cylexpansion.CylindricalExpansion`.

    Solution tests consist of comparing the calculated burn time to the
    analytic solution at a fixed point. Input tests check that invalid
    input raises the appropriate error expression.
    """

    def test_burntime_base_HE1(self):
        """Tests burntime solution at point in HE1 region.

        Uses default parameter values.
        """

        soln = CylindricalExpansion()(np.array([[0.9, 1.2]]), 0.6)

        answer = 1.0 + 0.4 * np.log(1.3 / 0.8)

        self.assertEqual(soln.burntime, answer)

    def test_burntime_base_HE2(self):
        """Tests burntime solution at point in HE2 region.

        Uses default parameter values.
        """

        soln = CylindricalExpansion()(np.array([[1.8, 2.4]]), 0.6)

        tmeet = 2.0 + 0.4 * np.log(1.8 / 0.8)
        answer = tmeet + 1.0 + 0.1 * np.log(2.9 / 1.9)

        self.assertEqual(soln.burntime, answer)

    def test_burntime_base_indet(self):
        """Tests burntime solution at point in HE2 region.

        Uses default parameter values.
        """

        soln = CylindricalExpansion()(np.array([[0.3, 0.4]]), 0.6)

        self.assertEqual(soln.burntime, 0.0)

    def test_burntime_1(self):
        """Tests burntime solution at points in each HE region.

        Uses default parameter values for :math:`r_2`, :math:`D_{{CJ}_1}`,
        :math:`D_{{CJ}_2}`, :math:`\alpha_1`, :math:`\alpha_2`, :math:`t_d`.
        The inner radius of HE1 is :math:`r_1=1.2`.
        """

        points = np.array([[0.9, 1.2], [1.8, 2.4]])

        soln = CylindricalExpansion(r_1=1.2)(points, 0.6)

        answer1 = 0.6 + 0.4 * np.log(1.3 / 1.0)
        tmeet = 1.6 + 0.4 * np.log(1.8 / 1.0)
        answer2 = tmeet + 1.0 + 0.1 * np.log(2.9 / 1.9)
        answer = np.array([answer1, answer2])

        abserr = 1.0e-15

        np.testing.assert_allclose(soln.burntime, answer, atol=abserr)

    def test_burntime_2(self):
        """Tests burntime solution at points in each HE region.

        Uses default parameter values for :math:`r_1`, :math:`D_{{CJ}_1}`,
        :math:`D_{{CJ}_2}`, :math:`\alpha_1`, :math:`\alpha_2`, :math:`t_d`.
        The radius of interface between HE1 and HE2 is :math:`r_2=2.5`.
        """

        points = np.array([[0.9, 1.2], [1.8, 2.4]])

        soln = CylindricalExpansion(r_2=2.5)(points, 0.6)

        answer1 = 1.0 + 0.4 * np.log(1.3 / 0.8)
        tmeet = 3.0 + 0.4 * np.log(2.3 / 0.8)
        answer2 = tmeet + 0.5 + 0.1 * np.log(2.9 / 2.4)
        answer = np.array([answer1, answer2])

        abserr = 1.0e-15

        np.testing.assert_allclose(soln.burntime, answer, atol=abserr)

    def test_burntime_DCJ1(self):
        """Tests burntime solution at points in each HE region.

        Uses default parameter values for :math:`r_1`, :math:`r_2`,
        :math:`D_{{CJ}_2}`, :math:`\alpha_1`, :math:`\alpha_2`, :math:`t_d`.
        The CJ detonation velocity of HE1 is :math:`D_{{CJ}_1}=1.0`.
        """

        points = np.array([[0.9, 1.2], [1.8, 2.4]])

        soln = CylindricalExpansion(D_CJ_1=1.0)(points, 0.6)

        answer1 = 0.5 + 0.1 * np.log(1.4 / 0.9)
        tmeet = 1.0 + 0.1 * np.log(1.9 / 0.9)
        answer2 = tmeet + 1.0 + 0.1 * np.log(2.9 / 1.9)
        answer = np.array([answer1, answer2])

        abserr = 1.0e-15

        np.testing.assert_allclose(soln.burntime, answer, atol=abserr)

    def test_burntime_DCJ2(self):
        """Tests burntime solution at points in each HE region.

        Uses default parameter values for :math:`r_1`, :math:`r_2`,
        :math:`D_{{CJ}_1}`, :math:`\alpha_1`, :math:`\alpha_2`, :math:`t_d`.
        The CJ detonation velocity of HE1 is :math:`D_{{CJ}_2}=0.5`.
        """

        points = np.array([[0.9, 1.2], [1.8, 2.4]])

        soln = CylindricalExpansion(D_CJ_2=0.5)(points, 0.6)

        answer1 = 1.0 + 0.4 * np.log(1.3 / 0.8)
        tmeet = 2.0 + 0.4 * np.log(1.8 / 0.8)
        answer2 = tmeet + 2.0 + 0.4 * np.log(2.8 / 1.8)
        answer = np.array([answer1, answer2])

        abserr = 1.0e-15

        np.testing.assert_allclose(soln.burntime, answer, atol=abserr)

    def test_burntime_alpha1(self):
        """Tests burntime solution at points in each HE region.

        Uses default parameter values for :math:`r_1`, :math:`r_2`,
        :math:`D_{{CJ}_1}`, :math:`D_{{CJ}_2}`, :math:`\alpha_2`, :math:`t_d`.
        The CJ detonation velocity of HE1 is :math:`\alpha_1=0.05`.
        """

        points = np.array([[0.9, 1.2], [1.8, 2.4]])

        soln = CylindricalExpansion(alpha_1=0.05)(points, 0.6)

        answer1 = 1.0 + 0.2 * np.log(1.4 / 0.9)
        tmeet = 2.0 + 0.2 * np.log(1.9 / 0.9)
        answer2 = tmeet + 1.0 + 0.1 * np.log(2.9 / 1.9)
        answer = np.array([answer1, answer2])

        abserr = 1.0e-15

        np.testing.assert_allclose(soln.burntime, answer, atol=abserr)

    def test_burntime_alpha2(self):
        """Tests burntime solution at points in each HE region.

        Uses default parameter values for :math:`r_1`, :math:`r_2`,
        :math:`D_{{CJ}_1}`, :math:`D_{{CJ}_2}`, :math:`\alpha_1`, :math:`t_d`.
        The CJ detonation velocity of HE1 is :math:`\alpha_2=0.05`.
        """

        points = np.array([[0.9, 1.2], [1.8, 2.4]])

        soln = CylindricalExpansion(alpha_2=0.05)(points, 0.6)

        answer1 = 1.0 + 0.4 * np.log(1.3 / 0.8)
        tmeet = 2.0 + 0.4 * np.log(1.8 / 0.8)
        answer2 = tmeet + 1.0 + 0.05 * np.log(2.95 / 1.95)
        answer = np.array([answer1, answer2])

        abserr = 1.0e-15

        np.testing.assert_allclose(soln.burntime, answer, atol=abserr)

    def test_burntime_tdet(self):
        """Tests burntime solution at points in each HE region.

        Uses default parameter values for :math:`r_1`, :math:`r_2`,
        :math:`D_{{CJ}_1}`, :math:`D_{{CJ}_2}`, :math:`\alpha_1`,
        :math:`\alpha_2`. Detonation time of HE1 is :math:`t_d=3.0`.
        """

        points = np.array([[0.9, 1.2], [1.8, 2.4]])

        soln = CylindricalExpansion(t_d=3.0)(points, 0.6)

        answer1 = 3.0 + 1.0 + 0.4 * np.log(1.3 / 0.8)
        tmeet = 3.0 + 2.0 + 0.4 * np.log(1.8 / 0.8)
        answer2 = tmeet + 1.0 + 0.1 * np.log(2.9 / 1.9)
        answer = np.array([answer1, answer2])

        abserr = 1.0e-15

        np.testing.assert_allclose(soln.burntime, answer, atol=abserr)

    def test_geometry_error(self):
        """Test for valid value of geometry."""

        self.assertRaisesRegexp(ValueError, "geometry must be 2",
                                CylindricalExpansion, geometry=1)

    def test_r1_neg_error(self):
        """Test for valid value of inner radius, :math:`r_1`."""

        self.assertRaisesRegexp(ValueError,
                                "Inner radius of HE1 must be > 0",
                                CylindricalExpansion, r_1=-1.0)

    def test_r1_zero_error(self):
        """Test for valid value of inner radius, :math:`r_1`."""

        self.assertRaisesRegexp(ValueError,
                                "Inner radius of HE1 must be > 0",
                                CylindricalExpansion, r_1=0.0)

    def test_r2_neg_error(self):
        """Test for valid value of HE interface radius, :math:`r_2`."""

        self.assertRaisesRegexp(ValueError, "Radius of interface between " +
                                "HE1 and HE2 must be > 0",
                                CylindricalExpansion, r_2=-1.0)

    def test_r2_zero_error(self):
        """Test for valid value of HE interface radius, :math:`r_2`."""

        self.assertRaisesRegexp(ValueError, "Radius of interface between " +
                                "HE1 and HE2 must be > 0",
                                CylindricalExpansion, r_2=0.0)

    def test_r2_smaller_error(self):
        """Test for valid value of HE interface radius, :math:`r_2`."""

        self.assertRaisesRegexp(ValueError, "Radius of interface between " +
                                "HE1 and HE2 must be > inner radius",
                                CylindricalExpansion, r_1=2.0, r_2=1.0)

    def test_DCJ1_neg_error(self):
        """Test for valid value of inner region HE CJ detonation velocity,
        :math:`D_{{CJ}_1}`."""

        self.assertRaisesRegexp(ValueError, "Detonation velocity for " +
                                "inner HE must be > 0",
                                CylindricalExpansion, D_CJ_1=-1.0)

    def test_DCJ1_zero_error(self):
        """Test for valid value of inner region HE CJ detonation velocity,
        :math:`D_{{CJ}_1}`."""

        self.assertRaisesRegexp(ValueError, "Detonation velocity for " +
                                "inner HE must be > 0",
                                CylindricalExpansion, D_CJ_1=0.0)

    def test_DCJ2_neg_error(self):
        """Tests for valid value of outer region HE CJ detonation velocity,
        :math:`D_{{CJ}_2}`."""

        self.assertRaisesRegexp(ValueError, "Detonation velocity for " +
                                "outer HE must be > 0",
                                CylindricalExpansion, D_CJ_2=-1.0)

    def test_DCJ2_zero_error(self):
        """Tests for valid value of outer region HE CJ detonation velocity,
        :math:`D_{{CJ}_2}`."""

        self.assertRaisesRegexp(ValueError, "Detonation velocity for " +
                                "outer HE must be > 0",
                                CylindricalExpansion, D_CJ_2=0.0)

    def test_alpha1_neg_error(self):
        """Test for valid value of inner region det velocity deviance
        coefficient, :math:`\alpha_1`."""

        self.assertRaisesRegexp(ValueError,
                                "Alpha for HE1 must be >= 0",
                                CylindricalExpansion, alpha_1=-1.0)

    def test_alpha2_neg_error(self):
        """Test for valid value of outer region det velocity deviance
        coefficient, :math:`\alpha_2`."""

        self.assertRaisesRegexp(ValueError,
                                "Alpha for HE2 must be >= 0",
                                CylindricalExpansion, alpha_2=-1.0)

    def test_omegac_neg_error(self):
        """Test for valid value of HE/inert edge angle, :math:`\omega_c`."""

        self.assertRaisesRegexp(ValueError, "omega_c must be > 0",
                                CylindricalExpansion, omega_c=-1.0)

    def test_omegac_zero_error(self):
        """Test for valid value of HE/inert edge angle, :math:`\omega_c`."""

        self.assertRaisesRegexp(ValueError, "omega_c must be > 0",
                                CylindricalExpansion, omega_c=0.0)

    def test_omegac_max_error(self):
        """Test for valid value of HE/inert edge angle, :math:`\omega_c`."""

        self.assertRaisesRegexp(ValueError, "omega_c must be < pi/2",
                                CylindricalExpansion, omega_c=2.0)

    def test_omegas_neg_error(self):
        """Test for valid value of HE free-surface angle, :math:`\omega_s`."""

        self.assertRaisesRegexp(ValueError, "omega_s must be > 0",
                                CylindricalExpansion, omega_s=-1.0)

    def test_omegas_zero_error(self):
        """Test for valid value of HE free-surface angle, :math:`\omega_s`."""

        self.assertRaisesRegexp(ValueError, "omega_s must be > 0",
                                CylindricalExpansion, omega_s=0.0)

    def test_omegas_max_error(self):
        """Test for valid value of HE free-surface angle, :math:`\omega_s`."""

        self.assertRaisesRegexp(ValueError, "omega_s must be < pi/2",
                                CylindricalExpansion, omega_s=2.0)

    def test_omegac_smaller_error(self):
        """Test for valid value of HE/inert edge angle, :math:`\omega_c`."""

        self.assertRaisesRegexp(ValueError,
                                "omega_c must be >= omega_s",
                                CylindricalExpansion,
                                omega_c=0.4, omega_s=0.5)


# class TestExplosiveArc(unittest.TestCase):
#     r"""Tests for :class:`exactpack.solvers.dsd.explosivearc.ExplosiveArc`.
#
#     Solution tests consist of comparing the calculated burn time to the
#     analytic solution at a fixed point. Input tests check that invalid
#     input raises the appropriate error expression.
#     """
#
#     def test_burntime_2d_base(self):
#         """Tests burntime solution at 2D point in LOS region.
#
#         Uses default parameter values.
#         """
#
#         soln = ExplosiveArc(geometry=2)(np.array([[4.0, 2.0]]), 0.6)
#
#         self.assertEqual(soln.burntime, 2.5)
#
#     def test_burntime_2d_detvel(self):
#         """Tests burntime solution at 2D point in LOS region.
#
#         Uses default parameter values for inert region radius, detonator
#         location and detonation time. HE detonation velocity is
#         :math:`D=1.0`.
#         """
#
#         soln = ExplosiveArc(geometry=2, D=1.0)(np.array([[4.0, 2.0]]), 0.6)
#
#         self.assertEqual(soln.burntime, 5.0)
#
#     def test_burntime_2d_dettime(self):
#         """Tests burntime solution at 2D point in LOS region.
#
#         Uses default parameter values for inert region radius, HE detonation
#         velocity and detonator location. Detonation time is :math:`t_d=-2.0`.
#         """
#
#         soln = ExplosiveArc(t_d=-2.0)(np.array([[4.0, 2.0]]), 0.6)
#
#         self.assertEqual(soln.burntime, 0.5)
#
#     def test_burntime_2d_detloc(self):
#         """Tests burntime solution at 2D point in LOS region.
#
#         Uses default parameter values for inert region radius, HE detonation
#         velocity and detonation time. Detonator location is
#         :math:`x_d=(0.0, -5.0)`.
#         """
#
#         soln = ExplosiveArc(geometry=2,
#                          x_d=(0.0, -5.0))(np.array([[4.0, -2.0]]), 0.6)
#
#         self.assertEqual(soln.burntime, 2.5)
#
#     def test_burntime_2d_inertR(self):
#         """Tests burntime solution at 2D point in LOS region.
#
#         Uses default parameter values for HE detonation velocity, detonation
#         time and detonator location. Inert region radius is :math:`R=4.0`.
#         """
#
#         soln = ExplosiveArc(geometry=2, R=4.0)(np.array([[4.0, 2.0]]), 0.6)
#
#         self.assertEqual(soln.burntime, 2.5)
#
#     def test_geometry_error(self):
#         """Test for valid value of geometry."""
#
#         self.assertRaisesRegexp(ValueError, "geometry must be 1",
#                                 ExplosiveArc, geometry=5)
#
#     def test_R_neg_error(self):
#         """Test for valid value of inner radius, :math:`R`."""
#
#         self.assertRaisesRegexp(ValueError,
#                                 "Inert obstacle radius must be > 0",
#                                 ExplosiveArc, R=-1.0)
#
#     def test_R_zero_error(self):
#         """Test for valid value of inner radius, :math:`R`."""
#
#         self.assertRaisesRegexp(ValueError,
#                                 "Inert obstacle radius must be > 0",
#                                 ExplosiveArc, R=0.0)
#
#     def test_D_CJ_neg_error(self):
#         """Test for valid value of HE detonation velocity, :math:`D`."""
#
#         self.assertRaisesRegexp(ValueError,
#                                 "Detonation velocity must be > 0",
#                                 ExplosiveArc, D_CH=-1.0)
#
#     def test_D_CJ_zero_error(self):
#         """Test for valid value of HE detonation velocity, :math:`D`."""
#
#         self.assertRaisesRegexp(ValueError,
#                                 "Detonation velocity must be > 0",
#                                 ExplosiveArc, D_CJ=0.0)
#
#     def test_detgeom_2d_error(self):
#         """Tests for valid geometry of detonator, :math:`x_d`."""
#
#         self.assertRaisesRegexp(ValueError, "Detonator location and " +
#                                 "geometry dimensions must be compatible",
#                                 ExplosiveArc, x_d=(0.0, 0.0, 0.0))
#
#     def test_detloc_2d_error(self):
#         """Tests for valid location of detonator, :math:`x_d`."""
#
#         self.assertRaisesRegexp(ValueError,
#                                 "Detonator must be outside of inert region",
#                                 ExplosiveArc, x_d=(0.0, 1.0))
#
#     @unittest.expectedFailure
#     def test_pts_in_inert(self):
#         """Tests that solution points are outside the inert region."""
#
#         soln = ExplosiveArc()(np.array([[1.0, 0.0]]), 0.6)
