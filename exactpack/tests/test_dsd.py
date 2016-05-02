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

from exactpack.solvers.dsd.ratestick import RateStick
from exactpack.solvers.dsd.cylexpansion import CylindricalExpansion
from exactpack.solvers.dsd.explosivearc import ExplosiveArc


class TestRateStick(unittest.TestCase):
    r"""Tests for
    :class:`exactpack.solvers.dsd.ratestick.RateStick`.

    Solution tests consist of comparing the calculated burn time to the
    analytic solution at a fixed point. Input tests check that invalid
    input raises the appropriate error expression.
    """

    def test_burntime_2d_base(self):
        """Tests burntime solution at 2D point.

        Uses default parameter values for all parameters.
        """

        soln = RateStick(geometry=2)(np.array([[3.0, 4.0]]), 0.6)

        self.assertEqual(soln.burntime, 5.0)

    def test_burntime_2d_detvel(self):
        """Tests burntime solution at 2D point.

        Uses default parameter values for detonator location and detonation
        time. HE detonation velocity is :math:`D=2.0`.
        """

        soln = RateStick(geometry=2, D=2.0)(np.array([[3.0, 4.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_2d_dettime(self):
        """Tests burntime solution at 2D point.

        Uses default parameter values for HE detonation velocity and
        detonator location. Detonation time is :math:`t_d=-2.0`.
        """

        soln = RateStick(geometry=2, t_d=-2.0)(np.array([[3.0, 4.0]]), 0.6)

        self.assertEqual(soln.burntime, 3.0)

    def test_burntime_2d_detloc(self):
        """Tests burntime solution at 2D point.

        Uses default parameter values for HE detonation velocity and
        detonation time. Detonator location is :math:`x_d=(1.0, 1.0)`.
        """

        soln = RateStick(geometry=2,
                         x_d=(1.0, 1.0))(np.array([[2.5, 3.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_geometry_error(self):
        """Test for valid value of geometry."""

        self.assertRaisesRegexp(ValueError, "geometry must be 1 or 2",
                                RateStick, geometry=5)

    def test_D_CJ_neg_error(self):
        """Test for valid value of HE detonation velocity, :math:`D`."""

        self.assertRaisesRegexp(ValueError, "Detonation velocity must be > 0",
                                RateStick, D_CJ=-1.0)

    def test_D_CJ_zero_error(self):
        """Test for valid value of HE detonation velocity, :math:`D`."""

        self.assertRaisesRegexp(ValueError, "Detonation velocity must be > 0",
                                RateStick, D_CJ=0.0)

    def test_detspec_2d_error(self):
        """Test for valid geometry of detonator, :math:`x_d`."""

        self.assertRaisesRegexp(ValueError,
                                "Detonator location and geometry " +
                                "dimensions must be compatible",
                                RateStick, x_d=(0.0, 0.0, 0.0))


class TestCylindricalExpansion(unittest.TestCase):
    r"""Tests for
    :class:`exactpack.solvers.dsd.cylexpansion.CylindricalExpansion`.

    Solution tests consist of comparing the calculated burn time to the
    analytic solution at a fixed point. Input tests check that invalid
    input raises the appropriate error expression.
    """

    def test_burntime_2d_base_t1(self):
        """Tests burntime solution at 2D point in :math:`t_1` region.

        Uses default parameter values.
        """

        soln = CylindricalExpansion(geometry=2)(np.array([[2.0, 11.5]]), 0.6)

        self.assertEqual(soln.burntime, 4.5)

    def test_burntime_2d_detvel1(self):
        """Tests burntime solution at 2D point in each region.

        Uses default parameter values for inner region radius, outer region
        HE detonation velocity, detonator locations and detonation times.
        HE detonation velocity in the inner region is :math:`D_1=1.5`.
        """

        points = np.array([[2.0, 11.5], [2.0, 6.5], [0.6, 0.8],
                           [4.0, 3.0], [2.0, -6.5], [2.0, -11.5]])

        answer = np.array([4.5, 3.5, 1.0 / 1.5, 4.0, 3.5, 4.5])

        soln = CylindricalExpansion(geometry=2, D1=1.5)(points, 0.6)

        self.assertTrue((soln.burntime == answer).all())

    def test_burntime_2d_detvel2(self):
        """Tests burntime solution at 2D point in each region.

        Uses default parameter values for inner region radius, inner region
        HE detonation velocity, detonator locations and detonation times.
        HE detonation velocity in the outer region is :math:`D_2=2.0`.
        """

        points = np.array([[2.0, 11.5], [2.0, 6.5], [0.6, 0.8],
                           [4.0, 3.0], [2.0, -6.5], [2.0, -11.5]])

        answer = np.array([3.25, 2.25, 0.5, 2.5, 2.25, 3.25])

        soln = CylindricalExpansion(geometry=2, D2=2.0)(points, 0.6)

        soln.burntime

        self.assertTrue((soln.burntime == answer).all())

    def test_burntime_2d_innerR(self):
        """Tests burntime solution at 2D point in each region.

        Uses default parameter values for HE detonation velocities, detonator
        locations and detonation times. The radius of the inner region is
        :math:`R=2.0`.
        """

        points = np.array([[2.0, 11.5], [2.0, 6.5], [0.6, 0.8],
                           [4.0, 3.0], [2.0, -6.5], [2.0, -11.5]])

        answer = np.array([4.5, 3.5, 0.5, 4.0, 3.5, 4.5])

        soln = CylindricalExpansion(geometry=2, R=2.0)(points, 0.6)

        self.assertTrue((soln.burntime == answer).all())

    def test_burntime_2d_dets1(self):
        """Tests burntime solution at 2D point in each region.

        Uses default parameter values for inner region radius, HE detonation
        velocities and detonation times. Detonator locations have been
        reversed, thus specified at
        **dets** :math:`=[-10.0, -5.0, 5.0, 10.0]`.
        """

        detz = [-10.0, -5.0, 5.0, 10.0]

        points = np.array([[2.0, 11.5], [2.0, 6.5], [0.6, 0.8],
                           [4.0, 3.0], [2.0, -6.5], [2.0, -11.5]])

        answer = np.array([4.5, 3.5, 0.5, 3.5, 3.5, 4.5])

        soln = CylindricalExpansion(geometry=2, dets=detz)(points, 0.6)

        self.assertTrue((soln.burntime == answer).all())

    def test_geometry_error(self):
        """Test for valid value of geometry."""

        self.assertRaisesRegexp(ValueError, "geometry must be 2",
                                CylindricalExpansion, geometry=1)

    def test_R_neg_error(self):
        """Test for valid value of inner radius, :math:`R`."""

        self.assertRaisesRegexp(ValueError, "Inner HE radius must be > 0",
                                CylindricalExpansion, R=-1.0)

    def test_R_zero_error(self):
        """Test for valid value of inner radius, :math:`R`."""

        self.assertRaisesRegexp(ValueError, "Inner HE radius must be > 0",
                                CylindricalExpansion, R=0.0)

    def test_D_CJ_1_neg_error(self):
        """Test for valid value of inner region HE detonation velocity,
        :math:`D_1`."""

        self.assertRaisesRegexp(ValueError,
                                "Detonation velocity 1 must be > 0",
                                CylindricalExpansion, D_CJ_1=-1.0)

    def test_D_CJ_1_zero_error(self):
        """Test for valid value of inner region HE detonation velocity,
        :math:`D_1`."""

        self.assertRaisesRegexp(ValueError,
                                "Detonation velocity 1 must be > 0",
                                CylindricalExpansion, D_CJ_1=0.0)

    def test_D_CJ_2_neg_error(self):
        """Tests for valid value of outer region HE detonation velocity,
        :math:`D_2`."""

        self.assertRaisesRegexp(ValueError,
                                "Detonation velocity 2 must be > 0",
                                CylindricalExpansion, D_CU_2=-1.0)

    def test_D_CJ_2_zero_error(self):
        """Tests for valid value of outer region HE detonation velocity,
        :math:`D_2`."""

        self.assertRaisesRegexp(ValueError,
                                "Detonation velocity 2 must be > 0",
                                CylindricalExpansion, D_CJ_2=0.0)


class TestExplosiveArc(unittest.TestCase):
    r"""Tests for :class:`exactpack.solvers.dsd.explosivearc.ExplosiveArc`.

    Solution tests consist of comparing the calculated burn time to the
    analytic solution at a fixed point. Input tests check that invalid
    input raises the appropriate error expression.
    """

    def test_burntime_2d_base(self):
        """Tests burntime solution at 2D point in LOS region.

        Uses default parameter values.
        """

        soln = ExplosiveArc(geometry=2)(np.array([[4.0, 2.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_2d_detvel(self):
        """Tests burntime solution at 2D point in LOS region.

        Uses default parameter values for inert region radius, detonator
        location and detonation time. HE detonation velocity is
        :math:`D=1.0`.
        """

        soln = ExplosiveArc(geometry=2, D=1.0)(np.array([[4.0, 2.0]]), 0.6)

        self.assertEqual(soln.burntime, 5.0)

    def test_burntime_2d_dettime(self):
        """Tests burntime solution at 2D point in LOS region.

        Uses default parameter values for inert region radius, HE detonation
        velocity and detonator location. Detonation time is :math:`t_d=-2.0`.
        """

        soln = ExplosiveArc(geometry=2, t_d=-2.0)(np.array([[4.0, 2.0]]), 0.6)

        self.assertEqual(soln.burntime, 0.5)

    def test_burntime_2d_detloc(self):
        """Tests burntime solution at 2D point in LOS region.

        Uses default parameter values for inert region radius, HE detonation
        velocity and detonation time. Detonator location is
        :math:`x_d=(0.0, -5.0)`.
        """

        soln = ExplosiveArc(geometry=2,
                         x_d=(0.0, -5.0))(np.array([[4.0, -2.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_2d_inertR(self):
        """Tests burntime solution at 2D point in LOS region.

        Uses default parameter values for HE detonation velocity, detonation
        time and detonator location. Inert region radius is :math:`R=4.0`.
        """

        soln = ExplosiveArc(geometry=2, R=4.0)(np.array([[4.0, 2.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_geometry_error(self):
        """Test for valid value of geometry."""

        self.assertRaisesRegexp(ValueError, "geometry must be 1",
                                ExplosiveArc, geometry=5)

    def test_R_neg_error(self):
        """Test for valid value of inner radius, :math:`R`."""

        self.assertRaisesRegexp(ValueError,
                                "Inert obstacle radius must be > 0",
                                ExplosiveArc, R=-1.0)

    def test_R_zero_error(self):
        """Test for valid value of inner radius, :math:`R`."""

        self.assertRaisesRegexp(ValueError,
                                "Inert obstacle radius must be > 0",
                                ExplosiveArc, R=0.0)

    def test_D_CJ_neg_error(self):
        """Test for valid value of HE detonation velocity, :math:`D`."""

        self.assertRaisesRegexp(ValueError,
                                "Detonation velocity must be > 0",
                                ExplosiveArc, D_CH=-1.0)

    def test_D_CJ_zero_error(self):
        """Test for valid value of HE detonation velocity, :math:`D`."""

        self.assertRaisesRegexp(ValueError,
                                "Detonation velocity must be > 0",
                                ExplosiveArc, D_CJ=0.0)

    def test_detgeom_2d_error(self):
        """Tests for valid geometry of detonator, :math:`x_d`."""

        self.assertRaisesRegexp(ValueError, "Detonator location and " +
                                "geometry dimensions must be compatible",
                                ExplosiveArc, x_d=(0.0, 0.0, 0.0))

    def test_detloc_2d_error(self):
        """Tests for valid location of detonator, :math:`x_d`."""

        self.assertRaisesRegexp(ValueError,
                                "Detonator must be outside of inert region",
                                ExplosiveArc, x_d=(0.0, 1.0))

    @unittest.expectedFailure
    def test_pts_in_inert(self):
        """Tests that solution points are outside the inert region."""

        soln = ExplosiveArc()(np.array([[1.0, 0.0]]), 0.6)
