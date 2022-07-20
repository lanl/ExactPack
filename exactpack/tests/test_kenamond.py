r"""Tests for the Kenamond exact solution solvers.

Test problems consist of comparison of the calculated burntime for a 2D or
3D point with the known analytic solution. Solver parameter inputs are
tested individually, both for validity and for new solutions.

A time value is passed to the ExactPack solver in order to maintain
a consistent format of input with the other solvers. Because the
burn time solution is not time dependent, this value is ignored by
the solver.
"""

import unittest

import numpy as np

from exactpack.solvers.kenamond.kenamond1 import Kenamond1
from exactpack.solvers.kenamond.kenamond2 import Kenamond2
from exactpack.solvers.kenamond.kenamond3 import Kenamond3


class TestKenamond1(unittest.TestCase):
    r"""Tests for
    :class:`exactpack.solvers.kenamond.kenamond1.Kenamond1`.

    Solution tests consist of comparing the calculated burn time to the
    analytic solution at a fixed point. Input tests check that invalid
    input raises the appropriate error expression.
    """

    def test_burntime_2d_base(self):
        """Tests burntime solution at 2D point.

        Uses default parameter values for all parameters.
        """

        soln = Kenamond1(geometry=2)(np.array([[3.0, 4.0]]), 0.6)

        self.assertEqual(soln.burntime, 5.0)

    def test_burntime_3d_base(self):
        """Tests burntime solution at 3D point (**geometry** :math:`=3`).

        Uses default parameter values for HE detonation velocity and
        detonation time. Detonator location is specified as the 3D origin.
        """

        soln = Kenamond1(geometry=3,
                         x_d=(0., 0., 0.))(np.array([[3.0, 4.0, 12.0]]), 0.6)

        self.assertEqual(soln.burntime, 13.0)

    def test_burntime_2d_detvel(self):
        """Tests burntime solution at 2D point.

        Uses default parameter values for detonator location and detonation
        time. HE detonation velocity is :math:`D=2.0`.
        """

        soln = Kenamond1(geometry=2, D=2.0)(np.array([[3.0, 4.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_3d_detvel(self):
        """Tests burntime solution at 3D point (**geometry** :math:`=3`).

        Uses default parameter value for detonation time. Detonator location
        is specified as the 3D origin. HE detonation velocity is
        :math:`D=2.0`.
        """

        soln = Kenamond1(geometry=3, x_d=(0., 0., 0.),
                         D=2.0)(np.array([[3.0, 4.0, 12.0]]), 0.6)

        self.assertEqual(soln.burntime, 6.5)

    def test_burntime_2d_dettime(self):
        """Tests burntime solution at 2D point.

        Uses default parameter values for HE detonation velocity and
        detonator location. Detonation time is :math:`t_d=-2.0`.
        """

        soln = Kenamond1(geometry=2, t_d=-2.0)(np.array([[3.0, 4.0]]), 0.6)

        self.assertEqual(soln.burntime, 3.0)

    def test_burntime_3d_dettime(self):
        """Tests burntime solution at 3D point (**geometry** :math:`=3`).

        Uses default parameter value for HE detonation velocity. Detonator
        location is specified as the 3D origin. Detonation time is
        :math:`t_d=-2.0`.
        """

        soln = Kenamond1(geometry=3, x_d=(0., 0., 0.),
                         t_d=-2.0)(np.array([[3.0, 4.0, 12.0]]), 0.6)

        self.assertEqual(soln.burntime, 11.0)

    def test_burntime_2d_detloc(self):
        """Tests burntime solution at 2D point.

        Uses default parameter values for HE detonation velocity and
        detonation time. Detonator location is :math:`x_d=(1.0, 1.0)`.
        """

        soln = Kenamond1(geometry=2,
                         x_d=(1.0, 1.0))(np.array([[2.5, 3.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_3d_detloc(self):
        """Tests burntime solution at 3D point (**geometry** :math:`=3`).

        Uses default parameter values for HE detonation velocity and
        detonation time. Detonator location is :math:`x_d=(1.0, 1.0, 1.0)`.
        """

        soln = Kenamond1(geometry=3,
                         x_d=(1., 1., 1.))(np.array([[2.5, 3.0, 7.0]]), 0.6)

        self.assertEqual(soln.burntime, 6.5)

    def test_geometry_error(self):
        """Test for valid value of geometry."""

        self.assertRaisesRegex(ValueError, "geometry must be 2 or 3",
                                Kenamond1, geometry=1)

    def test_D_neg_error(self):
        """Test for valid value of HE detonation velocity, :math:`D`."""

        self.assertRaisesRegex(ValueError, "Detonation velocity must be > 0",
                                Kenamond1, D=-1.0)

    def test_D_zero_error(self):
        """Test for valid value of HE detonation velocity, :math:`D`."""

        self.assertRaisesRegex(ValueError, "Detonation velocity must be > 0",
                                Kenamond1, D=0.0)

    def test_detspec_2d_error(self):
        """Test for valid geometry of detonator, :math:`x_d`."""

        self.assertRaisesRegex(ValueError,
                                "Detonator location and geometry " +
                                "dimensions must be compatible",
                                Kenamond1, x_d=(0.0, 0.0, 0.0))

    def test_detspec_3d_error(self):
        """Test for valid geometry of detonator, :math:`x_d`."""

        self.assertRaisesRegex(ValueError,
                                "Detonator location and geometry " +
                                "dimensions must be compatible",
                                Kenamond1, geometry=3, x_d=(0.0, 0.0))


class TestKenamond2(unittest.TestCase):
    r"""Tests for
    :class:`exactpack.solvers.kenamond.kenamond2.Kenamond2`.

    Solution tests consist of comparing the calculated burn time to the
    analytic solution at a fixed point. Input tests check that invalid
    input raises the appropriate error expression.
    """

    def test_burntime_2d_base_t1(self):
        """Tests burntime solution at 2D point in :math:`t_1` region.

        Uses default parameter values.
        """

        soln = Kenamond2(geometry=2)(np.array([[2.0, 11.5]]), 0.6)

        self.assertEqual(soln.burntime, 4.5)

    def test_burntime_2d_base_t2(self):
        """Tests burntime solution at 2D point in :math:`t_2` region.

        Uses default parameter values.
        """

        soln = Kenamond2(geometry=2)(np.array([[2.0, 6.5]]), 0.6)

        self.assertEqual(soln.burntime, 3.5)

    def test_burntime_2d_base_t3(self):
        """Tests burntime solution at 2D point in :math:`t_3` region.

        Uses default parameter values.
        """

        soln = Kenamond2(geometry=2)(np.array([[0.6, 0.8]]), 0.6)

        self.assertEqual(soln.burntime, 0.5)

    def test_burntime_2d_base_t4(self):
        """Tests burntime solution at 2D point in :math:`t_4` region.

        Uses default parameter values.
        """

        soln = Kenamond2(geometry=2)(np.array([[4.0, 3.0]]), 0.6)

        self.assertEqual(soln.burntime, 3.5)

    def test_burntime_2d_base_t5(self):
        """Tests burntime solution at 2D point in :math:`t_5` region.

        Uses default parameter values.
        """

        soln = Kenamond2(geometry=2)(np.array([[2.0, -6.5]]), 0.6)

        self.assertEqual(soln.burntime, 3.5)

    def test_burntime_2d_base_t6(self):
        """Tests burntime solution at 2D point in :math:`t_6` region.

        Uses default parameter values.
        """

        soln = Kenamond2(geometry=2)(np.array([[2.0, -11.5]]), 0.6)

        self.assertEqual(soln.burntime, 4.5)

    def test_burntime_3d_base_t1(self):
        """Tests burntime solution at 3D point in :math:`t_1` region
        (**geometry** :math:`=3`).

        Uses default parameter values for inner region radius, HE detonation
        velocities, detonation times and detonator locations.
        """

        soln = Kenamond2(geometry=3)(np.array([[2.0, 0.0, 11.5]]), 0.6)

        self.assertEqual(soln.burntime, 4.5)

    def test_burntime_3d_base_t2(self):
        """Tests burntime solution at 3D point in :math:`t_2` region
        (**geometry** :math:`=3`).

        Uses default parameter values for inner region radius, HE detonation
        velocities, detonation times and detonator locations.
        """

        soln = Kenamond2(geometry=3)(np.array([[2.0, 0.0, 6.5]]), 0.6)

        self.assertEqual(soln.burntime, 3.5)

    def test_burntime_3d_base_t3(self):
        """Tests burntime solution at 3D point in :math:`t_3` region
        (**geometry** :math:`=3`).

        Uses default parameter values for inner region radius, HE detonation
        velocities, detonation times and detonator locations.
        """

        soln = Kenamond2(geometry=3)(np.array([[0.6, 0.0, 0.8]]), 0.6)

        self.assertEqual(soln.burntime, 0.5)

    def test_burntime_3d_base_t4(self):
        """Tests burntime solution at 3D point in :math:`t_4` region
        (**geometry** :math:`=3`).

        Uses default parameter values for inner region radius, HE detonation
        velocities, detonation times and detonator locations.
        """

        soln = Kenamond2(geometry=3)(np.array([[0.0, 4.0, 3.0]]), 0.6)

        self.assertEqual(soln.burntime, 3.5)

    def test_burntime_3d_base_t5(self):
        """Tests burntime solution at 3D point in :math:`t_5` region
        (**geometry** :math:`=3`).

        Uses default parameter values for inner region radius, HE detonation
        velocities, detonation times and detonator locations.
        """

        soln = Kenamond2(geometry=3)(np.array([[0.0, 2.0, -6.5]]), 0.6)

        self.assertEqual(soln.burntime, 3.5)

    def test_burntime_3d_base_t6(self):
        """Tests burntime solution at 3D point in :math:`t_6` region
        (**geometry** :math:`=3`).

        Uses default parameter values for inner region radius, HE detonation
        velocities, detonation times and detonator locations.
        """

        soln = Kenamond2(geometry=3)(np.array([[0.0, 2.0, -11.5]]), 0.6)

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

        soln = Kenamond2(geometry=2, D1=1.5)(points, 0.6)

        self.assertTrue((soln.burntime == answer).all())

    def test_burntime_3d_detvel1(self):
        """Tests burntime solution at 3D point in each region
        (**geometry** :math:`=3`).

        Uses default parameter values for inner region radius, outer region
        HE detonation velocity, detonator locations and detonation times.
        HE detonation velocity in the inner region is :math:`D_1=1.5`.
        """

        points = np.array([[2.0, 0.0, 11.5], [2.0, 0.0, 6.5],
                           [0.6, 0.0, 0.8], [0.0, 4.0, 3.0],
                           [0.0, 2.0, -6.5], [0.0, 2.0, -11.5]])

        answer = np.array([4.5, 3.5, 1.0 / 1.5, 4.0, 3.5, 4.5])

        soln = Kenamond2(geometry=3, D1=1.5)(points, 0.6)

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

        soln = Kenamond2(geometry=2, D2=2.0)(points, 0.6)

        soln.burntime

        self.assertTrue((soln.burntime == answer).all())

    def test_burntime_3d_detvel2(self):
        """Tests burntime solution at 3D point in each region
        (**geometry** :math:`=3`).

        Uses default parameter values for inner region radius, inner region
        HE detonation velocity, detonator locations and detonation times.
        HE detonation velocity in the outer region is :math:`D_2=2.0`.
        """

        points = np.array([[2.0, 0.0, 11.5], [2.0, 0.0, 6.5],
                           [0.6, 0.0, 0.8], [0.0, 4.0, 3.0],
                           [0.0, 2.0, -6.5], [0.0, 2.0, -11.5]])

        answer = np.array([3.25, 2.25, 0.5, 2.5, 2.25, 3.25])

        soln = Kenamond2(geometry=3, D2=2.0)(points, 0.6)

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

        soln = Kenamond2(geometry=2, R=2.0)(points, 0.6)

        self.assertTrue((soln.burntime == answer).all())

    def test_burntime_3d_innerR(self):
        """Tests burntime solution at 3D point in each region
        (**geometry** :math:`=3`).

        Uses default parameter values for HE detonation velocities, detonator
        locations and detonation times. The radius of the inner region is
        :math:`R=2.0`.
        """

        points = np.array([[2.0, 0.0, 11.5], [2.0, 0.0, 6.5],
                           [0.6, 0.0, 0.8], [0.0, 4.0, 3.0],
                           [0.0, 2.0, -6.5], [0.0, 2.0, -11.5]])

        answer = np.array([4.5, 3.5, 0.5, 4.0, 3.5, 4.5])

        soln = Kenamond2(geometry=3, R=2.0)(points, 0.6)

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

        soln = Kenamond2(geometry=2, dets=detz)(points, 0.6)

        self.assertTrue((soln.burntime == answer).all())

    def test_burntime_3d_dets1(self):
        """Tests burntime solution at 3D point in each region
        (**geometry** :math:`=3`).

        Uses default parameter values for inner region radius, HE detonation
        velocities and detonation times. Detonator locations have been
        reversed, thus specified at
        **dets** :math:`=[-10.0, -5.0, 5.0, 10.0]`.
        """

        detz = [-10.0, -5.0, 5.0, 10.0]

        points = np.array([[2.0, 0.0, 11.5], [2.0, 0.0, 6.5],
                           [0.6, 0.0, 0.8], [0.0, 4.0, 3.0],
                           [0.0, 2.0, -6.5], [0.0, 2.0, -11.5]])

        answer = np.array([4.5, 3.5, 0.5, 3.5, 3.5, 4.5])

        soln = Kenamond2(geometry=3, dets=detz)(points, 0.6)

        self.assertTrue((soln.burntime == answer).all())

    def test_burntime_2d_dets2(self):
        """Tests burntime solution at 2D point in each region.

        Uses default parameter values for inner region radius, HE detonation
        velocities and detonation times. Detonator locations 1 and 2 have
        been moved to the same point. D2 ignites earlier than D1, thus the
        t1 solution region does not exist. Detonator locations are specified
        at **dets** :math:`=[9.0, 9.0, -5.0, -10.0]`.
        """

        detz = [9.0, 9.0, -5.0, -10.0]

        points = np.array([[2.0, 10.5], [2.0, 7.5], [0.6, 0.8],
                           [4.0, 3.0], [2.0, -6.5], [2.0, -11.5]])

        answer = np.array([3.5, 3.5, 0.5, 3.5, 3.5, 4.5])

        soln = Kenamond2(geometry=2, dets=detz)(points, 0.6)

        self.assertTrue((soln.burntime == answer).all())

    def test_burntime_3d_dets2(self):
        """Tests burntime solution at 3D point in each region
        (**geometry** :math:`=3`).

        Uses default parameter values for inner region radius, HE detonation
        velocities and detonation times. Detonator locations 1 and 2 have
        been moved to the same point. D2 ignites earlier than D1, thus the
        t1 solution region does not exist. Detonator locations are specified
        at **dets** :math:`=[9.0, 9.0, -5.0, -10.0]`.
        """

        detz = [9.0, 9.0, -5.0, -10.0]

        points = np.array([[2.0, 0.0, 10.5], [0.0, 2.0, 7.5],
                           [0.6, 0.0, 0.8], [4.0, 0.0, 3.0],
                           [0.0, 2.0, -6.5], [0.0, 2.0, -11.5]])

        answer = np.array([3.5, 3.5, 0.5, 3.5, 3.5, 4.5])

        soln = Kenamond2(geometry=3, dets=detz)(points, 0.6)

        self.assertTrue((soln.burntime == answer).all())

    def test_geometry_error(self):
        """Test for valid value of geometry."""

        self.assertRaisesRegex(ValueError, "geometry must be 2 or 3",
                                Kenamond2, geometry=1)

    def test_R_neg_error(self):
        """Test for valid value of inner radius, :math:`R`."""

        self.assertRaisesRegex(ValueError, "Inner HE radius must be > 0",
                                Kenamond2, R=-1.0)

    def test_R_zero_error(self):
        """Test for valid value of inner radius, :math:`R`."""

        self.assertRaisesRegex(ValueError, "Inner HE radius must be > 0",
                                Kenamond2, R=0.0)

    def test_D1_neg_error(self):
        """Test for valid value of inner region HE detonation velocity,
        :math:`D_1`."""

        self.assertRaisesRegex(ValueError,
                                "Detonation velocity 1 must be > 0",
                                Kenamond2, D1=-1.0)

    def test_D1_zero_error(self):
        """Test for valid value of inner region HE detonation velocity,
        :math:`D_1`."""

        self.assertRaisesRegex(ValueError,
                                "Detonation velocity 1 must be > 0",
                                Kenamond2, D1=0.0)

    def test_D2_neg_error(self):
        """Tests for valid value of outer region HE detonation velocity,
        :math:`D_2`."""

        self.assertRaisesRegex(ValueError,
                                "Detonation velocity 2 must be > 0",
                                Kenamond2, D2=-1.0)

    def test_D2_zero_error(self):
        """Tests for valid value of outer region HE detonation velocity,
        :math:`D_2`."""

        self.assertRaisesRegex(ValueError,
                                "Detonation velocity 2 must be > 0",
                                Kenamond2, D2=0.0)

    def test_D2_smaller_error(self):
        """Tests for valid value of outer region HE detonation velocity,
        :math:`D_2`."""

        self.assertRaisesRegex(ValueError,
                                "D1 must be > D2",
                                Kenamond2, D2=3.0)

    def test_dets_notenough_error(self):
        """Tests for valid number of detonators, **dets**."""

        self.assertRaisesRegex(ValueError,
                                "4 detonator locations must be specified",
                                Kenamond2, dets=[10.0, 5.0, -5.0])

    def test_dets_toomany_error(self):
        """Tests for valid number of detonators, **dets**."""

        self.assertRaisesRegex(ValueError,
                                "4 detonator locations must be specified",
                                Kenamond2,
                                dets=[10.0, 5.0, -5.0, -10.0, -15.0])

    def test_dets_2d_d1_error(self):
        """Tests for valid location of detonators, **dets**."""

        self.assertRaisesRegex(ValueError,
                                "Detonator 1 must be in outer HE region",
                                Kenamond2,
                                dets=[1.0, 5.0, -5.0, -10.0])

    def test_dets_2d_d2_error(self):
        """Tests for valid location of detonators, **dets**."""

        self.assertRaisesRegex(ValueError,
                                "Detonator 2 must be in outer HE region",
                                Kenamond2,
                                dets=[10.0, 2.0, -5.0, -10.0])

    def test_dets_2d_d4_error(self):
        """Tests for valid location of detonators, **dets**."""

        self.assertRaisesRegex(ValueError,
                                "Detonator 4 must be in outer HE region",
                                Kenamond2,
                                dets=[10.0, 5.0, -3.0, -10.0])

    def test_dets_2d_d5_error(self):
        """Tests for valid location of detonators, **dets**."""

        self.assertRaisesRegex(ValueError,
                                "Detonator 5 must be in outer HE region",
                                Kenamond2,
                                dets=[10.0, 5.0, -5.0, -1.0])

    def test_dets_3d_d1_error(self):
        """Tests for valid location of detonators, **dets**."""

        self.assertRaisesRegex(ValueError,
                                "Detonator 1 must be in outer HE region",
                                Kenamond2, geometry=3,
                                dets=[1.0, 5.0, -5.0, -10.0])

    def test_dets_3d_d2_error(self):
        """Tests for valid location of detonators, **dets**."""

        self.assertRaisesRegex(ValueError,
                                "Detonator 2 must be in outer HE region",
                                Kenamond2, geometry=3,
                                dets=[10.0, 2.0, -5.0, -10.0])

    def test_dets_3d_d4_error(self):
        """Tests for valid location of detonators, **dets**."""

        self.assertRaisesRegex(ValueError,
                                "Detonator 4 must be in outer HE region",
                                Kenamond2, geometry=3,
                                dets=[10.0, 5.0, -3.0, -10.0])

    def test_dets_3d_d5_error(self):
        """Tests for valid location of detonators, **dets**."""

        self.assertRaisesRegex(ValueError,
                                "Detonator 5 must be in outer HE region",
                                Kenamond2, geometry=3,
                                dets=[10.0, 5.0, -5.0, -1.0])

    def test_dettimes_notenough_error(self):
        """Tests for valid number of detonation times, :math:`t_{d_i}`."""

        self.assertRaisesRegex(ValueError, "5 detonation times must " +
                                "be specified", Kenamond2,
                                t_d=[2.0, 1.0, 0.0, 1.0])

    def test_dettimes_toomany_error(self):
        """Tests for valid number of detonation times, :math:`t_{d_i}`."""

        self.assertRaisesRegex(ValueError, "5 detonation times must " +
                                "be specified", Kenamond2,
                                t_d=[2.0, 1.0, 0.0, 1.0, 2.0, 3.0])

    #must allow D3 wave to exit inner region

    def test_dettimes_2d_t1_error(self):
        """Tests for valid detonation times, :math:`t_{d_1}`."""

        self.assertRaisesRegex(ValueError, "Detonation time 1 must be " +
                                "no less than -5.5",
                                Kenamond2, t_d=[-7.0, 1.0, 0.0, 1.0, 2.0])

    def test_dettimes_2d_t2_error(self):
        """Tests for valid detonation times, :math:`t_{d_2}`."""

        self.assertRaisesRegex(ValueError, "Detonation time 2 must be " +
                                "no less than -0.5",
                                Kenamond2, t_d=[2.0, -1.0, 0.0, 1.0, 2.0])

    def test_dettimes_2d_t4_error(self):
        """Tests for valid detonation times, :math:`t_{d_4}`."""

        self.assertRaisesRegex(ValueError, "Detonation time 4 must be " +
                                "no less than -0.5",
                                Kenamond2, t_d=[2.0, 1.0, 0.0, -1.0, 2.0])

    def test_dettimes_2d_t5_error(self):
        """Tests for valid detonation times, :math:`t_{d_5}`."""

        self.assertRaisesRegex(ValueError, "Detonation time 5 must be " +
                                "no less than -5.5",
                                Kenamond2, t_d=[2.0, 1.0, 0.0, 1.0, -7.0])

    def test_dettimes_3d_t1_error(self):
        """Tests for valid detonation times, :math:`t_{d_1}`."""

        self.assertRaisesRegex(ValueError, "Detonation time 1 must be " +
                                "no less than -5.5",
                                Kenamond2, geometry=3,
                                t_d=[-7.0, 1.0, 0.0, 1.0, 2.0])

    def test_dettimes_3d_t2_error(self):
        """Tests for valid detonation times, :math:`t_{d_2}`."""

        self.assertRaisesRegex(ValueError, "Detonation time 2 must be " +
                                "no less than -0.5",
                                Kenamond2, geometry=3,
                                t_d=[2.0, -1.0, 0.0, 1.0, 2.0])

    def test_dettimes_3d_t4_error(self):
        """Tests for valid detonation times, :math:`t_{d_3}`."""

        self.assertRaisesRegex(ValueError, "Detonation time 4 must be " +
                                "no less than -0.5",
                                Kenamond2, geometry=3,
                                t_d=[2.0, 1.0, 0.0, -1.0, 2.0])

    def test_dettimes_3d_t5_error(self):
        """Tests for valid detonation times, :math:`t_{d_4}`."""

        self.assertRaisesRegex(ValueError, "Detonation time 5 must be " +
                                "no less than -5.5",
                                Kenamond2, geometry=3,
                                t_d=[2.0, 1.0, 0.0, 1.0, -7.0])


class TestKenamond3(unittest.TestCase):
    r"""Tests for
    :class:`exactpack.solvers.kenamond.kenamond3.Kenamond3`.

    Solution tests consist of comparing the calculated burn time to the
    analytic solution at a fixed point. Input tests check that invalid
    input raises the appropriate error expression.
    """

    def test_burntime_2d_base_los(self):
        """Tests burntime solution at 2D point in LOS region.

        Uses default parameter values.
        """

        soln = Kenamond3(geometry=2)(np.array([[4.0, 2.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_2d_base_shadow(self):
        """Tests burntime solution at 2D point in shadow region.

        Uses default parameter values.
        """

        soln = Kenamond3(geometry=2)(np.array([[3.0, -4.0]]), 0.6)

        self.assertEqual(soln.burntime, 4.0 + 1.5 * np.arcsin(3.0 / 5.0))

    def test_burntime_3d_base_los(self):
        """Tests burntime solution at 3D point in LOS region.

        Uses default parameter values for inert region radius, HE detonation
        velocity and detonation time. Detonator location is specified as
        :math:`(0.0, 0.0, 5.0)`.
        """

        soln = Kenamond3(geometry=3,
                         x_d=(0.0, 0.0, 5.0))(np.array([[3.0, 4.0, 5.0]]),
                                              0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_3d_base_shadow(self):
        """Tests burntime solution at 3D point in shadow region.

        Uses default parameter values for inert region radius, HE detonation
        velocity and detonation time. Detonator location is specified as
        :math:`(0.0, 0.0, 5.0)`.
        """

        soln = Kenamond3(geometry=3,
                         x_d=(0.0, 0.0, 5.0))(np.array([[3.0, 0.0, -4.0]]),
                                              0.6)

        self.assertEqual(soln.burntime, 4.0 + 1.5 * np.arcsin(3.0 / 5.0))

    def test_burntime_2d_detvel_los(self):
        """Tests burntime solution at 2D point in LOS region.

        Uses default parameter values for inert region radius, detonator
        location and detonation time. HE detonation velocity is
        :math:`D=1.0`.
        """

        soln = Kenamond3(geometry=2, D=1.0)(np.array([[4.0, 2.0]]), 0.6)

        self.assertEqual(soln.burntime, 5.0)

    def test_burntime_2d_detvel_shadow(self):
        """Tests burntime solution at 2D point in shadow region.

        Uses default parameter values for inert region radius, detonator
        location and detonation time. HE detonation velocity is
        :math:`D=1.0`.
        """

        soln = Kenamond3(geometry=2, D=1.0)(np.array([[3.0, -4.0]]), 0.6)

        self.assertEqual(soln.burntime, 8.0 + 3.0 * np.arcsin(3.0 / 5.0))

    def test_burntime_3d_detvel_los(self):
        """Tests burntime solution at 3D point in LOS region.

        Uses default parameter values for inert region radius and detonation
        time. Detonator location is specified as :math:`(0.0, 0.0, 5.0)`. HE
        detonation velocity is :math:`D=1.0`.
        """

        soln = Kenamond3(geometry=3,
                         x_d=(0.0, 0.0, 5.0),
                         D=1.0)(np.array([[3.0, 4.0, 5.0]]), 0.6)

        self.assertEqual(soln.burntime, 5.0)

    def test_burntime_3d_detvel_shadow(self):
        """Tests burntime solution at 3D point in shadow region.

        Uses default parameter values for inert region radius and detonation
        time. Detonator location is specified as :math:`(0.0, 0.0, 5.0)`.
        HE detonation velocity is :math:`D=1.0`.
        """

        soln = Kenamond3(geometry=3,
                         x_d=(0.0, 0.0, 5.0),
                         D=1.0)(np.array([[3.0, 0.0, -4.0]]), 0.6)

        self.assertEqual(soln.burntime, 8.0 + 3.0 * np.arcsin(3.0 / 5.0))

    def test_burntime_2d_dettime_los(self):
        """Tests burntime solution at 2D point in LOS region.

        Uses default parameter values for inert region radius, HE detonation
        velocity and detonator location. Detonation time is :math:`t_d=-2.0`.
        """

        soln = Kenamond3(geometry=2, t_d=-2.0)(np.array([[4.0, 2.0]]), 0.6)

        self.assertEqual(soln.burntime, 0.5)

    def test_burntime_2d_dettime_shadow(self):
        """Tests burntime solution at 2D point in shadow region.

        Uses default parameter values for inert region radius, HE detonation
        velocity and detonator location. Detonation time is :math:`t_d=-2.0`.
        """

        soln = Kenamond3(geometry=2, t_d=-2.0)(np.array([[3.0, -4.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.0 + 1.5 * np.arcsin(3.0 / 5.0))

    def test_burntime_3d_dettime_los(self):
        """Tests burntime solution at 3D point in LOS region.

        Uses default parameter values for inert region radius and HE
        detonation velocity. Detonator location is specified as
        :math:`(0.0, 0.0, 5.0)`. Detonation time is :math:`t_d=-2.0`.
        """

        soln = Kenamond3(geometry=3,
                         x_d=(0.0, 0.0, 5.0),
                         t_d=-2.0)(np.array([[3.0, 4.0, 5.0]]), 0.6)

        self.assertEqual(soln.burntime, 0.5)

    def test_burntime_3d_dettime_shadow(self):
        """Tests burntime solution at 3D point in shadow region.

        Uses default parameter values for inert region radius and HE
        detonation velocity. Detonator location is specified as
        :math:`(0.0, 0.0, 5.0)`. Detonation time is :math:`t_d=-2.0`.
        """

        soln = Kenamond3(geometry=3,
                         x_d=(0.0, 0.0, 5.0),
                         t_d=-2.0)(np.array([[3.0, 0.0, -4.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.0 + 1.5 * np.arcsin(3.0 / 5.0))

    def test_burntime_2d_detloc_los(self):
        """Tests burntime solution at 2D point in LOS region.

        Uses default parameter values for inert region radius, HE detonation
        velocity and detonation time. Detonator location is
        :math:`x_d=(0.0, -5.0)`.
        """

        soln = Kenamond3(geometry=2,
                         x_d=(0.0, -5.0))(np.array([[4.0, -2.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_2d_detloc_shadow(self):
        """Tests burntime solution at 2D point in shadow region.

        Uses default parameter values for inert region radius, HE detonation
        velocity and detonation time. Detonator location is
        :math:`x_d=(0.0, -5.0)`.
        """

        soln = Kenamond3(geometry=2,
                         x_d=(0.0, -5.0))(np.array([[3.0, 4.0]]), 0.6)

        self.assertEqual(soln.burntime, 4.0 + 1.5 * np.arcsin(3.0 / 5.0))

    def test_burntime_3d_detloc_los(self):
        """Tests burntime solution at 3D point in LOS region.

        Uses default parameter values for inert region radius, HE detonation
        velocity and detonation time. Detonator location is
        :math:`x_d=(0.0, 0.0, -5.0)`.
        """

        soln = Kenamond3(geometry=3,
                         x_d=(0.0, 0.0, -5.0))(np.array([[3.0, 4.0, -5.0]]),
                                               0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_3d_detloc_shadow(self):
        """Tests burntime solution at 3D point in shadow region.

        Uses default parameter values for inert region radius, HE detonation
        velocity and detonation time. Detonator location is
        :math:`x_d=(0.0, 0.0, -5.0)`.
        """

        soln = Kenamond3(geometry=3,
                         x_d=(0.0, 0.0, -5.0))(np.array([[3.0, 0.0, 4.0]]),
                                               0.6)

        self.assertEqual(soln.burntime, 4.0 + 1.5 * np.arcsin(3.0 / 5.0))

    def test_burntime_2d_inertR_los(self):
        """Tests burntime solution at 2D point in LOS region.

        Uses default parameter values for HE detonation velocity, detonation
        time and detonator location. Inert region radius is :math:`R=4.0`.
        """

        soln = Kenamond3(geometry=2, R=4.0)(np.array([[4.0, 2.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_2d_inertR_shadow(self):
        """Tests burntime solution at 2D point in shadow region.

        Uses default parameter values for HE detonation velocity, detonation
        time and detonator location. Inert region radius is :math:`R=4.0`.
        """

        soln = Kenamond3(geometry=2, R=4.0)(np.array([[4.0, -2.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5 + 2.0 * np.arcsin(4.0 / 5.0))

    def test_burntime_3d_inertR_los(self):
        """Tests burntime solution at 3D point in LOS region.

        Uses default parameter values for HE detonation velocity and
        detonation time. Detonator location is specified as
        :math:`(0.0, 0.0, 5.0)`. Inert region radius is :math:`R=4.0`.
        """

        soln = Kenamond3(geometry=3,
                         x_d=(0.0, 0.0, 5.0),
                         R=4.0)(np.array([[3.0, 4.0, 5.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5)

    def test_burntime_3d_inertR_shadow(self):
        """Tests burntime solution at 3D point in shadow region.

        Uses default parameter values for HE detonation velocity and
        detonation time. Detonator location is specified as
        :math:`(0.0, 0.0, 5.0)`. Inert region radius is :math:`R=4.0`.
        """

        soln = Kenamond3(geometry=3,
                         x_d=(0.0, 0.0, 5.0),
                         R=4.0)(np.array([[4.0, 0.0, -2.0]]), 0.6)

        self.assertEqual(soln.burntime, 2.5 + 2.0 * np.arcsin(4.0 / 5.0))

    def test_geometry_error(self):
        """Test for valid value of geometry."""

        self.assertRaisesRegex(ValueError, "geometry must be 2 or 3",
                                Kenamond3, geometry=1)

    def test_R_neg_error(self):
        """Test for valid value of inner radius, :math:`R`."""

        self.assertRaisesRegex(ValueError,
                                "Inert obstacle radius must be > 0",
                                Kenamond3, R=-1.0)

    def test_R_zero_error(self):
        """Test for valid value of inner radius, :math:`R`."""

        self.assertRaisesRegex(ValueError,
                                "Inert obstacle radius must be > 0",
                                Kenamond3, R=0.0)

    def test_D_neg_error(self):
        """Test for valid value of HE detonation velocity, :math:`D`."""

        self.assertRaisesRegex(ValueError,
                                "Detonation velocity must be > 0",
                                Kenamond3, D=-1.0)

    def test_D_zero_error(self):
        """Test for valid value of HE detonation velocity, :math:`D`."""

        self.assertRaisesRegex(ValueError,
                                "Detonation velocity must be > 0",
                                Kenamond3, D=0.0)

    def test_detgeom_2d_error(self):
        """Tests for valid geometry of detonator, :math:`x_d`."""

        self.assertRaisesRegex(ValueError, "Detonator location and " +
                                "geometry dimensions must be compatible",
                                Kenamond3, x_d=(0.0, 0.0, 0.0))

    def test_detgeom_3d_error(self):
        """Tests for valid geometry of detonator, :math:`x_d`."""

        self.assertRaisesRegex(ValueError, "Detonator location and " +
                                "geometry dimensions must be compatible",
                                Kenamond3, geometry=3, x_d=(0.0, 0.0))

    def test_detloc_2d_error(self):
        """Tests for valid location of detonator, :math:`x_d`."""

        self.assertRaisesRegex(ValueError,
                                "Detonator must be outside of inert region",
                                Kenamond3, x_d=(0.0, 1.0))

    def test_detloc_3d_error(self):
        """Tests for valid location of detonator, :math:`x_d`."""

        self.assertRaisesRegex(ValueError,
                                "Detonator must be outside of inert region",
                                Kenamond3, geometry=3, x_d=(0.0, 0.0, 1.0))

    @unittest.expectedFailure
    def test_pts_in_inert(self):
        """Tests that solution points are outside the inert region."""

        soln = Kenamond3()(np.array([[1.0, 0.0]]), 0.6)
