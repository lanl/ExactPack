"""Unittests for the Guderley solver.
"""

import unittest

import numpy

from exactpack.solvers.guderley.ramsey import Guderley


class TestGuderleyRamsey(unittest.TestCase):
    """Tests for the Guderley problem :class:`exactpack.solvers.guderley.ramsey.Guderley`."""

    def test_density(self):
        """Regression test for density."""

        self.assertAlmostEqual(Guderley(gamma=3.)(numpy.array([1.]),
            0.).density[0], 2.0)

    def test_velocity(self):
        """Regression test for velocity."""

        self.assertAlmostEqual(Guderley(gamma=3.)(numpy.array([1.]),
            0.).velocity[0], -0.3182052970545358)

    def test_pressure(self):
        """Regression test for pressure."""

        self.assertAlmostEqual(Guderley(gamma=3.)(numpy.array([1.]),
            0.).pressure[0], 0.20250922214713074)

    def test_speed_of_sound(self):
        """Regression test for the speed of sound."""

        self.assertAlmostEqual(Guderley(gamma=3.)(numpy.array([1.]),
            0.).sound[0], 0.5511477417360032)

    def test_sie(self):
        """Regression test for specific internal energy."""

        self.assertAlmostEqual(Guderley(gamma=3.)(numpy.array([1.]),
            0.).sie[0], 0.050627305536782685)
