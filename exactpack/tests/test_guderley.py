"""Tests for the Su-Olson problem.
"""

import unittest

import numpy

from exactpack.solvers.guderley.ramsey import Guderley


class TestGuderleyRamsey(unittest.TestCase):

    def test_density(self):
        """Guderley problem: density"""

        self.assertAlmostEqual(Guderley(gamma=3.)(numpy.array([1.]), \
            0.).density[0], 2.0)

    def test_velocity(self):
        """Guderley problem: velocity"""

        self.assertAlmostEqual(Guderley(gamma=3.)(numpy.array([1.]), \
            0.).velocity[0], -0.3182052970545358)

    def test_pressure(self):
        """Guderley problem: pressure"""

        self.assertAlmostEqual(Guderley(gamma=3.)(numpy.array([1.]), \
            0.).pressure[0], 0.20250922214713074)

    def test_speed_of_sound(self):
        """Guderley problem: speed of sound"""

        self.assertAlmostEqual(Guderley(gamma=3.)(numpy.array([1.]), \
            0.).sound[0], 0.5511477417360032)

    def test_sie(self):
        """Guderley problem: sie"""

        self.assertAlmostEqual(Guderley(gamma=3.)(numpy.array([1.]), \
            0.).sie[0], 0.050627305536782685)
