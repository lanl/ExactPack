"""Unittests for the Timmes Cog8 solver.
"""

import unittest

import numpy

from exactpack.solvers.cog.cog8 import Cog8
from exactpack.solvers.cog.cog8_timmes import Cog8 as Cog8Timmes


class TestCog8(unittest.TestCase):
    """Timmes' implementation of the Cog8 problem :class:`exactpack.solvers.cog.cog8_timmes.Cog8`.

    The regression tests consist of comparing the values against the exact
    solution at :math:`r=0.1` and :math:`t=0.6`. Note: Cog8 currently crashes
    if the spatial interval is a single point; therefore, these tests really
    use :math:`r=[0.1,0.2]`, and the the first spatial point is selected. This
    is because Timmes' solution code returns cell averaged values.
    """

# Gamma = cv (gamma - 1) with cv=1.e12 and gamma=5/3
#       = 6.666666666666667e11

# position,density,temperature,sie,pressure,velocity
    def test_density_timmes(self):
        """Density regression test for Timmes' solver."""
        sol = Cog8Timmes(rho0=3.0, beta=2.0, temp0=100., alpha=-1., gamma=5./3., cv=1.e12)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r, t)
        self.assertAlmostEqual(solrt.density[0], 10.801498645123559)

    def test_density(self):
        """Density regression test for default solver."""
        sol = Cog8(rho0=3.0, beta=2.0, temp0=100., alpha=-1., gamma=5./3., Gamma=6.666666666666667e11)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r, t)
        self.assertAlmostEqual(solrt.density[0], 10.752341249425138)

    def test_temperature_timmes(self):
        """Temperature regression test for Timmes' solver."""
        sol = Cog8Timmes(rho0=3.0, beta=2.0, temp0=100., alpha=-1., gamma=5./3., cv=1.e12)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r, t)
        self.assertAlmostEqual(solrt.temperature[0], 357.1749456358788)

    def test_temperature(self):
        """Temperature regression test for default solver."""
        sol = Cog8(rho0=3.0, beta=2.0, temp0=100., alpha=-1., gamma=5./3., Gamma=6.666666666666667e11)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r, t)
        self.assertAlmostEqual(solrt.temperature[0], 358.80787280298523)

    def test_sie_timmes(self):
        """Specific internal energy regression test for Timmes' solver."""
        sol = Cog8Timmes(rho0=3.0, beta=2.0, temp0=100., alpha=-1., gamma=5./3., cv=1.e12)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r, t)
        scale = 10**15
        self.assertAlmostEqual(solrt.specific_internal_energy[0]/scale, 357174945635878.8/scale, places=7)

    def test_sie(self):
        """Specific internal energy regression test for default solver."""
        sol = Cog8(rho0=3.0, beta=2.0, temp0=100., alpha=-1., gamma=5./3., Gamma=6.666666666666667e11)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r, t)
        self.assertAlmostEqual(solrt.specific_internal_energy[0], 358807872802985.2)

    def test_pressure_timmes(self):
        """Pressure regression test for Timmes' solver."""
        sol = Cog8Timmes(rho0=3.0, beta=2.0, temp0=100., alpha=-1., gamma=5./3., cv=1.e12)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r, t)
        self.assertAlmostEqual(solrt.pressure[0], 2572016460905350.5)

    def test_pressure(self):
        """Pressure regression test for default solver."""
        sol = Cog8(rho0=3.0, beta=2.0, temp0=100., alpha=-1., gamma=5./3., Gamma=6.666666666666667e11)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r, t)
        self.assertAlmostEqual(solrt.pressure[0], 2572016460905351.0)

    def test_velocity_timmes(self):
        """Velocity regression test for Timmes' solver."""
        sol = Cog8Timmes(rho0=3.0, beta=2.0, temp0=100., alpha=-1., cv=1.e12, gamma=5./3.)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r, t)
        self.assertAlmostEqual(solrt.velocity[0], 0.17392955512357586)

    def test_velocity(self):
        """Velocity regression test for default solver."""
        sol = Cog8(rho0=3.0, beta=2.0, temp0=100., alpha=-1., gamma=5./3., Gamma=6.666666666666667e11)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r, t)
        self.assertAlmostEqual(solrt.velocity[0], 0.16666666666666669)

    def test_geometry_error(self):
        """Test Timmes' solver for valid value of geometry"""

        self.assertRaises(ValueError, Cog8Timmes, geometry=-1)

    def test_geometry_error(self):
        """Test default solver for valid value of geometry"""

        self.assertRaises(ValueError, Cog8, geometry=-1)
