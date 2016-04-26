"""Tests for the Cog8 problem. 
"""

import unittest

import numpy

from exactpack.solvers.cog.cog8 import Cog8
from exactpack.solvers.cog.cog8_timmes import Cog8 as Cog8Timmes


class TestCog8(unittest.TestCase):
    """Test for :class:`exactpack.cog.cog8_timmes.Cog8`.

    The tests consist of comparing the values against the exact
    solution at :math:`r=0.2` and :math:`t=2.0`.

    Note: Cog8 currently crashes if the spatial interval is a single
    point; therefore, these tests really use :math:`r=[0.2,0.3]` and
    the the first spatial point is selected. This is because Timmes'
    solution code returns cell averaged values. Fix this.

    """

# Gamma = cv (gamma - 1) with cv=1.e12 and gamma=5/3
#       = 6.666666666666667e11

# position,density,temperature,sie,pressure,velocity  
    def test_density_timmes(self):
        """cog8 problem: density timmes"""
        sol = Cog8Timmes(rho0=3.0,beta=2.0,temp0=100.,alpha=-1.,gamma=5./3.,cv=1.e12)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.density[0],10.801498645123559)

    def test_density(self):
        """cog8 problem: density"""
        sol = Cog8(rho0=3.0,beta=2.0,temp0=100.,alpha=-1.,gamma=5./3.,Gamma=6.666666666666667e11)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.density[0],10.752341249425138)

    def test_temperature_timmes(self):
        """cog8 problem: temperature timmes"""
        sol = Cog8Timmes(rho0=3.0,beta=2.0,temp0=100.,alpha=-1.,gamma=5./3.,cv=1.e12)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.temperature[0],357.1749456358788)

    def test_temperature(self):
        """cog8 problem: temperature"""
        sol = Cog8(rho0=3.0,beta=2.0,temp0=100.,alpha=-1.,gamma=5./3.,Gamma=6.666666666666667e11)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.temperature[0],358.80787280298523)

    def test_sie_timmes(self):
        """cog8 problem: sie timmes"""
        sol = Cog8Timmes(rho0=3.0,beta=2.0,temp0=100.,alpha=-1.,gamma=5./3.,cv=1.e12)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r,t)
        scale = 10**15
        self.assertAlmostEqual(solrt.sie[0]/scale,357174945635878.8/scale,places=7)

    def test_sie(self):
        """cog8 problem: sie"""
        sol = Cog8(rho0=3.0,beta=2.0,temp0=100.,alpha=-1.,gamma=5./3.,Gamma=6.666666666666667e11)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.sie[0],358807872802985.2)

    def test_pressure_timmes(self):
        """cog8 problem: pressure timmes"""
        sol = Cog8Timmes(rho0=3.0,beta=2.0,temp0=100.,alpha=-1.,gamma=5./3.,cv=1.e12)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.pressure[0],2572016460905350.5)

    def test_pressure(self):
        """cog8 problem: pressure"""
        sol = Cog8(rho0=3.0,beta=2.0,temp0=100.,alpha=-1.,gamma=5./3.,Gamma=6.666666666666667e11)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.pressure[0],2572016460905351.0)

    def test_velocity_timmes(self):
        """cog8 problem: velocity timmes"""
        sol = Cog8Timmes(rho0=3.0,beta=2.0,temp0=100.,alpha=-1.,cv=1.e12,gamma=5./3.)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.velocity[0],0.17392955512357586)

    def test_velocity(self):
        """cog8 problem: velocity"""
        sol = Cog8(rho0=3.0,beta=2.0,temp0=100.,alpha=-1.,gamma=5./3.,Gamma=6.666666666666667e11)
        r = numpy.array([0.1, 0.2])
        t = 0.6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.velocity[0],0.16666666666666669)


    def test_geometry_error(self):
        """Cog8 Problem Timmes: Test for valid value of geometry"""

        self.assertRaises(ValueError, Cog8Timmes, geometry=-1) 

    def test_geometry_error(self):
        """Cog8 Problem: Test for valid value of geometry"""

        self.assertRaises(ValueError, Cog8, geometry=-1) 

