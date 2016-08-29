"""Tests for the Mader rarefaction problem.
"""

import unittest

import numpy

from exactpack.solvers.mader.timmes import Mader


class TestMaderTimmes(unittest.TestCase):
    r"""Test for :class:`exactpack.mader.timmes.Mader`.

    The tests consist of comparing the values against the exact
    solution at :math:`r=0.7` and :math:`t=6.25 \times 10^{-6}`.

    Note: Mader currently returns a NaN if the spatial interval is a
    single point; therefore, these tests really use
    :math:`r=[0.7,0.8]` and the the first spatial point is
    selected. This is because Timmes' solution code returns cell
    averaged values.

    """

    def test_velocity(self):
        """Mader problem: velocity"""
        sol = Mader(p_cj=3.0e11, d_cj=8.0e5, gamma=3.0, u_piston=0.0)
        r = numpy.array([0.7, 0.8]) # r must contain 2 elements, otherwise
        t = 6.25e-6                 # the density and pressure are nan
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.velocity[0],144000.0)

    def test_pressure(self):
        """Mader problem: pressure"""
        sol = Mader(p_cj=3.0e11, d_cj=8.0e5, gamma=3.0, u_piston=0.0)
        r = numpy.array([0.7, 0.8])
        t = 6.25e-6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.pressure[0],223599111111.10834)


    def test_sound(self):
        """Mader problem: sound"""
        sol = Mader(p_cj=3.0e11, d_cj=8.0e5, gamma=3.0, u_piston=0.0)
        r = numpy.array([0.7, 0.8])
        t = 6.25e-6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.sound[0],544000.0)


    def test_density(self):
        """Mader problem: density"""
        sol = Mader(p_cj=3.0e11, d_cj=8.0e5, gamma=3.0, u_piston=0.0)
        r = numpy.array([0.7, 0.8])
        t = 6.25e-6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.density[0],2.26666666666663)



    def test_xdet(self):
        """Mader problem: xdet"""
        sol = Mader(p_cj=3.0e11, d_cj=8.0e5, gamma=3.0, u_piston=0.0)
        r = numpy.array([0.7, 0.8])
        t = 6.25e-6
        solrt = sol(r,t)
        self.assertAlmostEqual(solrt.xdet[0],4.3)





        
