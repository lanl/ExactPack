"""Tests for the Reinicke Meyer-ter-Vehn (RMTV) problem.
"""

import unittest

import numpy

from exactpack.solvers.rmtv.timmes import Rmtv


class TestRmtvTimmes(unittest.TestCase):
    r"""Test for :class:`exactpack.rmtv.timmes.Rmtv`.

    The comparisions are made at :math:`r=0.015` and
    :math:`t=xx`. Note: This problem is only calculated at a single
    time :math:`t=xx` 
    """

    def test_density(self):
        """Rmtv problem: density"""
        sol = Rmtv()
        r = numpy.array([0.015])
        solrt = sol._run(r)   # bypass wrapper
        self.assertAlmostEqual(solrt.density[0],3.21988174430864)

    def test_temperature(self):
        """Rmtv problem: temperature"""
        sol = Rmtv()
        r = numpy.array([0.015])
        solrt = sol._run(r)   # bypass wrapper
        self.assertAlmostEqual(solrt.temperature[0],4185.37719625862)

    @unittest.expectedFailure
    def test_energy(self):
        """Rmtv problem: energy"""
        sol = Rmtv()
        r = numpy.array([0.015])
        solrt = sol._run(r)   # bypass wrapper
        self.assertAlmostEqual(solrt.energy[0],1.674150878503448e+17)

    @unittest.expectedFailure
    def test_pressure(self):
        """Rmtv problem: pressure"""
        sol = Rmtv()
        r = numpy.array([0.015])
        solrt = sol._run(r)   # bypass wrapper
        self.assertAlmostEqual(solrt.pressure[0],1.347641962727881e+17)

    @unittest.expectedFailure
    def test_velocity(self):
        """Rmtv problem: velocity"""
        sol = Rmtv()
        r = numpy.array([0.015])
        solrt = sol._run(r)   # bypass wrapper
        self.assertAlmostEqual(solrt.velocity[0],14275298.14703435)



