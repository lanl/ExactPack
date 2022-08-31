"""Unittests for the Reinicke Meyer-ter-Vehn (RMTV) problem.
"""

import pytest

import numpy

from exactpack.solvers.rmtv.timmes import Rmtv


class TestRmtvTimmes():
    r"""Regression test for :class:`exactpack.solvers.rmtv.timmes.Rmtv`.

    The comparisions are made at :math:`r=0.015`.
    """

    def test_density(self):
        """Rmtv problem: density"""
        sol = Rmtv()
        r = numpy.array([0.015])
        solrt = sol._run(r)   # bypass wrapper
        numpy.testing.assert_allclose(solrt.density[0], 3.21988174430864)

    def test_temperature(self):
        """Rmtv problem: temperature"""
        sol = Rmtv()
        r = numpy.array([0.015])
        solrt = sol._run(r)   # bypass wrapper
        numpy.testing.assert_allclose(solrt.temperature[0], 4185.37719625862)

    def test_energy(self):
        """Rmtv problem: energy"""
        sol = Rmtv()
        r = numpy.array([0.015])
        solrt = sol._run(r)   # bypass wrapper
        numpy.testing.assert_allclose(solrt.energy[0], 1.674150878503448e+17)

    def test_pressure(self):
        """Rmtv problem: pressure"""
        sol = Rmtv()
        r = numpy.array([0.015])
        solrt = sol._run(r)   # bypass wrapper
        numpy.testing.assert_allclose(solrt.pressure[0], 1.347641962727881e+17)

    def test_velocity(self):
        """Rmtv problem: velocity"""
        sol = Rmtv()
        r = numpy.array([0.015])
        solrt = sol._run(r)   # bypass wrapper
        numpy.testing.assert_allclose(solrt.velocity[0], 14275298.14703435)



