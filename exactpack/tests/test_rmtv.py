"""Unittests for the Reinicke Meyer-ter-Vehn (RMTV) problem.

The expected data for these tests has been generated using the old Fortran
implementation of the solver by Timmes.
"""

import pytest

import numpy

from exactpack.solvers.rmtv import Rmtv


class TestRmtvTimmes():
    r"""Regression test for :class:`exactpack.solvers.rmtv.timmes.Rmtv`.

    The comparisions are made at :math:`r=0.015`.
    """
    r = numpy.linspace(0.001, 1.0, 20)
    sol = Rmtv()
    solrt = sol._run(r)

    def test_density(self):
        """Rmtv problem: density"""
        expected = [
            3.211666914010163, 3.3184753022475673, 3.654845603905064,
            4.298949329418528, 5.410581672991555, 7.292370931113524,
            10.528716893758011, 16.374385142429798, 28.715286739205517,
            5.474565677910465, 4.2122628187211815, 3.3635602195771566,
            2.755987338829082, 2.302613285717047, 1.9538321077584628,
            1.6789973067499615, 1.4579625670774448, 1.274947624933391,
            1.1207797303326805, 1.0
        ]
        numpy.testing.assert_allclose(self.solrt.density, expected, rtol=1.0e-5)

    def test_temperature(self):
        """Rmtv problem: temperature"""
        expected = [
            4185.385000517389, 4185.264166629872, 4184.846452941687,
            4183.90158846356, 4181.781013756959, 4176.58194344499,
            4162.014134218895, 4113.336162497097, 3894.7211006442008,
            3262.1663363690864, 3006.97961261811, 2788.2134941922163,
            2592.2222976879916, 2408.5753476598265, 2226.770592274654,
            2031.5984754168355, 1786.9260801977146, 1199.957273110956,
            0.0, 0.0
        ]
        numpy.testing.assert_allclose(self.solrt.temperature, expected, rtol=1.0e-5)

    def test_energy(self):
        """Rmtv problem: energy"""
        expected = [
            1.6741540002069555e+17, 1.674105666651949e+17, 1.673938581176675e+17,
            1.673560635385424e+17, 1.6727124055027834e+17, 1.670632777377996e+17,
            1.6648056536875584e+17, 1.6453344649988387e+17, 1.5578884402576803e+17,
            1.3048665345476346e+17, 1.202791845047244e+17, 1.1152853976768866e+17,
            1.0368889190751966e+17, 9.634301390639307e+16, 8.907082369098616e+16,
            8.126393901667342e+16, 7.147704320790858e+16, 4.799829092443824e+16,
            0.0, 0.0
        ]
        numpy.testing.assert_allclose(self.solrt.energy, expected, rtol=1.0e-5)

    def test_pressure(self):
        """Rmtv problem: pressure"""
        expected = [
            1.3442062528556106e+17, 1.388869577034298e+17, 1.5294967661551626e+17,
            1.7986380928078538e+17, 2.2625867713497443e+17, 3.045718475579187e+17,
            4.3820668527010106e+17, 6.735335054501166e+17, 1.1183803317423233e+18,
            1.7858943860721126e+17, 1.2666188418883886e+17, 9.378323992753162e+16,
            7.144131831858537e+16, 5.546017595172072e+16, 4.350735879798548e+16,
            3.4110483686221948e+16, 2.605271335062696e+16, 1.529882675374362e+16,
            0.0, 0.0
        ]
        numpy.testing.assert_allclose(self.solrt.pressure, expected, rtol=1.0e-5)

    def test_velocity(self):
        """Rmtv problem: velocity"""
        expected = [
            3342474.735365358, 51209749.4752897, 103036268.44135334,
            157764709.87829688, 216292189.3810227, 278908596.6378057,
            345363581.6793605, 415382223.3196282, 490835630.178072,
            94815009.58400244, 72147091.81828043, 56399196.08349457,
            44767837.89227895, 35802091.09985216, 28636426.79775544,
            22672181.24035581, 17326007.301569063, 10087816.49859422,
            0.0, 0.0
        ]
        numpy.testing.assert_allclose(self.solrt.velocity, expected, rtol=0.02)



