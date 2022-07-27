"""Unittests for the JWL-EOS Riemann problem.
"""

import unittest

import numpy

from exactpack.contrib.riemann_jwl.kamm import RiemannJWL

class TestJWLRiemann(unittest.TestCase):
    r"""Regression test for :class:`exactpack.contrib.riemann_jwl.kamm.RiemannJWL`.
    """

    # solution.dat
    # Shock
    # Location       Velocity       Pressure       Density        SIE            Sndspd         Time
    # 1.9000000E+01  0.0000000E+00  1.0000000E+00  9.5250000E-01  1.1779261E+00  1.4094146E+00  2.0000000E+01
    # 2.0000000E+01 -1.3299659E-01  1.1911644E+00  1.0445603E+00  1.2792983E+00  1.4692505E+00  2.0000000E+01
    #
    def test_shock_state(self):
        r"""Regression test for JWL Riemann problem (Lee): pre and post shock.
        """
        r = numpy.linspace(0., 100., 100)
        t = 20.0
        solver = RiemannJWL(interface_loc=50., 
                            rhol=0.9525, pl=1., ul=0., 
                            rhor=3.810 , pr=2., ur=0.,
                            rho0l=1.905, sie0l=0., gammal=0.8938, bigal=6.321e2, bigbl=-4.472e-2, r1l=1.13e1, r2l=1.13, 
                            rho0r=1.905, sie0r=0., gammar=0.8938, bigar=6.321e2, bigbr=-4.472e-2, r1r=1.13e1, r2r=1.13)
        solution = solver(r,t)
        i = 20
        self.assertAlmostEqual(solution.velocity[i-1], 0.0)
        self.assertAlmostEqual(solution.pressure[i-1], 1.0)
        self.assertAlmostEqual(solution.density[i-1], 9.52500E-01)
        self.assertAlmostEqual(solution.energy[i-1], 1.1779261E+00)
        self.assertAlmostEqual(solution.sound[i-1], 1.4094146E+00)
        self.assertAlmostEqual(solution.velocity[i], -1.3299659E-01)
        self.assertAlmostEqual(solution.pressure[i], 1.1911644E+00)
        self.assertAlmostEqual(solution.density[i], 1.0445603E+00)
        self.assertAlmostEqual(solution.energy[i], 1.2792983E+00)
        self.assertAlmostEqual(solution.sound[i], 1.4692505E+00)

    # solution.dat
    # Contact
    # Location       Velocity       Pressure       Density        SIE            Sndspd         Time
    # 4.7000000E+01 -1.3299659E-01  1.1911644E+00  1.0445603E+00  1.2792983E+00  1.4692505E+00  2.0000000E+01
    # 4.8000000E+01 -1.3299659E-01  1.1911644E+00  3.5156642E+00 -1.0148361E-03  1.5222130E+00  2.0000000E+01
    #
    def test_contact_state(self):
        r"""Regression test for JWL Riemann problem: pre and post contact.
        """
        r = numpy.linspace(0., 100., 100)
        t = 20.0
        solver = RiemannJWL(interface_loc=50., 
                            rhol=0.9525, pl=1., ul=0., 
                            rhor=3.810 , pr=2., ur=0.,
                            rho0l=1.905, sie0l=0., gammal=0.8938, bigal=6.321e2, bigbl=-4.472e-2, r1l=1.13e1, r2l=1.13, 
                            rho0r=1.905, sie0r=0., gammar=0.8938, bigar=6.321e2, bigbr=-4.472e-2, r1r=1.13e1, r2r=1.13)
        solution = solver(r,t)
        i = 47
        self.assertAlmostEqual(solution.velocity[i-1], -1.3299659E-01)
        self.assertAlmostEqual(solution.pressure[i-1], 1.1911644E+00)
        self.assertAlmostEqual(solution.density[i-1], 1.0445603E+00)
        self.assertAlmostEqual(solution.energy[i-1], 1.2792983E+00)
        self.assertAlmostEqual(solution.sound[i-1], 1.4692505E+00)
        self.assertAlmostEqual(solution.velocity[i], -1.3299659E-01)
        self.assertAlmostEqual(solution.pressure[i], 1.1911644E+00)
        self.assertAlmostEqual(solution.density[i], 3.5156642E+00)
        self.assertAlmostEqual(solution.energy[i], -1.0148361E-03) 
        self.assertAlmostEqual(solution.sound[i], 1.5222130E+00)
