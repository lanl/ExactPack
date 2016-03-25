r"""The following is a test of Timmes' Sedov code. 

"""

import numpy as np

import unittest

from exactpack.solvers.sedov.timmes import Sedov as SedovTimmes
from exactpack.solvers.sedov.kamm import Sedov as SedovKamm

class TestSedovTimmesVsKamm(unittest.TestCase):
    r"""This test compares Timmes vs Kamm, the latter of which has been verified
    against Sedov's book and LA-UR-00-6050.  """

    def test_sedov_timmes_vs_kamm(self):
        r"""Checks Timmes' against Kamm's Sedov solver. 
        """

        # construct spatial grid and choose time
        nmax   = 101
        rmin   = 0.6
        rmax   = 1.0
        r      = np.linspace(rmin, rmax, nmax)
        t      = 1.
        solvertimmes   = SedovTimmes(geometry=3, eblast=0.851072)
        solutiontimmes = solvertimmes(r,t)
        solverkamm     = SedovKamm(geometry=3, eblast=0.851072)
        solutionkamm   = solverkamm(r,t)
        tolerance =  4.e-4
        for i in range(nmax-1): # note: there is disagreement at nmax: 5.9999965521597574 != 0.0 for range(nmax)
            self.assertAlmostEqual(solutiontimmes.density[i],   solutionkamm.density[i],  delta=tolerance)
            self.assertAlmostEqual(solutiontimmes.velocity[i],  solutionkamm.velocity[i], delta=tolerance)
            self.assertAlmostEqual(solutiontimmes.pressure[i],  solutionkamm.pressure[i], delta=tolerance)

class TestSedovTimmesShock(unittest.TestCase):
    """Tests Timmes Sedov for pre and post shock values.
    """
    def test_shock_state(self):
        """Tests density, velocity, pressure, specific internal energy, and
        sound speed.
        """
 
        # construct spatial grid and choose time
        rmax   = 1.2
        r      = np.linspace(0.0, rmax, 101)
        t      = 1.
        solver = SedovTimmes(eblast=0.851072,gamma=1.4)
        solution = solver(r,t)

        i=84 # shock location
        self.assertAlmostEqual(solution.density[i-1],  5.52681913398)
        self.assertAlmostEqual(solution.density[i],    1.0000000E+00)
        self.assertAlmostEqual(solution.velocity[i-1], 0.330465254524)
        self.assertAlmostEqual(solution.velocity[i],   0.0)
        self.assertAlmostEqual(solution.pressure[i-1], 0.127634037837)
        self.assertAlmostEqual(solution.pressure[i],   1.4E-22)
        self.assertAlmostEqual(solution.sie[i-1],      0.0577339491049)
        self.assertAlmostEqual(solution.sie[i],        0.0)
        self.assertAlmostEqual(solution.sound[i-1],    0.179808263155)
        self.assertAlmostEqual(solution.sound[i],      0.0)


    def test_interpolate(self):
        r"""Sedov test: interpolation to large number of points.
        """

        # construct spatial grid and choose time
        rmax   = 1.2
        t      = 1.
        solver = SedovTimmes(eblast=0.851072,gamma=1.4)
        # 
        # Solve for small number of points
        r_small      = np.linspace(0.0, rmax, 501)
        solution_small = solver(r_small,t)
        #
        # Solve for large number of points
        r_big      = np.linspace(0.0, rmax, 6001)
        solution_big = solver(r_big,t)

        #  Interpolate small to big

        from scipy.interpolate import interp1d

        interpfcn = interp1d(r_small, solution_small.density)

        solution_big_interp = interpfcn(r_big)

        err = np.linalg.norm((solution_big_interp - solution_big.density),1)

        self.assertTrue(err<16.0)

    def test_geometry_assignment(self):
        r"""Sedov test: geometry assignment.
        """
        self.assertRaises(ValueError, SedovTimmes, geometry=-1) 
