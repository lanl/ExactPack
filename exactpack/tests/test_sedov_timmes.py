r"""The following is a test of Timmes' Sedov code.

"""

import numpy as np

import unittest

from exactpack.solvers.sedov.timmes import Sedov as SedovTimmes
from exactpack.solvers.sedov.kamm import Sedov as SedovKamm


class TestSedovTimmesVsKamm(unittest.TestCase):
    r"""This test compares Timmes vs Kamm, the latter of which has been verified
    against Sedov's book and LA-UR-00-6050.  """

    @unittest.expectedFailure
    def test_sedov_timmes_vs_kamm(self):
        r"""Checks Timmes' against Kamm's Sedov solver. Expected to fail because
        Kamm solver does not return correct values at shock for this choice of
        parameters.
        """

        # construct spatial grid and choose time
        nmax = 11
        rmin = 0.95
        rmax = 1.05
        r = np.linspace(rmin, rmax, nmax)
        t = 1.
        solvertimmes = SedovTimmes(geometry=3, eblast=0.851072)
        solutiontimmes = solvertimmes(r, t)
        solverkamm = SedovKamm(geometry=3, eblast=0.851072)
        solutionkamm = solverkamm(r, t)
        tolerance = 4.e-4
        for i in range(nmax):
            self.assertAlmostEqual(solutiontimmes.density[i],
                                   solutionkamm.density[i],  delta=tolerance)
            self.assertAlmostEqual(solutiontimmes.velocity[i],
                                   solutionkamm.velocity[i], delta=tolerance)
            self.assertAlmostEqual(solutiontimmes.pressure[i],
                                   solutionkamm.pressure[i], delta=tolerance)


class TestSedovTimmesShock(unittest.TestCase):
    """Tests Timmes Sedov for correct pre and post shock values.
    """
    # construct spatial grid and choose time
    rmax = 1.2
    r = np.linspace(0.0, rmax, 121)
    t = 1.
    solver = SedovTimmes(eblast=0.851072, gamma=1.4)
    solution = solver(r, t)

    ishock = 100  # shock location

    # analytic solution pre-shock (initial conditions)

    analytic_preshock = {
        'position': r[ishock+1],
        'density': 1.0,
        'specific_internal_energy': 0.0,
        'pressure': 0.0,
        'velocity': 0.0,
        'sound_speed': 0.0
        }

    # analytic solution  at the shock, from Kamm & Timmes 2007,
    # equations 13-18 (to 6 significant figures)

    analytic_postshock = {
        'position': 1.0,
        'density': 6.0,
        'specific_internal_energy': 5.5555e-2,
        'pressure': 1.33333e-1,
        'velocity': 3.33333e-1,
        'sound_speed': 1.76383e-1
        }

    def test_preshock_state(self):
        """Tests density, velocity, pressure, specific internal energy, and
        sound speed immediately before the shock.
        """

        for ikey in self.analytic_preshock.keys():
            self.assertAlmostEqual(self.solution[ikey][self.ishock+1],
                                   self.analytic_preshock[ikey], places=5)

    def test_postshock_state(self):
        """Tests density, velocity, pressure, specific internal energy, and
        sound speed immediately after the shock.
        """

        for ikey in self.analytic_postshock.keys():
            self.assertAlmostEqual(self.solution[ikey][self.ishock],
                                   self.analytic_postshock[ikey], places=5)

    def test_interpolate(self):
        r"""Sedov test: interpolation to large number of points.
        """

        # construct spatial grid and choose time
        rmax = 1.2
        t = 1.
        solver = SedovTimmes(eblast=0.851072, gamma=1.4)
        #
        # Solve for small number of points
        r_small = np.linspace(0.0, rmax, 501)
        solution_small = solver(r_small, t)
        #
        # Solve for large number of points
        r_big = np.linspace(0.0, rmax, 6001)
        solution_big = solver(r_big, t)

        #  Interpolate small to big

        from scipy.interpolate import interp1d

        interpfcn = interp1d(r_small, solution_small.density)

        solution_big_interp = interpfcn(r_big)

        err = np.linalg.norm((solution_big_interp - solution_big.density), 1)

        self.assertTrue(err < 16.0)

    def test_geometry_assignment(self):
        r"""Sedov test: geometry assignment.
        """
        self.assertRaises(ValueError, SedovTimmes, geometry=-1)
