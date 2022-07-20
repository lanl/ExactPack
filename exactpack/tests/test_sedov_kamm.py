"""The following test verifies the dimensionless self-similar variables of
Kamm's Fortran code against the values published in Sedov's book and
[Kamm2000]_, (Tables I, II, and III, for the spherical, cylindrical, and
planar cases respectively).
"""

import numpy as np

import unittest

from exactpack.solvers.sedov.kamm import Sedov

class TestSedovKammSelfSim(unittest.TestCase):
    r"""Tests for the Kamm's Sedov problem in self-similar variables.
    """

    def test_self_similar_sph(self):
        r"""Kamm's spherical Sedov: self-similar variables
        """

##############################################################################
# test dimensionless self-similar variables
# Comparison with Sedov's book and Kamm's paper, LA-UR-00-6055, Table 3, p 19 (Sph)
#                              0       1       2       3       4       5       6       7       8      9      10
#                              lambda  V-sedov Sedov-f Kamm-f  Sedov-g Kamm-g  Sedof-h Kamm-h  del-f  del-g  del-h
        sedovsph = np.array([0.9913, 0.3300, 0.9814, 0.9814, 0.8379, 0.8388, 0.9109, 0.9116, 1.e-4, 1.e-3, 1.e-3,
                             0.9773, 0.3250, 0.9529, 0.9529, 0.6457, 0.6454, 0.7993, 0.7992, 1.e-4, 1.e-3, 1.e-3,
                             0.9622, 0.3200, 0.9237, 0.9238, 0.4978, 0.4984, 0.7078, 0.7082, 1.e-3, 1.e-3, 1.e-3,
                             0.9342, 0.3120, 0.8744, 0.8745, 0.3241, 0.3248, 0.5923, 0.5929, 1.e-3, 1.e-3, 1.e-3,
                             0.9080, 0.3060, 0.8335, 0.8335, 0.2279, 0.2275, 0.5241, 0.5238, 1.e-4, 1.e-3, 1.e-3,
                             0.8747, 0.3000, 0.7872, 0.7872, 0.1509, 0.1508, 0.4674, 0.4674, 1.e-4, 1.e-3, 1.e-4,
                             0.8359, 0.2950, 0.7397, 0.7398, 0.0967, 0.0968, 0.4272, 0.4273, 1.e-3, 1.e-3, 1.e-3,
                             0.7950, 0.2915, 0.6952, 0.6952, 0.0621, 0.0620, 0.4021, 0.4021, 1.e-4, 1.e-4, 1.e-4,
                             0.7493, 0.2890, 0.6496, 0.6497, 0.0379, 0.0379, 0.3856, 0.3857, 1.e-3, 1.e-3, 1.e-3,
                             0.6788, 0.2870, 0.5844, 0.5844, 0.0174, 0.0174, 0.3732, 0.3732, 1.e-4, 1.e-4, 1.e-4,
                             0.5794, 0.2860, 0.4971, 0.4971, 0.0052, 0.0052, 0.3672, 0.3672, 1.e-4, 1.e-4, 1.e-4,
                             0.4560, 0.2857, 0.3909, 0.3909, 0.0009, 0.0009, 0.3656, 0.3656, 1.e-4, 1.e-4, 1.e-4,
                             0.3600, 0.2857, 0.3086, 0.3086, 0.0002, 0.0001, 0.3655, 0.3655, 1.e-4, 1.e-4, 1.e-4,
                             0.2960, 0.2857, 0.2538, 0.2537, 0.0000, 0.0000, 0.3655, 0.3655, 1.e-3, 1.e-4, 1.e-4,
                             0.2000, 0.2857, 0.1714, 0.1714, 0.0000, 0.0000, 0.3655, 0.3655, 1.e-4, 1.e-4, 1.e-4,
                             0.1040, 0.2857, 0.0892, 0.0891, 0.0000, 0.0000, 0.3655, 0.3655, 1.e-3, 1.e-4, 1.e-4]).reshape(16, 11)

        t = 1.
        solver = Sedov(geometry=3, eblast=0.851072)
        solution = solver.self_similar(sedovsph[:, 0], t)

        verbose = False
        for row in range(16):
            self.assertAlmostEqual(solution.f[row], sedovsph[row,2], delta=sedovsph[row,8])
            self.assertAlmostEqual(solution.g[row] , sedovsph[row,4], delta=sedovsph[row,9])
            self.assertAlmostEqual(solution.h[row], sedovsph[row,6], delta=sedovsph[row,10])
            self.assertAlmostEqual(solution.f[row], sedovsph[row,3], delta=1.e-4)
            self.assertAlmostEqual(solution.g[row] , sedovsph[row,5], delta=1.e-4)
            self.assertAlmostEqual(solution.h[row], sedovsph[row,7], delta=1.e-4)
            if (verbose):
                print("")
                print(row, sedovsph[row,0], sedovsph[row,1], solution[row][2])
                print("       f                 g                 h")
                print('Sedov: {0:.12f}    {1:.12f}    {2:.12f}'.format(sedovsph[row,2], sedovsph[row,4], sedovsph[row,6]))
                print('Kamm : {0:.12f}    {1:.12f}    {2:.12f}'.format(sedovsph[row,3], sedovsph[row,5], sedovsph[row,7]))
                print('ExPc : {0:.12f}    {1:.12f}    {2:.12f}'.format(solution.f[row], solution.g[row], solution.h[row]))


    def test_self_similar_cyl(self):
        r"""Kamm's cylindrical Sedov: self-similar variables
        """

##############################################################################
# test dimensionless self-similar variables
# Comparison with Sedov's book and Kamm's paper, LA-UR-00-6055, Table 2, p 18 (Cyl)
#                              0       1       2       3       4       5       6       7       8      9      10
#                              lambda  V-sedov Sedov-f Kamm-f  Sedov-g Kamm-g  Sedof-h Kamm-h  del-f  del-g  del-h
        sedovcyl = np.array([0.9998, 0.4166, 0.9996, 0.9996, 0.9973, 0.9972, 0.9985, 0.9984, 1.e-4, 1.e-3, 1.e-3,
                             0.9802, 0.4100, 0.9645, 0.9645, 0.7653, 0.7651, 0.8659, 0.8658, 1.e-4, 1.e-3, 1.e-3,
                             0.9644, 0.4050, 0.9374, 0.9374, 0.6285, 0.6281, 0.7832, 0.7829, 1.e-4, 1.e-3, 1.e-3,
                             0.9476, 0.4000, 0.9097, 0.9097, 0.5164, 0.5161, 0.7124, 0.7122, 1.e-4, 1.e-3, 1.e-3,
                             0.9295, 0.3950, 0.8812, 0.8812, 0.4234, 0.4233, 0.6514, 0.6513, 1.e-4, 1.e-3, 1.e-3,
                             0.9096, 0.3900, 0.8514, 0.8514, 0.3451, 0.3450, 0.5983, 0.5982, 1.e-4, 1.e-3, 1.e-3,
                             0.8725, 0.3820, 0.7998, 0.7999, 0.2427, 0.2427, 0.5266, 0.5266, 1.e-3, 1.e-4, 1.e-4,
                             0.8442, 0.3770, 0.7638, 0.7638, 0.1892, 0.1892, 0.4884, 0.4884, 1.e-4, 1.e-4, 1.e-4,
                             0.8094, 0.3720, 0.7226, 0.7226, 0.1414, 0.1415, 0.4545, 0.4545, 1.e-4, 1.e-3, 1.e-4,
                             0.7629, 0.3670, 0.6720, 0.6720, 0.0975, 0.0974, 0.4242, 0.4241, 1.e-4, 1.e-3, 1.e-3,
                             0.7242, 0.3640, 0.6327, 0.6327, 0.0718, 0.0718, 0.4074, 0.4074, 1.e-4, 1.e-4, 1.e-4,
                             0.6894, 0.3620, 0.5989, 0.5990, 0.0545, 0.0545, 0.3969, 0.3969, 1.e-3, 1.e-4, 1.e-4,
                             0.6390, 0.3600, 0.5521, 0.5521, 0.0362, 0.0362, 0.3867, 0.3867, 1.e-4, 1.e-4, 1.e-4,
                             0.5745, 0.3585, 0.4943, 0.4943, 0.0208, 0.0208, 0.3794, 0.3794, 1.e-4, 1.e-4, 1.e-4,
                             0.5180, 0.3578, 0.4448, 0.4448, 0.0123, 0.0123, 0.3760, 0.3760, 1.e-4, 1.e-4, 1.e-4,
                             0.4748, 0.3575, 0.4073, 0.4074, 0.0079, 0.0079, 0.3746, 0.3746, 1.e-3, 1.e-4, 1.e-4,
                             0.4222, 0.3573, 0.3621, 0.3620, 0.0044, 0.0044, 0.3737, 0.3737, 1.e-3, 1.e-4, 1.e-4,
                             0.3654, 0.3572, 0.3133, 0.3133, 0.0021, 0.0021, 0.3733, 0.3732, 1.e-4, 1.e-4, 1.e-3,
                             0.3000, 0.3572, 0.2571, 0.2572, 0.0008, 0.0008, 0.3730, 0.3730, 1.e-3, 1.e-4, 1.e-4,
                             0.2500, 0.3571, 0.2143, 0.2143, 0.0003, 0.0003, 0.3729, 0.3729, 1.e-4, 1.e-4, 1.e-4,
                             0.2000, 0.3571, 0.1714, 0.1714, 0.0001, 0.0001, 0.3729, 0.3729, 1.e-4, 1.e-4, 1.e-4,
                             0.1500, 0.3571, 0.1286, 0.1286, 0.0000, 0.0000, 0.3729, 0.3729, 1.e-4, 1.e-4, 1.e-4,
                             0.1000, 0.3571, 0.0857, 0.0857, 0.0000, 0.0000, 0.3729, 0.3729, 1.e-4, 1.e-4, 1.e-4]).reshape(23, 11)

        t = 1.
        solver = Sedov(geometry=2, eblast=0.311357)
        solution = solver.self_similar(sedovcyl[:, 0], t)

        verbose = False
        for row in range(23):
            self.assertAlmostEqual(solution.f[row], sedovcyl[row,2], delta=sedovcyl[row,8])
            self.assertAlmostEqual(solution.g[row] , sedovcyl[row,4], delta=sedovcyl[row,9])
            self.assertAlmostEqual(solution.h[row], sedovcyl[row,6], delta=sedovcyl[row,10])
            self.assertAlmostEqual(solution.f[row], sedovcyl[row,3], delta=1.e-4)
            self.assertAlmostEqual(solution.g[row] , sedovcyl[row,5], delta=1.e-4)
            self.assertAlmostEqual(solution.h[row], sedovcyl[row,7], delta=1.e-4)
            if (verbose):
                print("")
                print(row, sedovcyl[row,0], sedovcyl[row,1], solution[row][2])
                print("       f                 g                 h")
                print('Sedov: {0:.12f}    {1:.12f}    {2:.12f}'.format(sedovcyl[row,2], sedovcyl[row,4], sedovcyl[row,6]))
                print('Kamm : {0:.12f}    {1:.12f}    {2:.12f}'.format(sedovcyl[row,3], sedovcyl[row,5], sedovcyl[row,7]))
                print('ExPc : {0:.12f}    {1:.12f}    {2:.12f}'.format(solution.f[row], solution.g[row], solution.h[row]))




    def test_self_similar_pla(self):
        r"""Kamm's planar Sedov: self-similar variables
        """

##############################################################################
# test dimensionless self-similar variables
# Comparison with Sedov's book and Kamm's paper, LA-UR-00-6055, Table 1, p 17 (Pla)
#                              0       1       2       3       4       5       6       7       8      9      10
#                              lambda  V-sedov Sedov-f Kamm-f  Sedov-g Kamm-g  Sedof-h Kamm-h  del-f  del-g  del-h
        sedovpla = np.array([0.9797, 0.5500, 0.9699, 0.9699, 0.8625, 0.8620, 0.9162, 0.9159, 1.e-4, 1.e-3, 1.e-3,
                             0.9420, 0.5400, 0.9156, 0.9157, 0.6659, 0.6662, 0.7915, 0.7917, 1.e-3, 1.e-3, 1.e-3, 
                             0.9013, 0.5300, 0.8599, 0.8598, 0.5160, 0.5159, 0.6923, 0.6922, 1.e-3, 1.e-3, 1.e-3, 
                             0.8565, 0.5200, 0.8017, 0.8017, 0.3982, 0.3981, 0.6120, 0.6119, 1.e-4, 1.e-3, 1.e-3, 
                             0.8050, 0.5100, 0.7390, 0.7390, 0.3019, 0.3020, 0.5457, 0.5458, 1.e-4, 1.e-3, 1.e-3, 
                             0.7419, 0.5000, 0.6678, 0.6677, 0.2200, 0.2201, 0.4904, 0.4905, 1.e-3, 1.e-3, 1.e-3, 
                             0.7029, 0.4950, 0.6263, 0.6263, 0.1823, 0.1823, 0.4661, 0.4661, 1.e-4, 1.e-4, 1.e-4, 
                             0.6553, 0.4900, 0.5780, 0.5780, 0.1453, 0.1453, 0.4437, 0.4437, 1.e-4, 1.e-4, 1.e-4, 
                             0.5925, 0.4850, 0.5172, 0.5173, 0.1074, 0.1075, 0.4229, 0.4230, 1.e-3, 1.e-3, 1.e-3, 
                             0.5396, 0.4820, 0.4682, 0.4682, 0.0826, 0.0826, 0.4116, 0.4112, 1.e-4, 1.e-4, 1.e-3, 
                             0.4912, 0.4800, 0.4244, 0.4244, 0.0641, 0.0641, 0.4038, 0.4037, 1.e-4, 1.e-4, 1.e-3, 
                             0.4589, 0.4790, 0.3957, 0.3957, 0.0536, 0.0535, 0.4001, 0.4001, 1.e-4, 1.e-3, 1.e-4, 
                             0.4161, 0.4780, 0.3580, 0.3580, 0.0415, 0.0415, 0.3964, 0.3964, 1.e-4, 1.e-4, 1.e-4, 
                             0.3480, 0.4770, 0.2988, 0.2988, 0.0263, 0.0263, 0.3929, 0.3929, 1.e-4, 1.e-4, 1.e-4, 
                             0.2810, 0.4765, 0.2410, 0.2410, 0.0153, 0.0153, 0.3911, 0.3911, 1.e-4, 1.e-4, 1.e-4, 
                             0.2320, 0.4763, 0.1989, 0.1989, 0.0095, 0.0095, 0.3905, 0.3905, 1.e-4, 1.e-4, 1.e-4, 
                             0.1680, 0.4762, 0.1441, 0.1440, 0.0042, 0.0042, 0.3901, 0.3901, 1.e-3, 1.e-4, 1.e-4, 
                             0.1040, 0.4762, 0.0891, 0.0891, 0.0013, 0.0013, 0.3900, 0.3900, 1.e-4, 1.e-4, 1.e-4]).reshape(18, 11)

        t = 1.
        solver = Sedov(geometry=1, eblast=0.0673185)
        solution = solver.self_similar(sedovpla[:, 0], t)

        verbose = False
        for row in range(18):
            self.assertAlmostEqual(solution.f[row], sedovpla[row,2], delta=sedovpla[row,8])
            self.assertAlmostEqual(solution.g[row] , sedovpla[row,4], delta=sedovpla[row,9])
            self.assertAlmostEqual(solution.h[row], sedovpla[row,6], delta=sedovpla[row,10])
            self.assertAlmostEqual(solution.f[row], sedovpla[row,3], delta=1.e-4)
            self.assertAlmostEqual(solution.g[row] , sedovpla[row,5], delta=1.e-4)
            self.assertAlmostEqual(solution.h[row], sedovpla[row,7], delta=1.e-4)
            if (verbose):
                print("")
                print(row, sedovpla[row,0], sedovpla[row,1], solution[row][2])
                print("       f                 g                 h")
                print('Sedov: {0:.12f}    {1:.12f}    {2:.12f}'.format(sedovpla[row,2], sedovpla[row,4], sedovpla[row,6]))
                print('Kamm : {0:.12f}    {1:.12f}    {2:.12f}'.format(sedovpla[row,3], sedovpla[row,5], sedovpla[row,7]))
                print('ExPc : {0:.12f}    {1:.12f}    {2:.12f}'.format(solution.f[row], solution.g[row], solution.h[row]))

    def test_shock_state_interpolated(self):
        r"""Sedov Kamm Problem: pre and post shock values, with interpolation to
        large number of points
        """

        # construct sptial grid and choose time
        rmax   = 1.2
        t      = 1.
        solver = Sedov()
        # 
        # Solve for small number of points
        r_small      = np.linspace(0.0, rmax, 501)
        solution_small = solver(r_small,t)
        # Solve for large number of points
        r_big      = np.linspace(0.0, rmax, 6001)
        solution_big = solver(r_big,t)

        #  Interpolate small to big

        from scipy.interpolate import interp1d

        interpfcn = interp1d(r_small, solution_small.density)

        solution_big_interp = interpfcn(r_big)

        err = np.linalg.norm((solution_big_interp - solution_big.density),1)

        self.assertTrue(err<16.0)

    def test_geometry_error_sedov_kamm(self):
        """Sedov Kamm Problem: Test for valid value of geometry"""

        self.assertRaises(ValueError, Sedov, geometry=-1) 

class TestSedovKammShock(unittest.TestCase):
    """Tests Kamm Sedov for correct pre and post shock values.
    """
    # construct spatial grid and choose time
    rmax = 1.2
    r = np.linspace(0.0, rmax, 121)
    t = 1.
    solver = Sedov(eblast=0.851072, gamma=1.4, geometry=3)
    solution = solver(r, t)

    ishock = 100  # shock location

    # analytic solution pre-shock (initial conditions)

    analytic_preshock = {
        'position': r[ishock+1],
        'density': 1.0,
        'specific_internal_energy': 0.0,
        'pressure': 0.0,
        'velocity': 0.0,
        }

    # analytic solution  at the shock, from Kamm & Timmes 2007,
    # equations 13-18 (to 6 significant figures)

    analytic_postshock = {
        'position': 1.0,
        'density': 6.0,
        'specific_internal_energy': 5.5555e-2,
        'pressure': 1.33333e-1,
        'velocity': 3.33333e-1,
        }

    def test_preshock_state(self):
        """Tests density, velocity, pressure, specific internal energy, and
        sound speed immediately before the shock.
        """

        for ikey in self.analytic_preshock:
            self.assertAlmostEqual(self.solution[ikey][self.ishock+1],
                                   self.analytic_preshock[ikey], places=5)

    @unittest.expectedFailure
    def test_postshock_state(self):
        """Tests density, velocity, pressure, specific internal energy, and
        sound speed immediately after the shock.

        Currently, the Kamm solver does not return the correct value of the
        physical variables at the shock location, for at least some cases
        """

        for ikey in self.analytic_postshock:
            self.assertAlmostEqual(self.solution[ikey][self.ishock],
                                   self.analytic_postshock[ikey], places=5)

