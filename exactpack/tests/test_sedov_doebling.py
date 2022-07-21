''' Tests the solver implementation of the Doebling Sedov code
'''

import unittest

import numpy as np
import scipy.optimize as sci_opt

from exactpack.solvers.sedov.doebling import Sedov


def sedovFcnTable(solution, lamvec):
    '''helper function to find values of Sedov functions corresponding
       to a list of lambda values, lamvec'''

    nlam = len(lamvec)

    v = np.zeros(nlam)
    l_fun = np.zeros(nlam)
    f_fun = np.zeros(nlam)
    g_fun = np.zeros(nlam)
    h_fun = np.zeros(nlam)

    #   loop over lam values

    for i in range(nlam):

        # find v that corresponds to lam[i] by
        # minimizing (l_fun(v) - lam[i])**2

        vmin = solution.v0
        vmax = solution.v2
        lam_want = lamvec[i]

        v[i] = sci_opt.fminbound(sed_lam_min, vmin, vmax,
                                 args=(solution, lam_want), xtol=1e-16)
        v[i] = sci_opt.fmin(sed_lam_min, v[i], args=(solution, lam_want),
                            disp=False, xtol=1e-16, ftol=1e-16)

        # Compute all sedov functions at v[i]

        l_fun[i], dlamdv, f_fun[i], g_fun[i], h_fun[i] =\
            solution.sedov_funcs_standard(v[i])

    return l_fun, v, f_fun, g_fun, h_fun


def sed_lam_min(v, solution, lam_want):
    '''helper function to find value of v corresponding to
       value of lambda'''

    [l_fun, dlamdv, f_fun, g_fun, h_fun] = solution.sedov_funcs_standard(v)

    return (l_fun - lam_want)**2


class TestSedovDoeblingAssignments(unittest.TestCase):
    """Tests :class:`exactpack.solvers.sedov.doebling.Sedov`.

    These tests confirm proper assignment of variables, including default
    values
    """

    def test_defaults(self):

        # here are the defaults
        geometry = 3
        gamma = 7.0/5.0
        rho0 = 1.0
        omega = 0.0
        eblast = 0.851072

        solution = Sedov()

        self.assertEqual(solution.geometry, geometry)
        self.assertEqual(solution.gamma, gamma)
        self.assertEqual(solution.rho0, rho0)
        self.assertEqual(solution.omega, omega)
        self.assertEqual(solution.eblast, eblast)

    def test_assignment(self):
        # tests proper assignment of parameters
        #
        #  These values are made up and not physically meaningful
        #  This is only an arbitrary test case
        #

        # here are the defaults
        geometry = 1
        gamma = 1.1
        rho0 = 0.5
        omega = 0.2
        eblast = 3.14159

        solution = Sedov(geometry=geometry, gamma=gamma, rho0=rho0,
                         omega=omega, eblast=eblast)

        self.assertEqual(solution.geometry, geometry)
        self.assertEqual(solution.gamma, gamma)
        self.assertEqual(solution.rho0, rho0)
        self.assertEqual(solution.omega, omega)
        self.assertEqual(solution.eblast, eblast)

    #
    # Confirm that illegal parameter values raise an error
    #

    def test_illegal_value_geometry(self):
        self.assertRaises(ValueError, Sedov, geometry=-1.0)

    def test_illegal_value_gamma(self):
        self.assertRaises(ValueError, Sedov, gamma=0.8)

    def test_illegal_value_rho_0(self):
        self.assertRaises(ValueError, Sedov, rho0=-1.0)

    def test_illegal_value_omega(self):
        self.assertRaises(ValueError, Sedov, omega=-1.0)

    def test_illegal_value_eblast(self):
        self.assertRaises(ValueError, Sedov, eblast=-1.0)

    def test_illegal_combination_omega_geometry(self):
        self.assertRaises(ValueError, Sedov, omega=2.0, geometry=1.0)

    def test_illegal_value_time(self):
        solution = Sedov()
        self.assertRaises(ValueError, solution, r=[0., 1.], t=0.0)


class TestSedovDoeblingSpecialSingularities(unittest.TestCase):
    """Tests :class:`exactpack.solvers.sedov.doebling.Sedov`.

    These test the special sigularity cases of denom2=0 and denom3=0
    """

    def test_special_singularity_omega2(self):
        solution = Sedov(geometry=3, gamma=1.4, omega=2.71428)
        self.assertEqual(solution.special_singularity, 'omega2')

    def test_special_singularity_omega3(self):
        solution = Sedov(geometry=3, gamma=1.4, omega=1.8)
        self.assertEqual(solution.special_singularity, 'omega3')


class TestSedovDoeblingFunctionsTable1(unittest.TestCase):
    r''' Compare results to Kamm & Timmes, Table 1.
    Sedov Functions for gamma=1.4, planar geometry case'''

    # define vector of lambda values

    lamvec = [0.9797, 0.9420, 0.9013, 0.8565, 0.8050, 0.7419, 0.7029,
              0.6553, 0.5925, 0.5396, 0.4912, 0.4589, 0.4161, 0.3480,
              0.2810, 0.2320, 0.1680, 0.1040]

    # compute sedov function values for each lambda value

    solution = Sedov(gamma=1.4, geometry=1, omega=0.)

    l_fun, v, f_fun, g_fun, h_fun = sedovFcnTable(solution, lamvec)

    # compare results to Kamm & Timmes Table 1

    v_ref = [0.5500, 0.5400, 0.5300, 0.5200, 0.5100, 0.5000, 0.4950,
             0.4900, 0.4850, 0.4820, 0.4800, 0.4790, 0.4780, 0.4770,
             0.4765, 0.4763, 0.4762, 0.4762]

    f_fun_ref = [0.9699, 0.9157, 0.8598, 0.8017, 0.7390, 0.6677, 0.6263,
                 0.5780, 0.5173, 0.4682, 0.4244, 0.3957, 0.3580, 0.2988,
                 0.2410, 0.1989, 0.1440, 0.0891]

    g_fun_ref = [0.8620, 0.6662, 0.5159, 0.3981, 0.3020, 0.2201, 0.1823,
                 0.1453, 0.1075, 0.0826, 0.0641, 0.0535, 0.0415, 0.0263,
                 0.0153, 0.0095, 0.0042, 0.0013]

    h_fun_ref = [0.9159, 0.7917, 0.6922, 0.6119, 0.5458, 0.4905, 0.4661,
                 0.4437, 0.4230, 0.4112, 0.4037, 0.4001, 0.3964, 0.3929,
                 0.3911, 0.3905, 0.3901, 0.3900]

    # Convert computed Sedov functions to list and round to 4 decimal
    # places to agree with precision of reference table from Kamm & Timmes

    l_fun = [round(elem, 4) for elem in l_fun.tolist()]

    def test_sedov_functions_table1_l(self):
        self.assertListEqual(self.l_fun, self.lamvec)

    v = [round(elem, 4) for elem in v.tolist()]

    def test_sedov_functions_table1_v(self):
        self.assertListEqual(self.v, self.v_ref)

    f_fun = [round(elem, 4) for elem in f_fun.tolist()]

    def test_sedov_functions_table1_f(self):
        self.assertListEqual(self.f_fun, self.f_fun_ref)

    g_fun = [round(elem, 4) for elem in g_fun.tolist()]

    def test_sedov_functions_table1_g(self):
        self.assertListEqual(self.g_fun, self.g_fun_ref)

    h_fun = [round(elem, 4) for elem in h_fun.tolist()]

    def test_sedov_functions_table1_h(self):
        self.assertListEqual(self.h_fun, self.h_fun_ref)


class TestSedovDoeblingFunctionsTable2(unittest.TestCase):
    r''' Compare results to Kamm & Timmes,
    Table 2. Sedov Functions for gamma=1.4, cylindrical geometry case'''

    # define vector of lambda values

    lamvec = [0.9998, 0.9802, 0.9644, 0.9476, 0.9295, 0.9096, 0.8725,
              0.8442, 0.8094, 0.7629, 0.7242, 0.6894, 0.6390, 0.5745,
              0.5180, 0.4748, 0.4222, 0.3654, 0.3000, 0.2500, 0.2000,
              0.1500, 0.1000]

    # compute sedov function values for each lambda value

    solution = Sedov(gamma=1.4, geometry=2, omega=0.)

    l_fun, v, f_fun, g_fun, h_fun = sedovFcnTable(solution, lamvec)

    # compare results to Kamm & Timmes Table 2

    v_ref = [0.4166, 0.4100, 0.4050, 0.4000, 0.3950, 0.3900, 0.3820,
             0.3770, 0.3720, 0.3670, 0.3640, 0.3620, 0.3600, 0.3585,
             0.3578, 0.3575, 0.3573, 0.3572, 0.3572, 0.3571, 0.3571,
             0.3571, 0.3571]

    f_fun_ref = [0.9996, 0.9645, 0.9374, 0.9097, 0.8812, 0.8514, 0.7999,
                 0.7638, 0.7226, 0.6720, 0.6327, 0.5990, 0.5521, 0.4943,
                 0.4448, 0.4074, 0.3620, 0.3133, 0.2572, 0.2143, 0.1714,
                 0.1286, 0.0857]

    g_fun_ref = [0.9972, 0.7651, 0.6281, 0.5161, 0.4233, 0.3450, 0.2427,
                 0.1892, 0.1415, 0.0974, 0.0718, 0.0545, 0.0362, 0.0208,
                 0.0123, 0.0079, 0.0044, 0.0021, 0.0008, 0.0003, 0.0001,
                 0.0000, 0.0000]

    h_fun_ref = [0.9984, 0.8658, 0.7829, 0.7122, 0.6513, 0.5982, 0.5266,
                 0.4884, 0.4545, 0.4241, 0.4074, 0.3969, 0.3867, 0.3794,
                 0.3760, 0.3746, 0.3737, 0.3732, 0.3730, 0.3729, 0.3729,
                 0.3729, 0.3729]

    # Convert computed Sedov functions to list and round to 4 decimal
    # places to agree with precision of reference table from Kamm & Timmes

    l_fun = [round(elem, 4) for elem in l_fun.tolist()]

    def test_sedov_functions_table2_l(self):
        self.assertListEqual(self.l_fun, self.lamvec)

    v = [round(elem, 4) for elem in v.tolist()]

    def test_sedov_functions_table2_v(self):
        self.assertListEqual(self.v, self.v_ref)

    f_fun = [round(elem, 4) for elem in f_fun.tolist()]

    def test_sedov_functions_table2_f(self):
        self.assertListEqual(self.f_fun, self.f_fun_ref)

    g_fun = [round(elem, 4) for elem in g_fun.tolist()]

    def test_sedov_functions_table2_g(self):
        self.assertListEqual(self.g_fun, self.g_fun_ref)

    h_fun = [round(elem, 4) for elem in h_fun.tolist()]

    def test_sedov_functions_table2_h(self):
        self.assertListEqual(self.h_fun, self.h_fun_ref)


class TestSedovDoeblingFunctionsTable3(unittest.TestCase):
    r''' Compare results to Kamm & Timmes,
    Table 3. Sedov Functions for gamma=1.4, spherical geometry case'''

    # define vector of lambda values

    lamvec = [0.9913, 0.9773, 0.9622, 0.9342, 0.9080, 0.8747, 0.8359,
              0.7950, 0.7493, 0.6788, 0.5794, 0.4560, 0.3600, 0.2960,
              0.2000, 0.1040]

    # compute sedov function values for each lambda value

    solution = Sedov(gamma=1.4, geometry=3, omega=0.)

    l_fun, v, f_fun, g_fun, h_fun = sedovFcnTable(solution, lamvec)

    # compare results to Kamm & Timmes Table 3

    v_ref = [0.3300, 0.3250, 0.3200, 0.3120, 0.3060, 0.3000, 0.2950,
             0.2915, 0.2890, 0.2870, 0.2860, 0.2857, 0.2857, 0.2857,
             0.2857, 0.2857]

    f_fun_ref = [0.9814, 0.9529, 0.9238, 0.8745, 0.8335, 0.7872, 0.7398,
                 0.6952, 0.6497, 0.5844, 0.4971, 0.3909, 0.3086, 0.2537,
                 0.1714, 0.0891]

    g_fun_ref = [0.8388, 0.6454, 0.4984, 0.3248, 0.2275, 0.1508, 0.0968,
                 0.0620, 0.0379, 0.0174, 0.0052, 0.0009, 0.0001, 0.0000,
                 0.0000, 0.0000]

    h_fun_ref = [0.9116, 0.7992, 0.7082, 0.5929, 0.5238, 0.4674, 0.4273,
                 0.4021, 0.3857, 0.3732, 0.3672, 0.3656, 0.3655, 0.3655,
                 0.3655, 0.3655]

    # Convert computed Sedov functions to list and round to 4 decimal
    # places to agree with precision of reference table from Kamm & Timmes

    l_fun = [round(elem, 4) for elem in l_fun.tolist()]

    def test_sedov_functions_table3_l(self):
        self.assertListEqual(self.l_fun, self.lamvec)

    v = [round(elem, 4) for elem in v.tolist()]

    def test_sedov_functions_table3_v(self):
        self.assertListEqual(self.v, self.v_ref)

    f_fun = [round(elem, 4) for elem in f_fun.tolist()]

    def test_sedov_functions_table3_f(self):
        self.assertListEqual(self.f_fun, self.f_fun_ref)

    g_fun = [round(elem, 4) for elem in g_fun.tolist()]

    def test_sedov_functions_table3_g(self):
        self.assertListEqual(self.g_fun, self.g_fun_ref)

    h_fun = [round(elem, 4) for elem in h_fun.tolist()]

    def test_sedov_functions_table3_h(self):
        self.assertListEqual(self.h_fun, self.h_fun_ref)


class TestSedovDoeblingFunctionsTables45(unittest.TestCase):
    r''' Compare results to Kamm & Timmes, Tables 4 & 5
     Values of key variables for the gamma = 1.4 uniform density
     test cases at t=1s'''

    ##
    # Table 4
    ##

    # TODO: Need to improve accuracy of quadrature for second energy function
    # to account for singularity near lowest value of v. Want 6 places of
    # agreement with Kamm & Timmes Table 4 across all rows and columns.

    # Planar case (Row 1)

    solver_row1 = Sedov(gamma=1.4, geometry=1, omega=0., eblast=6.73185e-02)

    def test_sedov_functions_table4_row1_eval1(self):
        self.assertAlmostEqual(self.solver_row1.eval1, 0.197928, places=6)

    @unittest.expectedFailure
    def test_sedov_functions_table4_row1_eval2_fail(self):
        self.assertAlmostEqual(self.solver_row1.eval2, 0.175834, places=6)

    def test_sedov_functions_table4_row1_eval2(self):
        self.assertAlmostEqual(self.solver_row1.eval2, 0.175834, places=3)
        # Should be 6

    @unittest.expectedFailure
    def test_sedov_functions_table4_row1_alpha_fail(self):
        self.assertAlmostEqual(self.solver_row1.alpha, 0.538548, places=6)

    def test_sedov_functions_table4_row1_alpha(self):
        self.assertAlmostEqual(self.solver_row1.alpha, 0.538548, places=3)
        # Should be 6

    # Cylindrical case (Row 2)

    solver_row2 = Sedov(gamma=1.4, geometry=2, omega=0., eblast=0.311357)

    def test_sedov_functions_table4_row2_eval1(self):
        self.assertAlmostEqual(self.solver_row2.eval1, 6.54053e-02, places=6)

    @unittest.expectedFailure
    def test_sedov_functions_table4_row2_eval2_fail(self):
        self.assertAlmostEqual(self.solver_row2.eval2, 4.95650e-02, places=6)

    def test_sedov_functions_table4_row2_eval2(self):
        self.assertAlmostEqual(self.solver_row2.eval2, 4.95650e-02, places=4)
        # Should be 6

    @unittest.expectedFailure
    def test_sedov_functions_table4_row2_alpha_fail(self):
        self.assertAlmostEqual(self.solver_row2.alpha, 9.84041e-01, places=6)

    def test_sedov_functions_table4_row2_alpha(self):
        self.assertAlmostEqual(self.solver_row2.alpha, 9.84041e-01, places=4)
        # Should be 6

    # Spherical case (Row 3)

    solver_row3 = Sedov(gamma=1.4, geometry=3, omega=0., eblast=0.851072)

    def test_sedov_functions_table4_row3_eval1(self):
        self.assertAlmostEqual(self.solver_row3.eval1, 2.96269e-02, places=6)

    def test_sedov_functions_table4_row3_eval2(self):
        self.assertAlmostEqual(self.solver_row3.eval2, 2.11647e-02, places=6)

    @unittest.expectedFailure
    def test_sedov_functions_table4_row3_alpha_fail(self):
        self.assertAlmostEqual(self.solver_row3.alpha, 8.51060e-01, places=6)

    def test_sedov_functions_table4_row3_alpha(self):
        self.assertAlmostEqual(self.solver_row3.alpha, 8.51060e-01, places=4)
        # Should be 6

    ##
    #  Table 5
    ##

    # Planar case (Row 1)

    solution_row1 = solver_row1([1.0], 1.0)

    def test_sedov_functions_table5_row1_r2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].location, 0.5,
                               places=2)

    def test_sedov_functions_table5_row1_rho2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].density.right, 6.0,
                               places=2)

    @unittest.expectedFailure
    def test_sedov_functions_table5_row1_u2_fail(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].velocity.right,
                               2.77778e-01, places=6)

    def test_sedov_functions_table5_row1_u2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].velocity.right,
                               2.77778e-01, places=4)

    @unittest.expectedFailure
    def test_sedov_functions_table5_row1_e2_fail(self):
        self.assertAlmostEqual(
            self.solution_row1.jumps[0].specific_internal_energy.right,
            3.85802e-02, places=6)

    def test_sedov_functions_table5_row1_e2(self):
        self.assertAlmostEqual(
            self.solution_row1.jumps[0].specific_internal_energy.right,
            3.85802e-02, places=4)

    @unittest.expectedFailure
    def test_sedov_functions_table5_row1_p2_fail(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].pressure.right,
                               9.25926e-02, places=6)

    def test_sedov_functions_table5_row1_p2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].pressure.right,
                               9.25926e-02, places=4)

    # Cylindrical case (Row 2)

    solution_row2 = solver_row2([1.0], 1.0)

    def test_sedov_functions_table5_row2_r2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].location, 0.75,
                               places=3)

    def test_sedov_functions_table5_row2_rho2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].density.right, 6.0,
                               places=2)

    def test_sedov_functions_table5_row2_u2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].velocity.right,
                               3.125e-01, places=4)

    @unittest.expectedFailure
    def test_sedov_functions_table5_row2_e2_fail(self):
        self.assertAlmostEqual(
            self.solution_row2.jumps[0].specific_internal_energy.right,
            4.88281e-02, places=6)

    def test_sedov_functions_table5_row2_e2(self):
        self.assertAlmostEqual(
            self.solution_row2.jumps[0].specific_internal_energy.right,
            4.88281e-02, places=5)

    @unittest.expectedFailure
    def test_sedov_functions_table5_row2_p2_fail(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].pressure.right,
                               1.17188e-01, places=6)

    def test_sedov_functions_table5_row2_p2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].pressure.right,
                               1.17188e-01, places=5)

    # Spherical case (Row 3)

    solution_row3 = solver_row3([1.0], 1.0)

    def test_sedov_functions_table5_row3_r2(self):
        self.assertAlmostEqual(self.solution_row3.jumps[0].location, 1.0,
                               places=2)

    def test_sedov_functions_table5_row3_rho2(self):
        self.assertAlmostEqual(self.solution_row3.jumps[0].density.right, 6.0,
                               places=2)

    @unittest.expectedFailure
    def test_sedov_functions_table5_row3_u2_fail(self):
        self.assertAlmostEqual(self.solution_row3.jumps[0].velocity.right,
                               3.33334e-01, places=6)

    def test_sedov_functions_table5_row3_u2(self):
        self.assertAlmostEqual(self.solution_row3.jumps[0].velocity.right,
                               3.33334e-01, places=5)

    def test_sedov_functions_table5_row3_e2(self):
        self.assertAlmostEqual(
            self.solution_row3.jumps[0].specific_internal_energy.right,
            5.55559e-02, places=6)

    @unittest.expectedFailure
    def test_sedov_functions_table5_row3_p2_fail(self):
        self.assertAlmostEqual(self.solution_row3.jumps[0].pressure.right,
                               1.33334e-1, places=6)

    def test_sedov_functions_table5_row3_p2(self):
        self.assertAlmostEqual(self.solution_row3.jumps[0].pressure.right,
                               1.33334e-1, places=5)


class TestSedovDoeblingFunctionsTable67(unittest.TestCase):
    r''' Compare results to Kamm & Timmes, Tables 6 & 7.
     Values of key variables for the gamma = 1.4 singular test cases at t=1s'''

    ##
    #  Table 6
    ##

    # Cylindrical case (Row 1)

    solver_row1 = Sedov(gamma=1.4, geometry=2, omega=1.66667, eblast=2.45749)

    def test_sedov_functions_table6_row1_alpha(self):
        self.assertAlmostEqual(self.solver_row1.alpha, 4.80856, places=6)

    # Spherical case (Row 2)

    solver_row2 = Sedov(gamma=1.4, geometry=3, omega=2.33333, eblast=4.90875)

    @unittest.expectedFailure
    def test_sedov_functions_table6_row2_alpha_fail(self):
        self.assertAlmostEqual(self.solver_row2.alpha, 4.90875, places=6)

    def test_sedov_functions_table6_row2_alpha(self):
        self.assertAlmostEqual(self.solver_row2.alpha, 4.90875, places=4)
        # Should be 6

    ##
    #  Table 7
    ##

    # Cylindrical case (Row 1)

    solution_row1 = solver_row1([1.0], 1.0)

    def test_sedov_functions_table7_row1_r2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].location, 0.75,
                               places=3)

    def test_sedov_functions_table7_row1_rho2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].density.right,
                               9.69131, places=4)

    @unittest.expectedFailure
    def test_sedov_functions_table7_row1_u2_fail(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].velocity.right,
                               5.35714e-01, places=6)

    def test_sedov_functions_table7_row1_u2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].velocity.right,
                               5.35714e-01, places=5)

    def test_sedov_functions_table7_row1_e2(self):
        self.assertAlmostEqual(
            self.solution_row1.jumps[0].specific_internal_energy.right,
            1.43495e-01, places=6)

    @unittest.expectedFailure
    def test_sedov_functions_table7_row1_p2_fail(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].pressure.right,
                               5.56261e-1, places=6)

    def test_sedov_functions_table7_row1_p2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].pressure.right,
                               5.56261e-1, places=5)

    # Spherical case (Row 2)

    solution_row2 = solver_row2([1.0], 1.0)

    def test_sedov_functions_table7_row2_r2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].location,
                               1.00, places=3)

    def test_sedov_functions_table7_row2_rho2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].density.right,
                               6.00000, places=4)

    def test_sedov_functions_table7_row2_u2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].velocity.right,
                               6.25000e-01, places=6)

    @unittest.expectedFailure
    def test_sedov_functions_table7_row2_e2_fail(self):
        self.assertAlmostEqual(
            self.solution_row2.jumps[0].specific_internal_energy.right,
            1.95313e-01, places=6)

    def test_sedov_functions_table7_row2_e2(self):
        self.assertAlmostEqual(
            self.solution_row2.jumps[0].specific_internal_energy.right,
            1.95313e-01, places=5)

    @unittest.expectedFailure
    def test_sedov_functions_table7_row2_p2_fail(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].pressure.right,
                               4.68750e-1, places=6)

    def test_sedov_functions_table7_row2_p2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].pressure.right,
                               4.68750e-1, places=5)


class TestSedovDoeblingFunctionsTable89(unittest.TestCase):
    r''' Compare results to Kamm & Timmes, Tables 8 & 9.
     Values of key variables for the gamma = 1.4 vacuum test cases at t=1s'''

    ##
    #  Table 8
    ##

    # Cylindrical case (Row 1)

    solver_row1 = Sedov(gamma=1.4, geometry=2, omega=1.7, eblast=2.67315)

    def test_sedov_functions_table8_row1_eval1(self):
        self.assertAlmostEqual(self.solver_row1.eval1, 0.856238, places=6)

    def test_sedov_functions_table8_row1_eval2(self):
        self.assertAlmostEqual(self.solver_row1.eval2, 0.158561, places=6)

    @unittest.expectedFailure
    def test_sedov_functions_table8_row1_alpha_fail(self):
        self.assertAlmostEqual(self.solver_row1.alpha, 5.18062, places=6)

    def test_sedov_functions_table8_row1_alpha(self):
        self.assertAlmostEqual(self.solver_row1.alpha, 5.18062, places=5)
        # Should be 6

    # Spherical case (Row 2)

    solver_row2 = Sedov(gamma=1.4, geometry=3, omega=2.4, eblast=5.45670)

    def test_sedov_functions_table8_row2_eval1(self):
        self.assertAlmostEqual(self.solver_row2.eval1, 0.454265, places=6)

    def test_sedov_functions_table8_row2_eval2(self):
        self.assertAlmostEqual(self.solver_row2.eval2, 8.28391e-02, places=6)

    @unittest.expectedFailure
    def test_sedov_functions_table8_row2_alpha_fail(self):
        self.assertAlmostEqual(self.solver_row2.alpha, 5.45670, places=6)

    def test_sedov_functions_table8_row2_alpha(self):
        self.assertAlmostEqual(self.solver_row2.alpha, 5.45670, places=5)
        # Should be 6

    ##
    #  Table 9
    ##

    # Cylindrical case (Row 1)

    solution_row1 = solver_row1([1.0], 1.0)

    # Note about entry 1,1 of Table 9 in Kamm&Timmes 2007. Value in table is
    # 0.154090. I think this is actually lambda_v and not r_v. Correct r_v
    # value is lambda_v * (r2=0.75) = 0.115568. This is confirmed visually
    # by inspecting the vacuum boundaries in Figure 10. Also, Table 9 should
    # not have the word "uniform" in the title as omega!=0

    def test_sedov_functions_table9_row1_rv(self):
        self.assertAlmostEqual(self.solution_row1.jumps[1].location, 0.115568,
                               places=6)

    def test_sedov_functions_table9_row1_r2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].location, 0.75,
                               places=3)

    @unittest.expectedFailure
    def test_sedov_functions_table9_row1_rho2_fail(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].density.right,
                               9.78469, places=6)

    def test_sedov_functions_table9_row1_rho2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].density.right,
                               9.78469, places=4)  # Should be 6

    @unittest.expectedFailure
    def test_sedov_functions_table9_row1_u2_fail(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].velocity.right,
                               5.43478e-01, places=6)

    def test_sedov_functions_table9_row1_u2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].velocity.right,
                               5.43478e-01, places=5)  # Should be 6

    @unittest.expectedFailure
    def test_sedov_functions_table9_row1_e2_fail(self):
        self.assertAlmostEqual(
            self.solution_row1.jumps[0].specific_internal_energy.right,
            1.47684e-01, places=6)

    def test_sedov_functions_table9_row1_e2(self):
        self.assertAlmostEqual(
            self.solution_row1.jumps[0].specific_internal_energy.right,
            1.47684e-01, places=5)

    def test_sedov_functions_table9_row1_p2(self):
        self.assertAlmostEqual(self.solution_row1.jumps[0].pressure.right,
                               5.78018e-01, places=6)

    # Spherical case (Row 2)

    solution_row2 = solver_row2([1.0], 1.0)

    def test_sedov_functions_table9_row2_rv(self):
        self.assertAlmostEqual(self.solution_row2.jumps[1].location, 0.272644,
                               places=6)

    def test_sedov_functions_table9_row2_r2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].location, 1.00,
                               places=3)

    def test_sedov_functions_table9_row2_rho2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].density.right, 6.0,
                               places=2)

    def test_sedov_functions_table9_row2_u2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].velocity.right,
                               6.41026e-01, places=6)

    def test_sedov_functions_table9_row2_e2(self):
        self.assertAlmostEqual(
            self.solution_row2.jumps[0].specific_internal_energy.right,
            2.05457e-01, places=6)

    def test_sedov_functions_table9_row2_p2(self):
        self.assertAlmostEqual(self.solution_row2.jumps[0].pressure.right,
                               4.93097e-01, places=6)


class TestSedovDoeblingShock(unittest.TestCase):
    """Tests Doebling Sedov for correct pre and post shock values.
    """
    # construct spatial grid and choose time
    rmax = 1.2
    npts = 121
    r = np.linspace(0.0, rmax, npts)
    t = 1.
    solver = Sedov(eblast=0.851072, gamma=1.4, geometry=3.0, omega=0.0)
    solution = solver(r, t)

    # shock location (index closest to r=1.0)
    ishock = (np.abs(r-1.0)).argmin()

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

        for ikey in self.analytic_preshock:
            self.assertAlmostEqual(self.solution[ikey][self.ishock+1],
                                   self.analytic_preshock[ikey], places=5)

    def test_postshock_state(self):
        """Tests density, velocity, pressure, specific internal energy, and
        sound speed immediately after the shock.
        """

        for ikey in self.analytic_postshock:
            self.assertAlmostEqual(self.solution[ikey][self.ishock],
                                   self.analytic_postshock[ikey], places=5)

if __name__ == '__main__':
    unittest.main()
