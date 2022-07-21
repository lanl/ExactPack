#
#  tests the solver implementation for the Escape of HE Products test problem
#  (ehep for short)
#
#

import unittest

import numpy as np

from exactpack.solvers.ehep import EscapeOfHEProducts


class TestEHEPAssignments(unittest.TestCase):
    """Tests :class:`exactpack.solvers.ehep.EscapeOfHEProducts`.

    These tests confirm proper assignment of variables, including default
    values
    """

    def test_defaults(self):

        # here are the defaults
        D = 0.85       # Detonation velocity of HE
        rho_0 = 1.6    # Initial density of HE
        up = 0.05      # Velocity of piston (cm/us)
        xtilde = 1.0   # width of HE region (cm)
        xmax = 10.0    # maximum value of x (cm)
        tmax = 10.0    # maximum value of t (us)

        solution = EscapeOfHEProducts()

        self.assertEqual(solution.D, D)
        self.assertEqual(solution.rho_0, rho_0)
        self.assertEqual(solution.up, up)
        self.assertEqual(solution.xtilde, xtilde)
        self.assertEqual(solution.xmax, xmax)
        self.assertEqual(solution.tmax, tmax)

    def test_assignment(self):
        # tests proper assignment of parameters
        #
        #  These values are made up and not physically meaningful
        #  This is only an arbitrary test case
        #

        D = 2.00       # Detonation velocity of HE
        rho_0 = 1.75   # Initial density of HE
        up = 0.10      # Velocity of piston (cm/us)
        xtilde = 4.0   # width of HE region (cm)
        ttilde = 2.0   # time of shock wave arrival at xtilde (us)
        xmax = 5.5     # maximum value of x (cm)
        tmax = 7.5     # maximum value of t (us)

        solution = EscapeOfHEProducts(D=D, rho_0=rho_0, up=up,
                                      xtilde=xtilde, xmax=xmax, tmax=tmax)

        self.assertEqual(solution.D, D)
        self.assertEqual(solution.rho_0, rho_0)
        self.assertEqual(solution.up, up)
        self.assertEqual(solution.xtilde, xtilde)
        self.assertEqual(solution.ttilde, ttilde)
        self.assertEqual(solution.xmax, xmax)
        self.assertEqual(solution.tmax, tmax)

    #
    # Confirm that illegal parameter values raise an error
    #

    def test_illegal_value_D(self):
        self.assertRaises(ValueError, EscapeOfHEProducts,
                          D=-1.0)

    def test_illegal_value_rho_0(self):
        self.assertRaises(ValueError, EscapeOfHEProducts,
                          rho_0=-1.0)

    def test_illegal_value_up(self):
        self.assertRaises(ValueError, EscapeOfHEProducts,
                          up=-1.0)

    def test_illegal_value_up_2(self):
        self.assertRaises(ValueError, EscapeOfHEProducts,
                          D=1.0, gamma=3.0, up=0.3)

    def test_illegal_value_xtilde(self):
        self.assertRaises(ValueError, EscapeOfHEProducts,
                          xtilde=-1.0)

    def test_illegal_value_xmax(self):
        self.assertRaises(ValueError, EscapeOfHEProducts,
                          xmax=-1.0)

    def test_illegal_value_tmax(self):
        self.assertRaises(ValueError, EscapeOfHEProducts,
                          tmax=-1.0)


class TestEhepSolution(unittest.TestCase):
    """Tests :class:`exactpack.ehep.EscapeOfHEProducts`.

    These tests confirm proper solution values for specific cases
    """

    #  Define a set of corners for a polygon

    def getsoln(self):

        corn = []
        corn.append((5., 5.))
        corn.append((10., 5.))
        corn.append((10., 10.))
        corn.append((5., 10.))

        soln = EscapeOfHEProducts()

        return corn, soln

    #  test whether a set of points are on the boundaries of the polygon

    def test_point_on_boundary_1(self):
        """Test 1 for point_on_boundary function

        """

        corn, soln = self.getsoln()
        point = (7.5, 5.0)  # on
        self.assertTrue(EscapeOfHEProducts.point_on_boundary
                        (soln, corn, point))

    def test_point_on_boundary_2(self):
        """Test 2 for point_on_boundary function

        """

        corn, soln = self.getsoln()
        point = (10.0, 7.5)  # on
        self.assertTrue(EscapeOfHEProducts.point_on_boundary
                        (soln, corn, point))

    def test_point_on_boundary_3(self):
        """Test 3 for point_on_boundary function

        """

        corn, soln = self.getsoln()
        point = (5., 9.99)  # on
        self.assertTrue(EscapeOfHEProducts.point_on_boundary
                        (soln, corn, point))

    def test_point_on_boundary_4(self):
        """Test 4 for point_on_boundary function

        """

        corn, soln = self.getsoln()
        point = (2.5, 2.5)  # off
        self.assertFalse(EscapeOfHEProducts.point_on_boundary
                         (soln, corn, point))

    def test_point_on_boundary_5(self):
        """Test 5 for point_on_boundary function

        """

        corn, soln = self.getsoln()
        point = (4.99, 10.)  # off
        self.assertFalse(EscapeOfHEProducts.point_on_boundary
                         (soln, corn, point))

    def test_point_on_boundary_6(self):
        """Test 6 for point_on_boundary function

        """

        corn, soln = self.getsoln()
        point = (5.0, 10.01)  # off
        self.assertFalse(EscapeOfHEProducts.point_on_boundary
                         (soln, corn, point))

    def test_point_on_boundary_7(self):
        """Test 7 for point_on_boundary function

        """

        corn, soln = self.getsoln()
        point = (5.0, 10.0)  # on
        self.assertTrue(EscapeOfHEProducts.point_on_boundary
                        (soln, corn, point))

    def test_corners1(self):
        """ Tests proper calculation of polygon corners by comparing to
         results from hand calculations

        """

        D = 0.85       # Detonation velocity of HE
        rho_0 = 1.6    # Initial density of HE
        up = 0.05      # Velocity of piston (cm/us)
        xtilde = 1.0   # width of HE region (cm)
        xmax = 3.5     # maximum value of x (cm)
        tmax = 6.5     # maximum value of t (us)

        solution = EscapeOfHEProducts(D=D, rho_0=rho_0, up=up,
                                      xtilde=xtilde, xmax=xmax, tmax=tmax)

        self.assertSequenceEqual(
            solution.corners['00'],
            [(0., 0.), (0., 6.5), (0.325, 6.5)]
            )

    def test_corners2(self):
        """ Tests proper calculation of polygon corners by comparing to
         results from hand calculations

        """

        D = 0.85       # Detonation velocity of HE
        rho_0 = 1.6    # Initial density of HE
        up = 0.05      # Velocity of piston (cm/us)
        xtilde = 1.0   # width of HE region (cm)
        xmax = 3.5     # maximum value of x (cm)
        tmax = 6.5     # maximum value of t (us)

        solution = EscapeOfHEProducts(D=D, rho_0=rho_0, up=up,
                                      xtilde=xtilde, xmax=xmax, tmax=tmax)

        self.assertSequenceEqual(
            solution.corners['II'],
            [(1., 1.1764705882352942),
             (0.8289473684210525, 1.5789473684210527),
             (3.4125, 6.5),
             (3.5, 4.11764705882353)]
            )

    def test_0_regions(self):
        """ Tests proper evaluation of point locations inside
        regions 00, 0V, and 0H, and outside of all regions

        """

        rho_0 = 1.6

        solution = EscapeOfHEProducts(rho_0=rho_0)

        xvec = [0.2, 3.0, 0.75, -0.1]

        tvec = [5.5, 2.0, 0.25, 3.0]

        densoln = [0., 0., rho_0, 0.]

        regsoln = ['00', '0V', '0H', None]

        for i, x in enumerate(xvec):

            result = solution._run([x], tvec[i])

            self.assertEqual(result['sound_speed'], 0.0)
            self.assertEqual(result['pressure'], 0.0)
            self.assertEqual(result['density'], densoln[i])
            self.assertEqual(result['velocity'], 0.0)
            self.assertEqual(result['region'][0], regsoln[i])

    def test_fickett_table6_1(self):
        """ Tests proper solution values by comparing default case
        to the Table 6.1 in Fickett & Rivard

        """

        def compare_to_table(self, data, time):

            dataray = np.array([float(s) for s in data.split()]).\
                reshape((4, 6))

            x = dataray[:, 0]
            pressure = dataray[:, 2]
            velocity = dataray[:, 3]
            density = dataray[:, 4]
            sound_speed = dataray[:, 5]

            solution = EscapeOfHEProducts(xmax=5.0, tmax=7.0)

            for i in range(len(x)):
                result = solution._run([x[i]], time)
                np.testing.assert_array_almost_equal(result['density'],
                                       density[i], 4)
                np.testing.assert_array_almost_equal(result['pressure'],
                                       pressure[i], 4)
                np.testing.assert_array_almost_equal(result['velocity'],
                                       velocity[i], 4)
                np.testing.assert_array_almost_equal(result['sound_speed'],
                                       sound_speed[i], 4)

            return

        # Table 6.1, t=1.25

        #       'position','h','pressure','velocity','density','sound_speed'
        data = '''0.06250   0.      0.11955   0.05      1.58954      0.475
                  0.65625   0.94379 0.11955   0.05      1.58954      0.475
                  0.96875   1.50588 0.24094   0.175     2.00784      0.600
                  1.0625    1.60    0.        0.85      0.           0.
                  '''
        compare_to_table(self, data, time=1.25)

        # Table 6.1, t=2.0

        #       'position','h','pressure','velocity','density','sound_speed'
        data = ''' 0.1     0.      0.11955  0.05      1.58954       0.475
                   0.65    0.87425 0.11955  0.05      1.58954       0.475
                   1.05    1.34753 0.01395  0.29286   0.77684       0.23214
                   1.70    1.60    0.       0.85      0.            0.
                   '''
        compare_to_table(self, data, time=2.00)

        # Table 6.1, t=5.0

        #       'position','h','pressure','velocity','density','sound_speed'
        data = ''' 0.250   0.      0.01664  0.050     0.82373       0.24615
                   1.125    0.72076 0.01664  0.27885   0.82373       0.24615
                   2.625    1.4641  0.00014  0.475     0.16732       0.05
                   4.250    1.60    0.       0.85        0.            0.
                   '''
        compare_to_table(self, data, time=5.00)

    def test_fickett_table6_2(self):
        """ Tests proper solution values by comparing default case
        to the Table 6.2 in Fickett & Rivard

        """

        def compare_to_table(self, data):

            # Put the data in the desired format

            dataray = np.array([float(s) for s in data.split()]).\
                reshape((4, 6))

            t = dataray[:, 0]
            x = dataray[:, 1]
            pressure = dataray[:, 2]
            velocity = dataray[:, 3]
            density = dataray[:, 4]
            sound_speed = dataray[:, 5]

            # Initialize the solution

            solution = EscapeOfHEProducts(xmax=5.0, tmax=7.0)

            # Loop over each row in the data table

            for i in range(len(t)):

                result = solution._run([x[i]], t[i])
                #  Check for correct values of the physical variables
                np.testing.assert_array_almost_equal(result['density'],
                                       density[i], 4)
                np.testing.assert_array_almost_equal(result['pressure'],
                                       pressure[i], 4)
                np.testing.assert_array_almost_equal(result['velocity'],
                                       velocity[i], 4)
                np.testing.assert_array_almost_equal(result['sound_speed'],
                                       sound_speed[i], 4)

            return

        # Table 6.2, h=8

        #          t        x       p       u        rho             cs
        data = '''0.8     0.53464 0.18222  0.12165   1.8293        0.54665
                  1.6     0.58329 0.11955  0.05000   1.5895        0.47500
                  3.2     0.76484 0.03676  0.20439   1.0729        0.32061
                  6.4     1.64680 0.006525  0.30400   0.60296       0.18018
                  '''

        compare_to_table(self, data)

        # Table 6.2, h=13.6

        #          t        x       p       u        rho             cs
        data = '''1.2      0.88669 0.21985  0.15696   1.94746       0.58196
                  1.8      1.00686 0.02299  0.28518   0.91753       0.27418
                  3.0      1.41281 0.00371  0.37569   0.49965       0.14931
                  6.0      2.64671 0.00086  0.43320   0.30721       0.09180
                  '''
        compare_to_table(self, data)

        # Table 6.2, h=15.2

        #          t        x       p       u        rho             cs
        data = '''1.2      0.96657 0.25977  0.19024   2.05883       0.61524
                  1.3      0.99075 0.08176  0.34362   1.40045       0.41849
                  1.5      1.07639 0.01556  0.47686   0.80561       0.24074
                  6.0      3.56625 0.00003  0.56320   0.10432       0.03117
                  '''
        compare_to_table(self, data)

if __name__ == '__main__':
    unittest.main()
