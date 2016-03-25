"""Tests for the Riemann problem.
"""

import unittest

import numpy

from exactpack.solvers.riemann import *

class TestRiemannKamm(unittest.TestCase):
    r"""Test for :class:`exactpack.riemann.kamm.Riemann`.
    """

    def Riemann_test1_Sod(self):
        """Riemann problem 1: Sod"""

        #print "*** test1 ***"
        solver = Sod()
        t = 0.25
        # rarefaction
        r = numpy.array([0.19, 0.5])
        solution = solver(r,t)
        self.assertAlmostEqual(solution.density[0],  1.0)
        self.assertAlmostEqual(solution.pressure[0], 1.0)
        self.assertAlmostEqual(solution.velocity_x[0], 0.0)
        self.assertAlmostEqual(solution.density[1],  0.426319428191)
        self.assertAlmostEqual(solution.pressure[1], 0.303130178063)   
        self.assertAlmostEqual(solution.velocity_x[1], 0.927452620049)
        # contact
        r = numpy.array([0.7, 0.8])
        solution = solver(r,t)
        self.assertAlmostEqual(solution.density[0],  0.42631943)
        self.assertAlmostEqual(solution.pressure[0], 0.30313018)   
        self.assertAlmostEqual(solution.velocity_x[0], 0.92745262)
        self.assertAlmostEqual(solution.density[1],  0.26557371)
        self.assertAlmostEqual(solution.pressure[1], 0.30313018)
        self.assertAlmostEqual(solution.velocity_x[1], 0.92745262)
        # shock
        r = numpy.array([0.85, 1.0])
        solution = solver(r,t)
        # print "***"
        # print r, solution.density
        # print r, solution.pressure
        # print r, solution.velocity_x
        self.assertAlmostEqual(solution.density[0],  0.26557371)
        self.assertAlmostEqual(solution.pressure[0], 0.30313018)
        self.assertAlmostEqual(solution.velocity_x[0], 0.92745262)
        self.assertAlmostEqual(solution.density[1],  0.125)
        self.assertAlmostEqual(solution.pressure[1], 0.1)
        self.assertAlmostEqual(solution.velocity_x[1], 0.0)


    def Riemann_test2_Einfeldt(self):
        """Riemann problem 2: Einfeldt"""

        solver = Einfeldt()
        t = 0.15
        # left rarefaction
        r = numpy.array([0.05, 0.2])
        solution = solver(r,t)
        self.assertAlmostEqual(solution.density[0],  1.0)
        self.assertAlmostEqual(solution.pressure[0], 0.4)
        self.assertAlmostEqual(solution.velocity_x[0],-2.0)
        self.assertAlmostEqual(solution.density[1],  0.40187757)
        self.assertAlmostEqual(solution.pressure[1], 0.11163266)
        self.assertAlmostEqual(solution.velocity_x[1],-1.37639044)
        # center
        r = numpy.array([0.5])
        solution = solver(r,t)
        self.assertAlmostEqual(solution.density[0],  0.02185212)
        self.assertAlmostEqual(solution.pressure[0], 0.00189387)
        self.assertAlmostEqual(solution.velocity_x[0], 0.0)
        # right rarefaction
        r = numpy.array([0.8, 0.95])
        solution = solver(r,t)
        # print r, solution.density
        # print r, solution.pressure
        # print r, solution.velocity_x
        self.assertAlmostEqual(solution.density[0],  0.40187757)
        self.assertAlmostEqual(solution.pressure[0], 0.11163266)
        self.assertAlmostEqual(solution.velocity_x[0], 1.37639044)
        self.assertAlmostEqual(solution.density[1],  1.0)
        self.assertAlmostEqual(solution.pressure[1], 0.4)
        self.assertAlmostEqual(solution.velocity_x[1], 2.0)


    def Riemann_test3_StationaryContact(self):
        """Riemann problem 3: Stationary Contact"""

        solver = StationaryContact()
        t = 0.012
        # rarefaction
        r = numpy.array([0.0, 0.2])
        solution = solver(r,t)
        print "*** StationaryContact Not Working ***"
        print r, solution.density
        print r, solution.pressure
        print r, solution.velocity_x
        print "***", solution.density[0]
        print "***", solution.pressure[0]
        print "***", solution.velocity_x[0]        
        print "***", solution.density[1]
        print "***", solution.pressure[1]
        print "***", solution.velocity_x[1]        
        # self.assertAlmostEqual(solution.density[0],  1.0)
        # self.assertAlmostEqual(solution.pressure[0], 0.4)
        # self.assertAlmostEqual(solution.velocity_x[0],-2.0)
        # self.assertAlmostEqual(solution.density[1],  0.40187757)
        # self.assertAlmostEqual(solution.pressure[1], 0.11163266)
        # self.assertAlmostEqual(solution.velocity_x[1],-1.37639044)


    def Riemann_test4_SlowShock(self):
        """Riemann problem 4: Slow Shock"""

        solver = SlowShock()
        t = 1.0
        # rarefaction
        r = numpy.array([0.55, 0.65])
        solution = solver(r,t)
        self.assertAlmostEqual(solution.density[0],  3.85714275)
        self.assertAlmostEqual(solution.pressure[0], 10.33333251)
        self.assertAlmostEqual(solution.velocity_x[0], -0.81063134)
        self.assertAlmostEqual(solution.density[1],  1.00)
        self.assertAlmostEqual(solution.pressure[1], 1.00)
        self.assertAlmostEqual(solution.velocity_x[1],-3.44)


    def Riemann_test5_ShockContactShock(self):
        """Riemann problem 5: Shock-Contact-Shock"""

        solver = ShockContactShock()
        t = 0.3
        # shock
        r = numpy.array([0.1, 0.2])
        solution = solver(r,t)
        self.assertAlmostEqual(solution.density[0],  1.0)
        self.assertAlmostEqual(solution.pressure[0], 1.0)
        self.assertAlmostEqual(solution.velocity_x[0], 0.5)
        self.assertAlmostEqual(solution.density[1],  1.52071667) 
        self.assertAlmostEqual(solution.pressure[1], 1.81374997)
        self.assertAlmostEqual(solution.velocity_x[1],-0.02786405)
        # contact
        r = numpy.array([0.4, 0.6])
        solution = solver(r,t)
        self.assertAlmostEqual(solution.density[0],  1.52071667)
        self.assertAlmostEqual(solution.pressure[0], 1.81374997)
        self.assertAlmostEqual(solution.velocity_x[0],-0.02786405)
        self.assertAlmostEqual(solution.density[1],  1.90089584) 
        self.assertAlmostEqual(solution.pressure[1], 1.81374997)
        self.assertAlmostEqual(solution.velocity_x[1],-0.02786405)
        # contact
        r = numpy.array([0.6, 0.8])
        solution = solver(r,t)
        # print r, solution.density
        # print r, solution.pressure
        # print r, solution.velocity_x
        self.assertAlmostEqual(solution.density[0],  1.90089584) 
        self.assertAlmostEqual(solution.pressure[0], 1.81374997)
        self.assertAlmostEqual(solution.velocity_x[0],-0.02786405)
        self.assertAlmostEqual(solution.density[1],  1.25)
        self.assertAlmostEqual(solution.pressure[1], 1.00)
        self.assertAlmostEqual(solution.velocity_x[1],-0.50)


    def Riemann_test6_LeBlanc(self):
        """Riemann problem 6: Le Blanc"""

        solver = LeBlanc()
        t = 0.5
        # rarefaction
        r = numpy.array([0.1, 0.2])
        solution = solver(r,t)
        self.assertAlmostEqual(solution.density[0],  1.0) 
        self.assertAlmostEqual(solution.pressure[0], 0.0666667)
        self.assertAlmostEqual(solution.velocity_x[0], 0.0)
        self.assertAlmostEqual(solution.density[1],  0.72899991)
        self.assertAlmostEqual(solution.pressure[1], 0.03936601)
        self.assertAlmostEqual(solution.velocity_x[1], 0.10000006)
        # rarefaction
        r = numpy.array([0.4, 0.5])
        solution = solver(r,t)
        self.assertAlmostEqual(solution.density[0],  0.21600004)
        self.assertAlmostEqual(solution.pressure[0], 0.005184)
        self.assertAlmostEqual(solution.velocity_x[0], 0.40000006)
        self.assertAlmostEqual(solution.density[1],  0.15228709)
        self.assertAlmostEqual(solution.pressure[1], 0.00289521)
        self.assertAlmostEqual(solution.velocity_x[1], 0.465984)
        # contact
        r = numpy.array([0.5, 0.55])
        solution = solver(r,t)
        self.assertAlmostEqual(solution.density[0],  0.15228709)
        self.assertAlmostEqual(solution.pressure[0], 0.00289521)
        self.assertAlmostEqual(solution.velocity_x[0], 0.465984)
        self.assertAlmostEqual(solution.density[1],  0.04)
        self.assertAlmostEqual(solution.pressure[1], 0.00289521)
        self.assertAlmostEqual(solution.velocity_x[1], 0.465984)
        # contact
        r = numpy.array([0.55, 0.65])
        solution = solver(r,t)
        # print r, solution.density
        # print r, solution.pressure
        # print r, solution.velocity_x
        self.assertAlmostEqual(solution.density[0],  0.04)
        self.assertAlmostEqual(solution.pressure[0], 0.00289521)
        self.assertAlmostEqual(solution.velocity_x[0], 0.465984)
        self.assertAlmostEqual(solution.density[1],  0.01)
        self.assertAlmostEqual(solution.pressure[1], 6.66667000e-11)
        self.assertAlmostEqual(solution.velocity_x[1], 0.0)


