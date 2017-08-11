"""Tests for the Cog1 problem.
"""

import unittest

import numpy as np

from exactpack.solvers.cog.cog1 import Cog1
from exactpack.solvers.cog.cog2 import Cog2
from exactpack.solvers.cog.cog3 import Cog3
from exactpack.solvers.cog.cog4 import Cog4
from exactpack.solvers.cog.cog5 import Cog5
from exactpack.solvers.cog.cog6 import Cog6
from exactpack.solvers.cog.cog7 import Cog7
from exactpack.solvers.cog.cog8 import Cog8
from exactpack.solvers.cog.cog9 import Cog9
from exactpack.solvers.cog.cog10 import Cog10
from exactpack.solvers.cog.cog11 import Cog11
from exactpack.solvers.cog.cog12 import Cog12
from exactpack.solvers.cog.cog13 import Cog13
from exactpack.solvers.cog.cog14 import Cog14
from exactpack.solvers.cog.cog16 import Cog16
from exactpack.solvers.cog.cog17 import Cog17
from exactpack.solvers.cog.cog18 import Cog18
from exactpack.solvers.cog.cog19 import Cog19
from exactpack.solvers.cog.cog20 import Cog20
from exactpack.solvers.cog.cog21 import Cog21


# cog1 ##########################
class TestCog1(unittest.TestCase):
    """Test for Coggeshall problems: cog1

    """

    def test_cog1(self):
        """cog1 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog1(geometry=3, gamma=1.4, rho0=1.8, temp0=1.4, b=1.2, Gamma=40.)

        gold_density = np.array([
            1.80000000e+03, 5.32869105e+00, 5.07996520e-01, 1.13572322e-01,
            9.50045696e+03, 2.81250000e+01, 2.68122171e+00, 5.99438309e-01,
            1.85947418e+04, 5.50475747e+01, 5.24781341e+00, 1.17324889e+00,
            2.85280775e+04, 8.44540618e+01, 8.05120226e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            1.00000000e+00, 2.50000000e-01, 1.42857143e-01, 1.00000000e-01,
            4.00000000e+00, 1.00000000e+00, 5.71428571e-01, 4.00000000e-01,
            7.00000000e+00, 1.75000000e+00, 1.00000000e+00, 7.00000000e-01,
            1.00000000e+01, 2.50000000e+00, 1.42857143e+00, 1.00000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            2.21885047e+01, 2.21885047e+01, 2.21885047e+01, 2.21885047e+01,
            4.20393552e+00, 4.20393552e+00, 4.20393552e+00, 4.20393552e+00,
            2.14788185e+00, 2.14788185e+00, 2.14788185e+00, 2.14788185e+00,
            1.40000000e+00, 1.40000000e+00, 1.40000000e+00, 1.40000000e+00
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02,
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02,
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02,
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            2.21885047e+03, 2.21885047e+03, 2.21885047e+03, 2.21885047e+03,
            4.20393552e+02, 4.20393552e+02, 4.20393552e+02, 4.20393552e+02,
            2.14788185e+02, 2.14788185e+02, 2.14788185e+02, 2.14788185e+02,
            1.40000000e+02, 1.40000000e+02, 1.40000000e+02, 1.40000000e+02
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=4)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=7)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri]/10**6,  gold_pressure[ri, ti]/10**6, places=7)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=5)

    def test_geometry_error_cog1(self):
        """cog1 problem:"""
        self.assertRaises(ValueError, Cog1, geometry=-1)


# cog2 ##########################
class TestCog2(unittest.TestCase):
    """Test for Coggeshall problems: cog2

     """

    def test_cog2(self):
        """cog2 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog2(geometry=3, gamma=1.4, rho0=1.8, b=1.2, Gamma=40.)

        gold_density = np.array([
            4.78930511e+01, 1.25853422e+00, 2.89661083e-01, 1.13572322e-01,
            2.52781039e+02, 6.64258342e+00, 1.52884036e+00, 5.99438309e-01,
            4.94754956e+02, 1.30011771e+01, 2.99231837e+00, 1.17324889e+00,
            7.59053706e+02, 1.99464231e+01, 4.59081879e+00, 1.80000000e+00
         ]).reshape(npts, npts)

        gold_velocity = np.array([
            6.25000000e-01, 1.56250000e-01, 8.92857143e-02, 6.25000000e-02,
            2.50000000e+00, 6.25000000e-01, 3.57142857e-01, 2.50000000e-01,
            4.37500000e+00, 1.09375000e+00, 6.25000000e-01, 4.37500000e-01,
            6.25000000e+00, 1.56250000e+00, 8.92857143e-01, 6.25000000e-01
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            1.83105469e-03, 1.14440918e-04, 3.73684630e-05, 1.83105469e-05,
            2.92968750e-02, 1.83105469e-03, 5.97895408e-04, 2.92968750e-04,
            8.97216797e-02, 5.60760498e-03, 1.83105469e-03, 8.97216797e-04,
            1.83105469e-01, 1.14440918e-02, 3.73684630e-03, 1.83105469e-03
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            3.50779183e+00, 5.76111245e-03, 4.32967579e-04, 8.31828530e-05,
            2.96227780e+02, 4.86517341e-01, 3.65634653e-02, 7.02466769e-03,
            1.77560983e+03, 2.91621863e+00, 2.19163943e-01, 4.21063445e-02,
            5.55947539e+03, 9.13074790e+00, 6.86207368e-01, 1.31835938e-01
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            1.83105469e-01, 1.14440918e-02, 3.73684630e-03, 1.83105469e-03,
            2.92968750e+00, 1.83105469e-01, 5.97895408e-02, 2.92968750e-02,
            8.97216797e+00, 5.60760498e-01, 1.83105469e-01, 8.97216797e-02,
            1.83105469e+01, 1.14440918e+00, 3.73684630e-01, 1.83105469e-01
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=6)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=7)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri]/10**3,  gold_pressure[ri, ti]/10**3, places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=7)

    def test_geometry_error_cog2(self):
        """cog2 problem:"""
        self.assertRaises(ValueError, Cog2, geometry=-1)


# cog3 ##########################
class TestCog3(unittest.TestCase):
    """Test for Coggeshall problems: cog3

    """

    def test_cog3(self):
        """cog3 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog3(geometry=3, rho0=1.8, b=1.2, v=0.5, Gamma=40.)

        gold_density = np.array([
            6.41782459e+02, 9.19885676e+02, 1.31849920e+03, 1.88984368e+03,
            2.00557018e+01, 2.87464274e+01, 4.12030999e+01, 5.90576151e+01,
            4.95042794e+00, 7.09559398e+00, 1.01703236e+01, 1.45774239e+01,
            2.02949433e+00, 2.90893392e+00, 4.16946056e+00, 5.97621046e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([

            -2.40000000e-01, -2.40000000e-01, -2.40000000e-01, -2.40000000e-01,
            -9.60000000e-01, -9.60000000e-01, -9.60000000e-01, -9.60000000e-01,
            -1.68000000e+00, -1.68000000e+00, -1.68000000e+00, -1.68000000e+00,
            -2.40000000e+00, -2.40000000e+00, -2.40000000e+00, -2.40000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            2.88000000e-03, 2.88000000e-03, 2.88000000e-03, 2.88000000e-03,
            4.60800000e-02, 4.60800000e-02, 4.60800000e-02, 4.60800000e-02,
            1.41120000e-01, 1.41120000e-01, 1.41120000e-01, 1.41120000e-01,
            2.88000000e-01, 2.88000000e-01, 2.88000000e-01, 2.88000000e-01
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            7.39333393e+01, 1.05970830e+02, 1.51891108e+02, 2.17709992e+02,
            3.69666696e+01, 5.29854150e+01, 7.59455538e+01, 1.08854996e+02,
            2.79441756e+01, 4.00532089e+01, 5.74094424e+01, 8.22866425e+01,
            2.33797747e+01, 3.35109188e+01, 4.80321856e+01, 6.88459445e+01
        ]).reshape(npts, npts)

        # unphysical sign since gamma < 1
        gold_specific_internal_energy = np.array([
            -1.72800000e-01, -1.72800000e-01, -1.72800000e-01, -1.72800000e-01,
            -2.76480000e+00, -2.76480000e+00, -2.76480000e+00, -2.76480000e+00,
            -8.46720000e+00, -8.46720000e+00, -8.46720000e+00, -8.46720000e+00,
            -1.72800000e+01, -1.72800000e+01, -1.72800000e+01, -1.72800000e+01
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=7)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=7)

    def test_geometry_error_cog3(self):
        """cog3 problem:"""
        self.assertRaises(ValueError, Cog3, geometry=-1)


# cog4 ##########################
class TestCog4(unittest.TestCase):
    """Test for Coggeshall problems: cog4

    """

    def test_cog4(self):
        """cog4 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog4(geometry=3, gamma=1.4, rho0=1.8, u0=2.3, Gamma=40.)
        # T > 0 only when gamma < 1

        gold_density = np.array([
            8.35485990e+01, 8.35485990e+01, 8.35485990e+01, 8.35485990e+01,
            8.28907087e+00, 8.28907087e+00, 8.28907087e+00, 8.28907087e+00,
            3.26168817e+00, 3.26168817e+00, 3.26168817e+00, 3.26168817e+00,
            1.80000000e+00, 1.80000000e+00, 1.80000000e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            4.95519979e+00, 4.95519979e+00, 4.95519979e+00, 4.95519979e+00,
            3.12158026e+00, 3.12158026e+00, 3.12158026e+00, 3.12158026e+00,
            2.59037013e+00, 2.59037013e+00, 2.59037013e+00, 2.59037013e+00,
            2.30000000e+00, 2.30000000e+00, 2.30000000e+00, 2.30000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            -8.76928747e-02, -8.76928747e-02, -8.76928747e-02, -8.76928747e-02,
            -3.48009404e-02, -3.48009404e-02, -3.48009404e-02, -3.48009404e-02,
            -2.39643478e-02, -2.39643478e-02, -2.39643478e-02, -2.39643478e-02,
            -1.88928571e-02, -1.88928571e-02, -1.88928571e-02, -1.88928571e-02
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            -2.93064673e+02, -2.93064673e+02, -2.93064673e+02, -2.93064673e+02,
            -1.15386985e+01, -1.15386985e+01, -1.15386985e+01, -1.15386985e+01,
            -3.12656919e+00, -3.12656919e+00, -3.12656919e+00, -3.12656919e+00,
            -1.36028571e+00, -1.36028571e+00, -1.36028571e+00, -1.36028571e+00
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            -8.76928747e+00, -8.76928747e+00, -8.76928747e+00, -8.76928747e+00,
            -3.48009404e+00, -3.48009404e+00, -3.48009404e+00, -3.48009404e+00,
            -2.39643478e+00, -2.39643478e+00, -2.39643478e+00, -2.39643478e+00,
            -1.88928571e+00, -1.88928571e+00, -1.88928571e+00, -1.88928571e+00
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=7)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=7)

    def test_geometry_error_cog4(self):
        """cog4 problem:"""
        self.assertRaises(ValueError, Cog4, geometry=-1)


# cog5 ##########################
class TestCog5(unittest.TestCase):
    """Test for Coggeshall problems: cog5

    """

    def test_cog5(self):
        """cog5 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog5(rho0=1.8, u0=2.3, Gamma=40.)

        gold_density = np.array([
            1.80000000e+02, 1.80000000e+02, 1.80000000e+02, 1.80000000e+02,
            1.12500000e+01, 1.12500000e+01, 1.12500000e+01, 1.12500000e+01,
            3.67346939e+00, 3.67346939e+00, 3.67346939e+00, 3.67346939e+00,
            1.80000000e+00, 1.80000000e+00, 1.80000000e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            2.30000000e-01, 9.20000000e-01, 1.61000000e+00, 2.30000000e+00,
            2.30000000e-01, 9.20000000e-01, 1.61000000e+00, 2.30000000e+00,
            2.30000000e-01, 9.20000000e-01, 1.61000000e+00, 2.30000000e+00,
            2.30000000e-01, 9.20000000e-01, 1.61000000e+00, 2.30000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            5.75000000e-03, 5.75000000e-03, 5.75000000e-03, 5.75000000e-03,
            2.30000000e-02, 2.30000000e-02, 2.30000000e-02, 2.30000000e-02,
            4.02500000e-02, 4.02500000e-02, 4.02500000e-02, 4.02500000e-02,
            5.75000000e-02, 5.75000000e-02, 5.75000000e-02, 5.75000000e-02
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            4.14000000e+01, 4.14000000e+01, 4.14000000e+01, 4.14000000e+01,
            1.03500000e+01, 1.03500000e+01, 1.03500000e+01, 1.03500000e+01,
            5.91428571e+00, 5.91428571e+00, 5.91428571e+00, 5.91428571e+00,
            4.14000000e+00, 4.14000000e+00, 4.14000000e+00, 4.14000000e+00
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            -4.60000000e-01, -4.60000000e-01, -4.60000000e-01, -4.60000000e-01,
            -1.84000000e+00, -1.84000000e+00, -1.84000000e+00, -1.84000000e+00,
            -3.22000000e+00, -3.22000000e+00, -3.22000000e+00, -3.22000000e+00,
            -4.60000000e+00, -4.60000000e+00, -4.60000000e+00, -4.60000000e+00
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=7)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=7)

    def test_geometry_error_cog5(self):
        """cog5 problem:"""
        self.assertRaises(ValueError, Cog5, geometry=1)


# cog6 ##########################
class TestCog6(unittest.TestCase):
    """Test for Coggeshall problems: cog6

    """

    def test_cog6(self):
        """cog6 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog6(geometry=3, rho0=1.8, tau=1.25, b=1.2, Gamma=40.)

        gold_density = np.array([
            4.50926882e-02, 5.58182836e-02, 9.80478722e-02, 3.80202702e-01,
            2.38000635e-01, 2.94610667e-01, 5.17499772e-01, 2.00672189e+00,
            4.65826053e-01, 5.76625875e-01, 1.01287493e+00, 3.92765060e+00,
            7.14670946e-01, 8.84660178e-01, 1.55395405e+00, 6.02580674e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            -6.44122383e-03, -2.85204991e-02, -6.52680653e-02, -1.77777778e-01,
            -2.57648953e-02, -1.14081996e-01, -2.61072261e-01, -7.11111111e-01,
            -4.50885668e-02, -1.99643494e-01, -4.56876457e-01, -1.24444444e+00,
            -6.44122383e-02, -2.85204991e-01, -6.52680653e-01, -1.77777778e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            5.06461969e-05, 6.20589347e-05, 1.06124451e-04, 3.85802469e-04,
            8.10339150e-04, 9.92942956e-04, 1.69799121e-03, 6.17283951e-03,
            2.48166365e-03, 3.04088780e-03, 5.20009808e-03, 1.89043210e-02,
            5.06461969e-03, 6.20589347e-03, 1.06124451e-02, 3.85802469e-02
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            9.13509265e-05, 1.38560929e-04, 4.16211063e-04, 5.86732565e-03,
            7.71444929e-03, 1.17012635e-02, 3.51484025e-02, 4.95486887e-01,
            4.62409433e-02, 7.01381835e-02, 2.10681959e-01, 2.96998271e+00,
            1.44781462e-01, 2.19604273e-01, 6.59650080e-01, 9.29908448e+00,
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            3.03877181e-03, 3.72353608e-03, 6.36746703e-03, 2.31481481e-02,
            4.86203490e-02, 5.95765773e-02, 1.01879473e-01, 3.70370370e-01,
            1.48899819e-01, 1.82453268e-01, 3.12005885e-01, 1.13425926e+00,
            3.03877181e-01, 3.72353608e-01, 6.36746703e-01, 2.31481481e+00
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=7)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=7)

    def test_geometry_error_cog6(self):
        """cog6 problem:"""
        self.assertRaises(ValueError, Cog6, geometry=-1)


# cog7 ##########################
class TestCog7(unittest.TestCase):
    """Test for Coggeshall problems: cog7

    """

    def test_cog7(self):
        """cog7 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog7(geometry=3, tau=1.25, b=1.2, R0=2.0, Ri=0.1, Gamma=66.666666666667)

        gold_density = np.array([
            7.55343848e-06, 6.14785252e-04, 6.19266209e-03, 8.06444195e-02,
            1.15032360e-01, 1.45134848e-01, 2.66362483e-01, 1.11453425e+00,
            2.61140642e-01, 3.26003898e-01, 5.84035492e-01, 2.34525754e+00,
            4.20612885e-01, 5.23372531e-01, 9.30558350e-01, 3.68752897e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            -6.44122383e-03, -2.85204991e-02, -6.52680653e-02, -1.77777778e-01,
            -2.57648953e-02, -1.14081996e-01, -2.61072261e-01, -7.11111111e-01,
            -4.50885668e-02, -1.99643494e-01, -4.56876457e-01, -1.24444444e+00,
            -6.44122383e-02, -2.85204991e-01, -6.52680653e-01, -1.77777778e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            1.24611898e-07, 2.48745892e-06, 1.36278879e-05, 1.11102725e-04,
            4.04093368e-04, 5.01487465e-04, 8.83007362e-04, 3.37709102e-03,
            1.36614566e-03, 1.68347421e-03, 2.91689485e-03, 1.08539172e-02,
            2.87994877e-03, 3.54117650e-03, 6.10481778e-03, 2.25163915e-02
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            6.27498872e-11, 1.01950204e-07, 5.62619366e-06, 5.97320983e-04,
            3.09892091e-03, 4.85222048e-03, 1.56800022e-02, 2.50925574e-01,
            2.37837437e-02, 3.65879436e-02, 1.13571341e-01, 1.69701542e+00,
            8.07562375e-02, 1.23556967e-01, 3.78725944e-01, 5.53532305e+00
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            1.24611898e-05, 2.48745892e-04, 1.36278879e-03, 1.11102725e-02,
            4.04093368e-02, 5.01487465e-02, 8.83007362e-02, 3.37709102e-01,
            1.36614566e-01, 1.68347421e-01, 2.91689485e-01, 1.08539172e+00,
            2.87994877e-01, 3.54117650e-01, 6.10481778e-01, 2.25163915e+00
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=7)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=7)

    def test_geometry_error_cog7(self):
        """cog7 problem:"""
        self.assertRaises(ValueError, Cog7, geometry=-1)


# cog8 ##########################
class TestCog8(unittest.TestCase):
    """Test for Coggeshall problems: cog8

    """

    def test_cog8(self):
        """cog8 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog8(geometry=3, rho0=1.8, temp0=1.4, alpha=2.0, beta=1.0,
                   gamma=1.4, Gamma=40.)

        gold_density = np.array([
            1.80000000e+03, 1.77176398e+01, 2.74333623e+00, 8.35485990e-01,
            2.85732189e+03, 2.81250000e+01, 4.35477481e+00, 1.32625134e+00,
            3.44327613e+03, 3.38926256e+01, 5.24781341e+00, 1.59822720e+00,
            3.87798244e+03, 3.81714977e+01, 5.91033873e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            1.00000000e+00, 2.50000000e-01, 1.42857143e-01, 1.00000000e-01,
            4.00000000e+00, 1.00000000e+00, 5.71428571e-01, 4.00000000e-01,
            7.00000000e+00, 1.75000000e+00, 1.00000000e+00, 7.00000000e-01,
            1.00000000e+01, 2.50000000e+00, 1.42857143e+00, 1.00000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            2.21885047e+01, 6.67333167e+00, 4.10875016e+00, 3.01620857e+00,
            1.39778821e+01, 4.20393552e+00, 2.58835041e+00, 1.90009233e+00,
            1.15992174e+01, 3.48853724e+00, 2.14788185e+00, 1.57674703e+00,
            1.02989916e+01, 3.09748617e+00, 1.90711289e+00, 1.40000000e+00
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02,
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02,
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02,
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            2.21885047e+03, 6.67333167e+02, 4.10875016e+02, 3.01620857e+02,
            1.39778821e+03, 4.20393552e+02, 2.58835041e+02, 1.90009233e+02,
            1.15992174e+03, 3.48853724e+02, 2.14788185e+02, 1.57674703e+02,
            1.02989916e+03, 3.09748617e+02, 1.90711289e+02, 1.40000000e+02
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=7)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri]/10**5,  gold_pressure[ri, ti]/10**5, places=6)
            # specific_internal_energy
            scaled = 10**5
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri]/scaled,  gold_specific_internal_energy[ri, ti]/scaled, places=6)

    def test_geometry_error_cog8(self):
        """cog8 problem:"""
        self.assertRaises(ValueError, Cog8, geometry=-1)


# cog9 ##########################
class TestCog9(unittest.TestCase):
    """Test for Coggeshall problems: cog9

    """

    def test_cog9(self):
        """cog9 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog9(geometry=3, alpha=2.0, beta=1.0, rho0=1.8, gamma=1.4, Gamma=40.)

        gold_density = np.array([
            1.55873578e+04, 1.35985074e+05, 3.26015458e+05, 5.69209979e+05,
            7.61101456e+00, 6.63989619e+01, 1.59187235e+02, 2.77934560e+02,
            3.50536531e-01, 3.05810239e+00, 7.33160355e+00, 1.28006872e+01,
            4.92915534e-02, 4.30022561e-01, 1.03095140e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            6.25000000e-01, 1.56250000e-01, 8.92857143e-02, 6.25000000e-02,
            2.50000000e+00, 6.25000000e-01, 3.57142857e-01, 2.50000000e-01,
            4.37500000e+00, 1.09375000e+00, 6.25000000e-01, 4.37500000e-01,
            6.25000000e+00, 1.56250000e+00, 8.92857143e-01, 6.25000000e-01
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            -1.67410714e-03, -1.04631696e-04, -3.41654519e-05, -1.67410714e-05,
            -2.67857143e-02, -1.67410714e-03, -5.46647230e-04, -2.67857143e-04,
            -8.20312500e-02, -5.12695312e-03, -1.67410714e-03, -8.20312500e-04,
            -1.67410714e-01, -1.04631696e-02, -3.41654519e-03, -1.67410714e-03
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            -1.04379628e+03, -5.69133959e+02, -4.45538618e+02, -3.81167397e+02,
            -8.15465846e+00, -4.44635905e+00, -3.48077045e+00, -2.97787029e+00,
            -1.15019799e+00, -6.27149903e-01, -4.90955595e-01, -4.20022549e-01,
            -3.30077367e-01, -1.79975960e-01, -1.40891682e-01, -1.20535714e-01
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            -1.67410714e-01, -1.04631696e-02, -3.41654519e-03, -1.67410714e-03,
            -2.67857143e+00, -1.67410714e-01, -5.46647230e-02, -2.67857143e-02,
            -8.20312500e+00, -5.12695312e-01, -1.67410714e-01, -8.20312500e-02,
            -1.67410714e+01, -1.04631696e+00, -3.41654519e-01, -1.67410714e-01
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            scale = 10**6
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri]/scale,  gold_density[ri, ti]/scale, places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=7)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri]/10**5,  gold_pressure[ri, ti]/10**5, places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)

    def test_geometry_error_cog9(self):
        """cog9 problem:"""
        self.assertRaises(ValueError, Cog9, geometry=-1)


# cog10 ##########################
class TestCog10(unittest.TestCase):
    """Test for Coggeshall problems: cog10

    """

    def test_cog10(self):
        """cog10 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog10(geometry=3, gamma=1.4, temp0=1.4, beta=1.0, rho0=1.8, lambda0=0.1, Gamma=40.)

        gold_density = np.array([
            1.80000000e+02, 1.80000000e+02, 1.80000000e+02, 1.80000000e+02,
            1.12500000e+01, 1.12500000e+01, 1.12500000e+01, 1.12500000e+01,
            3.67346939e+00, 3.67346939e+00, 3.67346939e+00, 3.67346939e+00,
            1.80000000e+00, 1.80000000e+00, 1.80000000e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            2.35422226e+11, 2.35422226e+11, 2.35422226e+11, 2.35422226e+11,
            2.35422226e+11, 2.35422226e+11, 2.35422226e+11, 2.35422226e+11,
            2.35422226e+11, 2.35422226e+11, 2.35422226e+11, 2.35422226e+11,
            2.35422226e+11, 2.35422226e+11, 2.35422226e+11, 2.35422226e+11
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            1.40000000e-02, 1.40000000e-02, 1.40000000e-02, 1.40000000e-02,
            2.24000000e-01, 2.24000000e-01, 2.24000000e-01, 2.24000000e-01,
            6.86000000e-01, 6.86000000e-01, 6.86000000e-01, 6.86000000e-01,
            1.40000000e+00, 1.40000000e+00, 1.40000000e+00, 1.40000000e+00
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            1.00800000e+02, 1.00800000e+02, 1.00800000e+02, 1.00800000e+02,
            1.00800000e+02, 1.00800000e+02, 1.00800000e+02, 1.00800000e+02,
            1.00800000e+02, 1.00800000e+02, 1.00800000e+02, 1.00800000e+02,
            1.00800000e+02, 1.00800000e+02, 1.00800000e+02, 1.00800000e+02
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            1.40000000e+00, 1.40000000e+00, 1.40000000e+00, 1.40000000e+00,
            2.24000000e+01, 2.24000000e+01, 2.24000000e+01, 2.24000000e+01,
            6.86000000e+01, 6.86000000e+01, 6.86000000e+01, 6.86000000e+01,
            1.40000000e+02, 1.40000000e+02, 1.40000000e+02, 1.40000000e+02
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri]/10**11,  gold_velocity[ri, ti]/10**11, places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=7)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)

    def test_geometry_error_cog10(self):
        """cog10 problem:"""
        self.assertRaises(ValueError, Cog10, geometry=-1)


# cog11 ##########################
class TestCog11(unittest.TestCase):
    """Test for Coggeshall problems: cog11

    """

    def test_cog11(self):
        """cog11 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog11(geometry=3, gamma=1.4, rho0=1.8, temp0=1.4, beta=1.0, Gamma=40.)

        gold_density = np.array([
            1.80000000e+03, 8.52590569e+01, 2.48918295e+01, 1.13572322e+01,
            5.93778560e+02, 2.81250000e+01, 8.21124148e+00, 3.74648943e+00,
            3.79484527e+02, 1.79747183e+01, 5.24781341e+00, 2.39438550e+00,
            2.85280775e+02, 1.35126499e+01, 3.94508911e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            1.00000000e+00, 2.50000000e-01, 1.42857143e-01, 1.00000000e-01,
            4.00000000e+00, 1.00000000e+00, 5.71428571e-01, 4.00000000e-01,
            7.00000000e+00, 1.75000000e+00, 1.00000000e+00, 7.00000000e-01,
            1.00000000e+01, 2.50000000e+00, 1.42857143e+00, 1.00000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            2.21885047e+01, 1.38678154e+00, 4.52826626e-01, 2.21885047e-01,
            6.72629683e+01, 4.20393552e+00, 1.37271364e+00, 6.72629683e-01,
            1.05246211e+02, 6.57788816e+00, 2.14788185e+00, 1.05246211e+00,
            1.40000000e+02, 8.75000000e+00, 2.85714286e+00, 1.40000000e+00
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02,
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02,
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02,
            1.59757234e+06, 4.72942746e+03, 4.50867327e+02, 1.00800000e+02
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            2.21885047e+03, 1.38678154e+02, 4.52826626e+01, 2.21885047e+01,
            6.72629683e+03, 4.20393552e+02, 1.37271364e+02, 6.72629683e+01,
            1.05246211e+04, 6.57788816e+02, 2.14788185e+02, 1.05246211e+02,
            1.40000000e+04, 8.75000000e+02, 2.85714286e+02, 1.40000000e+02
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=6)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri]/10**5,  gold_pressure[ri, ti]/10**5, places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri]/10**6,  gold_specific_internal_energy[ri, ti]/10**6, places=6)

    def test_geometry_error_cog11(self):
        """cog11 problem:"""
        self.assertRaises(ValueError, Cog11, geometry=-1)


# cog12 ##########################
class TestCog12(unittest.TestCase):
    """Test for Coggeshall problems: cog12

    """

    def test_cog12(self):
        """cog12 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog12(geometry=3, gamma=1.4, rho0=1.8, u0=2.3, beta=1.0, Gamma=40.)

        gold_density = np.array([
            8.35485990e+01, 8.35485990e+01, 8.35485990e+01, 8.35485990e+01,
            8.28907087e+00, 8.28907087e+00, 8.28907087e+00, 8.28907087e+00,
            3.26168817e+00, 3.26168817e+00, 3.26168817e+00, 3.26168817e+00,
            1.80000000e+00, 1.80000000e+00, 1.80000000e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            4.95519979e+00, 4.95519979e+00, 4.95519979e+00, 4.95519979e+00,
            3.12158026e+00, 3.12158026e+00, 3.12158026e+00, 3.12158026e+00,
            2.59037013e+00, 2.59037013e+00, 2.59037013e+00, 2.59037013e+00,
            2.30000000e+00, 2.30000000e+00, 2.30000000e+00, 2.30000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            -8.76928747e-02, -8.76928747e-02, -8.76928747e-02, -8.76928747e-02,
            -3.48009404e-02, -3.48009404e-02, -3.48009404e-02, -3.48009404e-02,
            -2.39643478e-02, -2.39643478e-02, -2.39643478e-02, -2.39643478e-02,
            -1.88928571e-02, -1.88928571e-02, -1.88928571e-02, -1.88928571e-02
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            -2.93064673e+02, -2.93064673e+02, -2.93064673e+02, -2.93064673e+02,
            -1.15386985e+01, -1.15386985e+01, -1.15386985e+01, -1.15386985e+01,
            -3.12656919e+00, -3.12656919e+00, -3.12656919e+00, -3.12656919e+00,
            -1.36028571e+00, -1.36028571e+00, -1.36028571e+00, -1.36028571e+00
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            -8.76928747e+00, -8.76928747e+00, -8.76928747e+00, -8.76928747e+00,
            -3.48009404e+00, -3.48009404e+00, -3.48009404e+00, -3.48009404e+00,
            -2.39643478e+00, -2.39643478e+00, -2.39643478e+00, -2.39643478e+00,
            -1.88928571e+00, -1.88928571e+00, -1.88928571e+00, -1.88928571e+00
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=6)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)

    def test_geometry_error_cog12(self):
        """cog12 problem:"""
        self.assertRaises(ValueError, Cog12, geometry=-1)


# cog13 ##########################
class TestCog13(unittest.TestCase):
    """Test for Coggeshall problems: cog13

    """

    def test_cog13(self):
        """cog13 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog13(geometry=3, gamma=1.4, rho0=1.8, alpha=2.0, beta=1.0, lambda0=0.1, Gamma=40.)

        gold_density = np.array([
            1.80000000e+03, 7.08705591e+01, 1.92033536e+01, 8.35485990e+00,
            7.14330473e+02, 2.81250000e+01, 7.62085592e+00, 3.31562835e+00,
            4.91896590e+02, 1.93672146e+01, 5.24781341e+00, 2.28318172e+00,
            3.87798244e+02, 1.52685991e+01, 4.13723711e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            1.00000000e+00, 2.50000000e-01, 1.42857143e-01, 1.00000000e-01,
            4.00000000e+00, 1.00000000e+00, 5.71428571e-01, 4.00000000e-01,
            7.00000000e+00, 1.75000000e+00, 1.00000000e+00, 7.00000000e-01,
            1.00000000e+01, 2.50000000e+00, 1.42857143e+00, 1.00000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            3.15273980e-04, 5.00466248e-04, 6.03097428e-04, 6.79237200e-04,
            7.94440649e-04, 1.26109592e-03, 1.51971029e-03, 1.71157049e-03,
            1.15368388e-03, 1.83135900e-03, 2.20691786e-03, 2.48553657e-03,
            1.46337219e-03, 2.32295855e-03, 2.79933029e-03, 3.15273980e-03
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            2.26997266e+01, 1.41873291e+00, 4.63259726e-01, 2.26997266e-01,
            2.26997266e+01, 1.41873291e+00, 4.63259726e-01, 2.26997266e-01,
            2.26997266e+01, 1.41873291e+00, 4.63259726e-01, 2.26997266e-01,
            2.26997266e+01, 1.41873291e+00, 4.63259726e-01, 2.26997266e-01
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            3.15273980e-02, 5.00466248e-02, 6.03097428e-02, 6.79237200e-02,
            7.94440649e-02, 1.26109592e-01, 1.51971029e-01, 1.71157049e-01,
            1.15368388e-01, 1.83135900e-01, 2.20691786e-01, 2.48553657e-01,
            1.46337219e-01, 2.32295855e-01, 2.79933029e-01, 3.15273980e-01
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=6)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)

    def test_geometry_error_cog13(self):
        """cog13 problem:"""
        self.assertRaises(ValueError, Cog13, geometry=-1)


# cog14 ##########################
class TestCog14(unittest.TestCase):
    """Test for Coggeshall problems: cog14

    """

    def test_cog14(self):
        """cog14 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog14(geometry=3, gamma=1.4, rho0=1.8, alpha=2.0, beta=1.0, lambda0=0.1, Gamma=40.)

        gold_density = np.array([
            5.69209979e+02, 5.69209979e+02, 5.69209979e+02, 5.69209979e+02,
            1.77878118e+01, 1.77878118e+01, 1.77878118e+01, 1.77878118e+01,
            4.39063571e+00, 4.39063571e+00, 4.39063571e+00, 4.39063571e+00,
            1.80000000e+00, 1.80000000e+00, 1.80000000e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            2.01074758e-01, 2.01074758e-01, 2.01074758e-01, 2.01074758e-01,
            4.02149516e-01, 4.02149516e-01, 4.02149516e-01, 4.02149516e-01,
            5.31993805e-01, 5.31993805e-01, 5.31993805e-01, 5.31993805e-01,
            6.35854215e-01, 6.35854215e-01, 6.35854215e-01, 6.35854215e-01
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            3.36925486e-04, 3.36925486e-04, 3.36925486e-04, 3.36925486e-04,
            1.34770194e-03, 1.34770194e-03, 1.34770194e-03, 1.34770194e-03,
            2.35847840e-03, 2.35847840e-03, 2.35847840e-03, 2.35847840e-03,
            3.36925486e-03, 3.36925486e-03, 3.36925486e-03, 3.36925486e-03
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            7.67125395e+00, 7.67125395e+00, 7.67125395e+00, 7.67125395e+00,
            9.58906744e-01, 9.58906744e-01, 9.58906744e-01, 9.58906744e-01,
            4.14208780e-01, 4.14208780e-01, 4.14208780e-01, 4.14208780e-01,
            2.42586350e-01, 2.42586350e-01, 2.42586350e-01, 2.42586350e-01
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            3.36925486e-02, 3.36925486e-02, 3.36925486e-02, 3.36925486e-02,
            1.34770194e-01, 1.34770194e-01, 1.34770194e-01, 1.34770194e-01,
            2.35847840e-01, 2.35847840e-01, 2.35847840e-01, 2.35847840e-01,
            3.36925486e-01, 3.36925486e-01, 3.36925486e-01, 3.36925486e-01
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=6)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)

    def test_geometry_error_cog14(self):
        """cog14 problem:"""
        self.assertRaises(ValueError, Cog14, geometry=-1)


# cog16 ##########################
class TestCog16(unittest.TestCase):
    """Test for Coggeshall problems: cog16

    """

    def test_cog16(self):
        """cog16 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog16(geometry=3, gamma=1.4, u0=2.3, b=1.2, lambda0=0.1, Gamma=40.)

        gold_density = np.array([
            9.82644232e+21, 9.82644232e+21, 9.82644232e+21, 9.82644232e+21,
            1.16360167e+20, 1.16360167e+20, 1.16360167e+20, 1.16360167e+20,
            1.94125497e+19, 1.94125497e+19, 1.94125497e+19, 1.94125497e+19,
            6.20006595e+18, 6.20006595e+18, 6.20006595e+18, 6.20006595e+18
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            1.45120189e-01, 1.45120189e-01, 1.45120189e-01, 1.45120189e-01,
            7.65948951e-01, 7.65948951e-01, 7.65948951e-01, 7.65948951e-01,
            1.49915136e+00, 1.49915136e+00, 1.49915136e+00, 1.49915136e+00,
            2.30000000e+00, 2.30000000e+00, 2.30000000e+00, 2.30000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            7.89745100e-04, 7.89745100e-04, 7.89745100e-04, 7.89745100e-04,
            2.20004173e-02, 2.20004173e-02, 2.20004173e-02, 2.20004173e-02,
            8.42795554e-02, 8.42795554e-02, 8.42795554e-02, 8.42795554e-02,
            1.98375000e-01, 1.98375000e-01, 1.98375000e-01, 1.98375000e-01
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            3.10415387e+20, 3.10415387e+20, 3.10415387e+20, 3.10415387e+20,
            1.02398890e+20, 1.02398890e+20, 1.02398890e+20, 1.02398890e+20,
            6.54432424e+19, 6.54432424e+19, 6.54432424e+19, 6.54432424e+19,
            4.91975233e+19, 4.91975233e+19, 4.91975233e+19, 4.91975233e+19
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            7.89745100e-02, 7.89745100e-02, 7.89745100e-02, 7.89745100e-02,
            2.20004173e+00, 2.20004173e+00, 2.20004173e+00, 2.20004173e+00,
            8.42795554e+00, 8.42795554e+00, 8.42795554e+00, 8.42795554e+00,
            1.98375000e+01, 1.98375000e+01, 1.98375000e+01, 1.98375000e+01
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri]/10**21,  gold_density[ri, ti]/10**21, places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=6)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri]/10**20,  gold_pressure[ri, ti]/10**20, places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)

    def test_geometry_error_cog16(self):
        """cog16 problem:"""
        self.assertRaises(ValueError, Cog16, geometry=1)


# cog17 ##########################
class TestCog17(unittest.TestCase):
    """Test for Coggeshall problems: cog17

    """

    def test_cog17(self):
        """cog17 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog17(geometry=3, gamma=1.4, alpha=2.0, beta=1.0, lambda0=0.1, Gamma=40.)

        gold_density = np.array([
            2.02580193e-09, 3.31907389e-05, 1.66833500e-03, 2.02580193e-02,
            1.26612621e-10, 2.07442118e-06, 1.04270938e-04, 1.26612621e-03,
            4.13428966e-11, 6.77362018e-07, 3.40476531e-05, 4.13428966e-04,
            2.02580193e-11, 3.31907389e-07, 1.66833500e-05, 2.02580193e-04
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            -1.40000000e+00, -3.50000000e-01, -2.00000000e-01, -1.40000000e-01,
            -5.60000000e+00, -1.40000000e+00, -8.00000000e-01, -5.60000000e-01,
            -9.80000000e+00, -2.45000000e+00, -1.40000000e+00, -9.80000000e-01,
            -1.40000000e+01, -3.50000000e+00, -2.00000000e+00, -1.40000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            -2.10000000e-02, -1.31250000e-03, -4.28571429e-04, -2.10000000e-04,
            -3.36000000e-01, -2.10000000e-02, -6.85714286e-03, -3.36000000e-03,
            -1.02900000e+00, -6.43125000e-02, -2.10000000e-02, -1.02900000e-02,
            -2.10000000e+00, -1.31250000e-01, -4.28571429e-02, -2.10000000e-02
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            -1.70167362e-09, -1.74251379e-06, -2.86000286e-05, -1.70167362e-04,
            -1.70167362e-09, -1.74251379e-06, -2.86000286e-05, -1.70167362e-04,
            -1.70167362e-09, -1.74251379e-06, -2.86000286e-05, -1.70167362e-04,
            -1.70167362e-09, -1.74251379e-06, -2.86000286e-05, -1.70167362e-04
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            -2.10000000e+00, -1.31250000e-01, -4.28571429e-02, -2.10000000e-02,
            -3.36000000e+01, -2.10000000e+00, -6.85714286e-01, -3.36000000e-01,
            -1.02900000e+02, -6.43125000e+00, -2.10000000e+00, -1.02900000e+00,
            -2.10000000e+02, -1.31250000e+01, -4.28571429e+00, -2.10000000e+00
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri]/10**21,  gold_density[ri, ti]/10**21, places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=6)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri]/10**20,  gold_pressure[ri, ti]/10**20, places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)

    def test_geometry_error_cog17(self):
        """cog17 problem:"""
        self.assertRaises(ValueError, Cog17, geometry=-1)


# cog18 ##########################
class TestCog18(unittest.TestCase):
    """Test for Coggeshall problems: cog18

    """

    def test_cog18(self):
        """cog18 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog18(geometry=3, alpha=2.0, beta=1.0, rho0=1.8, tau=1.25, Gamma=40.)

        gold_density = np.array([
            9.86420334e+05, 8.68762552e+05, 6.21253904e+05, 2.77284545e+05,
            4.81650554e+02, 4.24200465e+02, 3.03346633e+02, 1.35392844e+02,
            2.21831286e+01, 1.95371798e+01, 1.39710778e+01, 6.23571767e+00,
            3.11933499e+00, 2.74726841e+00, 1.96457734e+00, 8.76850721e-01
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            -6.44122383e-03, -2.85204991e-02, -6.52680653e-02, -1.77777778e-01,
            -2.57648953e-02, -1.14081996e-01, -2.61072261e-01, -7.11111111e-01,
            -4.50885668e-02, -1.99643494e-01, -4.56876457e-01, -1.24444444e+00,
            -6.44122383e-02, -2.85204991e-01, -6.52680653e-01, -1.77777778e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            -2.77830566e-05, -3.40437585e-05, -5.82168414e-05, -2.11640212e-04,
            -4.44528905e-04, -5.44700136e-04, -9.31469463e-04, -3.38624339e-03,
            -1.36136977e-03, -1.66814417e-03, -2.85262523e-03, -1.03703704e-02,
            -2.77830566e-03, -3.40437585e-03, -5.82168414e-03, -2.11640212e-02
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            -1.82705146e+03, -1.97172950e+03, -2.41116267e+03, -3.91230398e+03,
            -1.42738395e+01, -1.54041367e+01, -1.88372084e+01, -3.05648749e+01,
            -2.01329604e+00, -2.17272216e+00, -2.65694994e+00, -4.31111345e+00,
            -5.77764402e-01, -6.23515615e-01, -7.62476584e-01, -1.23717915e+00
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            -2.77830566e-03, -3.40437585e-03, -5.82168414e-03, -2.11640212e-02,
            -4.44528905e-02, -5.44700136e-02, -9.31469463e-02, -3.38624339e-01,
            -1.36136977e-01, -1.66814417e-01, -2.85262523e-01, -1.03703704e+00,
            -2.77830566e-01, -3.40437585e-01, -5.82168414e-01, -2.11640212e+00
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri]/10**5,  gold_density[ri, ti]/10**5, places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri]/10**10,  gold_temperature[ri, ti]/10**10, places=6)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri]/10**4,  gold_pressure[ri, ti]/10**4, places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)

    def test_geometry_error_cog18(self):
        """cog18 problem:"""
        self.assertRaises(ValueError, Cog18, geometry=-1)


# cog19 ##########################
class TestCog19(unittest.TestCase):
    """Test for Coggeshall problems: cog19

    """

    def test_cog19(self):
        """cog19 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog19(geometry=3, gamma=1.4, rho0=1.8, u0=-2.3, Gamma=40.)

        # both regions
        gold_density = np.array([
            1.96020000e+01, 3.88800000e+02, 3.88800000e+02, 3.88800000e+02,
            4.46512500e+00, 1.96020000e+01, 4.54511250e+01, 3.88800000e+02,
            3.17718367e+00, 9.64065306e+00, 1.96020000e+01, 3.30612245e+01,
            2.72322000e+00, 6.63552000e+00, 1.22617800e+01, 1.96020000e+01
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            -2.30000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            -2.30000000e+00, -2.30000000e+00, -2.30000000e+00,  0.00000000e+00,
            -2.30000000e+00, -2.30000000e+00, -2.30000000e+00, -2.30000000e+00,
            -2.30000000e+00, -2.30000000e+00, -2.30000000e+00, -2.30000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            0.00000000e+00, 2.64500000e-02, 2.64500000e-02, 2.64500000e-02,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.64500000e-02,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            0.00000000e+00, 4.11350400e+02, 4.11350400e+02, 4.11350400e+02,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.11350400e+02,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            0.00000000e+00, 2.64500000e+00, 2.64500000e+00, 2.64500000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.64500000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=6)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)

    def test_geometry_error_cog19(self):
        """cog19 problem:"""
        self.assertRaises(ValueError, Cog19, geometry=-1)


# cog20 ##########################
class TestCog20(unittest.TestCase):
    """Test for Coggeshall problems: cog20

    """

    def test_cog20(self):
        """cog20 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog20(geometry=3, gamma=1.4, rho0=1.8, u0=2.3, a=0.3, Gamma=40.)

        # both regions
        gold_density = np.array([
            3.33306672e+00, 5.7052967693e+02, 7.8857859115e+02, 1.13352769679e+03,
            3.56233832e-01, 4.4638664500e+00, 3.3407347100e+01, 1.13352770000e+03,
            8.89112950e-01, 2.6089981000e-01, 6.1698973100e+00, 2.74171476200e+01,
            1.16933447e+00, 1.6904580000e-02, 1.3584726600e+00, 8.86880466000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            2.34020619, -0.03409091, -0.03797468, -0.04285714,
            2.24742268,  2.47727273,  2.75949367, -0.17142857,
            2.15463918,  2.37500000,  2.64556962,  2.98571429,
            2.06185567,  2.27272727,  2.53164557,  2.85714286
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            0.00000000e+00, 3.41554800e-02, 4.23810300e-02, 5.39795900e-02,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.39795900e-02,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            0.00000000e+00, 7.7946848941e+02, 1.33683087555e+03, 2.44749449634e+03,
            0.00000000e+00, 0.0000000000e+00, 0.00000000000e+00, 2.44749449634e+03,
            0.00000000e+00, 0.0000000000e+00, 0.00000000000e+00, 0.00000000000e+00,
            0.00000000e+00, 0.0000000000e+00, 0.00000000000e+00, 0.00000000000e+00,
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            0.00000000e+00, 3.4155475200e+00, 4.23810287000e+00, 5.39795918000e+00,
            0.00000000e+00, 0.0000000000e+00, 0.00000000000e+00, 5.39795918000e+00,
            0.00000000e+00, 0.0000000000e+00, 0.00000000000e+00, 0.00000000000e+00,
            0.00000000e+00, 0.0000000000e+00, 0.00000000000e+00, 0.00000000000e+00,
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=6)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)

    def test_geometry_error_cog20(self):
        """cog20 problem:"""
        self.assertRaises(ValueError, Cog20, geometry=-1)


# cog21 ##########################
class TestCog21(unittest.TestCase):
    """Test for Coggeshall problems: cog21

    """

    def test_cog21(self):
        """cog21 problem:"""

        npts = 4
        r = np.linspace(0.1, 1.0, npts)
        t = np.linspace(0.1, 1.0, npts)
        sol = Cog21(rho0=1.8, temp0=2.9, Gamma=400.)

        # both regions
        gold_density = np.array([
            2.70000000e+03, 1.80000000e+03, 1.80000000e+03, 1.80000000e+03,
            2.81250000e+01, 2.81250000e+01, 2.81250000e+01, 2.81250000e+01,
            5.24781341e+00, 5.24781341e+00, 5.24781341e+00, 5.24781341e+00,
            1.80000000e+00, 1.80000000e+00, 1.80000000e+00, 1.80000000e+00
        ]).reshape(npts, npts)

        gold_velocity = np.array([
            0.00000000e+00, 2.50000000e-01, 1.42857140e-01, 1.00000000e-01,
            4.00000000e+00, 1.00000000e+00, 5.71428570e-01, 4.00000000e-01,
            7.00000000e+00, 1.75000000e+00, 1.00000000e+00, 7.00000000e-01,
            1.00000000e+01, 2.50000000e+00, 1.42857143e+00, 1.00000000e+00
        ]).reshape(npts, npts)

        gold_temperature = np.array([
            2.90000000e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        ]).reshape(npts, npts)

        gold_pressure = np.array([
            3.13200000e+03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        ]).reshape(npts, npts)

        gold_specific_internal_energy = np.array([
            2.90000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        ]).reshape(npts, npts)

        for ti in range(npts):
            ts = t[ti]
            solrt = sol(r, ts)
            # density
            for ri in range(npts):
                self.assertAlmostEqual(solrt.density[ri],  gold_density[ri, ti], places=5)
            # velocity
            for ri in range(npts):
                self.assertAlmostEqual(solrt.velocity[ri],  gold_velocity[ri, ti], places=7)
            # temperature
            for ri in range(npts):
                self.assertAlmostEqual(solrt.temperature[ri],  gold_temperature[ri, ti], places=6)
            # pressure
            for ri in range(npts):
                self.assertAlmostEqual(solrt.pressure[ri],  gold_pressure[ri, ti], places=6)
            # specific_internal_energy
            for ri in range(npts):
                self.assertAlmostEqual(solrt.specific_internal_energy[ri],  gold_specific_internal_energy[ri, ti], places=6)
