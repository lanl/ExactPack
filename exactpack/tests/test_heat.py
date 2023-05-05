r"""Unit tests for the heat conduction solvers.
"""

import numpy as np

import pytest

from exactpack.solvers.heat import Rod1D
from exactpack.solvers.heat import PlanarSandwich
from exactpack.solvers.heat import PlanarSandwichHot
from exactpack.solvers.heat import PlanarSandwichHalf
from exactpack.solvers.heat import CylindricalSandwich
from exactpack.solvers.heat import Hutchens1
from exactpack.solvers.heat import Hutchens2


class TestHeatCylindricalSandwich():
    r"""Tests the Cylindrical Sandwich :class:`exactpack.solvers.heat.cylindrical_sandwich.CylindricalSandwich`."""

    def test_heat_cylindrical_sandwich_modes(self):
        r"""Regression tests for modes numbers."""
        kappa = 1.0
        a = 0.25
        b = 0.85
        T1 = 1.0
        T0 = 0.0
        Nsum = 3
        Msum = 4

        solver = CylindricalSandwich(Nsum=Nsum, Msum=Msum)

        # check alpha[n,m]
        alphax, betax = solver.alpha(Nsum, Msum, a, b)
        alpha1 = [3.4989852, 7.3870836, 11.61537219, 16.44862148]
        alpha2 = [6.25129196, 10.78395964, 14.42185509, 18.36175587]
        alpha3 = [8.82484366, 13.79272634, 17.8270296, 21.46822425]
        np.testing.assert_allclose(alphax[0], alpha1)
        np.testing.assert_allclose(alphax[1], alpha2)
        np.testing.assert_allclose(alphax[2], alpha3)

        # check beta[n,m]
        beta1 = [-5.10209162e-02, -4.24170214e-01, -9.42790889e-02, 1.03030103e+00]
        beta2 = [-2.28898139e-03, -1.03916648e-01, -4.22161618e-01, -3.58574815e-01]
        beta3 = [-7.93295896e-05, -9.47241486e-03, -1.05920956e-01, -3.85130036e-01]
        np.testing.assert_allclose(betax[0], beta1)
        np.testing.assert_allclose(betax[1], beta2)
        np.testing.assert_allclose(betax[2], beta3)
        # agrees with Mathematica to 1E-6, except for
        # k=8, m=98 and k=30, m=91

    def test_heat_cylindrical_sandwich_Rnm(self):
        r"""Regression tests for the modes Rnm."""
        kappa = 1.0
        a = 0.25
        b = 0.85
        T1 = 1.0
        T0 = 0.0
        Nsum = 3
        Msum = 4

        r0 = 0.5
        thetan = 5
        theta0 = np.linspace(0, np.pi/2, thetan)
        r, theta = np.meshgrid(r0, theta0)
        rtheta_list = (r, theta)
        t = 0.001

        solver = CylindricalSandwich(Nsum=Nsum, Msum=Msum)
        soln = solver(rtheta_list, t)

        alphax, betax = solver.alpha(Nsum, Msum, a, b)

        R = np.zeros(shape=(Nsum, Msum))
        R[0, 0] = 0.33245395
        R[0, 1] = 0.37989646
        R[0, 2] = -0.22673482
        R[0, 3] = -0.33822247
        R[1, 0] = 0.15110507
        R[1, 1] = 0.40690992
        R[1, 2] = -0.02749347
        R[1, 3] = -0.28941796
        R[2, 0] = 0.07733
        R[2, 1] = 0.3348765
        R[2, 2] = 0.19776377
        R[2, 3] = -0.24376106
        A = np.zeros(shape=(Nsum, Msum))
        A[0, 0] = 0.0871171306963
        A[0, 1] = 0.0354946414545
        A[0, 2] = 0.018762823829
        A[0, 3] = 0.0275386091655
        A[1, 0] = 0.0577388019134
        A[1, 1] = 0.0263055512547
        A[1, 2] = 0.0176425283465
        A[1, 3] = 0.0125640308931
        A[2, 0] = 0.0453074363113
        A[2, 1] = 0.0224030167834
        A[2, 2] = 0.0154832580728
        A[2, 3] = 0.0114801830117
        for n in range(Nsum):
            for m in range(Msum):
                k = 2 * (n + 1)
                alphanm = alphax[n, m]
                betanm = betax[n, m]
                Rnm = solver.R(r0, k, m, alphanm, betanm)
                Anm = solver.Anm_analytic(a, b, k, m, alphanm, betanm)
                assert Rnm == pytest.approx(R[n, m])
                assert Anm == pytest.approx(A[n, m])
                # Anm = solver.Anm_int(a, b, k, m, alphanm, betanm)  # error


class TestHeatRod1dBoundary():
    r"""Tests the 1D Rod :class:`exactpack.solvers.heat.rod1d.Rod1D`."""
    # construct spatial grid and select time
    kappa = 1.0
    T0 = 3.0
    T1 = 4.0
    L = 2.0
    t = 1.0
    Nsum = 1000
    Nx = 3001
    x = np.linspace(0.0, L, Nx)
    dx = float(L) / float(Nx)

    def test_BC1_homogeneous(self):
        r"""Checks Boundary Conditions."""        
        #
        # BC1 homogeneous
        # T(0) = 0
        # T(L) = 0
        solver = Rod1D(kappa=self.kappa, TL=self.T0, TR=self.T1, L=self.L, Nsum=self.Nsum,
                       alpha1=1.0, beta1=0.0, alpha2=1.0, beta2=0.0)
        soln = solver(self.x, self.t)
        bc1 = soln.temperature[0]
        bc2 = soln.temperature[-1]
        assert bc1 == pytest.approx(0.0)
        assert bc2 == pytest.approx(0.0)

    def test_BC1_non_homogeneous(self):
        #
        # BC1 nonhomogeneous
        # T(0) = Ta
        # T(L) = Tb
        solver = Rod1D(kappa=self.kappa, TL=0, TR=0, L=self.L, Nsum=self.Nsum,
                       alpha1=1.0, beta1=0.0, gamma1=self.T0, alpha2=1.0, beta2=0.0, gamma2=self.T1)
        soln = solver(self.x, self.t)
        bc1 = soln.temperature[0]
        bc2 = soln.temperature[-1]
        assert bc1 == pytest.approx(self.T0)
        assert bc2 == pytest.approx(self.T1)

    def test_BC2_homogeneous(self):
        #
        # BC2 homogeneous
        # partial_x T(0) = 0
        # partial_x T(L) = 0
        solver = Rod1D(kappa=self.kappa, TL=self.T0, TR=self.T1, L=self.L, Nsum=self.Nsum,
                       alpha1=0.0, beta1=1.0, alpha2=0.0, beta2=1.0)
        soln = solver(self.x, self.t)
        bc1 = (soln.temperature[1] - soln.temperature[0]) / self.dx
        bc2 = (soln.temperature[-1] - soln.temperature[-2]) / self.dx
        assert bc1 == pytest.approx(0.0, abs=1.e-3)
        assert bc2 == pytest.approx(0.0, abs=1e-3)

    def test_BC2_non_homogeneous(self):
        #
        # BC2 nonhomogeneous
        # partial_x T(0) = F
        # partial_x T(L) = F
        F = 1
        beta1 = 1.0
        beta2 = 1.0
        solver = Rod1D(kappa=self.kappa, TL=0, TR=0, L=self.L, Nsum=self.Nsum,
                       alpha1=0.0, beta1=beta1, gamma1=F, alpha2=0.0, beta2=beta2, gamma2=F)
        soln = solver(self.x, self.t)
        bc1 = (soln.temperature[1] - soln.temperature[0]) / self.dx
        bc2 = (soln.temperature[-1] - soln.temperature[-2]) / self.dx
        assert bc1 == pytest.approx(F, abs=1.e-3)
        assert bc2 == pytest.approx(F, abs=1e-3)

    def test_BC3_homogeneous(self):
        #
        # BC3 homogeneous
        # T(0) = 0
        # partial_x T(L) = 0
        solver3 = Rod1D(kappa=self.kappa, TL=self.T0, TR=self.T1, L=self.L, Nsum=self.Nsum,
                        alpha1=1.0, beta1=0.0, alpha2=0.0, beta2=1.0)
        soln3 = solver3(self.x, self.t)
        bc1 = soln3.temperature[0]
        bc2 = (soln3.temperature[-1] - soln3.temperature[-2]) / self.dx
        assert bc1 == pytest.approx(0.0)
        assert bc2 == pytest.approx(0.0, abs=1.e-3)

    def test_BC4_homogeneous(self):
        #
        # BC4 homogeneous
        # partial_x T(0) = 0
        # T(L) = 0
        solver4 = Rod1D(kappa=self.kappa, TL=self.T1, TR=self.T0, L=self.L, Nsum=self.Nsum,
                       alpha1=0.0, beta1=1.0, alpha2=1.0, beta2=0.0)  # reverse T0 and T1
        soln4 = solver4(self.x, self.t)
        bc1 = (soln4.temperature[0] - soln4.temperature[1]) / self.dx
        bc2 = soln4.temperature[-1]
        assert bc1 == pytest.approx(0.0, abs=1.e-3)
        assert bc2 == pytest.approx(0.0)
    
    def test_BC3_reverse_BC3(self):
        #
        # Test BC3 is the reverse of BC4
        solver3 = Rod1D(kappa=self.kappa, TL=self.T0, TR=self.T1, L=self.L, Nsum=self.Nsum,
                        alpha1=1.0, beta1=0.0, alpha2=0.0, beta2=1.0)
        soln3 = solver3(self.x, self.t)
        solver4 = Rod1D(kappa=self.kappa, TL=self.T1, TR=self.T0, L=self.L, Nsum=self.Nsum,
                       alpha1=0.0, beta1=1.0, alpha2=1.0, beta2=0.0)  # reverse T0 and T1
        soln4rev = solver4(self.x[-1::-1], self.t)
        for n in range(1, self.Nx):
            assert soln3.temperature[n]-soln4rev.temperature[n] == pytest.approx(0.0, abs=1.e-4)

    def test_BC3_non_homogeneous(self):
        #
        # BC3 nonhomogeneous
        # T(0) = T1
        # partial_x T(L) = F2
        T1 = 2
        F2 = -1
        solver3 = Rod1D(kappa=self.kappa, TL=0, TR=0, L=self.L, Nsum=self.Nsum,
                        alpha1=1.0, beta1=0.0, gamma1=T1, alpha2=0.0, beta2=1.0, gamma2=F2)
        soln3 = solver3(self.x, self.t)
        bc1 = soln3.temperature[0]
        bc2 = (soln3.temperature[-1] - soln3.temperature[-2]) / self.dx
        assert bc1 == pytest.approx(T1)
        assert bc2 == pytest.approx(F2, abs=1.e-3)

    def test_BC4_non_homogeneous(self):
        #
        # BC4 nonhomogeneous
        # partial_x T(0) = F1
        # T(L) = T2
        F1 = -1
        T2 = 2
        solver4 = Rod1D(kappa=self.kappa, TL=0, TR=0, L=self.L, Nsum=self.Nsum,
                        alpha1=0.0, beta1=1.0, gamma1=F1, alpha2=1.0, beta2=0.0, gamma2=T2)
        soln4 = solver4(self.x, self.t)
        bc1 = (soln4.temperature[1] - soln4.temperature[0]) / self.dx
        bc2 = soln4.temperature[-1]
        assert bc1 == pytest.approx(F1, abs=1.e-3)
        assert bc2 == pytest.approx(T2)

    def test_general_BC_homogeneous(self):
        #
        # general BC's homogeneous
        # alpha1 T(0) + beta1 partial_x T(0) = 0
        # alpha2 T(L) + beta2 partial_x T(L) = 0
        alpha1 = 1.0
        beta1 = -1.0
        alpha2 = 1.0
        beta2 = 2.0
        solver = Rod1D(alpha1=alpha1, beta1=beta1, alpha2=alpha2, beta2=beta2, L=2, TL=3, TR=4)
        soln = solver(self.x, self.t)
        bc1 = alpha1 * soln.temperature[0] + (beta1 * (soln.temperature[1] - soln.temperature[0]) / self.dx)
        bc2 = alpha2 * soln.temperature[-1] + (beta2 * (soln.temperature[-1] - soln.temperature[-2]) / self.dx)
        assert bc1 == pytest.approx(0.0, abs=1.e-4)
        assert bc2 == pytest.approx(0.0, abs=1.e-3)

    def test_general_BC_non_homogeneous(self):
        #
        # general BC's nonhomogeneous
        # alpha1 T(0) + beta1 partial_x T(0) = gamma1
        # alpha2 T(L) + beta2 partial_x T(L) = gamma2
        alpha1 = 1.0
        beta1 = -1.0
        gamma1 = 1.0
        alpha2 = 1.0
        beta2 = 2.0
        gamma2 = 1.0
        solver = Rod1D(alpha1=alpha1, beta1=beta1, gamma1=gamma1, alpha2=alpha2, beta2=beta2, gamma2=gamma2, L=2)
        soln = solver(self.x, self.t)
        bc1 = alpha1 * soln.temperature[0] + (beta1 * (soln.temperature[1] - soln.temperature[0]) / self.dx)
        bc2 = alpha2 * soln.temperature[-1] + (beta2 * (soln.temperature[-1] - soln.temperature[-2]) / self.dx)
        assert bc1 == pytest.approx(gamma1, abs=1.e-4)
        assert bc2 == pytest.approx(gamma2, abs=1.e-3)


class TestHeatRod1dRegression():
    # construct spatial grid and select time
    kappa = 1.0
    T0 = 3.0
    T1 = 4.0
    L = 2.0
    t = 1.0
    Nsum = 1000
    # regression tests
    x = [0.1, 0.5, 0.7]
    t = 0.1
    
    def test_heat_rod1d_regression1(self):
        r"""Regression tests for 1D Rod."""
        
        temp0 = [0.58073473, 2.45615774, 2.98282366]
        solver = Rod1D(kappa=self.kappa, TL=self.T0, TR=self.T1, L=self.L, Nsum=self.Nsum,
                       alpha1=1.0, beta1=0.0, alpha2=1.0, beta2=0.0)
        soln = solver(self.x, self.t)
        np.testing.assert_allclose(soln.temperature, temp0)

    def test_heat_rod1d_regression2(self):
        temp0 = [2.46926527, 0.79384226, 0.36717634]
        solver = Rod1D(kappa=self.kappa, TL=0, TR=0, L=self.L, Nsum=self.Nsum,
                       alpha1=1.0, beta1=0.0, gamma1=self.T0, alpha2=1.0, beta2=0.0, gamma2=self.T1)
        soln = solver(self.x, self.t)
        np.testing.assert_allclose(soln.temperature, temp0)
        
    def test_heat_rod1d_regression3(self):
        temp0 = [3.18285308, 3.27956288, 3.36103976]
        solver = Rod1D(kappa=self.kappa, TL=self.T0, TR=self.T1, L=self.L, Nsum=self.Nsum,
                       alpha1=0.0, beta1=1.0, alpha2=0.0, beta2=1.0)
        soln = solver(self.x, self.t)
        np.testing.assert_allclose(soln.temperature, temp0)

    def test_heat_rod1d_regression4(self):
        temp0 = [0.26570616, 0.05912576, 0.02207951]
        F = -1
        solver = Rod1D(kappa=self.kappa, TL=0, TR=0, L=self.L, Nsum=self.Nsum,
                       alpha1=0.0, beta1=1.0, gamma1=F, alpha2=0.0, beta2=1.0, gamma2=F)
        soln = solver(self.x, self.t)
        np.testing.assert_allclose(soln.temperature, temp0, rtol=2.0e-7)

    def test_heat_rod1d_regression5(self):
        temp0 = [0.58080926, 2.45929629, 2.99718884]
        solver = Rod1D(kappa=self.kappa, TL=self.T0, TR=self.T1, L=self.L, Nsum=self.Nsum,
                        alpha1=1.0, beta1=0.0, alpha2=0.0, beta2=1.0)
        soln = solver(self.x, self.t)
        np.testing.assert_allclose(soln.temperature, temp0)

    def test_heat_rod1d_regression6(self):
        temp0 = [3.81707325, 3.71800208, 3.62777238]
        solver = Rod1D(kappa=self.kappa, TL=self.T1, TR=self.T0, L=self.L, Nsum=self.Nsum,
                       alpha1=0.0, beta1=1.0, alpha2=1.0, beta2=0.0)  # reverse T0 and T1
        soln = solver(self.x, self.t)
        np.testing.assert_allclose(soln.temperature, temp0)

    def test_heat_rod1d_regression7(self):
        temp0 = [0.59627172, 0.51704211, 0.36237742]
        solver = Rod1D(alpha1=1, beta1=-1, alpha2=1, beta2=2, L=2, TL=3, TR=4)
        soln = solver(self.x, self.t)
        np.testing.assert_allclose(soln.temperature, temp0)

    def test_heat_rod1d_regression8(self):
        temp0 = [1.97797341, 1.99708713, 1.89143987]
        solver = Rod1D(alpha1=1, beta1=-1, gamma1=1.2, alpha2=1, beta2=2, gamma2=2.3, L=2)
        soln = solver(self.x, self.t)
        np.testing.assert_allclose(soln.temperature, temp0)


class TestRod1DFunctions():
    r"""Tests function definitions."""
    kappa = 1.0
    T0 = 3.0
    T1 = 4.0
    L = 2.0
    t = 1.0
    Nsum = 3
    Nx = 10

    def test_heat_rod1d_modes_BC1(self):
        r"""Regression test for the mode numbers and coefficients."""
        # BC1
        solver = Rod1D(kappa=self.kappa, TL=self.T0, TR=self.T1, L=self.L,
                       Nsum=self.Nsum, alpha1=1.0, beta1=0.0, alpha2=1.0,
                       beta2=0.0)
        solver.modes_BC1()
        kn0 = [0, 1.57079633, 3.14159265]
        An0 = [0, 0, 0]
        Bn0 = [0, 4.45633841, -0.31830989]
        np.testing.assert_allclose(solver.kn, kn0)
        np.testing.assert_allclose(solver.An, An0)
        np.testing.assert_allclose(solver.Bn, Bn0)

    def test_heat_rod1d_modes_BC2(self):
        # BC2
        F = -1
        solver = Rod1D(kappa=self.kappa, TL=0, TR=0, L=self.L, Nsum=self.Nsum,
                       alpha1=0.0, beta1=1.0, gamma1=F, alpha2=0.0, beta2=1.0,
                       gamma2=F)
        solver.modes_BC2()
        kn0 = [0, 1.57079633, 3.14159265]
        An0 = [1, -0.81056947, 0]
        Bn0 = [0, 0, 0]
        np.testing.assert_allclose(solver.kn, kn0)
        np.testing.assert_allclose(solver.An, An0)
        np.testing.assert_allclose(solver.Bn, Bn0)

    def test_heat_rod1d_modes_BC3(self):
        # BC3
        solver = Rod1D(kappa=self.kappa, TL=self.T0, TR=self.T1, L=self.L,
                       Nsum=self.Nsum, alpha1=1.0, beta1=0.0, alpha2=0.0,
                       beta2=1.0)
        solver.modes_BC3()
        kn0 = [0.78539816, 2.35619449, 3.92699082]
        An0 = [0, 0, 0]
        Bn0 = [4.6302881, 1.18317627, 0.79636651]
        np.testing.assert_allclose(solver.kn, kn0)
        np.testing.assert_allclose(solver.An, An0)
        np.testing.assert_allclose(solver.Bn, Bn0)

    def test_heat_rod1d_modes_BC4(self):
        # BC4: reverse T0 and T1
        solver = Rod1D(kappa=self.kappa, TL=self.T1, TR=self.T0, L=self.L,
                       Nsum=self.Nsum, alpha1=0.0, beta1=1.0, alpha2=1.0,
                       beta2=0.0)
        solver.modes_BC4()
        kn0 = [0.78539816, 2.35619449, 3.92699082]
        An0 = [4.6302881, -1.18317627, 0.79636651]
        Bn0 = [0, 0, 0]
        np.testing.assert_allclose(solver.kn, kn0)
        np.testing.assert_allclose(solver.An, An0)
        np.testing.assert_allclose(solver.Bn, Bn0)


class TestHeatPlanarSandwich():
    r"""Tests the planar sandwich :class:`exactpack.solvers.heat.planar_sandwich.PlanarSandwich`"""
    # construct spatial grid and select time
    t = 0.1
    L = 2.0
    y = [0.1, 0.5, 0.7, 1.5]


    def test_heat_planar_sandwich_regression_1d(self):
        #
        temp_1d = [8.23063274e-01, 2.63552477e-01, 1.17524868e-01, 7.96207473e-04]
        solver = PlanarSandwich(Nsum=20000, TB=1.0, TT=0.0)
        soln = solver(self.y, self.t)
        np.testing.assert_allclose(soln.temperature, temp_1d)

    def test_heat_planar_sandwich_regression_hot(self):
        #
        temp_hot = [2.73429384, 2.94087424, 2.97792049, 3.05912576]
        solver = PlanarSandwichHot(Nsum=20000, F=1.0, TL=3.0, TR=3.0)
        soln = solver(self.y, self.t)
        np.testing.assert_allclose(soln.temperature, temp_hot)

    def test_heat_planar_sandwich_regression_half(self):
        #
        temp_half = [2.17693856, 2.73654009, 2.88294825, 3.05842207]
        solver = PlanarSandwichHalf(Nsum=20000, TB=2.0, FT=1.0, TL=3.0, TR=3.0)
        soln = solver(self.y, self.t)
        np.testing.assert_allclose(soln.temperature, temp_half)



class TestHeatPlanarHutchens():
    r"""Tests the Hutchens heat solver :class:`exactpack.solvers.heat.hutchens1.Hutchens1`."""

    def test_heat_planar_hutchens1_regression(self):
        r""" Regression tests for Hutchens1."""
        b = 1.0
        r = [0.0, 0.25, 0.5, 0.75, 1.0]
        t = 0.1
        solver = Hutchens1()
        soln = solver(r, t)
        temp_h1 = [1.0, 1.0033374549970184, 1.107411979374676, 2.15419627072115,
                   4.999999999999999]
        np.testing.assert_allclose(soln.temperature, temp_h1)


class TestHeatPlanarHutchens():
    r"""Tests the Hutchens heat solver :class:`exactpack.solvers.heat.hutchens2.Hutchens2`."""

    def test_heat_planar_hutchens2_regression(self):
        r""" Regression tests for Hutchens2."""
        b = 1.0
        L = 2.0
        r0 = [0.0, 0.5, 1.0]
        z0 = 0.5
        r, z = np.meshgrid(r0, z0)
        rzlist = (r, z)
        t = 0  # dummy argument
        solver = Hutchens2()
        soln = solver(rzlist, t)
        temp_h2 = [-6969.729767307974, -6932.525739227717, -6807.659371413964]
        for n in range(len(r0)):
            assert soln[0, n][2] == pytest.approx(temp_h2[n])
