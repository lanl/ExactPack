r"""Tests for the radshocks solvers and the function files used by the solvers.
"""

import numpy as np
import pytest
import warnings

from exactpack.solvers.radshocks.nED_radshocks import ED_Solver
from exactpack.solvers.radshocks.nED_radshocks import nED_Solver
from exactpack.solvers.radshocks.nED_radshocks import Sn_Solver
from exactpack.solvers.radshocks.nED_radshocks import ie_Solver

import exactpack.solvers.radshocks.fnctn_ED as fnctn_ED
import exactpack.solvers.radshocks.fnctn_nED as fnctn_nED
import exactpack.solvers.radshocks.fnctn_2Tie as fnctn_2Tie

class Test_RadshockAssignments():
    r""" Tests for the equilibrium-diffusion functions contained in the file
         fnctn_ED.py.
    """
    def test_defaults(self):
        warnings.filterwarnings('ignore', 'divide by zero encountered in',
                                category=RuntimeWarning)
        M0 = 1.2
        prob = ED_Solver()
        assert prob.M0 == M0


class Test_fnctn_ED():
    r"""Test the equilibrium-divide helper functions.
    """
    sigA = 0.5
    sigS = 0.5
    M0 = 1.2
    P0 = 1.e-4
    gamma = 5./3.
    sigA= sigA
    sigS = sigS
    expDensity_abs = 0.
    expDensity_scat = 0.
    expTemp_abs = 0.
    expTemp_scat = 0.
    
    def test_sigA(self):
        assert fnctn_ED.sigma_a(1., self) == self.sigA

    def test_expDensity_abs(self):
        T = 1.1
        self.expDensity_abs = 1.
        rho = fnctn_ED.rho(T, self)
        val = self.sigA * rho**self.expDensity_abs
        np.testing.assert_allclose(fnctn_ED.sigma_a(T, self), val),
        self.expDensity_abs = 0.

    def test_expTemp_abs(self):
        T = 1.1
        self.expTemp_abs = 1.
        val = self.sigA * T**self.expTemp_abs
        np.testing.assert_allclose(fnctn_ED.sigma_a(T, self), val),
        self.expTemp_abs = 0.

    def test_sigS(self):
        assert fnctn_ED.sigma_s(1., self) == self.sigS

    def test_expDensity_scat(self):
        T = 1.1
        self.expDensity_scat = 1.
        rho = fnctn_ED.rho(T, self)
        val = self.sigS * rho**self.expDensity_scat
        np.testing.assert_allclose(fnctn_ED.sigma_s(T, self), val),
        self.expDensity_scat = 0.

    def test_expTemp_scat(self):
        T = 1.1
        self.expTemp_scat = 1.
        val = self.sigS * T**self.expTemp_scat
        np.testing.assert_allclose(fnctn_ED.sigma_s(T, self), val),
        self.expTemp_scat = 0.

    def test_sigT(self):
        assert fnctn_ED.sigma_t(1., self) == self.sigA + self.sigS


class Test_fnctn_nED():
    r"""Test the non-equilibrium-diffusion helper functions.
    """
    sigA = 0.5
    sigS = 0.5
    M0 = 1.2
    P0 = 1.e-4
    gamma = 5./3.
    sigA= sigA
    sigS = sigS
    expDensity_abs = 0.
    expDensity_scat = 0.
    expTemp_abs = 0.
    expTemp_scat = 0.
    Pr0 = 1./3.
    
    def test_sigT(self):
        assert fnctn_nED.sigma_t(1./3., 1.2, self) == self.sigA + self.sigS

    def test_mat_internal_energy(self):
        val = 1. / self.gamma / (self.gamma - 1.)
        assert fnctn_nED.mat_internal_energy(1. / 3., self.M0, self) == val


class Test_ConservationEquationsSatisfied():
    def test_ED_withContinuousShock(self):
        warnings.filterwarnings('ignore', 'divide by zero encountered in',
                                category=RuntimeWarning)
        self.prob_ED = ED_Solver()

        val  = self.prob_ED.Density * self.prob_ED.Speed
        val /= self.prob_ED.Sound_Speed[0]
        np.testing.assert_allclose(val, self.prob_ED.M0)

        val  = self.prob_ED.Density * self.prob_ED.Speed**2
        val += self.prob_ED.Pressure
        val /= self.prob_ED.Sound_Speed[0]**2
        Pr   = self.prob_ED.P0 * (self.prob_ED.Tm / self.prob_ED.Tref)**4 / 3.
        val += Pr
        np.testing.assert_allclose(val, val[0])

        val  = 0.5 * self.prob_ED.Density * self.prob_ED.Speed**2
        val += self.prob_ED.Density * self.prob_ED.SIE + self.prob_ED.Pressure
        val /= self.prob_ED.Sound_Speed[0]**2
        val *= self.prob_ED.Speed / self.prob_ED.Sound_Speed[0]
        Fr   = self.prob_ED.Fr / self.prob_ED.Sound_Speed[0]**2
        val += Fr
        np.testing.assert_allclose(val, val[0])

    def test_ED_withEmbeddedHydrodynamicShock(self):
        self.prob_ED = ED_Solver(M0 = 2.)

        val  = self.prob_ED.Density * self.prob_ED.Speed
        val /= self.prob_ED.Sound_Speed[0]
        np.testing.assert_allclose(val, self.prob_ED.M0)

        val  = self.prob_ED.Density * self.prob_ED.Speed**2
        val += self.prob_ED.Pressure
        val /= self.prob_ED.Sound_Speed[0]**2
        Pr   = self.prob_ED.P0 * (self.prob_ED.Tm / self.prob_ED.Tref)**4 / 3.
        val += Pr
        np.testing.assert_allclose(val, val[0])

        val  = 0.5 * self.prob_ED.Density * self.prob_ED.Speed**2
        val += self.prob_ED.Density * self.prob_ED.SIE + self.prob_ED.Pressure
        val /= self.prob_ED.Sound_Speed[0]**2
        val *= self.prob_ED.Speed / self.prob_ED.Sound_Speed[0]
        Fr   = self.prob_ED.Fr / self.prob_ED.Sound_Speed[0]**2
        val += Fr
        val  = val[:-1]
        np.testing.assert_allclose(val, val[0])

    def test_nED_withEmbeddedHydroShock(self):
        self.prob_nED = nED_Solver()

        val  = self.prob_nED.Density * self.prob_nED.Speed
        val /= self.prob_nED.Sound_Speed[0]
        np.testing.assert_allclose(val, self.prob_nED.M0)

        val  = self.prob_nED.Density * self.prob_nED.Speed**2
        val += self.prob_nED.Pressure
        val /= self.prob_nED.Sound_Speed[0]**2
        Pr   = self.prob_nED.P0 * (self.prob_nED.Tr / self.prob_nED.Tref)**4 / 3
        val += Pr
        np.testing.assert_allclose(val, val[0])

        val  = 0.5 * self.prob_nED.Density * self.prob_nED.Speed**2
        val += self.prob_nED.Density * self.prob_nED.SIE
        val += self.prob_nED.Pressure
        val /= self.prob_nED.Sound_Speed[0]**2
        val *= self.prob_nED.Speed / self.prob_nED.Sound_Speed[0]
        Fr   = self.prob_nED.Fr / self.prob_nED.Sound_Speed[0]**2
        val += Fr
        np.testing.assert_allclose(val, val[0])

    def test_nED_withContinuousShock(self):
        self.prob_nED = nED_Solver(M0 = 1.05, problem = 'LM_nED')

        val  = self.prob_nED.Density * self.prob_nED.Speed
        val /= self.prob_nED.Sound_Speed[0]
        np.testing.assert_allclose(val, self.prob_nED.M0)

        val  = self.prob_nED.Density * self.prob_nED.Speed**2
        val += self.prob_nED.Pressure
        val /= self.prob_nED.Sound_Speed[0]**2
        Pr   = self.prob_nED.P0 * (self.prob_nED.Tr / self.prob_nED.Tref)**4 / 3
        val += Pr
        np.testing.assert_allclose(val, val[0])

        val  = 0.5 * self.prob_nED.Density * self.prob_nED.Speed**2
        val += self.prob_nED.Density * self.prob_nED.SIE
        val += self.prob_nED.Pressure
        val /= self.prob_nED.Sound_Speed[0]**2
        val *= self.prob_nED.Speed / self.prob_nED.Sound_Speed[0]
        Fr   = self.prob_nED.Fr / self.prob_nED.Sound_Speed[0]**2
        val += Fr
        np.testing.assert_allclose(val, val[0])

    def test_Sn_works(self):
        self.prob_Sn = Sn_Solver()

        val  = self.prob_Sn.Density * self.prob_Sn.Speed
        val /= self.prob_Sn.Sound_Speed[0]
        np.testing.assert_allclose(val, self.prob_Sn.M0)

        val  = self.prob_Sn.Density * self.prob_Sn.Speed**2
        val += self.prob_Sn.Pressure
        val /= self.prob_Sn.Sound_Speed[0]**2
        Pr   = self.prob_Sn.P0 * (self.prob_Sn.Tr / self.prob_Sn.Tref)**4
        Pr  *= self.prob_Sn.VEF
        val += Pr
        np.testing.assert_allclose(val, val[0])

        val  = 0.5 * self.prob_Sn.Density * self.prob_Sn.Speed**2
        val += self.prob_Sn.Density * self.prob_Sn.SIE + self.prob_Sn.Pressure
        val /= self.prob_Sn.Sound_Speed[0]**2
        val *= self.prob_Sn.Speed / self.prob_Sn.Sound_Speed[0]
        Fr   = self.prob_Sn.Fr / self.prob_Sn.Sound_Speed[0]**2
        val += Fr
        np.testing.assert_allclose(val, val[0])

    def test_ie(self):
        self.prob_ie = ie_Solver()

        # print('\n')
        # print('test_MassFluxConservation_ie')
        val  = self.prob_ie.Density * self.prob_ie.Speed
        val /= self.prob_ie.Sound_Speed[0]
        np.testing.assert_allclose(val, self.prob_ie.M0)

        # print('\n')
        # print('test_MomentumFluxConservation_ie')
        val  = self.prob_ie.Density * self.prob_ie.Speed**2
        val += self.prob_ie.Pressure
        val /= self.prob_ie.Sound_Speed[0]**2
        np.testing.assert_allclose(val, val[0])

        # print('\n')
        # print('test_EnergyFluxConservation_ie')
        val  = 0.5 * self.prob_ie.Density * self.prob_ie.Speed**2
        val += self.prob_ie.Density * self.prob_ie.SIE + self.prob_ie.Pressure
        val /= self.prob_ie.Sound_Speed[0]**2
        val *= self.prob_ie.Speed / self.prob_ie.Sound_Speed[0]
        val += self.prob_ie.Fe
        np.testing.assert_allclose(val, val[0])

