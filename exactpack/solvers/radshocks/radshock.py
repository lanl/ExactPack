'''
The docstring for radshock.
'''

import os, copy, numpy, scipy, pickle, scipy.integrate, scipy.interpolate
import matplotlib.pyplot, importlib
try:
    import utils
except ImportError:
  from exactpack.solvers.radshocks import utils  
importlib.reload(utils)

# TOC:
# RadShock(object)
#   def __init__(self, various variables)
#
# IEShock(object)
#   def __init__(self, various variables)
#
# greyED_RadShock(RadShock)
#   def ED_driver(self)
#
# greyNED_RadShock(RadShock)
#   def nED_driver(self, epsilon = 1.,
#                  eps_precursor_ASP = 1.e-6, eps_relaxation_ASP = 1.e-6)
#
# greySn_RadShock(greyNED_RadShock)
#   def Sn_driver(self, Sn = 16, f_tol = 1.e-4)
#
# Shock_2Tie(IEShock)
#   def IE_driver(self)

class RadShock(object):
    def __init__(self, M0 = 1.2, rho0 = 1.,
                 Tref = 100., Cv = 1.4472799784454e12, gamma = 5. / 3.,
                 sigA = 577.35, sigS = 0.,
                 expDensity_abs = 0., expTemp_abs = 0.,
                 expDensity_scat = 0., expTemp_scat = 0.,
                 problem = 'nED', epsilon = 1., print_the_sources = 'False', 
                 eps_precursor_equil = 1.e-6,
                 eps_relaxation_equil = 1.e-6,
                 int_tol = 1.e-10, use_jac = 'True',
                 dxset = 5, numlev = 1, runtime = 1.e-7,
                 freezeWidth = 10., dumpnum = 100):
#: The initial Mach number.
        self.M0 = M0
#: The ambient, upstream equilibrium material density
        self.rho0 = rho0
#: The ambient, upstream equilibrium temperature
        self.Tref = Tref
#: The constant volume specific heat
        self.Cv = Cv
#: The adiabatic index, which is Cv / Cp
        self.gamma = gamma
#: The multiplicative factor for the absorption cross section
        self.sigA = sigA
#: The exponential power of density in the absorption cross section
        self.expDensity_abs = expDensity_abs
#: The exponential power of temperature in the absorption cross section
        self.expTemp_abs = expTemp_abs
#: The multiplicative factor for the scattering cross section
        self.sigS = sigS
#: The exponential power of density in the scattering cross section
        self.expDensity_scat = expDensity_scat
#: The exponential power of temperature in the scattering cross section
        self.expTemp_scat = expTemp_scat
        self.T0 = 1.
#: The speed of light in units of cm / s
        self.c = 2.99792458e10# [cm / s]
#: The radiation constant in units of erg / cm^3 / eV^4
        self.ar = 137.20172# [erg / cm^3 - eV^4]
#: The sound speed for an ideal gas
        self.sound = numpy.sqrt(self.gamma * (self.gamma - 1.) * self.Cv * \
                                self.Tref)
#: The ratio of the speed of light to the sound speed
        self.C0 = self.c / self.sound
#: A ratio proportional to the radiation pressure over an ideal kinetic energy
        self.P0 = self.ar * self.Tref**4 / (self.rho0 * self.sound**2)
#: A flag to print the residual source error
        self.print_the_sources = print_the_sources
#: The problem identifier allows the solution to differentiate between different nonequilibrium-diffusion rad-hydro models.  For example, the Lowrie-Morel source model as implemented in RAGE ("LM_nED"), the nonequilibrium-diffusion model that included first-order velocity corrections ("nED"), the flux-limited diffusion model (FLD) of Levermore-Pomraning ("FLD_LP"), the FLD model of Wilson's sum limiter ("FLD_1"), the FLD model of Larsen's square-root limiter ("FLD_2")
        self.problem = problem
        self.use_jac = use_jac
        self.eps_precursor_equil = eps_precursor_equil
        self.eps_relaxation_equil = eps_relaxation_equil
        self.int_tol = int_tol
        self.numlev = numlev
        self.runtime = runtime
        self.freezeWidth = freezeWidth
        self.dumpnum = dumpnum

class IEShock(object):
    def __init__(self, M0 = 1.4, rho0 = 1., Z = 1., Tref = 100., gamma = 5./3.,
                 Cv = 1.4472799784454e12, problem = 'ie',
                 print_the_sources = 'False', 
                 eps_precursor_equil = 1.e-6,
                 eps_relaxation_equil = 1.e-6,
                 eps_precursor_ASP = 1.e-6,
                 eps_relaxation_ASP = 1.e-6,
                 int_tol = 1.e-10, use_jac = False, epsilon = 1.):
        self.M0 = M0
        self.rho0 = rho0
        self.Z = Z
        self.Tref = Tref
        self.Cv = Cv
        self.gamma = gamma
        self.T0 = 1.
        self.Te0 = self.T0
        self.sound = numpy.sqrt(gamma * (gamma - 1.) * Cv * Tref)
        top = (3. * gamma - 1.) * Z + (gamma + 1.) * gamma
        bot = (3. - gamma) * Z + gamma + 1.
        self.Mc = numpy.sqrt(top / bot / gamma)
        self.print_the_sources = print_the_sources
        self.problem = problem
        self.eps_precursor_equil = eps_precursor_equil
        self.eps_relaxation_equil = eps_relaxation_equil
        self.eps_precursor_ASP = eps_precursor_ASP
        self.eps_relaxation_ASP = eps_relaxation_ASP
        self.int_tol = int_tol
        self.use_jac = use_jac
        self.epsilon = epsilon

class greyED_RadShock(RadShock):
    '''
    Define the grey equilibrium-diffusion radiative-shock problem,
    and drive the solution.
    '''
    def ED_driver(self):
        self.problem = 'ED'
        self.ED_profile = utils.ED_ShockProfiles(self)
        self.ED_profile.downstream_equilibrium()
        self.ED_profile.make_ED_solution()

class greyNED_RadShock(RadShock):
    '''
    Define the grey nonequilibrium-diffusion radiative-shock problem,
    and drive the solution.
    '''
# prob = radshock.greyNED_RadShock(M0 = 3., sigA = 44.93983839817290, sigS = 0.4006, expDensity_abs = 1, expTemp_abs = -3.5)
    def nED_driver(self, epsilon = 1.,
                   eps_precursor_ASP = 1.e-6, eps_relaxation_ASP = 1.e-6):
        self.Pr0 = self.T0**4 / 3.
        self.epsilon = epsilon
        self.eps_precursor_ASP = eps_precursor_ASP
        self.eps_relaxation_ASP = eps_relaxation_ASP
        self.eps_precursor_ASP_initially = min(eps_precursor_ASP, 1.e-3)
        self.eps_relaxation_ASP_initially = min(eps_relaxation_ASP, 1.e-3)
        self.nED_profile = utils.nED_ShockProfiles(self)
        self.nED_profile.downstream_equilibrium()
        self.nED_profile.make_2T_solution()

class greySn_RadShock(greyNED_RadShock):
    '''
    Define the grey-Sn radiative-shock problem, and drive the solution.
    '''
    def Sn_driver(self, Sn = 16, f_tol = 1.e-4, epsilon = 1.):
        self.nED_driver(epsilon = epsilon)
        self.Sn_profile = utils.Sn_ShockProfiles(self.nED_profile)
        self.Sn_profile.Sn = Sn
        self.Sn_profile.f_tol = f_tol
        self.Sn_profile.make_RT_solution()
        self.Sn_profile.continue_running()
        self.Sn_profile.update_dictionaries()
        if (self.Sn_profile.f_err[-1] > self.Sn_profile.f_tol):
            print_stmnt  = 'The greySn_RadShock solution for M0 = '
            print_stmnt += str(self.M0) + ' failed to converge.'
            print(print_stmnt)

class Shock_2Tie(IEShock):
    '''
    Define the ion-electron shock problem, and drive the solution.
    '''
    def IE_driver(self):
        if (self.M0 <= self.Mc):
          self.IE_profile = utils.IE_continuousShockProfiles(self)
          self.IE_profile.downstream_equilibrium()
          self.IE_profile.make_continuous_solution()
        else:
          self.IE_profile = utils.IE_discontinuousShockProfiles(self)
          self.IE_profile.downstream_equilibrium()
          self.IE_profile.make_2T_solution()
