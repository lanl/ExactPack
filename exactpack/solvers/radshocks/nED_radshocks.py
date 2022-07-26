r""" The radiative-shock solvers.
"""

# from ...base import ExactSolver, ExactSolution
from exactpack.base import ExactSolver, ExactSolution

from exactpack.solvers.radshocks import radshock
import numpy


class ED_Solver(ExactSolver):
    r"""Computes the solution to the equilibrium-diffusion radiative-shock
        problem. See [Lowrie2007]_ for the complete problem definition and
        solution method.  The problem default values are selected to be
        consistent with the :math:`{\cal M}_0 = 1.2` problem; see Fig. 5 in
        [Lowrie2007]_.

        Default values are: :math:`{\cal M}_0 = 1.2, \rho_0 = 1 \, g/cc, 
        T_{\text{ref}} = 100 \, eV`.
    """

    parameters = {
        'M0': "Mach number",
        'rho0': "ambient equilibrium initial density",
        'Tref': "ambient equilibrium initial temperature",
        'Cv': "ambient equilibrium heat-capacity at constant volume",
        'gamma': r"specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'sigA': "multiplier for absorption cross section",
        'sigS': "multiplier for scattering cross section",
        'expDensity_abs': "exponential power of the density in the absorption cross section",
        'expTemp_abs': "exponential power of the temperature in the absorption cross section",
        'expDensity_scat': "exponential power of the density in the scattering cross section",
        'expTemp_scat': "exponential power of the temperature in the scattering cross section",
        'eps_precursor_equil': "value of the small parameter epsilon that moves the solution out of equilibrium in the precursor region",
        'int_tol': "integrator tolerance"
        }

    M0 = 1.2
    rho0 = 1.
    Tref = 100.
    Cv = 1.4472799784454e12
    gamma = 5. / 3.
    sound = numpy.sqrt(gamma * (gamma - 1.) * Cv * Tref)
    sigA = 577.35
    sigS = 0.
    expDensity_abs = 0.
    expTemp_abs = 0.
    expDensity_scat = 0.
    expTemp_scat = 0.
    problem = 'ED'
    epsilon = 1.
    eps_precursor_equil = 1.e-6
    int_tol = 1.e-10
    # I do not allow L to be set in the code, but this could easily be changed
    L = 1. # cm
    geometry = 1 # convergence study requires us to set geometry

    def __init__(self, **kwargs):
        """Set default values if necessary and check for valid inputs.
        """
        super(ED_Solver, self).__init__(**kwargs)

        # instantiate the ED solver, and 'drive' the solver (below)
        prob = radshock.greyED_RadShock(
               M0 = self.M0,
               rho0 = self.rho0,
               Tref = self.Tref,
               Cv = self.Cv,
               gamma = self.gamma,
               sigA = self.sigA,
               sigS = self.sigS,
               expDensity_abs = self.expDensity_abs,
               expTemp_abs = self.expTemp_abs,
               expDensity_scat = self.expDensity_scat,
               expTemp_scat = self.expTemp_scat,
               eps_precursor_equil = self.eps_precursor_equil,
               int_tol = self.int_tol)
        prob.ED_driver()

        self.x = prob.ED_profile.x
        self.Fr = prob.C0 * prob.ar * prob.Tref**4 * prob.ED_profile.Fr
        self.Tm = prob.Tref * prob.ED_profile.Tm
        self.Density = prob.rho0 * prob.ED_profile.Density
        self.Speed = prob.sound * prob.ED_profile.Speed
        self.Mach = prob.ED_profile.Mach
        self.Pressure = prob.rho0 * prob.sound**2 * prob.ED_profile.Pressure
        self.SIE = self.Pressure / self.Density / (self.gamma - 1.)
        self.RADE = prob.ar * self.Tm**4
        self.Sound_Speed = self.Speed / self.Mach
        self.ar = prob.ar
        self.C0 = prob.C0
        self.P0 = prob.P0
        self.__prob = prob

    def _run(self, x, t):
        temperature = numpy.interp(x, self.x, self.Tm)
        density = numpy.interp(x, self.x, self.Density)
        velocity = numpy.interp(x, self.x, self.Speed)
        pressure = numpy.interp(x, self.x, self.Pressure)
        sie = numpy.interp(x, self.x, self.SIE)
        rade = numpy.interp(x, self.x, self.RADE)
        sound_speed = numpy.interp(x, self.x, self.Sound_Speed)

        return ExactSolution([x, temperature, density, velocity, pressure, sie,
                              rade, sound_speed],
                             names=['position',
                                    'temperature',
                                    'density',
                                    'velocity',
                                    'pressure',
                                    'sie',
                                    'rade',
                                    'sound_speed'])

class nED_Solver(ExactSolver):
    r"""Computes the solution to the nonequilibrium-diffusion radiative-shock
        problem.  See [Lowrie2008]_ or [Ferguson2017]_ for the complete problem
        definition.  The solution used here is given in [Ferguson2017]_.  The
        problem default values are selected to be consistent with the
        :math:`{\cal M}_0 = 1.2` problem; see Fig. 5 in [Lowrie2008]_ or Figure
        7 in [Ferguson2017]_.

        Default values are: :math:`{\cal M}_0 = 1.2, \rho_0 = 1 \, g/cc, 
        T_{\text{ref}} = 100 \, eV`.
    """

    parameters = {
        'M0': "Mach number",
        'rho0': "ambient equilibrium initial density",
        'Tref': "ambient equilibrium initial temperature",
        'Cv': "ambient equilibrium heat-capacity at constant volume",
        'gamma': r"specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'sigA': "multiplier for absorption cross section",
        'sigS': "multiplier for scattering cross section",
        'expDensity_abs': "exponential power of the density in the absorption cross section",
        'expTemp_abs': "exponential power of the temperature in the absorption cross section",
        'expDensity_scat': "exponential power of the density in the scattering cross section",
        'expTemp_scat': "exponential power of the temperature in the scattering cross section",
        'problem': "a flag to run with either the equations contained in the Lowrie-Edwards paper ('LM_nED') or the equations contained in the Ferguson-Morel-Lowrie paper ('nED'), or to run a flux-limited diffusion (FLD) problem; the FLD problems are for Levermore-Pomraning ('FLD_LP'), the Levermore-Pomraning polynomial variant ('FLD_poly'), Wilson''s sum flux-limiter ('FLD_1'), and Larsen''s square-root flux-limiter ('FLD_2')",
        'epsilon': "an asymptotically-small parameter for producing nonequilibrium-diffusion solutions in the equilibrium-diffusion limit",
        'eps_precursor_equil': "value of the small parameter epsilon that moves the solution out of equilibrium in the precursor region",
        'eps_relaxation_equil': "value of the small parameter epsilon that moves the solution out of equilibrium in the relaxation region",
        'eps_precursor_ASP': "value of the small parameter epsilon that determines how close to M = 1 the integration curve in the precursor region should get",
        'eps_relaxation_ASP': "value of the small parameter epsilon that determines how close to M = 1 the integration curve in the relaxation region should get",
        'int_tol': "integrator tolerance"
        }

    M0 = 1.2
    rho0 = 1.
    Tref = 100.
    Cv = 1.4472799784454e12
    gamma = 5. / 3.
    sound = numpy.sqrt(gamma * (gamma - 1.) * Cv * Tref)
    sigA = 577.35
    sigS = 0.
    expDensity_abs = 0.
    expTemp_abs = 0.
    expDensity_scat = 0.
    expTemp_scat = 0.
    problem = 'nED'
    epsilon = 1.
    eps_precursor_equil = 1.e-6
    eps_relaxation_equil = 1.e-6
    eps_precursor_ASP = 1.e-6
    eps_relaxation_ASP = 1.e-6
    int_tol = 1.e-10
    # I do not allow L to be set in the code, but this could easily be changed
    L = 1. # cm
    geometry = 1 # convergence study requires us to set geometry

    def __init__(self, **kwargs):
        super(nED_Solver, self).__init__(**kwargs)

        prob = radshock.greyNED_RadShock(
               M0 = self.M0,
               rho0 = self.rho0,
               Tref = self.Tref,
               Cv = self.Cv,
               gamma = self.gamma,
               sigA = self.sigA,
               sigS = self.sigS,
               expDensity_abs = self.expDensity_abs,
               expTemp_abs = self.expTemp_abs,
               expDensity_scat = self.expDensity_scat,
               expTemp_scat = self.expTemp_scat,
               problem = self.problem,
               epsilon = self.epsilon,
               eps_precursor_equil = self.eps_precursor_equil,
               eps_relaxation_equil = self.eps_relaxation_equil,
               int_tol = self.int_tol)
        prob.nED_driver(epsilon = self.epsilon)

        self.x = prob.nED_profile.x
        self.Tm = prob.Tref * prob.nED_profile.Tm
        self.Tr = prob.Tref * prob.nED_profile.Tr
        self.Fr = prob.C0 * prob.ar * prob.Tref**4 * prob.nED_profile.Fr
        self.Density = prob.rho0 * prob.nED_profile.Density
        self.Speed = prob.sound * prob.nED_profile.Speed
        self.Mach = prob.nED_profile.Mach
        self.Pressure = prob.rho0 * prob.sound**2 * prob.nED_profile.Pressure
        self.SIE = self.Pressure / self.Density / (self.gamma - 1.)
        self.RADE = prob.ar * self.Tr**4
        self.Sound_Speed = self.Speed / self.Mach
        self.C0 = prob.C0
        self.P0 = prob.P0
        self.problem = prob.problem
        self.__prob = prob

    def _run(self, x, t):
        temperature_mat = numpy.interp(x, self.x, self.Tm)
        temperature_rad = numpy.interp(x, self.x, self.Tr)
        density = numpy.interp(x, self.x, self.Density)
        velocity = numpy.interp(x, self.x, self.Speed)
        pressure = numpy.interp(x, self.x, self.Pressure)
        sie = numpy.interp(x, self.x, self.SIE)
        rade = numpy.interp(x, self.x, self.RADE)
        sound_speed = numpy.interp(x, self.x, self.Sound_Speed)

        return ExactSolution([x, temperature_mat, temperature_rad, density,
                              velocity, pressure, sie, rade, sound_speed],
                             names=['position',
                                    'temperature_mat',
                                    'temperature_rad',
                                    'density',
                                    'velocity',
                                    'pressure',
                                    'specific_internal_energy',
                                    'rade',
                                    'sound_speed'])

class Sn_Solver(ExactSolver):
    r"""Computes the solution to the :math:`\text{S}_{\text{n}}` radiative-shock
        problem.  See [Ferguson2017]_ for the complete problem definition.  The
        problem default values are selected to be consistent with the
        :math:`{\cal M}_0 = 1.2` problem; see Fig. 5 in [Lowrie2008]_ or Figure
        7 in [Ferguson2017]_.

        Default values are: :math:`{\cal M}_0 = 1.2, \, \rho_0 = 1 \, g/cc, \, T_{\text{ref}} = 100 \, eV`.
    """

    parameters = {
        'M0': "Mach number",
        'rho0': "ambient equilibrium initial density",
        'Tref': "ambient equilibrium initial temperature",
        'Cv': "ambient equilibrium heat-capacity at constant volume",
        'gamma': r"specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'sigA': "multiplier for absorption cross section",
        'sigS': "multiplier for scattering cross section",
        'expDensity_abs': "exponential power of the density in the absorption cross section",
        'expTemp_abs': "exponential power of the temperature in the absorption cross section",
        'expDensity_scat': "exponential power of the density in the scattering cross section",
        'expTemp_scat': "exponential power of the temperature in the scattering cross section",
        'problem': "a flag to run with either the equations contained in the Lowrie-Edwards paper ('LM_nED') or the equations contained in the Ferguson-Morel-Lowrie paper ('nED')",
        'epsilon': "an asymptotically-small parameter for producing nonequilibrium-diffusion solutions in the equilibrium-diffusion limit",
        'eps_precursor_equil': "value of the small parameter epsilon that moves the solution out of equilibrium in the precursor region",
        'eps_relaxation_equil': "value of the small parameter epsilon that moves the solution out of equilibrium in the relaxation region",
        'eps_precursor_ASP': "value of the small parameter epsilon that determines how close to M = 1 the integration curve in the precursor region should get",
        'eps_relaxation_ASP': "value of the small parameter epsilon that determines how close to M = 1 the integration curve in the relaxation region should get",
        'int_tol': "integrator tolerance",
        'Sn': "number of radiation-intensity directions for which to solve",
        'f_tol': "minimum relative convergence tolerance for the VEF"
        }

    M0 = 1.2
    rho0 = 1.
    Tref = 100.
    Cv = 1.4472799784454e12
    gamma = 5. / 3.
    sound = numpy.sqrt(gamma * (gamma - 1.) * Cv * Tref)
    sigA = 577.35
    sigS = 0.
    expDensity_abs = 0.
    expTemp_abs = 0.
    expDensity_scat = 0.
    expTemp_scat = 0.
    problem = 'nED'
    epsilon = 1.
    eps_precursor_equil = 1.e-6
    eps_relaxation_equil = 1.e-6
    eps_precursor_ASP = 1.e-6
    eps_relaxation_ASP = 1.e-6
    int_tol = 1.e-10
    Sn = 16
    f_tol = 1.e-4
    # I do not allow L to be set in the code, but this could easily be changed
    L = 1. # cm
    geometry = 1 # convergence study requires us to set geometry

    def __init__(self, **kwargs):
        super(Sn_Solver, self).__init__(**kwargs)

        prob = radshock.greySn_RadShock(
               M0 = self.M0,
               rho0 = self.rho0,
               Tref = self.Tref,
               Cv = self.Cv,
               gamma = self.gamma,
               sigA = self.sigA,
               sigS = self.sigS,
               expDensity_abs = self.expDensity_abs,
               expTemp_abs = self.expTemp_abs,
               expDensity_scat = self.expDensity_scat,
               expTemp_scat = self.expTemp_scat,
               problem = self.problem,
               eps_precursor_equil = self.eps_precursor_equil,
               eps_relaxation_equil = self.eps_relaxation_equil,
               int_tol = self.int_tol)
        prob.Sn_driver(Sn = self.Sn, f_tol = self.f_tol)

        self.x = prob.Sn_profile.x
        self.Tm = prob.Tref * prob.Sn_profile.Tm
        self.Tr = prob.Tref * prob.Sn_profile.Tr
        self.Fr = prob.C0 * prob.ar * prob.Tref**4 * prob.Sn_profile.Fr
        self.Density = prob.rho0 * prob.Sn_profile.Density
        self.Speed = prob.sound * prob.Sn_profile.Speed
        self.Mach = prob.Sn_profile.Mach
        self.Pressure = prob.rho0 * prob.sound**2 * prob.Sn_profile.Pressure
        self.SIE = self.Pressure / self.Density / (self.gamma - 1.)
        self.RADE = prob.ar * self.Tr**4
        self.Sound_Speed = self.Speed / self.Mach
        self.VEF = numpy.interp(self.x, prob.Sn_profile.x_RT, prob.Sn_profile.f)
        self.C0 = prob.C0
        self.P0 = prob.P0
        self.__prob = prob

    def _run(self, x, t):
        temperature_mat = numpy.interp(x, self.x, self.Tm)
        temperature_rad = numpy.interp(x, self.x, self.Tr)
        density = numpy.interp(x, self.x, self.Density)
        velocity = numpy.interp(x, self.x, self.Speed)
        pressure = numpy.interp(x, self.x, self.Pressure)
        sie = numpy.interp(x, self.x, self.SIE)
        rade = numpy.interp(x, self.x, self.RADE)
        sound_speed = numpy.interp(x, self.x, self.Sound_Speed)
        VEF = numpy.interp(x, self.x, self.VEF)

        return ExactSolution([x, temperature_mat, temperature_rad, density,
                              velocity, pressure, sie, rade, sound_speed, VEF],
                             names=['position',
                                    'temperature_mat',
                                    'temperature_rad',
                                    'density',
                                    'velocity',
                                    'pressure',
                                    'sie',
                                    'rade',
                                    'sound_speed',
                                    'VEF'])

class ie_Solver(ExactSolver):
    """Computes the solution to the ion-electron shock problem
    """

    parameters = {
        'M0': "Mach number",
        'rho0': "ambient equilibrium initial density",
        # 'kappa': "ion-diffusion coefficient",
        # 'R': "R",
        'Z': "number of available/free electrons",
        'Tref': "ambient equilibrium initial temperature",
        'Cv': "ambient equilibrium heat-capacity at constant volume",
        'gamma': r"specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'eps_precursor_equil': "value of the small parameter epsilon that moves the solution out of equilibrium in the precursor region",
        'eps_relaxation_equil': "value of the small parameter epsilon that moves the solution out of equilibrium in the relaxation region",
        'eps_precursor_ASP': "value of the small parameter epsilon that determines how close to M = 1 the integration curve in the precursor region should get",
        'eps_relaxation_ASP': "value of the small parameter epsilon that determines how close to M = 1 the integration curve in the relaxation region should get",
        'int_tol': "integrator tolerance",
        }

    M0 = 1.4
    rho0 = 1.
    # kappa = 1.
    # R = 1.
    Z = 1.
    Tref = 100.
    Cv = 1.4472799784454e12
    gamma = 5. / 3.
    sound = numpy.sqrt(gamma * (gamma - 1.) * Cv * Tref)
    problem = 'ie'
    eps_precursor_equil = 1.e-6
    eps_relaxation_equil = 1.e-6
    eps_precursor_ASP = 1.e-6
    eps_relaxation_ASP = 1.e-6
    int_tol = 1.e-10
    # I do not allow L to be set in the code, but this could easily be changed
    L = 1. # cm
    geometry = 1 # convergence study requires us to set geometry

    def __init__(self, **kwargs):
        super(ie_Solver, self).__init__(**kwargs)

        prob = radshock.Shock_2Tie(
               M0 = self.M0,
               rho0 = self.rho0,
               # kappa = self.kappa,
               # R = self.R,
               Z = self.Z,
               Tref = self.Tref,
               Cv = self.Cv,
               gamma = self.gamma,
               problem = self.problem,
               eps_precursor_equil = self.eps_precursor_equil,
               eps_relaxation_equil = self.eps_relaxation_equil,
               eps_precursor_ASP = self.eps_precursor_ASP,
               eps_relaxation_ASP = self.eps_relaxation_ASP,
               int_tol = self.int_tol)
        prob.IE_driver()

        self.x = prob.IE_profile.x
        self.Ti = prob.Tref * prob.IE_profile.Ti
        self.Tm = prob.Tref * prob.IE_profile.Tm
        self.Te = prob.Tref * prob.IE_profile.Te
        self.Density = prob.rho0 * prob.IE_profile.Density
        self.Speed = prob.sound * prob.IE_profile.Speed
        self.Mach = prob.IE_profile.Mach
        self.Pressure = prob.rho0 * prob.sound**2 * prob.IE_profile.Pressure
        self.SIE = self.Pressure / self.Density / (self.gamma - 1.)
        self.Sound_Speed = self.Speed / self.Mach
        self.Fe = prob.IE_profile.Fe
        self.__prob = prob

    def _run(self, x, t):
        temperature_ion = numpy.interp(x, self.x, self.Ti)
        temperature_mat = numpy.interp(x, self.x, self.Tm)
        temperature_elec = numpy.interp(x, self.x, self.Te)
        density = numpy.interp(x, self.x, self.Density)
        velocity = numpy.interp(x, self.x, self.Speed)
        pressure = numpy.interp(x, self.x, self.Pressure)
        sie = numpy.interp(x, self.x, self.SIE)
        sound_speed = numpy.interp(x, self.x, self.Sound_Speed)

        return ExactSolution([x, temperature_ion, temperature_mat,
                              temperature_elec, density, velocity, pressure,
                              sie, sound_speed],
                             names=['position',
                                    'temperature_ion',
                                    'temperature_mat',
                                    'temperature_elec',
                                    'density',
                                    'velocity',
                                    'pressure',
                                    'sie',
                                    'sound_speed'])
