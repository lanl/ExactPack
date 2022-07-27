r'''A Python implementation of the Sedov solver in double precision.

This is a pure Python implentation of the Timmes Sedov solver in ExactPack.
It uses SciPy optimization to find values of v that minimize the
error in lambda, and SciPy integration to evaluate the energy integrals.

This solver uses double precision (64 bit), while the Timmes and Kamm solvers
use quad precision (128 bit).
At small values of radius, the changes in the similarity variable become
too small to track with double precision arithmetic. Therefore, the values of
the physical variables are interpolated in that regime. The practical effect
of this is that the density and specific internal energy values,
for the Standard Sedov case at small
radius, are less accurate in this solver than in the Kamm or Timmes solvers,
and should not be trusted. It may be, however, that the impact of
this solution inaccuracy on a global error norm is low, due to the magnitude
of the solution at small radius being so much smaller than the magnitude of the
solution at larger radius. An asymptotic solution for the Sedov functions
at small values of radius may be able to mitigate this problem and should be
developed.
'''

import scipy.integrate as sci_int
import scipy.optimize as sci_opt
from scipy.interpolate import interp1d
import math
import numpy as np

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Sedov(ExactSolver):
    r'''Computes the solution to the Sedov problem. The solver reports the shock
    jump conditions in the :attr:`exactpack.base.ExactSolution.jumps`
    attribute of the return value.
    '''

    parameters = {
        'geometry': '1=planar, 2=cylindrical, 3=spherical',
        'gamma': 'specific heat ratio :math:`\gamma \equiv c_p/c_v`',
        'rho0': 'initial density',
        'omega': 'initial density power-law exponent,\
                 :math:`\\rho \equiv \\rho_0 r^{-\omega}`',
        'eblast': 'total amount of energy deposited at the origin\
                  at time zero',
        }

    # Default parameter values

    geometry = 3
    gamma = 7.0/5.0
    rho0 = 1.0
    omega = 0.0
    eblast = 0.851072

    def __init__(self, **kwargs):

        super(Sedov, self).__init__(**kwargs)

        # Check input parameters for acceptable values

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.gamma < 1:
            raise ValueError('gamma must be greater than 1')

        if self.rho0 < 0:
            raise ValueError('density must be greater than 0')

        if self.eblast < 0:
            raise ValueError('eblast must be greater than 0')

        # Omega must be between 0 and geometry (see Kamm&Timmes)
        if self.omega < 0 or self.omega >= self.geometry:
            raise ValueError('omega must be between 0 and geometry')

        # frequently used constants

        self.gamm1 = self.gamma - 1.0
        self.gamp1 = self.gamma + 1.0
        self.gpogm = self.gamp1 / self.gamm1
        self.xg2 = self.geometry + 2.0 - self.omega
        self.denom2 = 2.0*self.gamm1 + self.geometry - self.gamma*self.omega
        self.denom3 = self.geometry * (2.0 - self.gamma) - self.omega

        # shock-jump velocity

        self.v2 = 4 / (self.xg2*self.gamp1)
        self.vstar = 2 / (self.gamm1*self.geometry + 2)

        # Get solution type, depending on value of v2
        # options are 'standard', 'singular', and 'vacuum'

        osmall = 1.e-4
        if abs(self.v2 - self.vstar) <= osmall:
            self.solution_type = 'singular'
        elif self.v2 < self.vstar - osmall:
            self.solution_type = 'standard'
        elif self.v2 > self.vstar + osmall:
            self.solution_type = 'vacuum'

        # Check for special singularities,
        # options are 'omega2', 'omega3' and 'none'

        if abs(self.denom2) <= osmall:
            self.special_singularity = 'omega2'
            self.denom2 = 1.e-8
        elif abs(self.denom3) <= osmall:
            self.special_singularity = 'omega3'
            self.denom3 = 1.e-8
        else:
            self.special_singularity = 'none'

        # Various exponents, kamm equations 42-47, in terms of book's notation:
        # 0=beta6 a1=beta1  a2=-beta2 a3=beta3 a4=beta4 a5=-beta5

        self.a0 = 2.0/self.xg2
        self.a2 = -self.gamm1/self.denom2
        self.a1 = self.xg2*self.gamma/(2.0 + self.geometry*self.gamm1) *\
            (((2.0*(self.geometry*(2.0-self.gamma) - self.omega)) /
                (self.gamma*self.xg2*self.xg2))-self.a2)
        self.a3 = (self.geometry - self.omega) / self.denom2
        self.a4 = self.xg2 * (self.geometry - self.omega) *\
            self.a1 / self.denom3
        self.a5 = (self.omega*self.gamp1 - 2.0*self.geometry)/self.denom3

        # Frequent combinations, kamm equations 33-37

        self.a_val = 0.25 * self.xg2 * self.gamp1
        self.b_val = self.gpogm
        self.c_val = 0.5 * self.xg2 * self.gamma
        self.d_val = (self.xg2 * self.gamp1)/(self.xg2*self.gamp1 -
                                              2.0 * (2.0 + self.geometry *
                                                     self.gamm1))
        self.e_val = 0.5 * (2.0 + self.geometry * self.gamm1)

        # Evaluate the energy integrals
        # the singular case can be done by hand; save some cpu cycles
        # Kamm equations 80, 81, and 85

        if self.solution_type == 'singular':
            self.eval2 = self.gamp1/(self.geometry*(self.gamm1*self.geometry +
                                                    2.0)**2)
            self.eval1 = 2.0/self.gamm1 * self.eval2
            self.alpha = self.gpogm * 2**(self.geometry) /\
                (self.geometry*(self.gamm1*self.geometry + 2.0)**2)
            if self.geometry != 1:
                self.alpha *= math.pi

        else:
            # Post-shock origin

            self.v0 = 2.0 / (self.xg2 * self.gamma)
            self.vv = 2.0 / self.xg2
            self.rvv = 0.0

            if self.solution_type == 'standard':
                self.vmin = self.v0
            if self.solution_type == 'vacuum':
                self.vmin = self.vv

            # First energy integral

            self.eval1 = sci_int.quad(self.efun01, self.vmin, self.v2,
                                      epsabs=1e-12)[0]

            # Second energy integral

            self.eval2 = sci_int.quad(self.efun02, self.vmin, self.v2,
                                      epsabs=1e-12)[0]

            # Compute alpha

            if self.geometry == 1:
                self.alpha = 0.5 * self.eval1 + self.eval2 / self.gamm1
            else:
                self.alpha = (self.geometry - 1.0) * math.pi *\
                    (self.eval1 + 2.0 * self.eval2 / self.gamm1)

    def _run(self, r, t, npts=3001, vtol=1.e-8):

        # Time must be greater than zero
        if t <= 0.:
            raise ValueError("time must be greater than zero")

        # Re-map the desired list of points r to a list of points linearly
        # spaced between radius of 0.0 and max(r)

        r_eval = np.linspace(0.0, max(r), npts)[-1::-1]

        # Initialize arrays for physical variables and Sedov functions

        density = np.zeros(npts)
        velocity = np.zeros(npts)
        pressure = np.zeros(npts)
        specific_internal_energy = np.zeros(npts)
        sound_speed = np.zeros(npts)

        l_fun = np.zeros(npts)
        dlamdv = np.zeros(npts)
        f_fun = np.zeros(npts)
        g_fun = np.zeros(npts)
        h_fun = np.zeros(npts)

        vwant = np.zeros(npts)

        # Shock-jump conditions at time t

        # shock position

        self.r2 = (self.eblast/(self.alpha*self.rho0))**(1.0/self.xg2) *\
            t**(2.0/self.xg2)

        # pre-shock

        self.p1 = 0.
        self.u1 = 0.
        self.e1 = 0.
        self.cs1 = 0.

        # post-shock

        self.rho1 = self.rho0 * self.r2**(-self.omega)
        self.us = (2.0/self.xg2) * self.r2 / t
        self.u2 = 2.0 * self.us / self.gamp1
        self.rho2 = self.gpogm * self.rho1
        self.p2 = 2.0 * self.rho1 * self.us**2 / self.gamp1
        self.e2 = self.p2/(self.gamm1*self.rho2)
        self.cs2 = math.sqrt(self.gamma*self.p2/self.rho2)

        # Assign shock-jump conditions to JumpCondition object

        self.jumps = [JumpCondition(location=self.r2,
                                    description='Shock',
                                    density=(self.rho1, self.rho2),
                                    pressure=(self.p1, self.p2),
                                    specific_internal_energy=(self.e1,
                                                              self.e2),
                                    velocity=(self.u1, self.u2))]

        # For vacuum case, find radius corresponding to vv
        # and add second jump object for vacuum boundary

        if self.solution_type == 'vacuum':
            l_rvv = self.sedov_funcs_standard(self.vv)[0]
            self.rvv = l_rvv * self.r2
            self.jumps.append(JumpCondition(location=self.rvv,
                                            description='Vacuum Boundary'))

        # Loop over values of r_eval from highest to lowest
        # and compute the sedov functions.
        # When vwant stops changing, stop computing the sedov functions because
        # the solution has become invalid. Then add a solution point at the
        # origin. Then interpolate from r_eval back to r

        vconverged = False
        i = 0

        while not vconverged and i < npts:

            # index r_eval from largest to smallest value
            rwant = r_eval[i]
            # Compute physical solution at a single point (rwant,t)

            # if r is inside the shock
            if rwant <= self.r2:

                # Transition region between standard and vacuum cases
                # Kamm page 15 or equations 88-92
                # lambda = l_fun is book's zeta
                # f_fun is books V, g_fun is book's D, h_fun is book's P
                if self.solution_type == 'singular':
                    l_fun[i], dlamdv[i], f_fun[i], g_fun[i], h_fun[i] =\
                        self.sedov_funcs_singular(rwant)

                # for the vacuum case in the hole
                elif self.solution_type == 'vacuum' and rwant < self.rvv:
                    l_fun[i], dlamdv[i], f_fun[i], g_fun[i], h_fun[i] =\
                        self.sedov_funcs_vacuum()

                else:
                    #  solve for vwant at lam_want using sed_lam_min

                    self.lam_want = rwant/self.r2

                    #    First, set the bounds for v

                    if self.solution_type == 'standard':
                        vmin = self.v0
                        vmax = self.v2
                    elif self.solution_type == 'vacuum':
                        vmin = self.v2
                        vmax = self.vv

                    # Solve for initial guess of v_want at lam_want
                    # using scipy fminbound

                    vwant[i] = sci_opt.fminbound(
                        self.sed_lam_min, vmin, vmax, xtol=1.e-30,
                        maxfun=1000, disp=True)

                    # Compute Sedov functions at vwant
                    l_fun[i], dlamdv[i], f_fun[i], g_fun[i], h_fun[i] =\
                        self.sedov_funcs_standard(vwant[i])

                    # check to see if vwant is converged to package precision
                    if i > 0:
                        if abs(vwant[i]-vwant[i-1]) < vtol:
                            vconverged = True

                # Compute physical variables from Sedov functions
                density[i], velocity[i], pressure[i],\
                    specific_internal_energy[i], sound_speed[i] =\
                    self.physical(f_fun[i], g_fun[i], h_fun[i])

            # if outside the shock, solution is equal to inital conditions
            else:
                density[i] = self.rho0 * rwant**(-self.omega)
                velocity[i] = 0.0
                pressure[i] = 0.0
                specific_internal_energy[i] = 0.0
                sound_speed[i] = 0.0

            i += 1

        # Truncate solution vectors according to how many evaluations were made

        r_eval = r_eval[0:i-1]
        density = density[0:i-1]
        velocity = velocity[0:i-1]
        pressure = pressure[0:i-1]
        specific_internal_energy = specific_internal_energy[0:i-1]
        sound_speed = sound_speed[0:i-1]

        # Compute Sedov functions at origin

        if self.solution_type == 'singular':
            l_funo, dlamdvo, f_funo, g_funo, h_funo =\
                self.sedov_funcs_singular(rwant=0.0)
        elif self.solution_type == 'vacuum':
            l_funo, dlamdvo, f_funo, g_funo, h_funo =\
                self.sedov_funcs_vacuum()
        else:
            self.lam_want = 0.  # the origin
            vwant = sci_opt.fminbound(self.sed_lam_min, vmin, vmax,
                                      xtol=1.e-30, maxfun=1000, disp=True)
            l_funo, dlamdvo, f_funo, g_funo, h_funo =\
                self.sedov_funcs_standard(vwant)

        # Evaluate physical solution at origin point and append to result

        deno, velo, preso, sieo, sso = self.physical(f_funo, g_funo, h_funo)

        r_eval = np.append(r_eval, 0.)
        density = np.append(density, deno)
        velocity = np.append(velocity, 0.0)
        pressure = np.append(pressure, preso)
        specific_internal_energy = np.append(specific_internal_energy, sieo)
        sound_speed = np.append(sound_speed, sso)

        # interpolate from r_eval back to desired r list

        interp = interp1d(r_eval, density)
        density = interp(r)

        interp = interp1d(r_eval, velocity)
        velocity = interp(r)

        interp = interp1d(r_eval, pressure)
        pressure = interp(r)

        interp = interp1d(r_eval, specific_internal_energy)
        specific_internal_energy = interp(r)

        interp = interp1d(r_eval, sound_speed)
        sound_speed = interp(r)

        return ExactSolution([r, density, pressure, specific_internal_energy,
                              velocity, sound_speed],
                             names=['position',
                                    'density',
                                    'pressure',
                                    'specific_internal_energy',
                                    'velocity',
                                    'sound_speed'],
                             jumps=self.jumps)

    def sedov_funcs_standard(self, v):

        r''' Given similarity variable v, compute Sedov functions: f, g, h,
             :math:`\lambda`, and :math:`\frac{d\lambda}{dv}`
        '''

        # Frequent combinations and their derivative with v
        # Kamm equation 29-32, x4 a bit different to save a divide
        # x1 is book's F

        x1 = self.a_val * v
        dx1dv = self.a_val

        # o avoid singularity when c_val * v = 1.0:
        cbag = max(1e-30, self.c_val * v - 1.0)
        x2 = self.b_val * cbag
        dx2dv = self.b_val * self.c_val

        ebag = 1.0 - self.e_val * v
        x3 = self.d_val * ebag
        dx3dv = -self.d_val * self.e_val

        x4 = self.b_val * (1.0 - 0.5 * self.xg2 * v)
        x4 = max(x4, 1e-12)  # Weird math happens below if x4 is exactly zero
        dx4dv = -self.b_val * 0.5 * self.xg2

        # Transition region between standard and vacuum cases
        # Kamm page 15 or equations 88-92
        # lambda = l_fun is book's zeta
        # f_fun is books V, g_fun is book's D, h_fun is book's P

        # omega = omega2 = (2*(gamma -1) + xgeom)/gamma case, denom2 = 0
        # book expressions 20-22
        if self.special_singularity == 'omega2':
            beta0 = 1.0 / (2.0 * self.e_val)
            pp1 = self.gamm1 * beta0
            c6 = 0.5 * self.gamp1
            c2 = c6 / self.gamma
            y = 1.0 / (x1 - c2)
            z = (1.0 - x1)*y
            pp2 = self.gamp1 * beta0 * z
            dpp2dv = -self.gamp1 * beta0 * dx1dv * y * (1.0 + z)
            pp3 = (4.0 - self.geometry - 2.0*self.gamma) * beta0
            pp4 = -self.geometry * self.gamma * beta0

            l_fun = x1**(-self.a0) * x2**(pp1) * np.exp(pp2)
            dlamdv = (-self.a0*dx1dv/x1 + pp1*dx2dv/x2 + dpp2dv) * l_fun
            f_fun = x1 * l_fun
            g_fun = x1**(self.a0*self.omega) * x2**pp3 * x4**self.a5 *\
                np.exp(-2.0*pp2)
            h_fun = x1**(self.a0*self.geometry) * x2**pp4 * x4**(1.0 + self.a5)

        # omega = omega3 = xgeom*(2 - gamma) case, denom3 = 0
        # book expressions 23-25
        elif self.special_singularity == 'omega3':
            beta0 = 1.0 / (2.0 * self.e_val)
            pp1 = self.a3 + self.omega * self.a2
            pp2 = 1.0 - 4.0 * beta0
            c6 = 0.5 * self.gamp1
            pp3 = -self.geometry * self.gamma * self.gamp1 * beta0 *\
                (1.0 - x1)/(c6 - x1)
            pp4 = 2.0 * (self.geometry * self.gamm1 - self.gamma) * beta0

            l_fun = x1**(-self.a0) * x2**(-self.a2) * x4**(-self.a1)
            dlamdv = -(self.a0*dx1dv/x1 + self.a2*dx2dv/x2 +
                       self.a1*dx4dv/x4) * l_fun
            f_fun = x1 * l_fun
            g_fun = x1**(self.a0*self.omega) * x2**pp1 * x4**pp2 * np.exp(pp3)
            h_fun = x1**(self.a0*self.geometry) * x4**pp4 * np.exp(pp3)

        # for the standard or vacuum case not in the hole
        # kamm equations 38-41
        else:
            l_fun = x1**(-self.a0) * x2**(-self.a2) * x3**(-self.a1)
            dlamdv = -(self.a0*dx1dv/x1 + self.a2*dx2dv/x2 +
                       self.a1*dx3dv/x3) * l_fun
            f_fun = x1 * l_fun
            g_fun = x1**(self.a0 * self.omega) *\
                x2**(self.a3 + self.a2 * self.omega) *\
                x3**(self.a4 + self.a1 * self.omega) * x4**self.a5
            h_fun = x1**(self.a0*self.geometry) *\
                x3**(self.a4+self.a1*(self.omega-2.0))*x4**(1.0 + self.a5)

        return l_fun, dlamdv, f_fun, g_fun, h_fun

    def efun01(self, v):

        r'Integrand for first Sedov energy integral'

        [l_fun, dlamdv, f_fun, g_fun, h_fun] = self.sedov_funcs_standard(v)

        return dlamdv * l_fun**(self.geometry + 1.0) *\
            self.gpogm * g_fun * v**2

    def efun02(self, v):

        r'Integrand for second Sedov energy integral'

        [l_fun, dlamdv, f_fun, g_fun, h_fun] = self.sedov_funcs_standard(v)

        z = 8.0 / ((self.geometry + 2.0 - self.omega)**2 * self.gamp1)

        return dlamdv * l_fun**(self.geometry - 1.0) * h_fun * z

    def sed_lam_min(self, v):

        # Calculate the objective function (l_fun(v) - lam_want[i])**2

        [l_fun, dlamdv, f_fun, g_fun, h_fun] = self.sedov_funcs_standard(v)

        return (l_fun - self.lam_want)**2

    def physical(self, f_fun, g_fun, h_fun):
        '''Returns physical variables from single values of Sedov functions'''

        density = self.rho2 * g_fun
        velocity = self.u2 * f_fun
        pressure = self.p2 * h_fun
        specific_internal_energy = 0.
        sound_speed = 0.

        # Compute actual values for specific_internal_energy and
        # sound_speed only if density is greater than 0.

        if (density > 0.):
            specific_internal_energy = pressure /\
                (self.gamm1 * density)
            sound_speed = math.sqrt(self.gamma * pressure/density)

        return density, velocity, pressure, specific_internal_energy,\
            sound_speed

    def sedov_funcs_singular(self, rwant):
        l_fun = rwant/self.r2
        dlamdv = 0.0
        f_fun = l_fun
        g_fun = l_fun**(self.geometry - 2.0)
        h_fun = l_fun**self.geometry

        return l_fun, dlamdv, f_fun, g_fun, h_fun

    def sedov_funcs_vacuum(self):
        l_fun = 0.0
        dlamdv = 0.0
        f_fun = 0.0
        g_fun = 0.0
        h_fun = 0.0

        return l_fun, dlamdv, f_fun, g_fun, h_fun
