r""" ExactPack wrapper for the Riemann solvers.
"""

from exactpack.base import ExactSolver, ExactSolution, print_when_verbose

from exactpack.solvers.riemann import riemann
from numpy import interp, mgrid, array
import matplotlib.pyplot as plt

class IGEOS_Solver(ExactSolver):
    r"""Computes the analytic solution to the Riemann problem for an ideal-gas
        EOS. See [LoraClavijo2013]_ for the solution description. The problem
        default values are for the Sod shocktube, which is also Riemann problem
        #1.

        Default values are :math:`xmin=0, xd0=0.5, xmax=1, t=0.25, \rho_l=1, u_l=0, p_l=1, \gamma_l=1.4, \rho_r=0.125, u_r=0, p_r=0.1, \gamma_r=1.4`.
    """

    parameters = {
        'xmin': "At t=0, the left-most x-position.",
        'xd0': "At t=0, the location of the membrane separating the left and right states.",
        'xmax': "At t=0, the right-most x-position.",
        't': "The end time.",
        'pl': "The left-state initial pressure.",
        'rl': "The left-state initial density.",
        'ul': "The left-state initial velocity.",
        'gl': "The left-state adiabatic index.",
        'pr': "The right-state initial pressure.",
        'rr': "The right-state initial density.",
        'ur': "The right-state initial velocity.",
        'gr': "The right-state adiabatic index.",
        'A': "For a JWL EOS, the variable A.",
        'B': "For a JWL EOS, the variable B.",
        'R1': "For a JWL EOS, the variable R1.",
        'R2': "For a JWL EOS, the variable R2.",
        'r0': "For a JWL EOS, the variable r0.",
        'e0': "For a JWL EOS, the variable e0.",
        'num_x_pts': "The number of points in the spatial array.",
        'num_int_pts': "The number of integration points across a rarefaction state.",
        'int_tol': "The integration tolerance for integrating across a rarefaction.",
        'problem': "Flag/switch for defining mathematical function calls when integrating across rarefaction states. Default is 'igeos'; 'JWL' is currently an option."
    }

    xmin = 0.
    xd0  = 0.5
    xmax = 1.
    t    = 0.25
    rl   = 1.
    ul   = 0.
    pl   = 1.
    gl   = 1.4
    rr   = 0.125
    ur   = 0.
    pr   = 0.1
    gr   = 1.4
    A    = 0
    B    = 0
    R1   = 0
    R2   = 0
    r0   = 0
    e0   = 0
    problem = 'igeos'
    num_int_pts = 10001
    num_x_pts = 10001
    int_tol = 1.e-12
    # I do not allow L to be set in the code, but this could easily be changed
# Should we still be carrying around L and geometry?
    L = 1 # cm
    geometry = 1 # convergence study requires us to set geometry

    def __init__(self, **kwargs):
        """Set default values if necessary and check for valid inputs.
        """
        super().__init__(**kwargs)

    @print_when_verbose
    def _run(self, x, t):
        self.t = t
        prob = riemann.RiemannIGEOS(
               xmin = self.xmin,
               xd0 = self.xd0,
               xmax = self.xmax,
               t = t,
               rl = self.rl,
               ul = self.ul,
               pl = self.pl,
               gl = self.gl,
               rr = self.rr,
               ur = self.ur,
               pr = self.pr,
               gr = self.gr,
               A = self.A,
               B = self.B,
               R1 = self.R1,
               R2 = self.R2,
               r0 = self.r0,
               e0 = self.e0,
               problem = self.problem,
               num_int_pts = self.num_int_pts,
               num_x_pts = self.num_x_pts,
               int_tol = self.int_tol)
        prob.driver(x)

        self.x = prob.x
        self.p = prob.p
        self.r = prob.r
        self.u = prob.u
        self.e = prob.e
        self.Vregs = prob.Vregs
        self.soln_type = prob.soln_type

        pressure = interp(x, self.x, self.p)
        density = interp(x, self.x, self.r)
        velocity = interp(x, self.x, self.u)
        sie = interp(x, self.x, self.e)

        return ExactSolution([x, pressure, density, velocity, sie],
                             names=['position',
                                    'pressure',
                                    'density',
                                    'velocity',
                                    'specific_internal_energy'])


    def _streakplot(self, soln, xs, t, N=21, var_str='pressure'):
        X, T = mgrid[xs[0]:xs[-1]:complex(0,N), 0:t:complex(0,N)]
        T[:,0] += T[0,:][1] / T[0,:][-1] * 1.e-4
        Z = [interp(x, (xs - self.xd0) * t / T[0][-1], soln[var_str])
             for x in (X[:,0] - self.xd0) for t in T[0]]
        Z = array(Z)
        Z.resize(N,N)
        zmin = min(Z[0])
        zmin = [min(min(z), zmin) for z in Z]
        zmin = min(zmin)
        zmax = max(Z[0])
        zmax = [max(max(z), zmax) for z in Z]
        zmax = max(zmax)
        fig, ax = plt.subplots(1,1)
        c = ax.pcolor(X, T, Z, shading='auto', vmin=zmin, vmax=zmax)
        morphology = self.soln_type.split('-')[-1]
        ii = 0
        if (morphology[0] == 'R'):
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], '--k')
            ii += 1
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], '--k')
        elif (morphology[0] == 'S'):
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], 'k')
        ii += 1
        plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], ':k')
        ii += 1
        if (morphology[2] == 'R'):
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], '--k')
            ii += 1
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], '--k')
        elif (morphology[2] == 'S'):
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], 'k')
        plt.xlim((xs[0], xs[-1]))
        plt.ylim((0., t))
        plt.xlabel('position')
        plt.ylabel('time')
        plt.title(' '.join(var_str.split('_')))
        fig.colorbar(c, ax=ax)
        plt.show()


class GenEOS_Solver(ExactSolver):
    r"""Computes the semi-analytic solution to the Riemann problem for a
        general EOS. See [MenikoffPlohr1989]_ for the solution description. The
        problem default values are for the Sod shocktube, which is also Riemann
        problem #1.

        Default values are :math:`xmin=0, xd0=0.5, xmax=1, t=0.25, \rho_l=1, u_l=0, p_l=1, \gamma_l=1.4, \rho_r=0.125, u_r=0, p_r=0.1, \gamma_r=1.4`.
    """

    parameters = {
        'xmin': "At t=0, the left-most x-position.",
        'xd0': "At t=0, the location of the membrane separating the left and right states.",
        'xmax': "At t=0, the right-most x-position.",
        't': "The end time.",
        'pl': "The left-state initial pressure.",
        'rl': "The left-state initial density.",
        'ul': "The left-state initial velocity.",
        'gl': "The left-state adiabatic index.",
        'pr': "The right-state initial pressure.",
        'rr': "The right-state initial density.",
        'ur': "The right-state initial velocity.",
        'gr': "The right-state adiabatic index.",
        'A': "For a JWL EOS, the variable A.",
        'B': "For a JWL EOS, the variable B.",
        'R1': "For a JWL EOS, the variable R1.",
        'R2': "For a JWL EOS, the variable R2.",
        'r0': "For a JWL EOS, the variable r0.",
        'e0': "For a JWL EOS, the variable e0.",
        'num_x_pts': "The number of points in the spatial array.",
        'num_int_pts': "The number of integration points across a rarefaction state.",
        'int_tol': "The integration tolerance for integrating across a rarefaction.",
        'problem': "Flag/switch for defining mathematical function calls when integrating across rarefaction states. Default is 'igeos'; 'JWL' is currently an option."
    }

    xmin = 0.
    xd0  = 0.5
    xmax = 1.
    t    = 0.25
    rl   = 1.
    ul   = 0.
    pl   = 1.
    gl   = 1.4
    rr   = 0.125
    ur   = 0.
    pr   = 0.1
    gr   = 1.4
    A    = 0
    B    = 0
    R1   = 0
    R2   = 0
    r0   = 0
    e0   = 0
    problem = 'igeos'
    num_int_pts = 10001
    num_x_pts = 10001
    int_tol = 1.e-12
    # I do not allow L to be set in the code, but this could easily be changed
# Should we still be carrying around L and geometry?
    L = 1 # cm
    geometry = 1 # convergence study requires us to set geometry

    def __init__(self, **kwargs):
        """Set default values if necessary and check for valid inputs.
        """
        super().__init__(**kwargs)

    @print_when_verbose
    def _run(self, x, t):
        self.t = t
        # instantiate the GenEOS solver, and 'drive' the solver (below)
        prob = riemann.RiemannGenEOS(
               xmin = self.xmin,
               xd0 = self.xd0,
               xmax = self.xmax,
               t = t,
               rl = self.rl,
               ul = self.ul,
               pl = self.pl,
               gl = self.gl,
               rr = self.rr,
               ur = self.ur,
               pr = self.pr,
               gr = self.gr,
               A = self.A,
               B = self.B,
               R1 = self.R1,
               R2 = self.R2,
               r0 = self.r0,
               e0 = self.e0,
               problem = self.problem,
               num_int_pts = self.num_int_pts,
               num_x_pts = self.num_x_pts,
               int_tol = self.int_tol)
        prob.driver()

        self.x = prob.x
        self.p = prob.p
        self.r = prob.r
        self.u = prob.u
        self.e = prob.e
        self.Vregs = prob.Vregs
        self.soln_type = prob.soln_type

        pressure = interp(x, self.x, self.p)
        density = interp(x, self.x, self.r)
        velocity = interp(x, self.x, self.u)
        sie = interp(x, self.x, self.e)

        return ExactSolution([x, pressure, density, velocity, sie],
                             names=['position',
                                    'pressure',
                                    'density',
                                    'velocity',
                                    'specific_internal_energy'])


    def _streakplot(self, soln, xs, t, N=21, var_str='pressure'):
        X, T = mgrid[xs[0]:xs[-1]:complex(0,N), 0:t:complex(0,N)]
        T[:,0] += T[0,:][1] / T[0,:][-1] * 1.e-4
        Z = [interp(x, (xs - self.xd0) * t / T[0][-1], soln[var_str])
             for x in (X[:,0] - self.xd0) for t in T[0]]
        Z = array(Z)
        Z.resize(N,N)
        zmin = min(Z[0])
        zmin = [min(min(z), zmin) for z in Z]
        zmin = min(zmin)
        zmax = max(Z[0])
        zmax = [max(max(z), zmax) for z in Z]
        zmax = max(zmax)
        fig, ax = plt.subplots(1,1)
        ax.set_aspect('equal', adjustable='box')
        c = ax.pcolor(X, T, Z, shading='auto', vmin=zmin, vmax=zmax)
        morphology = self.soln_type.split('-')[-1]
        ii = 0
        if (morphology[0] == 'R'):
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], '--k')
            ii += 1
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], '--k')
        elif (morphology[0] == 'S'):
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], 'k')
        ii += 1
        plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], ':k')
        ii += 1
        if (morphology[2] == 'R'):
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], '--k')
            ii += 1
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], '--k')
        elif (morphology[2] == 'S'):
            plt.plot([self.xd0, self.Vregs[ii]*t + self.xd0], [0., t], 'k')
        plt.xlim((xs[0], xs[-1]))
        plt.ylim((0., t))
        plt.xlabel('position')
        plt.ylabel('time')
        plt.title(' '.join(var_str.split('_')))
        fig.colorbar(c, ax=ax)
        plt.show()
