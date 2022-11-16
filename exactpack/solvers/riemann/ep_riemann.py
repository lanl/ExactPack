r""" ExactPack wrapper for the Riemann solvers.
"""

from exactpack.base import ExactSolver, ExactSolution

from exactpack.solvers.riemann import riemann
from numpy import interp

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
        super(IGEOS_Solver, self).__init__(**kwargs)


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

        pressure = interp(x, self.x, self.p)
        density = interp(x, self.x, self.r)
        velocity = interp(x, self.x, self.u)
        sie = interp(x, self.x, self.e)

        return ExactSolution([x, pressure, density, velocity, sie],
                             names=['position',
                                    'pressure',
                                    'density',
                                    'velocity',
                                    'sie'])


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
        super(GenEOS_Solver, self).__init__(**kwargs)


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

        pressure = interp(x, self.x, self.p)
        density = interp(x, self.x, self.r)
        velocity = interp(x, self.x, self.u)
        sie = interp(x, self.x, self.e)

        return ExactSolution([x, pressure, density, velocity, sie],
                             names=['position',
                                    'pressure',
                                    'density',
                                    'velocity',
                                    'sie'])
