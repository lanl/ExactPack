"""A Fortran based Riemann solver.

This is a Python interface to the Riemann solution code from `Frank Timmes'
website <http://cococubed.asu.edu/code_pages/exact_riemann.shtml>`_.
"""

from ...base import ExactSolver, ExactSolution
from _timmes import riemann


class Riemann(ExactSolver):
    """ Computes the solution to the Riemann problem.
    """

    parameters = {
        'interface_loc': 'initial interface location :math:`r_0`',
        'gamma': 'specific heat ratio :math:`\gamma \equiv c_p/c_v`',
        'rhol': 'density on left in Eq. :eq:`riemannL`',
        'pl': 'pressure on left in Eq. :eq:`riemannL`',
        'ul': 'velocity on left in Eq. :eq:`riemannL`',
        'rhor': 'density on right in Eq. :eq:`riemannR`',
        'pr': 'pressure on right in Eq. :eq:`riemannR`',
        'ur': 'velocity on right in Eq. :eq:`riemannR`',
        }

    geometry = 1
    gamma = 1.4
    interface_loc = 0.5
    rhol = 1.0
    pl = 1.0
    ul = 0.0
    rhor = 0.125
    pr = 0.1
    ur = 0.0

    def _run(self, r, t):

        rho, p, u = riemann(t=t,
                            x=r,
                            xi=self.interface_loc,
                            gamma=self.gamma,
                            rhol=self.rhol,
                            pl=self.pl,
                            ul=self.ul,
                            rhor=self.rhor,
                            pr=self.pr,
                            ur=self.ur
        )

        return ExactSolution([r, rho, p, u],
                             names=['position',
                                    'density',
                                    'pressure',
                                    'velocity'
                                ]
        )
