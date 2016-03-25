"""A Fortran based Riemann solver from Jim Kamm.
"""

from ...base import ExactSolver, ExactSolution
from _kamm import riemann_kamm


class Riemann(ExactSolver):
    """ Computes the solution to the Riemann problem.
    """

    parameters = {
        'interface_loc': 'initial interface location :math:`r_0`',
        'gammal': 'left specific heat ratio :math:`\gamma \equiv c_p/c_v`',
        'gammar': 'right specific heat ratio :math:`\gamma \equiv c_p/c_v`',
        'rhol': 'density on left in Eq. :eq:`riemannL`',
        'pl': 'pressure on left in Eq. :eq:`riemannL`',
        'ul': 'velocity on left in Eq. :eq:`riemannL`',
        'rhor': 'density on right in Eq. :eq:`riemannR`',
        'pr': 'pressure on right in Eq. :eq:`riemannR`',
        'ur': 'velocity on right in Eq. :eq:`riemannR`',
        }

    geometry = 1
    gammal = 1.4
    gammar = 1.4
    interface_loc = 0.5
    rhol = 1.0
    pl = 1.0
    ul = 0.0
    rhor = 0.125
    pr = 0.1
    ur = 0.0

    def _run(self, r, t):

        rho, p, u, sound, sie, entropy = riemann_kamm(time=t,
                                                 x=r,
                                                 xd0=self.interface_loc,
                                                 gammal=self.gammal,
                                                 gammar=self.gammar,
                                                 rhol=self.rhol,
                                                 pl=self.pl,
                                                 ul=self.ul,
                                                 rhor=self.rhor,
                                                 pr=self.pr,
                                                 ur=self.ur)

        return ExactSolution([r, rho, p, u, sound, sie, entropy],
                             names=['position',
                                    'density',
                                    'pressure',
                                    'velocity_x',
                                    'sound',
                                    'energy',
                                    'entropy'])
