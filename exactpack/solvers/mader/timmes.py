r"""A Fortran based Mader solver.

This is a Python interface to the Mader rarefaction solution code from
`Frank Timmes' website <http://cococubed.asu.edu/research_pages/mader.shtml>`_.
Timmes' solution code is released under LA-CC-05-101.

For a CJ detonation speed of 0.8 cm/s, it takes 6.25 :math:`\mu` s to
travel 5 cm. This time has been hardwired into Timmes' code.
"""

from ...base import ExactSolver, ExactSolution
from .rarefaction import mader


class Mader(ExactSolver):
    """ Computes the solution to the Mader problem.
    """

    parameters = {
        'p_cj': 'Chapman-Jouget pressure',
        'd_cj': 'Chapman-Jouget density',
        'gamma': r'ratio of specific heats :math:`\gamma \equiv c_p/c_v`',
        'u_piston': 'speed of piston',
        }

    p_cj = 3.0e11  # 0.3 Mbar
    d_cj = 8.0e5   # 0.8 cm/us
    gamma = 3.0
    u_piston = 0.0
#
# IC from Fig. 14 LA-UR-05-6865
# 5cm slab, gamma=3, temp=0.025 eV, 1.875 g/cc, 0.8 cm/us
# It takes 6.25 us for detnation to reach x=0.
#
    def _run(self, r, t):

        u, p, c, rho, xdet = mader(t=t,
                                   x=r,
                                   p_cj=self.p_cj,
                                   d_cj=self.d_cj,
                                   gamma=self.gamma,
                                   u_piston=self.u_piston)

        return ExactSolution([r, u, p, c, rho, xdet],
                             names=['position',
                                    'velocity',
                                    'pressure',
                                    'sound',
                                    'density',
                                    'xdet'])
