r"""A Cog8 solver in Fortran implemented by Frank Timmes.

This is a Python interface to the Riemann solution code from `Frank
Timmes website <http://cococubed.asu.edu/code_pages/cog8.shtml>`_.
Timmes' code is released under LA-CC-05-101. The exact solution is
given in the cog8 solver.

"""

from ...base import ExactSolver, ExactSolution
from _timmes import cog8_timmes


class Cog8Timmes(ExactSolver):
    """ Computes the solution to the Cog8 problem.
    """
    
    parameters = {
        'rho0': 'initial density of the gas',
        'temp0': 'temperature of the gas',
        'alpha': r'dimensionless constant :math:`\alpha` in Eq. :eq:`lambdaDef`',
        'beta': r'dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`',
        'gamma': 'ratio of specific heats :math:`\gamma \equiv c_p/c_v`',
        'cv': 'specific heat at constant volume [erg/g/eV]',
        }

    rho0 = 3.0
    temp0 = 2000.0
    alpha = -1.0
    beta = 2.0
    gamma = 5.0 / 3.0
    cv = 1.0

    def _run(self, r, t):

        den, tev, ener, pres, vel = cog8_timmes(t=t,
                                         r=r,
                                         rho0=self.rho0,
                                         temp0=self.temp0,
                                         alpha=self.alpha,
                                         beta=self.beta,
                                         gamma=self.gamma,
                                         cv=self.cv)

        return ExactSolution([r, den, tev, ener, pres, vel],
                             names=['position',
                                    'density',
                                    'temperature',
                                    'sie',
                                    'pressure',
                                    'velocity'])
