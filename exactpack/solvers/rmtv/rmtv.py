"""A Fortran based RMTV solver.

This is a Python interface to the Riemann solution code from `Frank
Timmes website <http://cococubed.asu.edu/research_pages/rmtv.shtml>`_.
This code is released under LA-CC-05-101.
"""

from ...base import ExactSolver, ExactSolution
# from ._timmes import rmtv
from .timmes import rmtv


class Rmtv(ExactSolver):
    """ Computes the solution to the RMTV problem.
    """
    parameters = {
        'aval': r'power :math:`a` in the thermal conductivity :eq:`chidef`',
        'bval': r'power :math:`b` in the thermal conductivity :eq:`chidef`',
        'chi0': r'coefficient :math:`\chi_0` in the thermal conductivity \
        :eq:`chidef`',
        'gamma': r'ratio of specific heats :math:`\gamma \equiv c_v/c_p`',
        'bigamma': r'The |Gruneisen| gas coefficient :math:`\Gamma` defined in \
        Eq. :eq:`BigGamma`',
        'rf': 'position of the heat front',
        'xif': 'dimensionless position of the heat front',
        'xis': 'dimensionless position of the shock front',
        'beta0': 'eigenvalue of the problem',
        'g0': 'heat front scaling parameter',
        }

    aval = -2.0
    bval = 6.5
    chi0 = 1.0
    gamma = 1.25
    bigamma = 1.0
    rf = 0.9
    xif = 2.0
    xis = 1.0
    beta0 = 7.197534e7 # LA-UR-05-6865 p. 31
    g0 = 1.0

    def _run(self, r, t=None):
        # The 't' parameter is required by the ExactPAck API but is not used
        # here. Ideally we would have some way of converting 't' to a value of
        # 'rf'.

        den, tev, ener, pres, vel = rmtv(r=r,
                                         aval_in=self.aval,
                                         bval_in=self.bval,
                                         chi0=self.chi0,
                                         gamma=self.gamma,
                                         bigamma=self.bigamma,
                                         rf=self.rf,
                                         xif_in=self.xif,
                                         xis=self.xis,
                                         beta0_in=self.beta0,
                                         g0=self.g0)

        return ExactSolution([r, den, tev, ener, pres, vel],
                             names=['position',
                                    'density',
                                    'temperature',
                                    'energy',
                                    'pressure',
                                    'velocity'])
