"""A Fortran based Guderley solver.

This is a Python interface to Scott Ramsey's Guderley fortran code.
"""
from ...base import ExactSolver, ExactSolution
# from ._ramsey import guderley_1d
from .ramsey import guderley_1d


class Guderley(ExactSolver):
    """ Computes the solution to the Guderley problem.
    """

    parameters = {
        'geometry': '1=planar, 2=cylindrical, 3=spherical',
        'gamma': 'specific heat ratio',
        'rho0': 'initial uniform density',
        }
    geometry = 3
    gamma = 1.4
    rho0 = 1.0

    def _run(self, r, t):

        den, vel, pres, snd, sie = guderley_1d(t=t,
                                               r=r,
                                               ngeom=self.geometry,
                                               gamma=self.gamma,
                                               rho0=self.rho0)

        return ExactSolution([r, den, vel, pres, snd, sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'pressure',
                                    'sound',
                                    'sie'])
