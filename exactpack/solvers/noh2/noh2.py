r"""Noh's 2nd solution in Python.

This is a pure Python implementation of the Noh2 solution using Numpy.
The package includes specialized classes for specific geometries.
"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class Noh2(ExactSolver):

    r"""Computes the solution to the general Noh2 problem.

    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'rho0': "initial density",
        'e0': "initial internal energy",
        }

    geometry = 3
    gamma = 5.0 / 3.0
    rho0 = 1.0
    e0 = 1.0

    def __init__(self, **kwargs):

        super(Noh2, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

    def _run(self, r, t):

        if t > 1:
            raise ValueError("The time t must be less than 1")
        
        delta = self.geometry
        density = np.ones(shape=r.shape) * self.rho0 / (1 - t)**delta
        velocity = -np.ones(shape=r.shape) * r / (1 - t)
        sie = np.ones(shape=r.shape) * self.e0 / (1 - t)**((self.gamma - 1) * delta)
        pressure = np.ones(shape=r.shape) * density * sie * (self.gamma - 1)

        return ExactSolution([r, density, pressure, sie, velocity],
                             names=['position',
                                    'density',
                                    'pressure',
                                    'specific_internal_energy',
                                    'velocity'],
        )


class PlanarNoh2(Noh2):
    """The standard planar Noh2 problem.

    The planar Noh2 problem as defined in [Noh1987]_, with a default value
    of :math:`\gamma=5/3`.
    """

    parameters = {'gamma':  Noh2.parameters['gamma'],
                  'rho0': Noh2.parameters['rho0'],
                  'e0': Noh2.parameters['e0']
                  }
    geometry = 1
    gamma = 5.0 / 3.0
    rho0 = 1
    e0 = 1


class CylindricalNoh2(Noh2):
    """The standard cylindrical Noh2 problem.

    The cylindrical Noh2 problem as defined in [Noh1987]_, with a default
    value of :math:`\gamma=5/3`.
    """

    parameters = {'gamma':  Noh2.parameters['gamma'],
                  'rho0': Noh2.parameters['rho0'],
                  'e0': Noh2.parameters['e0']
                  }
    geometry = 2
    gamma = 5.0 / 3.0
    rho0 = 1
    e0 = 1

class SphericalNoh2(Noh2):
    """The standard spherical Noh2 problem.

    The spherical Noh2 problem as defined in [Noh1987]_, with a default
    value of :math:`\gamma=5/3`.
    """

    parameters = {'gamma':  Noh2.parameters['gamma'],
                  'rho0': Noh2.parameters['rho0'],
                  'e0': Noh2.parameters['e0']
                  }
    geometry = 3
    gamma = 5.0 / 3.0
    rho0 = 1
    e0 = 1
