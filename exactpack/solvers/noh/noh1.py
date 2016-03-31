r"""A Noh solver in Python.

This is a pure Python implementation of the Noh solution using Numpy.
The package includes specialized classes for specific geometries.
"""

import numpy as np

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Noh(ExactSolver):

    r"""Computes the solution to the general Noh problem.

    This is a base class with default values, which can be used to solve the Noh
    problem for a perfect-gas with any specific heat ratio, :math:`\gamma`.  It
    supports one-dimensional planar, two-dimensional cylindrical, and
    three-dimensional spherical shocks. The default values for the parameters
    are :math:`\rho_0=1` and :math:`u_0=1`, although the user is free to
    set these parameters. The default geometry is spherical, with 
    :math:`\gamma=5/3`.

    The solver reports jumps in the
    :attr:`exactpack.base.ExactSolution.jumps` attribute of the return value.

    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'u0': "incident velocity (negative)",
        'rho0': "density"
        }

    geometry = 3
    gamma = 5.0 / 3.0
    u0 = -1.0
    rho0 = 1.0

    def __init__(self, **kwargs):

        super(Noh, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.u0 >= 0:
            raise ValueError("Incident velocity must be negative")

    def _run(self, r, t):

        shock_location = abs(self.u0) * t * (self.gamma - 1) / 2

        density = np.where(r < shock_location,
                           self.rho0 * ((self.gamma + 1) / (self.gamma - 1)) ** \
                           self.geometry * np.ones(shape=r.shape),
                           self.rho0 * (1 + abs(self.u0) * t / r) ** (self.geometry - 1))
        pressure = np.where(r < shock_location,
                            (self.rho0 * self.u0 * self.u0) * 4.0 ** self.geometry / 3.0 * \
                            np.ones(shape=r.shape),
                            np.zeros(shape=r.shape))
        sie = np.where(r < shock_location,
                      (self.u0 * self.u0) * (1.0 / 2.0) * np.ones(shape=r.shape),
                       np.zeros(shape=r.shape))
        velocity = np.where(r < shock_location,
                            np.zeros(shape=r.shape),
                            self.u0 * np.ones(shape=r.shape))

        return ExactSolution([r, density, pressure, sie, velocity],
                             names=['position',
                                    'density',
                                    'pressure',
                                    'sie',
                                    'velocity'],
                             jumps=[JumpCondition(shock_location,
                                    "Shock",
                                    density=(((self.gamma + 1) / \
                                    (self.gamma - 1)) ** self.geometry,
                                    4.0 ** (self.geometry - 1)),
                                    pressure=(4.0 ** self.geometry / 3.0, 0),
                                    sie=(1.0 / 2.0, 0),
                                    velocity=(0, -1))
                                    ]
        )


class PlanarNoh(Noh):
    """The standard planar Noh problem.

    The planar Noh problem as defined in [Noh]_, with a default value
    of :math:`\gamma=5/3`.
    """

    parameters = {'gamma':  Noh.parameters['gamma']}
    geometry = 1
    gamma = 5.0 / 3.0


class CylindricalNoh(Noh):
    """The standard cylindrical Noh problem.

    The cylindrical Noh problem as defined in [Noh]_, with a default
    value of :math:`\gamma=5/3`.
    """

    parameters = {'gamma': Noh.parameters['gamma']}
    geometry = 2
    gamma = 5.0 / 3.0


class SphericalNoh(Noh):
    """The standard spherical Noh problem.

    The spherical Noh problem as defined in [Noh]_, with a default
    value of :math:`\gamma=5/3`.
    """

    parameters = {'gamma': Noh.parameters['gamma']}
    geometry = 3
    gamma = 5.0 / 3.0
