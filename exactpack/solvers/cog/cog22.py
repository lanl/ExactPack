r"""A Cog22 solver in Python.

*** not yet implemented ***

This is a pure Python implementation of the Cog22 solution using Numpy.
The solution contains a shock located at

.. math::

  R_\text{shock} = t \, r^{-k (\gamma - 1)/(\gamma + 1)} \,
  \frac{\rho_2 u_2 - \rho_1 u_1}{\rho^2 - \rho_1^2} \ ,

where

.. math::

  &\frac{1 - \gamma}{2 \gamma} \big( \rho_1 u_1^2 - \rho_2 u_2^2  \big)
  =
  \frac{\rho_1 \rho_2}{\rho_1 - \rho_2} \big( u_1 - u_2 \big)^2
  \\[5pt]
  & u_1^2 - u_2^2 = \frac{\gamma (u_1 - u_2)^2  }{(\rho_1 -
  \rho_2)^2} \ 
  (\rho_2^2 - \rho_1^2)  

Solve for :math:`u_2` and :math:`\rho_2` in terms of :math:`u_1` 
and :math:`\rho_1`.

Region 1:

.. math::

  \rho(r,t) &= \rho_1 \, r^{-2k / (\gamma + 1)}
  \\
  u(r,t) &= u_1 \, r^{-k(\gamma - 1)/(\gamma + 1)}
  \\
  T(r,t) &= \frac{u_1^2 (1 - \gamma)}{2 \gamma \Gamma} \, 
  r^{-2 k (\gamma - 1) / (\gamma + 1)}

Region 2:

.. math::

  \rho(r,t) &=  \rho_2 \, r^{-2k / (\gamma + 1)}
  \\
  u(r,t) &= u_2 \, r^{-k(\gamma - 1)/(\gamma + 1)}
  \\
  T(r,t) &=  \frac{u_2^2 (1 - \gamma)}{2 \gamma \Gamma} \, 
  r^{-2 k (\gamma - 1) / (\gamma + 1)}

Free parameters: :math:`\rho_1`, :math:`u_1`, and :math:`k`,
with :math:`T > 0` only for :math:`\gamma < 1`.

"""

import numpy as np
from math import sqrt

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog22(ExactSolver):
    """Computes the solution to the Cog22 problem.
    ** not yet implemented **
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'cv': "specific heat at constant volume",
        'rho1': "density coefficient",
        'u1': "velocity coefficient",
        }

    geometry = 3
    gamma = 5.0 / 3.0
    cv = 1.
    rho0 = 1.
    u0 = 1.

    def __init__(self, **kwargs):

        if 'geometry' in kwargs and not kwargs['geometry'] in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        super(Cog22, self).__init__(**kwargs)

    def _run(self, r, t):

        bigGamma = self.cv * (self.gamma - 1)   # gas constant
        k = self.geometry - 1.

        shock_location = 0
        density = 1
        velocity = 0
        temperature = 0 
        pressure = bigGamma * density * temperature
        sie = pressure / density / (self.gamma - 1)

        return ExactSolution([r, density, velocity, temperature, pressure,
                             sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'temperature',
                                    'pressure',
                                    'sie'])


class PlanarCog22(Cog22):
    """The planar Cog22 problem with a default value of :math:`\gamma=5/3`.
    """

    parameters = {'gamma': Cog22.parameters['gamma']}
    geometry = 1
    gamma = 5.0 / 3.0


class CylindricalCog22(Cog22):
    """The cylindrical Cog22 problem with a default value of
    :math:`\gamma=5/3`.
    """

    parameters = {'gamma': Cog22.parameters['gamma']}
    geometry = 2
    gamma = 5.0 / 3.0


class SphericalCog22(Cog22):
    """The spherical Cog22 problem with a default value of
    :math:`\gamma=5/3`.
    """

    parameters = {'gamma': Cog22.parameters['gamma']}
    geometry = 3
    gamma = 5.0 / 3.0
