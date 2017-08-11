r"""A Cog19 solver in Python.

This is a pure Python implementation of the Cog19 solution using Numpy.
The solution contains a shock located at

.. math::

  R(t) = - \frac{1}{2} (\gamma - 1)\, u_0 t

Region 1:

.. math::

  \rho(r,t) &= \rho_0  \left(\frac{\gamma + 1}{\gamma - 1}\right)^{k + 1}
  \\
  u(r,t) &= 0
  \\
  T(r,t) &= \frac{u_0^2 \,(\gamma - 1)}{2 \Gamma}

Region 2:

.. math::

  \rho(r,t) &= \rho_0  \left( \frac{r - u_0 t}{r} \right)^k
  \\
  u(r,t) &= u_0 
  \\
  T(r,t) &= 0

Free parameters: :math:`k`, :math:`\rho_0`, and :math:`u_0` (with
:math:`u_0 < 0`), :math:`\gamma`, and :math:`\Gamma`.

"""

import numpy as np
from math import sqrt

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog19(ExactSolver):
    """Computes the solution to the Cog19 problem. No conduction.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'rho0': "density coefficient",
        'u0': "velocity coefficient",
        'Gamma': "Gruneisen gas parameter",
        }
    geometry = 3
    gamma = 1.4
    rho0 = 1.8
    u0 = -2.3
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog19, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.u0 > 0:
            raise ValueError("u0 must be strictly negative")
                
    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1.
        shock_location = -(self.gamma - 1) * self.u0 * t / 2
        density = self.rho0 * np.where(r < shock_location,
                           pow((self.gamma + 1) / (self.gamma - 1), k + 1),
                           pow((r - self.u0 * t) / r, k)) * \
                           np.ones(shape=r.shape)
        velocity = np.where(r < shock_location, 0, self.u0) * \
            np.ones(shape=r.shape)
        temperature = np.where(r < shock_location,
                               pow(self.u0, 2) * (self.gamma - 1) /
                               (2 * bigGamma), 0) * np.ones(shape=r.shape)
        pressure = bigGamma * density * temperature
        sie = pressure / density / (self.gamma - 1)

        return ExactSolution([r, density, velocity, temperature, pressure,
                             sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'temperature',
                                    'pressure',
                                    'specific_internal_energy'])
    
class PlanarCog19(Cog19):
    """The planar Cog19 problem.
    """

    parameters = {
        'gamma': Cog19.parameters['gamma'],
        'rho0': Cog19.parameters['rho0'],
        'u0': Cog19.parameters['u0'],
        'Gamma': Cog19.parameters['Gamma'],  
        }    
    geometry = 1


class CylindricalCog19(Cog19):
    """The cylindrical Cog19 problem.
    """

    parameters = {
        'gamma': Cog19.parameters['gamma'],
        'rho0': Cog19.parameters['rho0'],
        'u0': Cog19.parameters['u0'],
        'Gamma': Cog19.parameters['Gamma'],  
        }    
    geometry = 2


class SphericalCog19(Cog19):
    """The spherical Cog19 problem.
    """

    parameters = {
        'gamma': Cog19.parameters['gamma'],
        'rho0': Cog19.parameters['rho0'],
        'u0': Cog19.parameters['u0'],
        'Gamma': Cog19.parameters['Gamma'],  
        }    
    geometry = 3
