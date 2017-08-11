r"""A Cog20 solver in Python.

This is a pure Python implementation of the Cog20 solution using Numpy.
The solution contains a shock located at

.. math::

  R_\text{shock} = \frac{u_0 (\gamma - 1)}{4 a} \, 
  \frac{t (1 - 2 a t)}{1 - a t}

Region 1:

.. math::

  \rho(r,t) &=  \rho_0  \left(\frac{\gamma + 1}
  {\gamma - 1}\right)^{k + 1} \Big(1 - a t\Big)^{-k -1}
  \\
  u(r,t) &= - \frac{a r}{1 - a t}
  \\
  T(r,t) &= \frac{u_0^2 (\gamma - 1)}{2\Gamma} 
  \Big(1 - a t \Big)^{-2}

Region 2:

.. math::

  \rho(r,t) &= \rho_0 \Big(1 - a t \Big)^{-k - 1} 
  \left(\frac{r - u_0 t}{r}\right)^k 
  \\
  u(r,t) &= \frac{u_0  - a r}{1 - a t}
  \\
  T(r,t) &= 0

Free parameters: :math:`a`, :math:`k`, :math:`u_0`, and :math:`\rho_0`,
:math:`\gamma`, and :math:`\Gamma`. 

"""

import numpy as np
from math import sqrt

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog20(ExactSolver):
    """Computes the solution to the Cog20 problem. No conduction.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'rho0': "density coefficient",
        'u0': "velocity coefficient",
        'a': "free parameter with dimensions of inverse time",
        'Gamma': "Gruneisen gas parameter",
        }
    geometry = 3
    gamma = 1.4
    rho0 = 1.8
    u0 = 2.3
    a = 0.3
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog20, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.a == 0.0:
            raise ValueError("parameter a cannot be zero")
                
    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1.
        c1 = 1 - self.a * t
        shock_location = self.u0 * (self.gamma - 1) / (4 * self.a)
        shock_location = shock_location * t * (1 - 2 * self.a * t) / c1

        density = (self.rho0 / pow(c1, k + 1)) * \
                   np.where(r < shock_location,
                   pow((self.gamma + 1) / (self.gamma - 1), k + 1),
                            pow((r - self.u0 * t) / r, k)) * \
                           np.ones(shape=r.shape)
        velocity = np.where(r < shock_location,
                            -self.a * r / c1,
                            (self.u0 - self.a * r) / c1) * \
                             np.ones(shape=r.shape)
        temperature = np.where(r < shock_location,
                               pow(self.u0, 2) * (self.gamma - 1) / \
                               (2 * bigGamma) / pow(c1, 2), 0) * \
                               np.ones(shape=r.shape)
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

    
class PlanarCog20(Cog20):
    """The planar Cog20 problem.
    """

    parameters = {
        'gamma': Cog20.parameters['gamma'],
        'rho0': Cog20.parameters['rho0'],
        'u0': Cog20.parameters['u0'],
        'a': Cog20.parameters['a'],
        'Gamma': Cog20.parameters['Gamma'],  
        }        
    geometry = 1


class CylindricalCog20(Cog20):
    """The cylindrical Cog20 problem.
    """

    parameters = {
        'gamma': Cog20.parameters['gamma'],
        'rho0': Cog20.parameters['rho0'],
        'u0': Cog20.parameters['u0'],
        'a': Cog20.parameters['a'],
        'Gamma': Cog20.parameters['Gamma'],  
        }        
    geometry = 2


class SphericalCog20(Cog20):
    """The spherical Cog20 problem.
    """

    parameters = {
        'gamma': Cog20.parameters['gamma'],
        'rho0': Cog20.parameters['rho0'],
        'u0': Cog20.parameters['u0'],
        'a': Cog20.parameters['a'],
        'Gamma': Cog20.parameters['Gamma'],  
        }        
    geometry = 3
