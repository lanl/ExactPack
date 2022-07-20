r"""A Cog18 solver in Python.

This is a pure Python implementation of the Cog18 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^{-(2 \beta + k + 7)/\alpha} \cdot
  \Big(\tau^2 - t^2 \Big)^{-(k + 1)/2 + (2 \beta + k + 7)/2\alpha}
  \\
  u(r,t) &= - \frac{r t}{\tau^2 - t^2}
  \\
  T(r,t) &= \frac{\alpha \tau^2}{\Gamma (2 \alpha - 2 \beta - k - 7)}
  \cdot \frac{r^2}{(\tau^2 - t^2)^2}

with

.. math::

   \gamma = \frac{k+3}{k+1} \ .

Free parameters: :math:`\alpha`, :math:`\beta`, :math:`k`, :math:`\tau`,
:math:`\rho_0`, and :math:`\Gamma`.

"""

import numpy as np
from math import sqrt

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog18(ExactSolver):
    """Computes the solution to the Cog18 problem.
       Note: choose alpha, beta, tau correctly.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'alpha': r"dimensionless constant :math:`\alpha` in Eq. :eq:`lambdaDef`",
        'beta': r"dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`",
        'rho0': "density coefficient",
        'tau': "free parameter of dimension time",
        'Gamma': "Gruneisen gas parameter",
        }
    geometry = 3
    alpha = 2.0
    beta = 1.0
    rho0 = 1.8
    tau = 1.25
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog18, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.alpha in [0]:
            raise ValueError("alpha cannot equal 0")

        if self.alpha < -2.0 or self.alpha > -1.0:
            print("*** warning: alpha lies outside range [-2,-1] ***")
        if self.beta < 1.0 or self.beta > 3.0:
            print("*** warning: beta lies outside range [1,3] ***")

    def _run(self, r, t):

        k = self.geometry - 1.
        gamma = (k + 3) / (k + 1)
        bigGamma = self.Gamma
        c1 = -(2 * self.beta + k + 7) / self.alpha
        c2 = -(k + 1) / 2
        c3 = c2 - c1 / 2
        x1 = pow(self.tau, 2) - pow(t, 2)
        temp0 = self.alpha * pow(self.tau, 2) / bigGamma / \
                (2 * self.alpha - 2 * self.beta - k - 7)
        
        density = self.rho0 * pow(r, c1) * pow(x1, c3) * np.ones(shape=r.shape)
        velocity = - (r * t / x1) * np.ones(shape=r.shape)
        temperature = temp0 * (pow(r, 2) / pow(x1, 2)) * np.ones(shape=r.shape)
        pressure = bigGamma * density * temperature
        sie = pressure / density / (gamma - 1)

        return ExactSolution([r, density, velocity, temperature, pressure,
                             sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'temperature',
                                    'pressure',
                                    'specific_internal_energy'])

    
class PlanarCog18(Cog18):
    """The planar Cog18 problem.
    """

    parameters = {
        'alpha': Cog18.parameters['alpha'],
        'beta': Cog18.parameters['beta'],
        'rho0': Cog18.parameters['rho0'],
        'tau': Cog18.parameters['tau'],
        'Gamma': Cog18.parameters['Gamma'],  
        }
    geometry = 1


class CylindricalCog18(Cog18):
    """The cylindrical Cog18 problem.
    """

    parameters = {
        'alpha': Cog18.parameters['alpha'],
        'beta': Cog18.parameters['beta'],
        'rho0': Cog18.parameters['rho0'],
        'tau': Cog18.parameters['tau'],
        'Gamma': Cog18.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog18(Cog18):
    """The spherical Cog18 problem.
    """

    parameters = {
        'alpha': Cog18.parameters['alpha'],
        'beta': Cog18.parameters['beta'],
        'rho0': Cog18.parameters['rho0'],
        'tau': Cog18.parameters['tau'],
        'Gamma': Cog18.parameters['Gamma'],  
        }
    geometry = 3
