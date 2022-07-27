r"""A Cog14 solver in Python.

This is a pure Python implementation of the Cog14 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= 
  \rho_0 \, r^{-k - b}
  \\
  u(r,t) &= \left(\frac{\Gamma T_0 (k - b)}{b}\right)^{1/2} \cdot r^b
  \\
  T(r,t) &= T_0 \, r^{2 b}
  \\
  b &= \frac{k - 1 - \alpha k}{2 + \alpha - 2(\beta + 4)}  
  \\
  T_0 &=  
  \left[ \frac{b}{\Gamma(k - b)} \left( \frac{4 c a \lambda_0}{3}\,
  \frac{\gamma - 1}{\Gamma} \right)^2 \cdot
  \frac{16 \, \rho_0^{2\alpha - 2} \, b^4}{\left[ 
  2 b + (\gamma - 1)(k + b)\right]^2}
  \right]^{-1/(5 + 2 \beta)}

Free parameters: :math:`\alpha`, :math:`\beta` :math:`k`, :math:`\rho_0`,
:math:`\lambda_0`, and :math:`\Gamma`. 

"""

import numpy as np
from math import sqrt

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog14(ExactSolver):
    """Computes the solution to the Cog14 problem.
    
    Computes the solution to the Cog14 problem with defaults geometry = 3,
    gamma = 1.4, rho0 = 1.8, alpha = 2.0, beta = 1.0, lambda0 = 0.1, Gamma = 40.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'rho0': "density coefficient",
        'alpha': r"dimensionless constant :math:`\alpha` in Eq. :eq:`lambdaDef`",
        'beta': r"dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`",
        'lambda0': r"constant :math:`\lambda_0` in Eq. :eq:`lambdaDef`",
        'Gamma': "|Gruneisen| gas parameter",
        }

    geometry = 3
    gamma = 1.4
    rho0 = 1.8
    alpha = 2.0
    beta = 1.0
    lambda0 = 0.1
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog14, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.alpha < -2.0 or self.alpha > -1.0:
            print("*** warning: alpha lies outside range [-2,-1] ***")
        if self.beta < 1.0 or self.beta > 3.0:
            print("*** warning: beta lies outside range [1,3] ***")

    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1.
        c = 2.997e10    # speed of light [cm/s]
        a = 1.3720e+02  # erg cm^-3 ev^-4
                        # a = 7.5657e-15 erg cm^-3 K^-4
                        #   = 1.3720e+02 erg cm^-3 ev^-4 using k_B
                        #   = 8.6173324e-5 eV K^-1
        b = (k - 1 - self.alpha * k) / (2 + self.alpha - 2 * (self.beta + 4))
        c1 = -k - b
        temp0 = b / bigGamma / (k - b)
        x1 = 4 * c * self.lambda0 * a * (self.gamma - 1) / 3 / bigGamma
        temp0 = temp0 * pow(x1, 2)
        x2 = 2 * b + (self.gamma - 1) * (k + b)
        x3 = pow(self.rho0, 2 * self.alpha - 2) * 16 * pow(b, 4) / pow(x2, 2)
        temp0 = temp0 * x3
        x4 = -1. / (5 + 2 * self.beta)
        temp0 = pow(temp0, x4)
        c2 = sqrt(bigGamma * temp0 * (k - b) / b)

        density = self.rho0 * pow(r, c1) * np.ones(shape=r.shape)
        velocity = c2 * pow(r, b) * np.ones(shape=r.shape)
        temperature = temp0 * pow(r, 2 * b) * np.ones(shape=r.shape)  # [eV]
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


class PlanarCog14(Cog14):
    """The planar Cog14 problem.
    """

    parameters = {
        'gamma': Cog14.parameters['gamma'],
        'rho0': Cog14.parameters['rho0'],
        'alpha': Cog14.parameters['alpha'],
        'beta': Cog14.parameters['beta'],
        'lambda0': Cog14.parameters['lambda0'],
        'Gamma': Cog14.parameters['Gamma'],  
        }
    geometry = 1


class CylindricalCog14(Cog14):
    """The cylindrical Cog14 problem.
    """

    parameters = {
        'gamma': Cog14.parameters['gamma'],
        'rho0': Cog14.parameters['rho0'],
        'alpha': Cog14.parameters['alpha'],
        'beta': Cog14.parameters['beta'],
        'lambda0': Cog14.parameters['lambda0'],
        'Gamma': Cog14.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog14(Cog14):
    """The spherical Cog14 problem.
    """

    parameters = {
        'gamma': Cog14.parameters['gamma'],
        'rho0': Cog14.parameters['rho0'],
        'alpha': Cog14.parameters['alpha'],
        'beta': Cog14.parameters['beta'],
        'lambda0': Cog14.parameters['lambda0'],
        'Gamma': Cog14.parameters['Gamma'],  
        }
    geometry = 3

