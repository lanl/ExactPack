r"""A Cog16 solver in Python.

This is a pure Python implementation of the Cog16 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^{-k - b}
  \\
  u(r,t) &= u_0 \, r^b
  \\
  T(r,t) &= \frac{u_0^2 b}{\Gamma (k - b)} \cdot t^{2b}
  \\
  \rho_0 &= \left( \frac{16 c a \lambda_0 (\gamma - 1)}{3}\right)^k
  \cdot
  \\&
  \frac{b^{(5k - 1)/2}}{ u_0 (k - b)^{(k - 1)/2} \Gamma^{(3k - 1)/2} 
  \Big[2b + (\gamma - 1) (k + b) \Big]^k }
  \\
  \alpha &= 1 - \frac{1}{k}  ~~~\beta = \frac{1}{2}\,\alpha - 3 ~~(k \ne 0)

Free parameters: :math:`b`, :math:`k` and :math:`u_0`, :math:`\gamma`,
:math:`\lambda_0`, and :math:`\Gamma`.

"""

import numpy as np
from math import sqrt

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog16(ExactSolver):
    """Computes the solution to the Cog16 problem.
       Incomplete: We need to solve for beta as a function of alpha.
    """

    parameters = {
        'geometry': "2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'u0': "velocity coefficient",
        'b': r"dimensionless constant",
        'lambda0': r"constant :math:`\lambda_0` in Eq. :eq:`lambdaDef`",
        'Gamma': "Gruneisen gas parameter",
        }
    geometry = 3
    gamma = 1.4
    u0 = 2.3
    b = 1.2
    lambda0 = 0.1
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog16, self).__init__(**kwargs)

        if self.geometry not in [2, 3]:
            raise ValueError("geometry must be 2, or 3")

    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1.
        c = 2.997e10     # speed of light [cm/s]
        a = 1.3720e+02   # erg cm^-3 ev^-4
                         # a   = 7.5657e-15 erg cm^-3 K^-4
                         #     = 1.3720e+02 erg cm^-3 ev^-4
                         # k_B = 8.6173324e-5 eV K^-1
        alpha = 1. - 1 / k
        beta = alpha / 2. - 3.
        c1 = -k - self.b
        c2 = (5 * k - 1) / 2
        c3 = (k - 1) / 2
        c4 = (3 * k - 1) / 2
        x1 = 2 * self.b + (self.gamma - 1) * (k + self.b)
        temp0 = pow(self.u0, 2) * self.b / bigGamma / (k - self.b)
        rho0 = 16 * c * self.lambda0 * a * (self.gamma - 1) / 3
        rho0 = pow(rho0, k)
        rho0 = rho0 * pow(self.b, c2) / (self.u0 * pow(k - self.b, c3) * \
               pow(bigGamma, c4) * pow(x1, k))

        density = rho0 * pow(r, c1) * np.ones(shape=r.shape)
        velocity = self.u0 * pow(r, self.b) * np.ones(shape=r.shape)
        temperature = temp0 * pow(r, 2 * self.b) * np.ones(shape=r.shape)
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



class CylindricalCog16(Cog16):
    """The cylindrical Cog16 problem.
    """

    parameters = {
        'gamma': Cog16.parameters['gamma'],
        'u0': Cog16.parameters['u0'],
        'b': Cog16.parameters['b'],
        'lambda0': Cog16.parameters['lambda0'],
        'Gamma': Cog16.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog16(Cog16):
    """The spherical Cog16 problem.
    """

    parameters = {
        'gamma': Cog16.parameters['gamma'],
        'u0': Cog16.parameters['u0'],
        'b': Cog16.parameters['b'],
        'lambda0': Cog16.parameters['lambda0'],
        'Gamma': Cog16.parameters['Gamma'],  
        }
    geometry = 3
