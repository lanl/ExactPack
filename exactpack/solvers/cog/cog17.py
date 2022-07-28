r"""A Cog17 solver in Python.

This is a pure Python implementation of the Cog17 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^{(2 \beta - 4) / (1 - \alpha)} \, t^{(2
  \beta + 5) / (\alpha - 1)}
  \\
  u(r,t) &= u_0 \,\frac{r}{t}
  \\
  T(r,t) &= T_0 \, \left( \frac{r}{t} \right)^2
  \\
  u_0 &= \frac{2 \beta + 5}{2 \beta - 4 + (1 - \alpha) (k + 1)}
  \\
  T_0 &= \frac{(\alpha - 1) (2 \beta + 5)}{\Gamma \Big[
  2 \beta - 4 + (1 - \alpha) (k + 1)\Big]^2} \cdot
  \frac{9 - (1 - \alpha) (k + 1)}{2 \beta - 4 + 2 (1 - \alpha)}
  \\
  \rho_0 &=   \left[ \frac{3 \Gamma}{4 c a \lambda_0 (\gamma - 1)} \,
  T_0^{-\beta - 3} \cdot
  \frac{1}{2}\,\frac{-2 + u_0 \left[2 + (\gamma - 1) (k + 1)
  \right]}{\alpha (2 \beta - 4) / (1 - \alpha) + 
  2 \beta + k + 7}
  \right]^{1/(\alpha - 1)}

Free parameters: :math:`\alpha`, :math:`\beta`, :math:`k`, :math:`\lambda_0`,
and :math:`\Gamma`.

"""

import numpy as np
from math import sqrt

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog17(ExactSolver):
    """Computes the solution to the Cog17 problem.

    Computes the solution to the Cog17 problem with defaults geometry = 3, 
    gamma = 1.4, alpha = 2.0, beta = 1.0, lambda0 = 0.1, Gamma = 40.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': r"specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'alpha': r"dimensionless constant :math:`\alpha` in Eq. :eq:`lambdaDef`",
        'beta': r"dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`",
        'lambda0': r"constant :math:`\lambda_0` in Eq. :eq:`lambdaDef`",
        'Gamma': "|Gruneisen| gas parameter",
    }

    geometry = 3
    gamma = 1.4
    alpha = 2.0
    beta = 1.0
    lambda0 = 0.1
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog17, self).__init__(**kwargs)

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
                        # a   = 7.5657e-15 erg cm^-3 K^-4
                        #     = 1.3720e+02 erg cm^-3 ev^-4
                        # k_B = 8.6173324e-5 eV K^-1
        x1 = 2 * self.beta - 4
        x2 = 2 * self.beta + 5
        c0 = 1 / (1 - self.alpha)
        c1 = x1 * c0
        c2 = x2 * c0
        c3 = -self.beta - 3
        x3 = x1 + (1 - self.alpha) * (k + 1)
        x4 = 9 - (1 - self.alpha) * (k + 1)
        x5 = x1 + 2 * (1 - self.alpha)

        u0 = x2 / x3
        temp0 = x2 * (self.alpha - 1) / bigGamma / pow(x3, 2)
        temp0 = temp0 * x4 / x5
        x6 = -2 + u0 * (2 + (self.gamma - 1) * (k + 1))
        x7 = 2 * (self.alpha * x1 * c0 + 2 * self.beta + k + 7)
        x8 = 3 / (4 * c * self.lambda0 * a)
        x8 = x8 * bigGamma / (self.gamma - 1)
        x8 = x8 * pow(temp0, c3)
        rho0 = (x6 / x7) * x8
        rho0 = pow(rho0, c0)

        density = rho0 * pow(r, c1) * pow(t, c2) * np.ones(shape=r.shape)
        velocity = u0 * (r / t) * np.ones(shape=r.shape)
        temperature = temp0 * pow((r / t), 2) * np.ones(shape=r.shape)
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



class PlanarCog17(Cog17):
    """The planar Cog17 problem.
    """

    parameters = {
        'gamma': Cog17.parameters['gamma'],
        'alpha': Cog17.parameters['alpha'],
        'beta': Cog17.parameters['beta'],
        'lambda0': Cog17.parameters['lambda0'],
        'Gamma': Cog17.parameters['Gamma'],  
        }
    geometry = 1


class CylindricalCog17(Cog17):
    """The cylindrical Cog17 problem.
    """

    parameters = {
        'gamma': Cog17.parameters['gamma'],
        'alpha': Cog17.parameters['alpha'],
        'beta': Cog17.parameters['beta'],
        'lambda0': Cog17.parameters['lambda0'],
        'Gamma': Cog17.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog17(Cog17):
    """The spherical Cog17 problem.
    """

    parameters = {
        'gamma': Cog17.parameters['gamma'],
        'alpha': Cog17.parameters['alpha'],
        'beta': Cog17.parameters['beta'],
        'lambda0': Cog17.parameters['lambda0'],
        'Gamma': Cog17.parameters['Gamma'],  
        }
    geometry = 3
