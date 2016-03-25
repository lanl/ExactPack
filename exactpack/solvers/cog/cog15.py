r"""A Cog15 solver in Python.

This is a pure Python implementation of the Cog15 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^{2/(\alpha - \beta - 4)} \, t^{-k - 1 - 2(\alpha - \beta - 4)}
  \\
  u(r,t) &= \frac{r}{t}
  \\
  T(r,t) &= T_0\, r^{-2/(\alpha - \beta - 4)} \, t^{-2}
  \\
  \frac{2}{\alpha - \beta - 4} &= \frac{k + 4 - \alpha(k + 1) - 2 (\beta + 4)}{\alpha - 1}  
  \\
  T_0 &=  \left[ \frac{3}{4 c a \lambda_0} \, \frac{\Gamma}{2(k + 1) (\gamma - 1)} \cdot \rho_0^{1 - \alpha}
  \cdot 
  \Big(2 + \left[2 - (\gamma - 1) (k + 1) \right] (\alpha - \beta - 4) \Big)
  \right]^{1/(\beta + 3)}

Free parameters: :math:`k`, :math:`\rho_0`, and either :math:`\alpha`,
:math:`\lambda_0`, and :math:`\Gamma` (with :math:`\beta` a function of
:math:`k` and :math:`\alpha`).

"""

import numpy as np
from math import sqrt
from scipy.optimize import minimize

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog15(ExactSolver):
    """Computes the solution to the Cog15 problem.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'rho0': "density coefficient",
        'alpha': r"dimensionless constant :math:`\alpha` in Eq. :eq:`lambdaDef`",
        'lambda0': r"constant :math:`\lambda_0` in Eq. :eq:`lambdaDef`",
        'Gamma': "Gruneisen gas parameter",
        }
    geometry = 3
    gamma = 1.4
    rho0 = 1.8
    alpha = 8.0
    lambda0 = 0.1
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog15, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1.
        c = 2.997e10    # speed of light [cm/s]
        a = 1.3720e+02  # erg cm^-3 ev^-4
                        # a = 7.5657e-15 erg cm^-3 K^-4
                        #   = 1.3720e+02 erg cm^-3 ev^-4 using k_B
                        #   = 8.6173324e-5 eV K^-1

        # solve for beta given alpha
        def cog15(xalpha,xbeta,k):
            return np.abs((k+4-xalpha*(k+1)-2*(xbeta+4))/(xalpha-1)-2/(xalpha-xbeta-4))
        xalpha = self.alpha
        res = minimize(cog15, 0., args=(xalpha, k))
        beta = res.x[0]

        c1 = 2/(self.alpha - beta -4)
        c3 = 1. / (beta + 3)
        temp0 = 3 / (4 * c * self.lambda0 * a) \
                * bigGamma / (2 * (k + 1) * (self.gamma - 1)) \
                * pow(self.rho0, 1 - self.alpha)

        density = self.rho0 * pow(r, 2/(self.alpha-beta-4)) \
                  * pow(t, -k -1-2/(self.alpha-beta-4)) \
                  * np.ones(shape=r.shape)
        velocity = (r / t) * np.ones(shape=r.shape)
        temperature = temp0 * pow(r, -c1) * pow(t, -2) * np.ones(shape=r.shape)
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


class PlanarCog15(Cog15):
    """The planar Cog15 problem.
    """

    parameters = {
        'gamma': Cog15.parameters['gamma'],
        'rho0': Cog15.parameters['rho0'],
        'alpha': Cog15.parameters['alpha'],
        'lambda0': Cog15.parameters['lambda0'],
        'Gamma': Cog15.parameters['Gamma'],  
        }
    geometry = 1


class CylindricalCog15(Cog15):
    """The cylindrical Cog15 problem.
    """

    parameters = {
        'gamma': Cog15.parameters['gamma'],
        'rho0': Cog15.parameters['rho0'],
        'alpha': Cog15.parameters['alpha'],
        'lambda0': Cog15.parameters['lambda0'],
        'Gamma': Cog15.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog15(Cog15):
    """The spherical Cog15 problem.
    """

    parameters = {
        'gamma': Cog15.parameters['gamma'],
        'rho0': Cog15.parameters['rho0'],
        'alpha': Cog15.parameters['alpha'],
        'lambda0': Cog15.parameters['lambda0'],
        'Gamma': Cog15.parameters['Gamma'],  
        }
    geometry = 3
