r"""A Cog13 solver in Python.

This is a pure Python implementation of the Cog13 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^{2/(\alpha - \beta - 4)} \,
   t^{-2/(\alpha - \beta - 4) - k - 1}
  \\
  u(r,t) &= \frac{r}{t}
  \\
  T(r,t) &= T_0 \, r^{-2/(\alpha - \beta - 4)} \cdot
  t^{[\alpha (k + 1) - k - 2]/(\beta + 3) 
  + 
  2(\alpha - 1)/[(\beta + 3)(\alpha - \beta - 4)]}
  \\
  T_0 &=   
  \left[ \frac{3\Gamma}{4 c a \lambda_0 (\gamma - 1)} \cdot
  \frac{\alpha - 1 + (\beta + 3)(\gamma - 1)}{\beta + 3} \cdot
  \rho_0^{1 - \alpha}\,\frac{\beta + 4 - \alpha}{2}
  \right]^{1/(\beta + 3)}

Free parameters: :math:`\alpha`, :math:`\beta` :math:`k`,
:math:`\rho_0`, :math:`\lambda_0`, and :math:`\Gamma`. 

"""

import numpy as np

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog13(ExactSolver):
    """Computes the solution to the Cog13 problem.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'rho0': "density coefficient",
        'alpha': r"dimensionless constant :math:`\alpha` in Eq. :eq:`lambdaDef`",
        'beta': r"dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`",
        'lambda0': r"constant :math:`\lambda_0` in Eq. :eq:`lambdaDef`",
        'Gamma': "Gruneisen gas parameter",
        }
    geometry = 3
    gamma = 1.4
    rho0 = 1.8
    alpha = 2.0
    beta = 1.0
    lambda0 = 0.1
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog13, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.gamma == 1.0:
            raise ValueError("gamma cannot be one")

        if self.alpha < -2.0 or self.alpha > -1.0:
            print("*** warning: alpha lies outside range [-2,-1] ***")
        if self.beta < 1.0 or self.beta > 3.0:
            print("*** warning: beta lies outside range [1,3] ***")
        
    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1.
        c1 = 2 / (self.alpha - self.beta - 4)
        c2 = -c1 - k - 1
        c3 = (self.alpha * (k + 1) - k - 2) / (self.beta + 3)
        c4 = 2 * (self.alpha - 1) / (self.beta + 3) / (self.alpha - self.beta - 4)
        c5 = c3 + c4
        c = 2.997e10    # speed of light [cm/s]
        a = 1.3720e+02  # erg cm^-3 ev^-4
        # a = 7.5657e-15 erg cm^-3 K^-4
        #   = 1.3720e+02 erg cm^-3 ev^-4 using k_B = 8.6173324e-5 eV K^-1
        c6 = 3 * bigGamma / (4 * c * self.lambda0 * a * (self.gamma - 1))
        c7 = 1. / (self.beta + 3)
        c8 = (self.alpha - 1 + (self.beta + 3) * (self.gamma - 1)) * c7
        c9 = pow(self.rho0, 1 - self.alpha) * (self.beta + 4 - self.alpha) / 2
        temp0 = pow((c6 * c8 * c9), c7)  # see coggeshall p761
        density = self.rho0 * pow(r, c1) * pow(t, c2) * np.ones(shape=r.shape)
        velocity = (r / t) * np.ones(shape=r.shape)  # speed [cm/s]
        temperature = temp0 * pow(r, -c1) * pow(t, c5) * \
            np.ones(shape=r.shape)  # temperature [eV]
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


class PlanarCog13(Cog13):
    """The planar Cog13 problem.
    """
    
    parameters = {
        'gamma': Cog13.parameters['gamma'],
        'rho0': Cog13.parameters['rho0'],
        'alpha': Cog13.parameters['alpha'],
        'beta': Cog13.parameters['beta'],
        'lambda0': Cog13.parameters['lambda0'],
        'Gamma': Cog13.parameters['Gamma'],  
        }
    geometry = 1


class CylindricalCog13(Cog13):
    """The cylindrical Cog13 problem.
    :math:`\gamma=5/3`.
    """

    parameters = {
        'gamma': Cog13.parameters['gamma'],
        'rho0': Cog13.parameters['rho0'],
        'alpha': Cog13.parameters['alpha'],
        'beta': Cog13.parameters['beta'],
        'lambda0': Cog13.parameters['lambda0'],
        'Gamma': Cog13.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog13(Cog13):
    """The spherical Cog13 problem.
    """

    parameters = {
        'gamma': Cog13.parameters['gamma'],
        'rho0': Cog13.parameters['rho0'],
        'alpha': Cog13.parameters['alpha'],
        'beta': Cog13.parameters['beta'],
        'lambda0': Cog13.parameters['lambda0'],
        'Gamma': Cog13.parameters['Gamma'],  
        }
    geometry = 3
