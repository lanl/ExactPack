r"""A Cog10 solver in Python.

This is a pure Python implementation of the Cog10 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^{-k}
  \\
  u(r,t) &=
  \frac{4 c a \lambda_0}{3}\,\frac{\gamma - 1}{\Gamma \gamma} 
  \cdot k \rho_0^{\alpha - 1} \, T_0^{\beta + 3}
  \\
  T(r,t) &= T_0 \, r^k
  \\[5pt]
  \alpha &= \beta + 4  - 1/k ~~(k \ne 0)  

Free parameters: :math:`\beta`, :math:`k`, :math:`\rho_0`, :math:`T_0`,
:math:`\lambda_0`, :math:`\gamma`, and :math:`\Gamma`.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog10(ExactSolver):
    """Computes the solution to the Cog10 problem.

    Computes the solution to the Cog10 problem with defaults geometry = 3,
    gamma = 1.4, beta = 1.0, lambda0 = 0.1, rho0 = 1.8, temp0 = 1.4, Gamma = 40.
    """

    parameters = {
        'geometry': "2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'beta': r"dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`",
        'lambda0': r"constant :math:`\lambda_0` in Eq. :eq:`lambdaDef`",
        'rho0': "density coefficient",
        'temp0': "temperature coefficient",
        'Gamma': "|Gruneisen| gas parameter",
        }
    geometry = 3
    gamma = 1.4
    beta = 1.0
    lambda0 = 0.1
    rho0 = 1.8        
    temp0 = 1.4
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog10, self).__init__(**kwargs)

        if self.geometry not in [2, 3]:
            raise ValueError("geometry must be 2, or 3")

        if self.beta < 1.0 or self.beta > 3.0:
            print("*** warning: beta lies outside range [1,3] ***")
        
    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1.
        alpha = self.beta + 4 - 1 / k
        if alpha < -2.0 or alpha > -1.0:
            print("*** warning: alpha lies outside range [-2,-1] ***")
        c = 2.997e10   # speed of light [cm/s]
        a = 1.3720e+02  # erg cm^-3 ev^-4
        c1 = 4 * c * self.lambda0 * a / 3
        c2 = (self.gamma - 1) / (bigGamma * self.gamma)
        c3 = c1 * c2 * k * pow(self.rho0, alpha - 1) * \
             pow(self.temp0, self.beta + 3)

# a = 7.5657e-15 erg cm^-3 K^-4
#   = 1.3720e+02 erg cm^-3 ev^-4 using k_B = 8.6173324e-5 eV K^-1

        density = self.rho0 * pow(r, -k) * np.ones(shape=r.shape)
        velocity = c3 * np.ones(shape=r.shape)  # speed [cm/s] # problem here
        temperature = self.temp0 * pow(r, k) * np.ones(shape=r.shape)  # [eV]
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

    
class CylindricalCog10(Cog10):
    """The cylindrical Cog10 problem. 
    """
    parameters = {
        'gamma': Cog10.parameters['gamma'],
        'beta': Cog10.parameters['beta'],
        'lambda0': Cog10.parameters['lambda0'],
        'rho0': Cog10.parameters['rho0'],
        'temp0': Cog10.parameters['temp0'],
        'Gamma': Cog10.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog10(Cog10):
    """The spherical Cog10 problem. 
    """

    parameters = {
        'gamma': Cog10.parameters['gamma'],
        'beta': Cog10.parameters['beta'],
        'lambda0': Cog10.parameters['lambda0'],
        'rho0': Cog10.parameters['rho0'],
        'temp0': Cog10.parameters['temp0'],
        'Gamma': Cog10.parameters['Gamma'],  
        }
    geometry = 3
