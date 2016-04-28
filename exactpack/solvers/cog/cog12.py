r"""A Cog12 solver in Python.

This is a pure Python implementation of the Cog12 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^{-2k/(\gamma + 1)}
  \\
  u(r,t) &=u_0 \, r^{k(1 - \gamma)/(1 + \gamma)}
  \\
  T(r,t) &= \frac{u_o^2\, (1 - \gamma)}{2 \Gamma \gamma}
  \cdot
  r^{2 k (1 - \gamma)/(1 + \gamma)}
  \\ 
  \alpha &= (\beta + 4)(1 - \gamma)  + \frac{(k -1)(\gamma + 1)}{2k}
  ~~(k \ne 0)

Free parameters: :math:`k`, :math:`\rho_0`, :math:`u_0`, :math:`\gamma`,
:math:`\Gamma`, and :math:`\beta` (with :math:`\alpha` a function of
:math:`k`, :math:`\beta`, and :math:`\gamma`). Note that :math:`T > 0`
only when :math:`\gamma < 1`.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class Cog12(ExactSolver):
    """Computes the solution to the Cog12 problem.
    """

    parameters = {
        'geometry': "2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'beta': r"dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`",
        'rho0': "density coefficient",
        'u0': "velocity coefficient",
        'Gamma': "Gruneisen gas parameter",
        }
    geometry = 3
    gamma = 1.4
    beta = 1.0
    rho0 = 1.8
    u0 = 2.3
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog12, self).__init__(**kwargs)

        if self.geometry not in [2, 3]:
            raise ValueError("geometry must be 2, or 3")

    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1.

        c1 = -2 * k / (self.gamma + 1)
        c2 = k * (1 - self.gamma) / (1 + self.gamma)
        c3 = 2 - (self.gamma - 1) * (k + 1)
        temp0 = pow(self.u0, 2) * (1 - self.gamma) / \
                (2 * bigGamma * self.gamma)
#       alpha = (self.beta + 4) * (1 + self.gamma) + (k - 1) / c3
#       this is never used

# a = 7.5657e-15 erg cm^-3 K^-4
#   = 1.3720e+02 erg cm^-3 ev^-4 using k_B = 8.6173324e-5 eV K^-1

        density = self.rho0 * pow(r, c1) * np.ones(shape=r.shape)
        velocity = self.u0 * pow(r, c2) * np.ones(shape=r.shape)
        temperature = temp0 * pow(r, 2 * c2) * np.ones(shape=r.shape)  # [eV]
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


    
class PlanarCog12(Cog12):
    """The planar Cog12 problem.
    """

    parameters = {
        'gamma': Cog12.parameters['gamma'],
        'beta': Cog12.parameters['beta'],
        'rho0': Cog12.parameters['rho0'],
        'u0': Cog12.parameters['u0'],
        'Gamma': Cog12.parameters['Gamma'],  
        }
    geometry = 1

    
class CylindricalCog12(Cog12):
    """The cylindrical Cog12 problem. 
    """

    parameters = {
        'gamma': Cog12.parameters['gamma'],
        'beta': Cog12.parameters['beta'],
        'rho0': Cog12.parameters['rho0'],
        'u0': Cog12.parameters['u0'],
        'Gamma': Cog12.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog12(Cog12):
    """The spherical Cog12 problem. 
    """

    parameters = {
        'gamma': Cog12.parameters['gamma'],
        'beta': Cog12.parameters['beta'],
        'rho0': Cog12.parameters['rho0'],
        'u0': Cog12.parameters['u0'],
        'Gamma': Cog12.parameters['Gamma'],  
        }
    geometry = 3
