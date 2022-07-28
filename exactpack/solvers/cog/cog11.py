r"""A Cog11 solver in Python.

This is a pure Python implementation of the Cog11 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= 
  \rho_0 \, r^{(\gamma - 1)(k + 1) - 2} \, t^{1 - k - (\gamma - 1)(k + 1)}
  \\
  u(r,t) &= \frac{r}{t}
  \\
  T(r,t) &= T_0 \, r^{2 - (\gamma -1)(k + 1)} \, t^{-2}
  \\[5pt]
  \alpha &= \beta + 4 + \frac{k -1}{2 - (\gamma - 1)(k + 1)}

Free parameters: :math:`k`, :math:`\rho_0`, :math:`T_0`, :math:`\gamma`,
and :math:`\beta` (with :math:`\alpha` a function of :math:`k`,
:math:`\beta` and :math:`\gamma`).

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class Cog11(ExactSolver):
    """Computes the solution to the Cog11 problem.

    Computes the solution to the Cog11 problem with defaults  geometry = 3, 
    gamma = 1.4, rho0 = 1.8, temp0 = 1.4, beta = 1.0, cv = 1.e2.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': r"specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'beta': r"dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`",
        'rho0': "density coefficient",
        'temp0': "temperature coefficient",
        'Gamma': "|Gruneisen| gas parameter",
        }

    geometry = 3
    gamma = 1.4
    rho0 = 1.8
    temp0 = 1.4
    beta = 1.0
    cv = 1.e2

    def __init__(self, **kwargs):

        super(Cog11, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.beta < 1.0 or self.beta > 3.0:
            print("*** warning: beta lies outside range [1,3] ***")
        
    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1.
        c1 = (self.gamma - 1) * (k + 1)
        c2 = 1 - k - c1
        c3 = 2 - c1
        alpha = self.beta + 4 + (k - 1) / c3
        if alpha < -2.0 or alpha > -1.0:
            print("*** warning: alpha lies outside range [-2,-1] ***")

# a = 7.5657e-15 erg cm^-3 K^-4
#   = 1.3720e+02 erg cm^-3 ev^-4 using k_B = 8.6173324e-5 eV K^-1
        density = self.rho0 * pow(r, c1 -2) * pow(t, c2) * \
            np.ones(shape=r.shape)  # mass density [g/cc]
        velocity = (r / t) * np.ones(shape=r.shape)  # speed [cm/s]
        temperature = self.temp0 * pow(r, c3) * pow(t, -2) * \
            np.ones(shape=r.shape)  # temperature [eV]
        pressure = bigGamma * density * temperature  # [eV]
        sie = pressure / density / (self.gamma - 1)

        return ExactSolution([r, density, velocity, temperature, pressure,
                             sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'temperature',
                                    'pressure',
                                    'specific_internal_energy'])


class PlanarCog11(Cog11):
    """The planar Cog11 problem.
    """

    parameters = {
        'gamma': Cog11.parameters['gamma'],
        'beta': Cog11.parameters['beta'],
        'rho0': Cog11.parameters['rho0'],
        'temp0': Cog11.parameters['temp0'],
        'Gamma': Cog11.parameters['Gamma'],  
        }
    geometry = 1


class CylindricalCog11(Cog11):
    """The cylindrical Cog11 problem.
    """

    parameters = {
        'gamma': Cog11.parameters['gamma'],
        'beta': Cog11.parameters['beta'],
        'rho0': Cog11.parameters['rho0'],
        'temp0': Cog11.parameters['temp0'],
        'Gamma': Cog11.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog11(Cog11):
    """The spherical Cog11 problem.
    """

    parameters = {
        'gamma': Cog11.parameters['gamma'],
        'beta': Cog11.parameters['beta'],
        'rho0': Cog11.parameters['rho0'],
        'temp0': Cog11.parameters['temp0'],
        'Gamma': Cog11.parameters['Gamma'],  
        }
    geometry = 3
