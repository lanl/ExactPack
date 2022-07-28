r"""A Cog2 solver in Python.

This is a pure Python implementation of the Cog2 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^b\, t^{-2(b + k + 1) / [2 + (\gamma - 1)(k + 1)]}
  \\[3pt]
  u(r,t) &= \frac{2}{2 + (\gamma - 1)(k + 1)} \,\frac{r}{t}
  \\[3pt]
  T(r,t) &= \frac{2(\gamma - 1)(k + 1)}{\Gamma (b + 2) 
  [2 + (\gamma - 1)(k + 1)]^2}\, \left(\frac{r}{t}\right)^2

Free parameters: :math:`b`, :math:`k`, :math:`\rho_0`, :math:`\gamma`,
and :math:`\Gamma`.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class Cog2(ExactSolver):
    """Computes the solution to the Cog2 problem.

    Computes the solution to the Cog2 problem with defaults geometry = 3, gamma = 1.4,
    rho0 = 1.8, b = 1.2, Gamma = 40.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': r"specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'rho0': "density coefficient",
        'b': "free dimensionless parameter",
        'Gamma': "|Gruneisen| gas parameter",
        }

    geometry = 3
    gamma = 1.4
    rho0 = 1.8
    b = 1.2
    Gamma = 40.

    def __init__(self, **kwargs):

        super(Cog2, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1.
        c1 = 2 + (self.gamma - 1) * (k + 1)
        c2 = -2 * (self.b + k + 1) / c1
        u1 = 2 / c1
        t1 = 2 * (self.gamma - 1) * (k + 1) / bigGamma / (self.b + 2) / \
            pow(c1, 2)

        density = self.rho0 * pow(r, self.b) * pow(t, c2) * \
            np.ones(shape=r.shape)
        velocity = u1 * (r / t) * np.ones(shape=r.shape)
        temperature = t1 * pow((r / t), 2) * \
            np.ones(shape=r.shape)  # [eV]
        pressure = bigGamma * density * temperature  # pressure [dyn/cm^2]
        sie = pressure / density / (self.gamma - 1)  # specific energy [erg/g]

        return ExactSolution([r, density, velocity, temperature, pressure,
                              sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'temperature',
                                    'pressure',
                                    'specific_internal_energy'])


class PlanarCog2(Cog2):
    """The planar Cog2.
    """

    parameters = {
        'gamma': Cog2.parameters['gamma'],
        'rho0': Cog2.parameters['rho0'], 
        'b': Cog2.parameters['b'],
        'Gamma': Cog2.parameters['Gamma'],
        }
    geometry = 1


class CylindricalCog2(Cog2):
    """The cylindrical Cog2.
    """
    parameters = {
        'gamma': Cog2.parameters['gamma'],
        'rho0': Cog2.parameters['rho0'], 
        'b': Cog2.parameters['b'],
        'Gamma': Cog2.parameters['Gamma'],
        }
    geometry = 2


class SphericalCog2(Cog2):
    """The spherical Cog2.
    """

    parameters = {
        'gamma': Cog2.parameters['gamma'],
        'rho0': Cog2.parameters['rho0'], 
        'b': Cog2.parameters['b'],
        'Gamma': Cog2.parameters['Gamma'],        
        }
    geometry = 3
