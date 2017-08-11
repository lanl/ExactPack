r"""A Cog3 solver in Python. 

This is a pure Python implementation of the Cog3 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^{b - k -1}\, e^{b t}
  \\
  u(r,t) &= -\frac{b}{v} \cdot r
  \\
  T(r,t) &= \frac{b^2}{ v^2\, \Gamma (k - v - 1)} \cdot r^2
  \\[5pt]
  \gamma &= \frac{k - 1}{k + 1}

Free parameters: :math:`v`, :math:`b`, :math:`k`, :math:`\rho_0`,
and :math:`\Gamma`. Note that :math:`\gamma < 1`. 

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class Cog3(ExactSolver):
    """Computes the solution to the Cog3 problem.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'rho0': "density coefficient",
        'b': "free dimensionless parameter",
        'v': "free parameter with dimensions of velocity",
        'Gamma': "Gruneisen gas parameter"
        }
    geometry = 3
    rho0 = 1.8
    b = 1.2
    v = 0.5
    Gamma = 40.

    def __init__(self, **kwargs):

        super(Cog3, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

    def _run(self, r, t):

        k = self.geometry - 1.
        gamma = (k - 1) / (k + 1)
        bigGamma = self.Gamma
        c1 = k - self.v - 1
        c2 = self.v - k - 1
        ee = 2.718281828459045
        density = self.rho0 * pow(r, c2) * pow(ee, self.b * t) * \
            np.ones(shape=r.shape)  # mass density [g/cc]
        velocity = -(self.b * r / self.v) * \
            np.ones(shape=r.shape)  # speed [cm/s]
        temperature = pow((self.b * r / self.v), 2) / (bigGamma * c1) * \
            np.ones(shape=r.shape)  # temperature [eV]
        pressure = bigGamma * density * temperature  # pressure [dyn/cm^2]
        sie = pressure / density / (gamma - 1)  # specific energy [erg/g]

        return ExactSolution([r, density, velocity, temperature, pressure,
                              sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'temperature',
                                    'pressure',
                                    'specific_internal_energy'])


class PlanarCog3(Cog3):
    """The planar Cog3 problem.
    """

    parameters = {
        'rho0': Cog3.parameters['rho0'], 
        'b': Cog3.parameters['b'],
        'v': Cog3.parameters['v'],
        'Gamma': Cog3.parameters['Gamma'],  
        }
    geometry = 1


class CylindricalCog3(Cog3):
    """The cylindrical Cog3 problem.
    """

    parameters = {
        'rho0': Cog3.parameters['rho0'], 
        'b': Cog3.parameters['b'],
        'v': Cog3.parameters['v'],
        'Gamma': Cog3.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog3(Cog3):
    """The spherical Cog3 problem.
    """

    parameters = {
        'rho0': Cog3.parameters['rho0'], 
        'b': Cog3.parameters['b'],
        'v': Cog3.parameters['v'],
        'Gamma': Cog3.parameters['Gamma'],  
        }
    geometry = 3
