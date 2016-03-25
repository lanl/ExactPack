r"""A Cog1 solver in Python.

This is a pure Python implementation of the Cog1 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^b\, t^{-b -k -1}
  \\[3pt]
  u(r,t) &= \frac{r}{t}
  \\[3pt]
  T(r,t) &= T_0\, r^{-b}\, t^{b - (\gamma-1)(k+1)}  

Free parameters: :math:`b`, :math:`k`, :math:`\rho_0`, :math:`T_0`,
and :math:`\gamma`.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class Cog1(ExactSolver):
    """Computes the solution to the Cog1 problem.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'rho0': "density coefficient",
        'temp0': "temperature coefficient",
        'b': "free param",
        'Gamma': "Gruneisen gas parameter",
        }
    geometry = 3
    gamma = 1.4
    rho0 = 1.8
    temp0 = 1.4
    b = 1.2
    Gamma = 40.

    def __init__(self, **kwargs):

        super(Cog1, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1
        c1 = self.b
        c2 = -self.b - k - 1
        c3 = self.b - (self.gamma - 1) * (k + 1)
        density = self.rho0 * pow(r, c1) * pow(t, c2) * np.ones(shape=r.shape)
        velocity = (r / t) * np.ones(shape=r.shape)
        temperature = self.temp0 * pow(r, -c1) * pow(t, c3) * \
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
                                    'sie'])


class PlanarCog1(Cog1):
    """The planar Cog1.
    """

    parameters = {
        'gamma': Cog1.parameters['gamma'],
        'rho0': Cog1.parameters['rho0'], 
        'temp0': Cog1.parameters['temp0'], 
        'b': Cog1.parameters['b'],
        'Gamma': Cog1.parameters['Gamma'],        
        }
    geometry = 1


class CylindricalCog1(Cog1):
    """The cylindrical Cog1.
    """

    parameters = {
        'gamma': Cog1.parameters['gamma'],
        'rho0': Cog1.parameters['rho0'], 
        'temp0': Cog1.parameters['temp0'], 
        'b': Cog1.parameters['b'],
        'Gamma': Cog1.parameters['Gamma'],        
        }
    geometry = 2


class SphericalCog1(Cog1):
    """The spherical Cog1.
    """

    parameters = {
        'gamma': Cog1.parameters['gamma'],
        'rho0': Cog1.parameters['rho0'], 
        'temp0': Cog1.parameters['temp0'], 
        'b': Cog1.parameters['b'],
        'Gamma': Cog1.parameters['Gamma'],        
        }
    geometry = 3
