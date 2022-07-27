r"""A Cog4 solver in Python.

This is a pure Python implementation of the Cog4 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^{-2 k / (\gamma + 1)}
  \\
  u(r,t) &= u_0\, r^{-k (\gamma - 1) / (\gamma + 1)}
  \\
  T(r,t) &= \frac{u_0^2 (1 - \gamma)}{2 \gamma \Gamma}\,
  r^{-2k (\gamma - 1) / (\gamma + 1)}

Free parameters: :math:`k`, :math:`u_0`, :math:`\rho_0`, :math:`\gamma`,
and :math:`\Gamma`. The only physical solutions for which :math:`T > 0`
are for :math:`\gamma < 1`.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog4(ExactSolver):
    """Computes the solution to the Cog4 problem.

    Computes the solution to the Cog4 problem with defaults geometry = 3, gamma = 1.4, 
    rho0 = 1.4, u0 = 2.3, Gamma = 40.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v` (must be < 1)",
        'rho0': "density coefficient",
        'u0': "velocity coefficient",
        'Gamma': "|Gruneisen| gas parameter",
        }

    geometry = 3
    gamma = 1.4
    rho0 = 1.4
    u0 = 2.3
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog4, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.gamma >= 1:
            print("*** warning: gamma > 1 gives T < 0 ***")

    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1
        x1 = -k / (self.gamma + 1)
        x2 = -k * (self.gamma - 1) / (self.gamma + 1)
        t1 = pow(self.u0, 2) * (1 - self.gamma) / (2 * \
            self.gamma * bigGamma)
        density = self.rho0 * pow(r, 2 * x1) * \
            np.ones(shape=r.shape)  # mass density [g/cc]
        velocity = self.u0 * pow(r, x2) * \
            np.ones(shape=r.shape)   # speed [cm/s]
        temperature = t1 * pow(r,  2 * x2) * \
            np.ones(shape=r.shape)  # temperature [eV]

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


class PlanarCog4(Cog4):
    """The planar Cog4.
    """

    parameters = {
        'gamma': Cog4.parameters['gamma'], 
        'rho0': Cog4.parameters['rho0'],
        'u0': Cog4.parameters['u0'],
        'Gamma': Cog4.parameters['Gamma'],
        }
    geometry = 1


class CylindricalCog4(Cog4):
    """The cylindrical Cog4.
    """

    parameters = {
        'gamma': Cog4.parameters['gamma'], 
        'rho0': Cog4.parameters['rho0'],
        'u0': Cog4.parameters['u0'],
        'Gamma': Cog4.parameters['Gamma'],        
        }
    geometry = 2


class SphericalCog4(Cog4):
    """The spherical Cog4.
    """

    parameters = {
        'gamma': Cog4.parameters['gamma'], 
        'rho0': Cog4.parameters['rho0'],
        'u0': Cog4.parameters['u0'],
        'Gamma': Cog4.parameters['Gamma'],        
        }
    geometry = 3
