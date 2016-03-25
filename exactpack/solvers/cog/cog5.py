r"""A Cog5 solver in Python.

This is a pure Python implementation of the Cog5 solution using Numpy.

This is a pure Python implementation of the Cog4 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0\, r^{-2}
  \\
  u(r,t) &=  u_0\, t
  \\
  T(r,t) &=  \frac{u_0}{\Gamma} \cdot r
  \\
  k &= 2 ~\textrm{and}~ \gamma = \frac{1}{2}

Free parameters: :math:`u_0`, :math:`\rho_0`, and :math:`\Gamma`.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog5(ExactSolver):
    """Computes the solution to the Cog5 problem.
    """

    parameters = {
        'rho0': "density coefficient",
        'u0': "velocity coefficient",
        'Gamma': "Gruneisen gas parameter",
        }
    rho0 = 1.8
    u0 = 2.3
    Gamma = 40.
    
    def __init__(self, **kwargs):

        super(Cog5, self).__init__(**kwargs)

    def _run(self, r, t):

        geometry = 3
        k = geometry - 1.
        gamma = 1. / 2.
        bigGamma = self.Gamma

        density = self.rho0 * pow(r, -2) * \
            np.ones(shape=r.shape)  # mass density [g/cc]
        velocity = self.u0 * t * np.ones(shape=r.shape)  # speed [cm/s]
        temperature = (self.u0 * r / bigGamma) * \
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
                                    'sie'])
