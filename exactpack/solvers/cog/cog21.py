r"""A Cog21 solver in Python.

This is a pure Python implementation of the Cog21 solution using Numpy.
The solution contains a shock located at

.. math::

  R_\text{shock} &= \frac{2}{\Gamma T_0 \,t^2}   ~~\textrm{with}~~
  \\
  k &= 2 ~~\textrm{and}~~ \gamma = 5 \ .

Region 1:

.. math::

  \rho(r,t) &= \frac{3}{2}\, \rho_0 r^{-3}
  \\
  u(r,t) &= 0
  \\
  T(r,t) &= T_0\, r^3

Region 2:

.. math::

  \rho(r,t) &= \rho_0\, r^{-3}
  \\
  u(r,t) &= \frac{r}{t}
  \\
  T(r,t) &= 0

Free parameters: :math:`\rho_0`, :math:`T_0`, and :math:`\Gamma`. 

"""

import numpy as np
from math import sqrt

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition

class Cog21(ExactSolver):
    """omputes the solution to the Cog21 problem.

    Computes the solution to the Cog21 problem with defaults rho0 = 1.8,
    temp0 = 2.9, Gamma = 400.
    """

    parameters = {
        'rho0': "density coefficient",
        'temp0': "temperature coefficient",
        'Gamma': "|Gruneisen| gas parameter",
        }

    rho0 = 1.8
    temp0 = 2.9
    Gamma = 400.
    
    def __init__(self, **kwargs):

        super(Cog21, self).__init__(**kwargs)

    def _run(self, r, t):

        gamma = 5.0
        geometry = 3
        k = geometry - 1.
        bigGamma = self.Gamma
        shock_location = 2 / (bigGamma * self.temp0 * pow(t, 2))
        density = self.rho0 * pow(r, -3) * np.where(r < shock_location,
                  1.5, 1) * np.ones(shape=r.shape)
        velocity = np.where(r < shock_location,
                               0, r / t) * np.ones(shape=r.shape)
        temperature = np.where(r < shock_location,
                               self.temp0 * pow(r, 3), 0) * \
                               np.ones(shape=r.shape)
        pressure = bigGamma * density * temperature
        sie = pressure / density / (gamma - 1)

        return ExactSolution([r, density, velocity, temperature, pressure,
                             sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'temperature',
                                    'pressure',
                                    'specific_internal_energy'])
