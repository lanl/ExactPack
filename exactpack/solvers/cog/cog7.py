r"""A Cog7 solver in Python.

This is a pure Python implementation of the Cog7 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= 
  \frac{R_0^{b/\gamma} \, \tau^{[(k + 1)\gamma - 1 - b]/(\gamma - 1)} }
  {\left( R_0^{2 - b/\gamma} - R_i^{2 - b/\gamma} \right)^{1/(\gamma - 1)}}
  \cdot 
  r^{-k - 1}
  \cdot
  \left( \frac{r}{\left(\tau^2 - t^2\right)^{1/2}}\right)^{k + 1 - b / \gamma}
  \cdot
  \\&
  \left[
  \left( \frac{r}{\left(\tau^2 - t^2\right)^{1/2}}\right)^{2 - b / \gamma}
  -
  \left(\frac{R_i}{\tau}\right)^{2 - b / \gamma}
  \right]^{1/(\gamma - 1)}
  \\[10pt]
  u(r,t) &=- \frac{r t}{\tau^2 - t^2}
  \\[10pt]
  T(r,t) &= \frac{\tau^2 (\gamma - 1)}{\Gamma (2 \gamma - b)}
  \cdot
  r^{-2}
  \left( \frac{r}{\left(\tau^2 - t^2\right)^{1/2}}\right)^{2 + b / \gamma}
  \cdot 
  \\&
  \left[
  \left( \frac{r}{\left(\tau^2 - t^2\right)^{1/2}}\right)^{2 - b / \gamma}
  -
  \left(\frac{R_i}{\tau}\right)^{2 - b / \gamma}
  \right]
  \\[5pt]
 \gamma &= \frac{k + 3}{k + 1}

Free parameters: :math:`b`, :math:`k`, :math:`\tau`, :math:`R_0`,
:math:`R_i`, and :math:`\Gamma`. For :math:`b = 0`, :math:`k = 2` (spherical
with :math:`\gamma = 5/3`), this becomes Kidder's 1976 solution
[R.E. Kidder, Nucl. Fusion **16** (1976) 33].

"""

import numpy as np
from math import sqrt

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog7(ExactSolver):
    """Computes the solution to the Cog7 problem.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'tau': "free parameter",
        'b': "free dimensionless parameter",
        'R0': "free parameter with dimensions of length",
        'Ri': "free parameter with dimensions of length",
        'Gamma': "Gruneisen gas parameter",
        }
    geometry = 3
    tau = 1.25
    b = 1.2
    R0 = 2.0
    Ri = 0.1
    Gamma = 40.

    def __init__(self, **kwargs):

        super(Cog7, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

    def _run(self, r, t):

        k = self.geometry - 1.
        gamma = (k + 3) / (k + 1)
        bigGamma = self.Gamma
        x1 = pow(self.tau, 2) - pow(t, 2)
        x2 = sqrt(x1)
        c1 = 2 - self.b / gamma
        x3 = pow(r / x2, c1) - pow(self.Ri / self.tau, c1)
        c2 = 1 / (gamma - 1)
        c3 = k + 1 - self.b / gamma
        c4 = 2 + self.b / gamma
        c5 = ((k + 1) * gamma - 1 - self.b) / (gamma - 1)
        x4 = pow(r / x2, c3)
        x5 = pow(r / x2, c4)
        x6 = pow((self.tau / r), 2) * (gamma - 1) / \
             bigGamma / (2 * gamma - self.b)
        x7 = pow(self.R0, self.b / gamma) / pow(pow(self.R0, c1) - \
             pow(self.Ri, c1), c2)
        rho0 = x7 * x4 * pow(x3, c2) * pow(self.tau, c5) * pow(r, -k - 1)
        temp0 = x6 * x5 * x3

        density = rho0 * np.ones(shape=r.shape)
        velocity = - (r * t / x1) * np.ones(shape=r.shape)
        temperature = temp0 * np.ones(shape=r.shape)
        pressure = bigGamma * density * temperature
        sie = pressure / density / (gamma - 1)

        return ExactSolution([r, density, velocity, temperature, pressure,
                             sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'temperature',
                                    'pressure',
                                    'sie'])

    
class PlanarCog7(Cog7):
    """The planar Cog7.
    """

    parameters = {
        'tau': Cog7.parameters['tau'],                 
        'b': Cog7.parameters['b'],
        'R0': Cog7.parameters['R0'],
        'Ri': Cog7.parameters['Ri'],
        'Gamma': Cog7.parameters['Gamma'],        
        }
    geometry = 1


class CylindricalCog7(Cog7):
    """The cylindrical Cog7.
    """

    parameters = {
        'tau': Cog7.parameters['tau'],                 
        'b': Cog7.parameters['b'],
        'R0': Cog7.parameters['R0'],
        'Ri': Cog7.parameters['Ri'],
        'Gamma': Cog7.parameters['Gamma'],        
        }
    geometry = 2


class SphericalCog7(Cog7):
    """The spherical Cog7.
    """

    parameters = {
        'tau': Cog7.parameters['tau'],                 
        'b': Cog7.parameters['b'],
        'R0': Cog7.parameters['R0'],
        'Ri': Cog7.parameters['Ri'],
        'Gamma': Cog7.parameters['Gamma'],        
        }
    geometry = 3
    

class Kidder76(Cog7):
    """Cog7 reduces to the 1976 Kidder solution for geometry=3, b=0.
    """

    parameters = {
        'tau': Cog7.parameters['tau'],                 
        'R0': Cog7.parameters['R0'],
        'Ri': Cog7.parameters['Ri'],
        'Gamma': Cog7.parameters['Gamma'],
        }
    geometry = 3
    b = 0.

