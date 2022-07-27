r"""A Cog6 solver in Python.

This is a pure Python implementation of the Cog6 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \frac{r^b}{(\tau^2 - t^2)^{(k + 1 + b)/2}}
  \\
  u(r,t) &= -\frac{r t}{\tau^2 - t^2}
  \\
  T(r,t) &= \frac{\tau^2}{\Gamma (b + 2)} \cdot 
  \frac{r^2}{(\tau^2 - t^2)^2}
  \\[5pt]
 \gamma &= \frac{k + 3}{k + 1}

Free parameters: :math:`b`, :math:`k`, :math:`\rho_0`, :math:`\tau`,
and :math:`\Gamma`. For :math:`b=3`, :math:`k=2` (spherical with
:math:`\gamma = 5/3`), this becomes the 1974 Kidder solution [Kidder1974]_ .

"""

import numpy as np

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog6(ExactSolver):
    """Computes the solution to the Cog6 problem.

    Computes the solution to the Cog6 problem with defaults geometry = 3, rho0 = 1.8,
    tau = 1.25, b = 1.2, Gamma = 40.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'rho0': "density coefficient",
        'tau': "free parameter with dimensions of time",
        'b': "free dimensionless parameter",
        'Gamma': "|Gruneisen| gas parameter",
        }
        
    geometry = 3
    rho0 = 1.8
    tau = 1.25
    b = 1.2
    Gamma = 40.

    def __init__(self, **kwargs):

        super(Cog6, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

    def _run(self, r, t):

        k = self.geometry - 1
        gamma = (k + 3.) / (k + 1.)
        bigGamma = self.Gamma
        c1 = (k + 1 + self.b) / 2
        x1 = pow(self.tau, 2) - pow(t, 2)

        density = self.rho0 * pow(r, self.b) / pow(x1, c1) * \
            np.ones(shape=r.shape)  # mass density [g/cc]
        velocity = -(r * t / x1) * np.ones(shape=r.shape)  # speed [cm/s]
        temperature = pow(self.tau * r / x1, 2) / (bigGamma * (self.b + 2)) * \
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
    
class PlanarCog6(Cog6):
    """The planar Cog6 problem.
    """

    parameters = {
        'rho0': Cog6.parameters['rho0'],
        'tau': Cog6.parameters['tau'],                 
        'b': Cog6.parameters['b'],
        'Gamma': Cog6.parameters['Gamma'],  
        }
    geometry = 1


class CylindricalCog6(Cog6):
    """The cylindrical Cog6 problem.
    """

    parameters = {
        'rho0': Cog6.parameters['rho0'],
        'tau': Cog6.parameters['tau'],                 
        'b': Cog6.parameters['b'],
        'Gamma': Cog6.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog6(Cog6):
    """The spherical Cog6 problem.
    """

    parameters = {
        'rho0': Cog6.parameters['rho0'],
        'tau': Cog6.parameters['tau'],                 
        'b': Cog6.parameters['b'],
        'Gamma': Cog6.parameters['Gamma'],  
        }
    geometry = 3

    
class Kidder74(Cog6):
    """Cog6 reduces to the 1974 Kidder solution for geometry=3, b=3,
    and :math:`\gamma = 5/3`.
    """

    parameters = {
        'rho0': Cog6.parameters['rho0'],
        'tau': Cog6.parameters['tau'],                 
        'Gamma': Cog6.parameters['Gamma'],  
        }
    geometry = 3
    b = 3.

    
