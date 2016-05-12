r"""A Cog9 solver in Python.

This is a pure Python implementation of the Cog9 solution using Numpy.
The exact solution takes the form,

.. math::

  \rho(r,t) &= \rho_0 \, r^{-(2\beta + k + 7)/\alpha} \,
  t^{-2[\alpha(k + 1) - 2 \beta - k - 7]/\alpha[2 + (\gamma - 1)(k + 1)]}
  \\
  u(r,t) &= \frac{2}{2 + (\gamma - 1)(k + 1)}  \, \frac{r}{t}
  \\
  T(r,t) &= 
  \frac{2\alpha(\gamma - 1)(k + 1)}{\Gamma [2 + (\gamma - 1)(k + 1)]^2 
  (2\alpha - 2\beta - k - 7)}
  \cdot 
  \left(\frac{r}{t}\right)^2

Free parameters: :math:`\alpha`, :math:`\beta`, :math:`k`, :math:`\rho_0`,
:math:`\gamma`, and :math:`\Gamma`.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog9(ExactSolver):
    """Computes the solution to the Cog9 problem.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'alpha': r"dimensionless constant :math:`\alpha` in Eq. :eq:`lambdaDef`",
        'beta': r"dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`",
        'rho0': "density coefficient",
        'Gamma': "Gruneisen gas parameter",
        }
    geometry = 3
    gamma = 1.4
    alpha = 2.0
    beta = 1.0
    rho0 = 1.8
    Gamma = 40.

    def __init__(self, **kwargs):

        super(Cog9, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.alpha < -2.0 or self.alpha > -1.0:
            print "*** warning: alpha lies outside range [-2,-1] ***"
        if self.beta < 1.0 or self.beta > 3.0:
            print "*** warning: beta lies outside range [1,3] ***"

    def _run(self, r, t):

        bigGamma = self.Gamma
        k = self.geometry - 1
        c1 = 2 * self.beta + k + 7
        c2 = -c1 / self.alpha
        c3 = 2 + (self.gamma - 1) * (k + 1)
        c4 = -2 * (self.alpha * (k + 1) - c1) / self.alpha / c3
        c5 = 2 * self.alpha * (self.gamma - 1) * (k + 1) / bigGamma / \
             pow(c3, 2) / (2 * self.alpha - 2 * self.beta - k - 7)

        density = self.rho0 * pow(r, c2) * pow(t, c4) * \
            np.ones(shape=r.shape)    # mass density [g/cc]
        velocity = (2 / c3) * (r / t) * np.ones(shape=r.shape)
        temperature = c5 * pow((r / t), 2) * np.ones(shape=r.shape)
        pressure = bigGamma * density * temperature  # pressure [dyn/cm^2]
        sie = pressure / density / (self.gamma - 1)

        return ExactSolution([r, density, velocity, temperature, pressure,
                             sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'temperature',
                                    'pressure',
                                    'sie'])


class PlanarCog9(Cog9):
    """The planar Cog9 problem.
    """

    parameters = {
        'gamma': Cog9.parameters['gamma'],
        'alpha': Cog9.parameters['alpha'],                 
        'beta': Cog9.parameters['beta'],
        'rho0': Cog9.parameters['rho0'],
        'Gamma': Cog9.parameters['Gamma'],  
        }
    geometry = 1


class CylindricalCog9(Cog9):
    """The cylindrical Cog9 problem.
    """

    parameters = {
        'gamma': Cog9.parameters['gamma'],
        'alpha': Cog9.parameters['alpha'],                 
        'beta': Cog9.parameters['beta'],
        'rho0': Cog9.parameters['rho0'],
        'Gamma': Cog9.parameters['Gamma'],  
        }
    geometry = 2


class SphericalCog9(Cog9):
    """The spherical Cog9 problem.
    """

    parameters = {
        'gamma': Cog9.parameters['gamma'],
        'alpha': Cog9.parameters['alpha'],                 
        'beta': Cog9.parameters['beta'],
        'rho0': Cog9.parameters['rho0'],
        'Gamma': Cog9.parameters['Gamma'],  
        }
    geometry = 3
