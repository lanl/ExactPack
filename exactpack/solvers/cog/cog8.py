r"""A Cog8 solver in Python.

This is a pure Python implementation of the Cog8 solution using Numpy.
The solution takes a particularly simple analytic form,

.. math::

   \rho(r,t) &= \rho_0 \, r^{(k-1)/(\beta - \alpha + 4)} t^{-(k+1)-
   (k-1)/(\beta - \alpha +4) }
   \\
   u(r,t) &= \frac{r}{t}
   \\
   T(r,t) &= T_0 \,  r^{(1-k)/(\beta - \alpha + 4)}
   t^{(1-\gamma)(k+1) + (k-1)/(\beta - \alpha +4) }   \ .

Free parameters: :math:`\alpha`, :math:`\beta`, :math:`k`,
:math:`\rho_0`, :math:`T_0`, and :math:`\gamma`.

For the values :math:`\alpha=-1`, :math:`\beta=2`, :math:`\gamma=5/3`,
and :math:`k=2` (spherical), the solution takes the form:

.. math::
   \rho(r,t) &= \rho_0\, r^{1/7} t^{-22/7}
   \\
   u(r,t) &= r t^{-1} 
   \\
   T(r,t) &= \rho_0\, r^{-1/7} t^{-13/7} \ .

"""

import numpy as np

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Cog8(ExactSolver):
    """Computes the solution to the Cog8 problem.

    Computes the solution to the Cog8 problem with defaults geometry = 3, gamma = 1.4,
    alpha = 2.0, beta = 1.0, rho0 = 1.8, temp0 = 1.4, Gamma = 40.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': r"specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'alpha': r"dimensionless constant :math:`\alpha` in Eq. :eq:`lambdaDef`",
        'beta': r"dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`",
        'rho0': "density coefficient",
        'temp0': "temperature coefficient",
        'Gamma': "|Gruneisen| gas parameter",
        }

    geometry = 3
    gamma = 1.4
    alpha = 2.0
    beta = 1.0
    rho0 = 1.8
    temp0 = 1.4
    Gamma = 40.

    def __init__(self, **kwargs):

        super(Cog8, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        if self.alpha < -2.0 or self.alpha > -1.0:
            print("*** warning: alpha lies outside range [-2,-1] ***")
        if self.beta < 1.0 or self.beta > 3.0:
            print("*** warning: beta lies outside range [1,3] ***")
    
    def _run(self, r, t):

        # No valid solution at t=0
        if t <= 0:
            nan_array = np.empty(len(r))
            nan_array[:] = np.nan
            density = nan_array
            velocity = nan_array
            temperature = nan_array
            pressure = nan_array
            sie = nan_array

        else:
            bigGamma = self.Gamma
            k = self.geometry - 1.
            c1 = (k - 1) / (self.beta - self.alpha + 4)
            c2 = (k + 1) + c1
            c3 = (1 - self.gamma) * (k + 1) + c1

            density = self.rho0 * pow(r, c1) * pow(t, -c2) * \
                np.ones(shape=r.shape)    # mass density [g/cc]
            velocity = (r / t) * np.ones(shape=r.shape)  # speed [cm/s]
            temperature = self.temp0 * pow(r, -c1) * pow(t, c3) * \
                np.ones(shape=r.shape)   # temperature [eV]
            pressure = bigGamma * density * temperature  # pressure [dyn/cm^2]
            sie = pressure / density / (self.gamma - 1)

        return ExactSolution([r, density, velocity, temperature, pressure,
                             sie],
                             names=['position',
                                    'density',
                                    'velocity',
                                    'temperature',
                                    'pressure',
                                    'specific_internal_energy'])


class PlanarCog8(Cog8):
    """The planar Cog8 problem.
    """

    parameters = {
        'gamma': Cog8.parameters['gamma'],
        'alpha': Cog8.parameters['alpha'],                 
        'beta': Cog8.parameters['beta'],
        'rho0': Cog8.parameters['rho0'],
        'temp0': Cog8.parameters['temp0'],
        'Gamma': Cog8.parameters['Gamma'],        
        }
    geometry = 1


class CylindricalCog8(Cog8):
    """The cylindrical Cog8 problem.
    """

    parameters = {
        'gamma': Cog8.parameters['gamma'],
        'alpha': Cog8.parameters['alpha'],                 
        'beta': Cog8.parameters['beta'],
        'rho0': Cog8.parameters['rho0'],
        'temp0': Cog8.parameters['temp0'],
        'Gamma': Cog8.parameters['Gamma'],
        }
    geometry = 2


class SphericalCog8(Cog8):
    """The spherical Cog8 problem.
    """

    parameters = {
        'gamma': Cog8.parameters['gamma'],
        'alpha': Cog8.parameters['alpha'],                 
        'beta': Cog8.parameters['beta'],
        'rho0': Cog8.parameters['rho0'],
        'temp0': Cog8.parameters['temp0'],
        'Gamma': Cog8.parameters['Gamma'],
        }
    geometry = 3
