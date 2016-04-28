r"""An implementation of the Cog8 solution in Fortran written by Frank Timmes.

This is a Fortran based solver for the Cog8 solution, as implemented by Frank Timmes.
The original Fortran source code is available at `Frank Timmes' website
<http://cococubed.asu.edu/research_pages/cog8.shtml>`_, under release LA-CC-05-101.
The exact solution is quite simple, and takes the form

.. math::

   \rho(r,t) &= \rho_0 \, r^{(k-1)/(\beta - \alpha + 4)} t^{-(k+1)-
   (k-1)/(\beta - \alpha +4) }
   \\
   u(r,t) &= \frac{r}{t}
   \\
   T(r,t) &= T_0 \,  r^{(1-k)/(\beta - \alpha + 4)}
   t^{(1-\gamma)(k+1) + (k-1)/(\beta - \alpha +4) }   \ .

Free parameters: :math:`k`, :math:`\gamma`, :math:`c_v`, :math:`\alpha`,
:math:`\beta`, :math:`\rho_0`, and :math:`T_0`. For the specific values
:math:`\alpha=-1`, :math:`\beta=2`, :math:`\gamma=5/3`, and
:math:`k=2` (spherical), the solution takes the simple form,

.. math::
   \rho(r,t) &= \rho_0\, r^{1/7} t^{-22/7}
   \\
   u(r,t) &= r t^{-1} 
   \\
   T(r,t) &= \rho_0\, r^{-1/7} t^{-13/7} \ .

"""

from ...base import ExactSolver, ExactSolution
from _timmes import cog8_timmes


class Cog8(ExactSolver):
    """ Computes the solution to the Cog8 problem.
    """
    
    parameters = {
        'rho0': 'initial density of the gas',
        'temp0': 'temperature of the gas',
        'alpha': r'dimensionless constant :math:`\alpha` in Eq. :eq:`lambdaDef`',
        'beta': r'dimensionless constant :math:`\beta` in Eq. :eq:`lambdaDef`',
        'gamma': 'ratio of specific heats :math:`\gamma \equiv c_p/c_v`',
        'cv': 'specific heat at constant volume [erg/g/eV]',
        }

    gamma = 1.4        
    alpha = 2.0
    beta = 1.0
    rho0 = 1.8
    temp0 = 1.4
    cv = 1.0
                
    def _run(self, r, t):

        den, tev, ener, pres, vel = cog8_timmes(t=t,
                                         r=r,
                                         rho0=self.rho0,
                                         temp0=self.temp0,
                                         alpha=self.alpha,
                                         beta=self.beta,
                                         gamma=self.gamma,
                                         cv=self.cv)

        return ExactSolution([r, den, tev, ener, pres, vel],
                             names=['position',
                                    'density',
                                    'temperature',
                                    'sie',
                                    'pressure',
                                    'velocity'])
