"""A Fortran based Sedov solver.

This is a Python interface for the Sedov problem. The Fortran source code was
adapted from `Frank Timmes website <http://cococubed.asu.edu/research_pages/sedov.shtml>`_.

"""

from ...base import ExactSolver, ExactSolution
from _timmes import sed_1d
from numpy import linspace
from scipy.interpolate import interp1d


class Sedov(ExactSolver):
    """ Computes the solution to the Sedov problem.
    """

    parameters = {
        'geometry': '1=planar, 2=cylindrical, 3=spherical',
        'gamma': 'specific heat ratio :math:`\gamma \equiv c_p/c_v`',
        'eblast': 'total amount of energy deposited at the origin at \
        time zero',
        }

    geometry = 3
    gamma = 7.0 / 5.0
    eblast = 0.851072

    def __init__(self, interpolation_points=5000, **kwargs):
        """Initialize the Sedov solver class.

        :param integer interpolation_points: interpolate the Sedov solution
          from a representation using this number of points (faster, but
          not quite as accurate).  If 0, then don't do interpolation.
        """

        if 'geometry' in kwargs and not kwargs['geometry'] in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")
        self.interpolation_points = interpolation_points

        super(Sedov, self).__init__(**kwargs)

    def _run(self, r, t):
        binterp = False
        if self.interpolation_points and len(r) > self.interpolation_points:
            rlong = r
            r = linspace(min(rlong), max(rlong),
                         self.interpolation_points)
            binterp = True
        soln = sed_1d(time=t,
                      xpos=r,
                      eblast=self.eblast,
                      omega_in=0.0,
                      xgeom_in=self.geometry,
                      rho0=1.0,
                      vel0=0.0,
                      ener0=0.0,
                      pres0=1.4e-22,
                      cs0=0.0,
                      gam0=self.gamma)
        if binterp:
            soln = interp1d(r, soln)(rlong)
            r = rlong

        den, ener, pres, vel, cs = soln

        return ExactSolution([r, den, ener, pres, vel, cs],
                             names=['position',
                                    'density',
                                    'sie',
                                    'pressure',
                                    'velocity',
                                    'sound'])

