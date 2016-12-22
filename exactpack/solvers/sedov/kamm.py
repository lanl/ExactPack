"""A Fortran based Sedov solver.

NOTE: Currently, the Kamm solver does not return the correct value of the
physical variables at the shock location, for at least some cases. The
development team recommends not using the Kamm solver until this is resolved.
(see unit test exactpack.tests.test_sedov_kamm.TestSedovKammShock)

This is  Python interface for the Sedov problem. The Fortran
source is Jim Kamm's original Fortran Sedov solver, from which Timmes' source
code was based. The Kamm code also provides output in dimensionless self
similar variables :math:`\lambda`, :math:`V`, :math:`f`, :math:`g`, :math:`h`.
To access the selfsimilar variables: solver = Sedov() followed by solution =
solver.self_similar(lam, t), where lam is a spatial array in dimensionless
selfsimilar form.

The selfsimilar output has been verified against the values reported in Sedov's
book and Tables 1, 2, and 3 of the paper LA-UR-00-6055. Timmes' code has been
chosen for the default, as it has better behavior in certain asymptotic
regimes.

"""

from ...base import ExactSolver, ExactSolution
from _kamm import sedov_kamm_1d
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
        'rho0': 'initial density',
        'omega': 'initial density power-law exponent,\
                  :math:`\\rho \equiv \\rho_0 r^{-\omega}`',
        }

    geometry = 3
    gamma = 7.0 / 5.0
    eblast = 0.851072
    rho0 = 1.0
    omega = 0.0

    def __init__(self, interpolation_points=5000, **kwargs):
        """Initialize the Sedov solver class.

        :param integer interpolation_points: interpolate the Sedov solution
          from a representation using this number of points (faster, but
          not quite as accurate).  If 0, then don't do interpolation.
        """

        super(Sedov, self).__init__(**kwargs)

        if self.geometry not in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        self.interpolation_points = interpolation_points

    def _run(self, r, t):
        binterp = False
        if self.interpolation_points and len(r) > self.interpolation_points:
            rlong = r
            r = linspace(min(rlong), max(rlong), self.interpolation_points)
            binterp = True
        soln = sedov_kamm_1d(time=t,
                             xval=r,
                             eblast=self.eblast,
                             rho1=self.rho0,
                             omega=self.omega,
                             u1=0.0,
                             e1=0.0,
                             p1=0.0,
                             c1=0.0,
                             gamval=self.gamma,
                             icoord=self.geometry,
                             self_sim=False
                             )

        if binterp:
            soln = interp1d(r, soln)(rlong)
            r = rlong

        den, ener, pres, vel = soln

        return ExactSolution([r, den, ener, pres, vel],
                             names=['position',
                                    'density',
                                    'specific_internal_energy',
                                    'pressure',
                                    'velocity'
                                    ]
                             )

    # Kamm's Sedov provides access to similarity variables. See LA-UR-00-6055
    # for more details about these variables.
    def self_similar(self, lam, t):
        soln = sedov_kamm_1d(time=t,
                             xval=lam,
                             eblast=self.eblast,
                             rho1=1.0,
                             omega=0.0,
                             u1=0.0,
                             e1=0.0,
                             p1=0.0,
                             c1=0.0,
                             gamval=self.gamma,
                             icoord=self.geometry,
                             self_sim=True
                             )

        g, V, h, f = soln

        return ExactSolution([lam, g, V, h, f],
                             names=['lam',
                                    'g',
                                    'V',
                                    'h',
                                    'f'
                                    ]
                             )
