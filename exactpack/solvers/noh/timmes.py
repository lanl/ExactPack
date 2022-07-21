"""A Fortran based Noh solver.

This is a Python interface to the Noh solution code from `Frank Timmes'
website <http://cococubed.asu.edu/research_pages/noh.shtml>`_.  The code
was developed at Los Alamos National Laboratory, and released under
LA-CC-05-101.  A full description of the solution and code listing is
in a Los Alamos report [Timmes2005]_.
"""

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition
from ._timmes import noh_1d


class Noh(ExactSolver):
    r"""Computes the solution to the general Noh problem.

    The functionality of this class is completely superseded by
    :class:`exactpack.solvers.noh1.Noh`, the only difference being
    that this class is a wrapper to a Fortran library.  This version
    is provided primarily as a template example to developers to
    demonstrate how to add an exact solution that is computed by an
    external Fortran code. The default geometry is spherical, with 
    :math:`\gamma=5/3`. The parameters values for the density and
    gas velocity are are :math:`\rho_0=1` and :math:`u_0=1`.
    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        }

    geometry = 3
    gamma = 5.0/3.0

    def __init__(self, **kwargs):

        if 'geometry' in kwargs and not kwargs['geometry'] in [1, 2, 3]:
            raise ValueError("geometry must be 1, 2, or 3")

        super(Noh, self).__init__(**kwargs)

    def _run(self, r, t):

        density, sie, pressure, velocity, jumps = noh_1d(
            t, r, 1.0, -1.0, self.gamma, self.geometry)

        return ExactSolution([r, density, sie, pressure, velocity],
                             names=['position',
                                    'density',
                                    'sie',
                                    'pressure',
                                    'velocity'],
                             jumps=[JumpCondition(jumps[0],
                                    "Shock",
                                    density=(jumps[1], jumps[2]),
                                    sie=(jumps[3], jumps[4]),
                                    pressure=(jumps[5], jumps[6]),
                                    velocity=(jumps[7], jumps[8]))
                                    ]
        )
