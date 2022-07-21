r"""The Planar Sandwich [Dawes2016]_ implemented in Fortran. This is a python interface
to Alan Dawes' fortran code.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution
from ._dawes import planar_sand


class PlanarSandwichDawes(ExactSolver):

    r"""Computes the solution to Planar Sandwich.
    This is direct python interface to fortran source code provided Alan Dawes.
    """

    parameters = {
        'kappa': "Thermal diffusivity",
        'Nsum': "Number of terms to include in the sum.",
        'L': "Length of the square in the y-direction",
        'TB': "Temperature BC on the bottom of the square at y=0",
        'TT': "Temperature BC on the top of the square at y=L",
        }

    kappa = 1.0
    Nsum = 10000
    L = 2.0
    TB = 1.0
    TT = 0.0

    def __init__(self, **kwargs):

        super(PlanarSandwichDawes, self).__init__(**kwargs)

    def _run(self, y, t):
        temperature = planar_sand(t, self.Nsum, self.L, self.TB, self.TT, y)

        return ExactSolution([y, temperature],
                                names=['position', 'temperature']
                             )
