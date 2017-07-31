r"""The Half Planar Sandwich

"""

import numpy as np

from ...base import ExactSolver, ExactSolution
from . import Rod1D

class PlanarSandwichHalf(Rod1D):

    r"""Computes the solution to the Hot Planar Sandwich heat conduction problem.
    """

    parameters = {
        'kappa': Rod1D.parameters['kappa'],
        'Nsum': Rod1D.parameters['Nsum'],
        'L': Rod1D.parameters['L'],
        'TB': "Temperature at the bottom ",
        'FT': "The flux at the top",
        'TL': Rod1D.parameters['TL'],
        'TR': Rod1D.parameters['TR'],
        }

    alpha1 = 1
    beta1 = 0
    alpha2 = 0
    beta2 = 1
    Nsum = 10000
    L = 2.0
    TB = 1.0
    FT = 0.0
    TL = 3.0
    TR = 3.0
    geometry = 2

    def __init__(self, **kwargs):
        # We need to rename gamma1 and gamma2 for this case, and we need these
        # set BEFORE the call to the initializer of the base class.  So we check
        # if they are set in the kwargs dictionary, and if so use those values,
        # and otherwise use the defaults defined above.
        self.gamma1 = kwargs.get("TB", self.TB)
        self.gamma2 = kwargs.get("FT", self.FT)
        super(PlanarSandwichHalf, self).__init__(**kwargs)
