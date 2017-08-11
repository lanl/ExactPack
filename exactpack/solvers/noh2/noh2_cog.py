r"""Noh's 2nd solution in Python in terms of Cog1.

This is a pure Python Noh2 solution using the Cog1 solution.
"""

import numpy as np

from ...base import ExactSolver, ExactSolution
from ..cog.cog1 import Cog1


class Noh2Cog(Cog1):
    r"""Computes the solution to the general Noh2 problem in terms of Cog1.

    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'gamma': "specific heat ratio :math:`\gamma \equiv c_p/c_v`",
        'rho0': "initial density",
        'e0': "initial internal energy",
        }

    geometry = 3
    gamma = 5.0 / 3.0
    rho0 = 1.0
    e0 = 1.0
    b = 0
    Gamma = 1.0

    def __init__(self, **kwargs):
        super(Noh2Cog, self).__init__(**kwargs)
        self.temp0 = self.e0*((self.gamma - 1)/self.Gamma)

    def _run(self, r, t):
        if t>1:
            raise ValueError("The time t must be less than 1")
        tau = 1.0 - t
        soln = super(Noh2Cog, self)._run(r, tau)
        soln.velocity *= -1
        return soln
