r"""A DSD Rate Stick solver in Python.

This is a pure Python implementation of the Rate Stick solution using Numpy.

***********description here***************

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class RateStick(ExactSolver):
    r"""Computes the numerical solution to the Rate Stick Problem.

    ******description of solver use here****************

    ******description of default values here************
    **** Assume time of specified initial wave configuration is 0. ****

    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical",
        'R': "radius of HE",
        'D_CJ': "nominal detonation velocity of the HE",
        'alpha': "coefficient of linear detonation velocity deviance",
        'omega_c': "DSD edge angle between HE and inert",
        'omega_s': "DSD free-surface angle (HE and vacuum)",
        'omega_fix': "DSD fixed-surface angle (rigid boundary)"
        'x_d': "detonator location, enter as a tuple: " +
        "(:math:`x`, :math:`y` [, :math:`z`])",
        'r_d': "initial detonation front radius",
        }

    # Default values

    geometry = 2       # geometry = n + 1 (n is used in Bdzil document)
    R = 1.0
    D_CJ = 1.0
    alpha = 0.1
    omega_c = 0.7854     # pi / 4
    omega_s = 0.5
    omega_fix = 1.5708    # pi / 2
    x_d = (0.0, -24.0)
    r_d = np.sqrt(626.0)

    def __init__(self, **kwargs):

        """Input evaluation points as an :math:`N` x **geometry** 2D array of
        positions:
        [ [:math:`x_0`, :math:`y_0` (, :math:`z_0`)],
        [:math:`x_1`, :math:`y_1` (, :math:`z_1`)], ...,
        [:math:`x_N`, :math:`y_N` (, :math:`z_N`)] ].

        A time value must be input in order to use the ExactPack
        machinery. The time value is ignored in the calculation.

        """

        super(RateStick, self).__init__(**kwargs)

        self.x_d = np.array(self.x_d)

        if self.geometry not in [1, 2]:
            raise ValueError("geometry must be 1 or 2")

        if self.D_CJ <= 0:
            raise ValueError('Detonation velocity must be > 0')

        if len(self.x_d) != 2:
            raise ValueError('Detonator location and geometry dimensions ' +
                             'must be compatible')

        # *** Other tests on input values here ****

    def _run(self, xylist, t):

        # ****  Solver here *****

        return ExactSolution([xylist[:, 0], xylist[:, 1], btime],
                             names=['position_x',
                                    'position_y',
                                    'burntime'],
                             jumps=[])
