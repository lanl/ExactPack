r"""A DSD Cylindrical Expansion solver in Python.

This is a pure Python implementation of the Cylindrical Expansion solution
using Numpy.

***********description here***************

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class CylindricalExpansion(ExactSolver):
    r"""Computes the numerical solution to the Cylindrical Expansion Problem.

    ******description of solver use here****************

    ******description of default values here************
    **** Assume time of specified initial wave configuration is 0. ****

    """

    parameters = {
        'geometry': "2=cylindrical",
        'D_CJ_1': "nominal detonation velocity of inner HE",
        'D_CJ_2': "nominal detonation velocity of outer HE",
        'alpha_1': "coefficient of linear det vel deviance, inner HE",
        'alpha_2': "coefficient of linear det vel deviance, inner HE",
        'omega_c': "DSD edge angle between HE and inert",
        'omega_s': "DSD free-surface angle (HE and vacuum)",
        'omega_fix': "DSD fixed-surface angle (rigid boundary)"
        'x_d': "detonator location, enter as a tuple: " +
        "(:math:`x`, :math:`y` [, :math:`z`])",
        'r_d': "initial detonation front radius",
        }

    # Default values

    geometry = 2
    D_CJ_1 = 0.5
    D_CJ_2 = 1.0
    alpha_1 = 0.1
    alpha_2 = 0.1
    omega_c = 0.7854     # pi / 4
    omega_s = 0.5
    omega_fix = 1.5708    # pi / 2
    x_d = (0.0, 0.0)
    r_d = 1.0

    def __init__(self, **kwargs):

        """Input evaluation points as an :math:`N` x **geometry** 2D array of
        positions:
        [ [:math:`x_0`, :math:`y_0` (, :math:`z_0`)],
        [:math:`x_1`, :math:`y_1` (, :math:`z_1`)], ...,
        [:math:`x_N`, :math:`y_N` (, :math:`z_N`)] ].

        A time value must be input in order to use the ExactPack
        machinery. The time value is ignored in the calculation.

        """

        super(CylindricalExpansion, self).__init__(**kwargs)

        self.x_d = np.array(self.x_d)

        if self.geometry not in [2]:
            raise ValueError("geometry must be 2")

        if self.D_CJ_1 <= 0:
            raise ValueError('Detonation velocity for inner HE must be > 0')

        if self.D_CJ_2 <= 0:
            raise ValueError('Detonation velocity for outer HE must be > 0')

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
