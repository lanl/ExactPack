r"""A Kenamond1 solver in Python.

This is a pure Python implementation of the Kenamond1 solution using Numpy.

The Kenamond HE Problem 1 is used to test a code's ability to calculate
burn time tables for an unobstructed line-of-sight, single-point initiation
of a single HE region. An infinite medium of a single HE with constant
detonation velocity :math:`D` is ignited at time :math:`t=t_d` by a
single point detonator located at :math:`\vec{x}=\vec{x}_d`.

The HE light time solution for spherical propagation at a specified
detonation velocity becomes:

.. math:: t(\vec{x}) = t_d + \frac{|| \vec{x} - \vec{x}_d ||}{D}

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class Kenamond1(ExactSolver):
    r"""Computes the general solution to the Kenamond HE Problem 1.

    Can be used to solve the Kenamond1 problem for a high explosive
    with any detonation velocity, :math:`D`, detonator location,
    :math:`\vec{x}_d`, and detonation time, :math:`t_d`. Supports
    2D and 3D Cartesian solutions.

    Default values are selected to be consistent with the problem 
    definition in [Kenamond2011]_. Default values are: **geometry** \ 
    :math:`=2`, :math:`D=1.0`, :math:`x_d=(0.0, 0.0)`, and \ 
    :math:`t_d=0.0`.

    """

    parameters = {
        'geometry': "2=two-dimensional, 3=three-dimensional",
        'D': "detonation velocity of the HE",
        'x_d': "detonator location, enter as a tuple: " +
        "(:math:`x`, :math:`y` [, :math:`z`])",
        't_d': "detonation time"
        }

    # Default values

    geometry = 2
    D = 1.0
    x_d = (0.0, 0.0)
    t_d = 0.0

    def __init__(self, **kwargs):

        """Input evaluation points as an :math:`N` x **geometry** 2D array of
        positions:
        [ [:math:`x_0`, :math:`y_0` (, :math:`z_0`)],
        [:math:`x_1`, :math:`y_1` (, :math:`z_1`)], ...,
        [:math:`x_N`, :math:`y_N` (, :math:`z_N`)] ].

        A time value must be input in order to use the ExactPack
        machinery. The time value is ignored in the calculation.

        """

        super(Kenamond1, self).__init__(**kwargs)

        self.x_d = np.array(self.x_d)

        if self.geometry not in [2, 3]:
            raise ValueError("geometry must be 2 or 3")

        if self.D <= 0:
            raise ValueError('Detonation velocity must be > 0')

        if len(self.x_d) != self.geometry:
            raise ValueError('Detonator location and geometry dimensions ' +
                             'must be compatible')

    def _run(self, xylist, t):

        delta = xylist - self.x_d
        btime = np.zeros(len(delta))

        for index, vec in enumerate(delta):
            btime[index] = self.t_d + np.sqrt(np.dot(vec, vec)) / self.D

        if self.geometry == 2:
            return ExactSolution([xylist[:, 0], xylist[:, 1], btime],
                                 names=['position_x',
                                        'position_y',
                                        'burntime'],
                                 jumps=[])
        elif self.geometry == 3:
            return ExactSolution([xylist[:, 0], xylist[:, 1],
                                  xylist[:, 2], btime],
                                 names=['position_x',
                                        'position_y',
                                        'position_z',
                                        'burntime'],
                                 jumps=[])
