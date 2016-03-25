r"""A Kenamond3 solver in Python.

This is a pure Python implementation of the Kenamond3 solution using Numpy.

The Kenamond HE Problem 3 is used to test a code's ability to calculate
burn time tables for a single-point initiation of a single HE region
surrounding an inert region. An infinite medium of a single HE with
constant detonation velocity :math:`D` surrounds an inert spherical
obstacle of radius :math:`R` centered at the origin. A single point
detonator located at :math:`\vec{x}=\vec{x}_d` is ignited at time
:math:`t=t_d`. The detonator must be located outside of the inert region.

The HE material can be divided into two solution regions: the material
in the line-of-sight of the detonator and the material in the shadow of
the inert object. Let :math:`t_1` designate the line-of-sight region
and :math:`t_2` designate the shadow region.

The burntime solution at the point :math:`\vec{p}=` (:math:`x`, :math:`y`
[, :math:`z`]) is as follows:

.. math::

   t(\vec{p}) = \left\{ \begin{array}{c}
                         t_1 \;\; \mathrm{if} \;\; \theta \le 0  \\
                          t_2 \;\; \mathrm{if} \;\; \theta > 0
                         \end{array}
               \right.

where,

.. math::

   t_1(\vec{p}) = t_d + \frac{||\vec{p}-\vec{x}_{d}||}{D}

   t_2(\vec{p}) = t_d + \frac{l_{da} + l_{ab} + l_{bp}}{D}

   \theta = \pi - \alpha - \beta - \psi

   \alpha = \arccos \left(- \frac{\vec{p} \cdot \vec{x}_{d}}
   {||\vec{p}|| ||\vec{x}_{d}||}\right)

   \beta = \arccos \left(\frac{R}{l_{op}}\right)

   \psi = \arccos \left(\frac{R}{l_{od}}\right)

   l_{da} = \sqrt{{l_{od}}^{2} - R^2}

   l_{ab} = R \theta

   l_{bp} = \sqrt{{l_{op}}^{2} - R^2}

   l_{op} = ||\vec{p}|| = \sqrt{x^2 + y^2 (+ z^2)}

   l_{od} = ||\vec{x}_{d}|| = \sqrt{{x_d}^2 + {y_d}^2 (+ {z_d}^2)}

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class Kenamond3(ExactSolver):
    r"""Computes the general solution to the Kenamond HE Problem 3.

    The inert obstacle is assumed to be a spherical region centered at
    the origin. The solver will only accept points outside of or on the
    surface of the inert region for burn time calculation.

    Default values are selected to be consistent with the problem
    definition in [Kenamond]_. Default values are
    **geometry** :math:`=2`, :math:`R=3.0`, :math:`D=2.0`,
    :math:`x_d=(0.0, 5.0)`, and :math:`t_d=0.0`.

    """

    parameters = {
        'geometry': "2=two-dimensional, 3=three-dimensional",
        'R': "radius of inert obstacle",
        'D': "detonation velocity of the HE",
        'x_d': "detonator location, enter as a tuple: " +
        "(:math:`x`, :math:`y` [, :math:`z`])",
        't_d': "detonation time"
        }

    # Default values

    geometry = 2
    R = 3.0
    D = 2.0
    x_d = (0.0, 5.0)
    t_d = 0.0

    def __init__(self, **kwargs):

        """Input evaluation points as an :math:`N` x **geometry** 2D array of
        positions: [ [:math:`x_0`, :math:`y_0` (, :math:`z_0`)],
        [:math:`x_1`, :math:`y_1` (, :math:`z_1`)], ...,
        [:math:`x_N`, :math:`y_N` (, :math:`z_N`)] ]. These
        points must be outside of or on the surface of the inert region.

        A time value must be input in order to use the ExactPack
        machinery. The time value is ignored in the calculation.

        """

        super(Kenamond3, self).__init__(**kwargs)

        self.x_d = np.array(self.x_d)

        if self.geometry not in [2, 3]:
            raise ValueError("geometry must be 2 or 3")

        if self.R <= 0:
            raise ValueError('Inert obstacle radius must be > 0')

        if self.D <= 0:
            raise ValueError('Detonation velocity must be > 0')

        if len(self.x_d) != self.geometry:
            raise ValueError('Detonator location and geometry dimensions ' +
                             'must be compatible')

        if np.linalg.norm(self.x_d) <= self.R:
            raise ValueError('Detonator must be outside of inert region')

    def _run(self, xylist, t):

        btime = np.zeros(len(xylist))

        for index, vec in enumerate(xylist):
            l_op = np.sqrt(np.dot(vec, vec))
            if l_op < self.R:
                raise ValueError('HE grid points must be outside of ' +
                                 'inert region')
            l_od = np.linalg.norm(self.x_d)

            # assume in line-of-sight of detonator
            temp = vec - self.x_d
            btime[index] = self.t_d + np.sqrt(np.dot(temp, temp)) / self.D

            # determine if in inert shadow and change time
            l_da = np.sqrt(l_od ** 2 - self.R ** 2)
            l_bp = np.sqrt(l_op ** 2 - self.R ** 2)
            psi = np.arccos(self.R / l_od)
            beta = np.arccos(self.R / l_op)
            alpha = np.arccos(-np.dot(vec, self.x_d) / (l_od * l_op))
            theta = np.pi - alpha - beta - psi
            if theta > 0:
                l_ab = self.R * theta
                btime[index] = self.t_d + (l_da + l_ab + l_bp) / self.D

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
