r"""A Kenamond2 solver in Python.

This is a pure Python implementation of the Kenamond2 solution using Numpy.

The Kenamond HE Problem 2 is used to test a code's ability to calculate burn
time tables for an unobstructed line-of-sight, multi-point initiation of a
multiple HE region. An HE sphere of radius :math:`R` centered at the origin
with constant detonation velocity :math:`D_1` is surrounded by an infinite
medium of a second HE with constant detonation velocity :math:`D_2`. Five
point detonators located at :math:`\vec{x}=\vec{x}_{d_i}` are ignited at
times :math:`t=t_{d_i}`. The detonators are located on the :math:`y`-axis
in the 2D test and on the :math:`z`-axis in the 3D test.

This solver implementation is based on some specific conditions.

* The detonation velocity of the inner HE region is higher than the
  detonation velocity of the outer HE region: :math:`D_1 > D_2`.
* Detonator 3 is located at the origin.
* Only detonator 3 is located inside the inner HE region.
* The detonation times for the other detonators must allow the burn wave from
  detonator 3 to exit the inner HE region before interaction.


In order to meet the last criteria, each detonation time must satisfy the
following condition:

.. math::
    t_{d_i} \ge t_{d_3} + R \left(\frac{1}{D_1}+\frac{1}{D_2}\right)
    - \frac{|a_{d_i}|}{D_2}

where :math:`i=1,2,4,5` and :math:`a_{d_i}` is the axial location of
detonator :math:`d_i` (:math:`a_{d_i}=y_{d_i}` in 2D and
:math:`a_{d_i}=z_{d_i}` in 3D).

With these conditions, the burntime solution at the point
:math:`\vec{p}=` (:math:`x`, :math:`y` [, :math:`z`]) is as follows;

.. math:: t(\vec{p}) = \min(t_1,t_2,\max(t_3,t_4),t_5,t_6)

where,

.. math::

   t_1(\vec{p}) = t_{d_1} + \frac{||\vec{p}-\vec{x}_{d_1}||}{D_2}

   t_2(\vec{p}) = t_{d_2} + \frac{||\vec{p}-\vec{x}_{d_2}||}{D_2}

   t_3(\vec{p}) = t_{d_3} + \frac{||\vec{p}-\vec{x}_{d_3}||}{D_1}

   t_4(\vec{p}) = t_{d_3} + \frac{||\vec{p}-\vec{x}_{d_3}||}{D_2} +
   R \left(\frac{1}{D_1} - \frac{1}{D_2}\right)

   t_5(\vec{p}) = t_{d_4} + \frac{||\vec{p}-\vec{x}_{d_4}||}{D_2}

   t_6(\vec{p}) = t_{d_5} + \frac{||\vec{p}-\vec{x}_{d_5}||}{D_2}

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class Kenamond2(ExactSolver):
    r"""Computes the general solution to the Kenamond HE Problem 2.

    The detonator locations are on the :math:`y`-axis in 2D and on
    the :math:`z`-axis in 3D. Detonator 3 will be located at the origin.
    User will specify the other four :math:`y` or :math:`z` axial detonator
    locations and all five detonation times.

    Default values are selected to be consistent with the problem
    definition in [Kenamond]_. Default values are
    **geometry** :math:`=2`, :math:`R=3.0`, :math:`D_1=2.0`,
    :math:`D_2 = 1.0`, :math:`t_d=[2.0, 1.0, 0.0, 1,0, 2.0]`, and
    **dets** :math:`=[10.0, 5.0, -5.0, -10.0]`.

    """

    parameters = {
        'geometry': "2=two-dimensional, 3=three-dimensional",
        'R': "radius of inner HE",
        'D1': "detonation velocity of the inner HE",
        'D2': "detonation velocity of the outer HE, D2 < D1",
        'dets': "axial detonator locations, enter as a list: " +
        "[:math:`a_{d_1}`, :math:`a_{d_2}`, :math:`a_{d_4}`, " +
        ":math:`a_{d_5}`]. Detonator 3 will be automatically inserted " +
        "at the origin.",
        't_d': "detonation times, enter as a list: " +
        "[:math:`t_{d_1}`, :math:`t_{d_2}`, :math:`t_{d_3}`, " +
        ":math:`t_{d_4}`, :math:`t_{d_5}`]"
        }

    # Default values

    geometry = 2
    R = 3.0
    D1 = 2.0
    D2 = 1.0
    dets = [10.0, 5.0, -5.0, -10.0]
    t_d = [2.0, 1.0, 0.0, 1.0, 2.0]

    def __init__(self, **kwargs):

        """Input evaluation points as an :math:`N` x **geometry** 2D array of
        positions:
        [ [:math:`x_0`, :math:`y_0` (, :math:`z_0`)],
        [:math:`x_1`, :math:`y_1` (, :math:`z_1`)], ...,
        [:math:`x_N`, :math:`y_N` (, :math:`z_N`)] ].

        A time value must be input in order to use the ExactPack
        machinery. The time value is ignored in the calculation.

        """

        super(Kenamond2, self).__init__(**kwargs)

        if self.geometry not in [2, 3]:
            raise ValueError("geometry must be 2 or 3")

        if self.R <= 0:
            raise ValueError('Inner HE radius must be > 0')

        if self.D1 <= 0:
            raise ValueError('Detonation velocity 1 must be > 0')

        if self.D2 <= 0:
            raise ValueError('Detonation velocity 2 must be > 0')

        if self.D1 < self.D2:
            raise ValueError('D1 must be > D2')

        if len(self.dets) != 4:
            raise ValueError('4 detonator locations must be specified')

        detz = np.array(self.dets)
        detnumber = np.array([1, 2, 4, 5])

        for index, det in enumerate(detz):
            if abs(det) <= self.R:
                raise ValueError('Detonator ' + str(detnumber[index]) +
                                     ' must be in outer HE region')

        if len(self.t_d) != 5:
            raise ValueError('5 detonation times must be specified')

        t2check = np.delete(np.array(self.t_d), 2)
        tcompare = self.t_d[2] + self.R * (1.0 / self.D1 + 1.0 / self.D2)  \
                   - abs(detz) / self.D2
        for i in range(4):
            if t2check[i] < tcompare[i]:
                raise ValueError('Detonation time ' + str(detnumber[i]) +
                                 ' must be no less than ' + str(tcompare[i]))

        # Make '5 x geometry' 2D array of dets
        self.dets = np.zeros((5, self.geometry))
        self.dets[0, -1] = detz[0]
        self.dets[1, -1] = detz[1]
        self.dets[3, -1] = detz[2]
        self.dets[4, -1] = detz[3]

    def _run(self, xylist, t):

        btime = np.zeros(len(xylist))

        for index, vec in enumerate(xylist):
            temp = vec - self.dets[2]                         # det 3
            dtop = np.sqrt(np.dot(temp, temp))
            btime[index] = self.t_d[2] + dtop / self.D1       # bt3
            btcheck = self.t_d[2] + dtop / self.D2  \
                  + self.R * (1.0 / self.D1 - 1.0 / self.D2)  # bt4
            btime[index] = max(btime[index], btcheck)

            temp = vec - self.dets[0]                         # det 1
            dtop = np.sqrt(np.dot(temp, temp))
            btcheck = self.t_d[0] + dtop / self.D2            # bt1
            btime[index] = min(btime[index], btcheck)

            temp = vec - self.dets[1]                         # det 2
            dtop = np.sqrt(np.dot(temp, temp))
            btcheck = self.t_d[1] + dtop / self.D2            # bt2
            btime[index] = min(btime[index], btcheck)

            temp = vec - self.dets[3]                         # det 4
            dtop = np.sqrt(np.dot(temp, temp))
            btcheck = self.t_d[3] + dtop / self.D2            # bt5
            btime[index] = min(btime[index], btcheck)

            temp = vec - self.dets[4]                         # det 5
            dtop = np.sqrt(np.dot(temp, temp))
            btcheck = self.t_d[4] + dtop / self.D2            # bt6
            btime[index] = min(btime[index], btcheck)

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
