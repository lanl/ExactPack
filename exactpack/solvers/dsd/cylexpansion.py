r"""A DSD Cylindrical Expansion solver in Python.

This is a pure Python implementation of the Cylindrical Expansion solution
using Numpy.

The DSD Cylindrical Expansion problem is used to test a code's algorithm for
updating the solution for the level-set equation. It is insensitive to any
issues there may be with boundary conditions, as the only boundaries are
parallel to level sets of the governing equation. It also tests the level-set
solution algorithm for a multiple HE region.

A cylindrical HE tube with inner radius :math:`r_1` and outer radius
:math:`r_2` is surrounded by an adjoining cylindrical tube of a second HE.
Thus, the second HE tube has an inner radius of :math:`r_2` and extends
indefinitely in the radial direction. At time :math:`t_d`, the inner HE is
initiated by a circle detonator of radius :math:`r_1`. Without loss of
generality, the entire system is assumed to be centered at the origin.

The only internal boundary in the complete system (between the HE materials)
is parallel to the burn front. Thus, none of the boundary angles need to be
defined for this solver.

For each HE material, the velocity of the detonation wave in the shock-normal
direction, :math:`D_n`, is described by a linear deviation from the nominal
constant Chapman-Jouguet detonation shock speed :math:`D_{CJ}`. The deviation
depends on the curvature, :math:`\kappa`, of the detonation shock front. For
the inner HE material,

.. math:: D_{n_1} = D_{{CJ}_1} - \alpha_1 \kappa

For the outer HE material,

.. math:: D_{n_2} = D_{{CJ}_2} - \alpha_2 \kappa


Under the linear detonation velocity condition and the cylindrical HE
configuration given above, the detonation trajectory is described by

.. math:: \frac{dr}{dt} = D_{CJ} - \alpha \kappa = D_{CJ} - \frac{\alpha}{r}

The solution to the 1D ODE in each material is given by

.. math::
    D_{{CJ}_i} (t-t_{0_i}) = (r-r_{0_i}) + \frac{\alpha_i}{D_{{CJ}_i} }
    ln \left(\frac{r-\frac{\alpha_i}{D_{{CJ}_i}}}{r_{0_i}-\frac{\alpha_i}
    {D_{{CJ}_i}}} \right), r>\frac{\alpha_i}{D_{{CJ}_i}}, i = 1, 2

where :math:`(t_{0_i}, r_{0_i})` are the time and radius when detonation is
initiated in each of the explosives.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class CylindricalExpansion(ExactSolver):
    r"""Computes the numerical solution to the Cylindrical Expansion Problem.

    The HE regions are assumed to be two concentric cylindrical tubes centered
    at the origin, modeled in the :math:`r \theta`-plane. All radii are assumed
    to be positive and large enough to avoid the singularity at the origin,
    i.e. :math:`r_1 > \frac{\alpha_1}{D_{{CJ}_1}}` and :math:`r_2 >
    \frac{\alpha_2}{D_{{CJ}_2}}`. No boundary angles are necessary.

    The nominal detonation velocities of both HEs, :math:`D_{{CJ}_i}`, must
    be positive. The linear coefficients, :math:`\alpha_i`, of detonation
    velocity deviance must also be positive.

    Default values are selected to be consistent with the problem definition
    in [Bdzil]_. Default values are **geometry** :math:`=2`, :math:`r_1=1.0`,
    :math:`r_2=2.0`, :math:`D_{{CJ}_1}=0.5`, :math:`D_{{CJ}_2}=1.0`,
    :math:`\alpha_1=0.1`, :math:`\alpha_2=0.1` and :math:`t_d=0.0`.
    """

    parameters = {
        'geometry': "2=cylindrical",
        'r_1': "inner radius of HE1",
        'r_2': "radius of interface between HE1 and HE2",
        'D_CJ_1': "nominal detonation velocity of inner HE",
        'D_CJ_2': "nominal detonation velocity of outer HE",
        'alpha_1': "linear detonation velocity deviance coefficient for HE1",
        'alpha_2': "linear detonation velocity deviance coefficient for HE2",
        't_d': "initial detonation time",
        }

    # Default values

    geometry = 2
    r_1 = 1.0
    r_2 = 2.0
    D_CJ_1 = 0.5
    D_CJ_2 = 1.0
    alpha_1 = 0.1
    alpha_2 = 0.1
    t_d = 0.0

    def __init__(self, **kwargs):

        """Input evaluation points as an :math:`N` x :math:`2` 2D array of
        positions:
        [ [:math:`x_0`, :math:`y_0`],
        [:math:`x_1`, :math:`y_1`], ...,
        [:math:`x_N`, :math:`y_N`] ].

        A time value must be input in order to use the ExactPack
        machinery. The time value is ignored in the calculation.

        """

        super(CylindricalExpansion, self).__init__(**kwargs)

        if self.geometry not in [2]:
            raise ValueError("geometry must be 2")

        if self.r_1 <= 0:
            raise ValueError('Inner radius of HE1 must be > 0')

        if self.r_2 <= 0:
            raise ValueError('Radius of interface between HE1 and HE2 ' +
                             'must be > 0')

        if self.r_2 <= self.r_1:
            raise ValueError('Radius of interface between HE1 and HE2 ' +
                             'must be > inner radius')

        if self.D_CJ_1 <= 0:
            raise ValueError('Detonation velocity for inner HE must be > 0')

        if self.D_CJ_2 <= 0:
            raise ValueError('Detonation velocity for outer HE must be > 0')

        if self.alpha_1 < 0:
            raise ValueError('Alpha for HE1 must be >= 0')

        if self.alpha_2 < 0:
            raise ValueError('Alpha for HE2 must be >= 0')

    def _run(self, xylist, t):

        veldev_1 = self.alpha_1 / self.D_CJ_1
        t_mid = self.t_d + ((self.r_2 - self.r_1) +
                            veldev_1 * np.log((self.r_2 - veldev_1) /
                                              (self.r_1 - veldev_1))
                            ) / self.D_CJ_1
        veldev_2 = self.alpha_2 / self.D_CJ_2

        btime = np.zeros(len(xylist))

        for index, vec in enumerate(xylist):
            rpt = np.sqrt(np.dot(vec, vec))

            if rpt < self.r_1:
                btime[index] = self.t_d

            elif rpt < self.r_2:
                btime[index] = self.t_d + ((rpt - self.r_1) +
                                           veldev_1 *
                                           np.log((rpt - veldev_1) /
                                                  (self.r_1 - veldev_1))
                                           ) / self.D_CJ_1
            else:
                btime[index] = t_mid + ((rpt - self.r_2) +
                                        veldev_2 *
                                        np.log((rpt - veldev_2) /
                                               (self.r_2 - veldev_2))
                                        ) / self.D_CJ_2

        return ExactSolution([xylist[:, 0], xylist[:, 1], btime],
                             names=['position_x',
                                    'position_y',
                                    'burntime'],
                             jumps=[])
