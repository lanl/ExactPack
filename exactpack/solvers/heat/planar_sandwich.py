r"""The Planar Sandwich [Dawes2016]_ implemented in python, which is the default in ExactPack.

This is the planar sandwich heat conduction problem, a heat flow problem in 2D rectangular
coordinates x and y. See Ref. [Dawes2016]_ for details. The problem consists of three
material layers aligned along the :math:`y`-direction. The outer two layers do not conduct
heat (:math:`\kappa=0`), while the inner layer is heat conducting with :math:`\kappa > 0`,
forming a sandwich of conducting and non-conducting materials. When the initial temperatures
and boundary conditions do not vary along the :math:`x`-direction, on both the upper and
lower boundaries, symmetry arguments imply that heat only flows in the :math:`y`-direction,

.. math::
  \frac{\partial T}{\partial t} = \kappa \, \frac{\partial^2 T}{\partial y^2}
  \ ,

and the problem reduces to a 1D heat equation.
The lower side of the sandwich along :math:`y=0` is held at a constant temperature
:math:`T_0`, while the temperature of the upper portion of the sandwich at
:math:`y=L` is :math:`T_L`, providing the boundary condtions

.. math::
  T(0,t) &= T_0
  \\[8pt]
  T(L,t) &= T_L
  \ .

As for initial conditions, the sandwich starts at zero temperature,

.. math::
  T(x,0) = 0
  \ .

Variable separation and Fourier analysis yield the solution

.. math::
  T(y,t)
  &=
  T_0 +  \frac{T_L - T_0}{L}\, y +
  \sum_{n=1}^\infty \left[\frac{2 T_L\,(-1)^n - 2 T_0}{n \pi}\right]
  \sin k_n y \,
  e^{- \kappa \, k_n^2 t }
  \\[5pt]
  k_n
  &=
  \frac{n \pi}{L}
  \ .

Not surprisingly, this is the solution BC1 of the 1D rod (modulo a relative sign).

"""

import numpy as np

from ...base import ExactSolver, ExactSolution
from . import Rod1D


class PlanarSandwich(Rod1D):

    r"""Computes the solution to the Planar Sandwich heat conduction problem.
    This is direct python interface to fortran source code provided Alan Dawes.
    """

    parameters = {
        'kappa': Rod1D.parameters['kappa'],
        'Nsum': Rod1D.parameters['Nsum'],
        'L': Rod1D.parameters['L'],
        'TT': "The boundary condition at the top.",
        'TB': "The boundary condition at the bottom",
        'TL': Rod1D.parameters['TL'],
        'TR': Rod1D.parameters['TR'],
        }

    alpha1 = 1
    beta1 = 0
    alpha2 = 1
    beta2 = 0
    Nsum = 10000
    TB = 1.0
    TT = 0.0
    TL = 0.0
    TR = 0.0
    geometry = 2

    def __init__(self, **kwargs):
        # We need to rename gamma1 and gamma2 for this case, and we need these
        # set BEFORE the call to the initializer of the base class.  So we check
        # if they are set in the kwargs dictionary, and if so use those values,
        # and otherwise use the defaults defined above.
        self.gamma1 = kwargs.get("TB", self.TB)
        self.gamma2 = kwargs.get("TT", self.TT)
        super(PlanarSandwich, self).__init__(**kwargs)
