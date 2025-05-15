r"""Heat flow in a 1D rod with boundary and initial conditions.

This problem involves heat conduction in a 1D rod of length :math:`L`.
The heat conduction equation for the temperature profile :math:`T(x,t)` is

.. math::
  \frac{\partial T}{\partial t} = \kappa \, \frac{\partial^2 T}{\partial x^2}
  \ ,
 :label: DE

where :math:`\kappa` is the (constant) thermal conductivity of the rod. We place
the ends of the rod at :math:`x=0` and :math:`x=L`, and use a linear combination
of Dirichlet and Neumann and boundary conditions,

.. math::
  \alpha_1 T(0,t) + \beta_1 \partial_x T(0,t) &= 0
  \\
  \alpha_2 T(0,t) + \beta_2 \partial_x T(0,t) &= 0
  \ ,
 :label: BCs

with :math:`t > 0`. Finally, we must provide a continuous initial temperature
profile :math:`T(x,0)` to obtain a solution to the differential equation.
For simplicity we shall consider only two cases: :math:`T(x,0)` is a constant
temperature :math:`T_0`, or :math:`T(x,0)` varies linearly between :math:`T_0`
and :math:`T_1` over the length of the rod,

.. math::
  T(x,0) = T_0 + \frac{T_1 - T_0}{L} \cdot x
  \ ,
 :label: IC

for :math:`x \in (0,L)`. Note that :math:`T(x=0,0)=T_0` and :math:`T(x=L,0)=T_1`.

In general, the boundary conditions (BC's) give a transcendental equation, but there
are four special cases that are particularly simple.

BC1: The first boundary condition choice is

.. math::
  T(0,t) &= 0
  \\[5pt]
  T(L,t) &= 0
  \ ,
 :label: BC1

which corresponds to :math:`\alpha_1 = 1`, :math:`\beta_1 = 0`, :math:`\alpha_2 = 1`,
and :math:`\beta_2 = 0`. The solution takes the form

.. math::
  T(x,t)
  &=
  \sum_{n=1}^\infty
  \left[\frac{2 T_0 - 2 T_1 \, (-1)^n}{n \pi}\right]\,
  \sin k_n x \, e^{- \kappa \, k_n^2 t}
  \\[5pt]
  k_n
  &=
  \frac{n\pi}{L}
  \ .
 :label: BC1sol

BC2: The second boundary condition choice is

.. math::
  \partial_x T(0,t) &= 0
  \\[5pt]
  \partial_x T(L,t) &= 0
  \ ,
 :label: BC2

which corresponds to :math:`\alpha_1 = 0`, :math:`\beta_1 = 1`, :math:`\alpha_2 = 0`,
and :math:`\beta_2 = 1`. The solution takes the form

.. math::
  T(x,t)
  &=
  \frac{1}{2}(T_0 + T_1) + \sum_{n=1}^\infty
  \left[ 2(T_1 - T_0) \, \frac{ 1 - (-1)^n}{n^2 \pi^2}  \right]\,
  \cos k_n x \, e^{- \kappa \, k_n^2 t}
  \\[5pt]
  k_n
  &=
  \frac{n\pi}{L}
  \ .
 :label: BC2sol

For the case of constant initial condition :math:`T(x,0)=T_0`, or equivalently
for :math:`T_1 = T_0`, note that the solution reduces to the time independent
form :math:`T(x,t) = T_0`. This is because the BC's prevent heat from flowing
across the boundaries, and the initial temperature profile remains fixed.

BC3: The third boundary condition choice is

.. math::
  T(0,t) &= 0
  \\[5pt]
  \partial_x T(L,t) &= 0
  \ ,
 :label: BC3

which corresponds to :math:`\alpha_1 = 1`, :math:`\beta_1 = 0`, :math:`\alpha_2 = 0`, and
:math:`\beta_2 = 1`. The solution takes the form

.. math::
  T(x,t) &= \sum_{n=0}^\infty
  \left[\frac{4T_1}{(2n + 1)\pi} - \frac{8(T_1 - T_0)}{(2n+1)^2 \pi^2} \right]
  \sin k_n x \, e^{-\kappa \, k_n^2 t}
  \\[5pt]
  k_n &= \frac{(2 n + 1) \pi}{2 L}
  \ .
 :label: BC3sol

BC4: The fourth boundary condition choice is

.. math::
  \partial_x T(0,t) &= 0
  \\[5pt]
  T(L,t) &= 0
  \ ,
 :label: BC4

which corresponds to :math:`\alpha_1 = 0`, :math:`\beta_1 = 1`, :math:`\alpha_2 = 1`, and
:math:`\beta_2 = 0`. On physical grounds, the solution must be identical to the one for BC3,
except that the rod has been inverted. In terms of a series expansion, we have

.. math::
  T(x,t) &= \sum_{n=0}^\infty
  \left[
  \frac{4T_0\, (-1)^n}{(2n+1)\pi}
  -
  \frac{8(T_1 - T_0)}{(2n+1)^2 \pi^2} \Big[1 - (-1)^n\Big]
  \right]
  \cos k_n x \, e^{-\kappa \, k_n^2 t}
  \\[5pt]
  k_n &= \frac{(2 n + 1) \pi}{2 L}
  \ .
 :label: BC4sol

The solutions for BC3 and BC4 appear to be quite different; however, they
are indeed just a reflection across the midpoint of the rod. Since the
infinite sums in ExactPack must be truncated at some order, comparing BC3
against BC4 provides a good metric to determine whether the truncation
has been performed to sufficient accuracy. The first 100 terms seems gives
reasonable results.

General BC: The solution is of the form

.. math::
  T(x,t) = \sum_n \Big[A_n \cos k_n x  + B_n \sin k_n x \Big]\,
  e^{-\kappa \, k_n^2 t}
  \ ,

where the wave numbers are

.. math::
   k_n = \frac{\mu_n}{L}
   \ ,

with :math:`\mu_n` being the solutions to

.. math::
  \tan \mu
  = \frac{(\alpha_2 \bar\beta_1 - \alpha_1 \bar\beta_2)\mu}
  {\alpha_1 \alpha_2 + \bar\beta_1 \bar\beta_2 \mu^2}
  \,\,\,\, {\rm for}\,\,\, \bar\beta_i = \beta_i/L
  \ .

Case I: :math:`\alpha_1 \ne 0`. The coefficients :math:`A_n` and
:math:`B_n` are related by

.. math::
  A_n = - \frac{\beta_1 k_n}{\alpha_1}\, B_n
  \ ,

which leads us to write

.. math::
  X_n(x)
  &=
  \sin k_n x - \frac{\beta_1 k_n}{\alpha_1} \, \cos k_n x
  \\[5pt]
  T(x,t)
  &=
  \sum_{n=1}^\infty B_n \, X_n(x) \, e^{-\kappa \, k_n^2 t}
  \ .

Note that :math:`n=0` does not contribute since :math:`\mu_0=0` means
that :math:`k_0=0`, and therefore :math:`A_0=0` and :math:`\sin k_0 x
= 0` (for :math:`\alpha_1 \ne 0`). The modes :math:`X_n` are orthogonal
for :math:`n \ne m`, and calculating the normalization :math:`N_n` is
straightforward but tedious:

.. math::
  \int_0^L dx \, X_n(x) \, X_m(x)
  &= N_n \, \delta_{nm}
  \\[5pt]
  {\rm where}\,\,\,
  N_n
  &=
  \frac{1}{4 \alpha_1 k_n^2}\, \Big[ -2 \alpha_1 \beta_1 k_n  +
  2 (\beta_1^2 k_n^2 + \alpha_1^2) k_n L \, +
  \\
  & 2 \alpha_1 \beta_1 k_n \cos 2 k_n L + (\beta_1^2 k_n^2 - \alpha_1^2)
  \sin 2 k_n L \Big]
  \ .

The coefficients :math:`B_n` can be calculated from

.. math::
  B_n = \frac{1}{N_n}\int_0^L dx \, T(x,0)\, X_n(x)
  \ .

Since the expression for :math:`B_n` is rather long for a constant plus
a linear initial condition, :math:`T(x,0) = T_0 + (T_1 - T_0)\, x / L`,
we divide the result into two pieces:

.. math::
  T^{(0)}(x,0) &= T_0:
  \\[3pt]
  B_n^{(0)}
  &=
  \frac{T_0}{N_n}\Big[ \frac{1 - \cos k_n L}{k_n} -
  \frac{\beta_1 }{\alpha_1}\, \sin k_n L
  \Big]
  \ ,

and

.. math::
  T^{(1)}(x,0) &= \frac{T_1 - T_0}{L}\, x:
  \\[5pt]
  B_n^{(1)}
  &=
  \frac{T_1-T_0}{N_n \, L} \, \frac{1}{\alpha_1 k_n^2}
  \Big[
  \beta_1 k_n - (\alpha_1 k_n L + \beta_1 k_n) \cos k_n L \,+
  \\
  & \hskip3.5cm (\alpha_1 - \beta_1 k)n^2 L) \sin k_n L
  \Big]
  \ .

Since the boundary conditions are homogenous, we may add the two
solutions the constant plus linear initial condition, :math:`T(x,0)
= T^{(0)}(x,0) + X^{(1)}(x,0)`, in which case :math:`B_n = B_n^{(0)}
+ B_n^{(1)}`.

Case II: :math:`\alpha_1 = 0`. We may assume that :math:`\beta_2 \ne 0`,
otherwise this is BC4, which has already been treated. The wave numbers
are given by :math:`k_n = \mu_n/L`, where :math:`\mu_n` are the solutions
to

.. math::
  \tan \mu
  = \frac{\alpha_2}{\bar\beta_2}\,\frac{1}{\mu}
  \ .

The coefficients of the sin-function vanish,

.. math::
  B_n = 0
  \ ,

which leads us to write

.. math::
  X_n(x)
  &=
  \cos k_n x
  \\[5pt]
  T(x,t)
  &=
  \sum_{n=1}^\infty A_n \, X_n(x) \, e^{-\kappa \, k_n^2 t}
  \ .

The orthogonality relation is

.. math::
  \int_0^L dx \, X_n(x) \, X_m(x)
  &= N_n \, \delta_{nm}
  \\[5pt]
  {\rm where}\,\,\,
  N_n
  &= \frac{1}{4 k_n}\Big[ 2 k_n L + sin 2 k_n L  \Big]
  \ .

The coefficients :math:`B_n` can be calculated from

.. math::
  A_n
  &= \frac{1}{N_n}\int_0^L dx \, T(x,0)\, X_n(x)
  \\[5pt]
  &=
  T_0 \, \frac{sin k_n L}{k_n} + \frac{(T_1 - T)}{L k_n^2}
  \Big[ -1 + \cos k_n L + k_n L \, \sin k_n L \Big]
  \ .

"""

import numpy as np
from scipy.optimize import fsolve

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Rod1D(ExactSolver):

    r"""Computes the solution to the 1D heat conduction problem om rod for boundary
    conditions given by linear combinations of Neumann and Dirichlet.

    The four boundary condition types BC1, BC2, BC3, BC4 have been implements for
    homogeneous and nonhomogeneous cases. BC1 is equivalent to the planar sandwich,
    and BC2 to the hot planar sandwich. Note that BC3 and BC4 are physically equivalent
    correspond to the same physical system, which one might call the planar half sandwich,
    with constant temperature at one end, and constant heat flux at the other. This is
    a new test problem, not mentioned by Dawes et al. The general case for rod1D might
    be called PlanarSandwichTheWorks.

    """

    parameters = {
        'Nsum': "Number of terms to include in the sum.",
        'kappa': "Thermal diffusivity",
        'TL': "Value of IC profile for the left end of the rod at x=0",
        'TR': "Value of IC profile for the right end of the rod at x=L",
        'L': "Length of the rod",
        'alpha1': "BC parameter for temperature at x=0",
        'beta1': "BC parameter for flux at x=0",
        'gamma1': "nonhomogeneous BC parameter at x=0",
        'alpha2': "BC parameter for temperature at x=L",
        'beta2': "BC parameter for flux at x=L",
        'gamma2': "nonhomogeneous BC parameter for x=L",
        }

    kappa = 1.0
    TL = 3.0
    TR = 3.0
    L = 2.0
    Nsum = 100
    alpha1 = 1.0
    beta1 = 0.0
    gamma1 = 0.0
    alpha2 = 1.0
    beta2 = 0.0
    gamma2 = 0.0

    def modes_BC1(self):
        r"""Computes coefficients :math:`A_n` and :math:`B_n` and the modes :math:`k_n`
        for the first boundary condition special case:

        :math:`\alpha_1 \ne 0` , :math:`\beta_1 = 0` , :math:`\alpha_2 \ne 0` , and :math:`\beta_2 = 0`
        """

        T1 = self.gamma1 / self.alpha1
        T2 = self.gamma2 / self.alpha2
        Ta = self.TL - T1
        Tb = self.TR - T2
        for n in range(self.Nsum):
            self.kn[n] = n * np.pi / self.L
            if n != 0:
                self.Bn[n] = 2 * Ta * (1 - (-1)**n) / float(n * np.pi)  # constant
                self.Bn[n] = self.Bn[n] + 2 * (Ta - Tb) * (-1)**n / (n * np.pi)  # linear

    def modes_BC2(self):
        r"""Computes coefficients :math:`A_n` and :math:`B_n` and the modes :math:`k_n`
        for the second boundary condition special case:

        :math:`\alpha_1 = 0` , :math:`\beta_1 \ne 0` , :math:`\alpha_2 = 0` , and :math:`\beta_2 \ne 0`
        """

        F1 = self.gamma1 / self.beta1
        Ta = self.TL
        Tb = self.TR - F1 * self.L
        for n in range(self.Nsum):
            self.kn[n] = n * np.pi / self.L
            if n == 0:
                self.An[n] = float(Ta + Tb) / 2
            else:
                self.An[n] = 2 * (Ta - Tb) * (1 - (-1)**n) / (n * np.pi)**2

    def modes_BC3(self):
        r"""Computes coefficients :math:`A_n` and :math:`B_n` and the modes :math:`k_n`
        for the third boundary condition special case:

        :math:`\alpha_1 \ne 0` , :math:`\beta_1 = 0` , :math:`\alpha_2 = 0` , and :math:`\beta_2 \ne 0`
        """

        T1 = self.gamma1 / self.alpha1
        F2 = self.gamma2 / self.beta2
        Ta = self.TL - T1
        Tb = self.TR - (T1 + F2 * self.L)
        for n in range(self.Nsum):
            self.kn[n] = (2 * n + 1) * np.pi / (2 * self.L)
            self.Bn[n] = 4 * Ta / ((2 * n + 1) * np.pi)  # constant
            self.Bn[n] = self.Bn[n] + 8 * (Tb - Ta) * (-1)**n / ((2 * n + 1) * np.pi)**2  # linear

    def modes_BC4(self):
        r"""Computes coefficients :math:`A_n` and :math:`B_n` and the modes :math:`k_n`
        for the fourth boundary condition special case:

        :math:`\alpha_1 = 0` , :math:`\beta_1 \ne 0` , :math:`\alpha_2 \ne 0` , and :math:`\beta_2 = 0`
        """

        F1 = self.gamma1 / self.beta1
        T2 = self.gamma2 / self.alpha2
        Ta = self.TL - (T2 - F1 * self.L)
        Tb = self.TR - T2
        for n in range(self.Nsum):
            self.kn[n] = (2 * n + 1) * np.pi / (2 * self.L)
            self.An[n] = 4 * Ta * (-1)**n / ((2 * n + 1) * np.pi)  # constant
            self.An[n] = self.An[n] - 8 * (Tb - Ta) / ((2 * n + 1) * np.pi)**2 + \
                4 * (Tb - Ta) * (-1)**n / ((2 * n + 1) * np.pi)  # linear

    # otherwise
    def modes_BCgen(self):
        r"""Computes coefficients :math:`A_n` and :math:`B_n` and the modes :math:`k_n`
        for the general boundary condition:

        :math:`\alpha_1 \ne 0` , :math:`\beta_1 \ne 0` , :math:`\alpha_2 \ne 0` , and :math:`\beta_2 \ne 0`
        """
        a1 = float(self.alpha1)
        b1 = float(self.beta1) / self.L
        a2 = float(self.alpha2)
        b2 = float(self.beta2) / self.L
        if (a1 != 0.0):
            func = lambda mu: np.tan(mu) - (a2 * b1 - a1 * b2) * mu / (a1 * a2 + b1 * b2 * mu**2)
            for n in range(self.Nsum):
                if n != 0:
                    muinit = n * np.pi  # initial guess for wave number
                    mu = fsolve(func, muinit)[0]
                    self.kn[n] = mu / self.L  # wave number
                    Nn = (-2 * a1 * b1 * mu + 2 * (b1**2 * mu**2 + a1**2) * mu +
                            2 * a1 * b1 * mu * np.cos(2 * mu) + (b1**2 * mu**2 - a1**2) *
                            np.sin(2 * mu)) / (4 * a1**2 * self.kn[n])  # normalization
                    tmp = self.TL * (1 - np.cos(mu)) / self.kn[n] - (b1 * self.L / a1) * np.sin(mu)  # const
                    tmp = tmp + ((self.TR - self.TL) / (self.L * a1 * self.kn[n]**2)) * \
                        (b1 * mu - (a1*mu + b1 * mu) * np.cos(mu) + (a1 - b1**2 * mu**2) * np.sin(mu))
                    self.Bn[n] = tmp / Nn
                    self.An[n] = -(b1 * mu / a1) * self.Bn[n]
        else:
            a = a2 / b2  # b2 =/= 0, as a1 = 0 and b2 = 0 is case B4 above
            func = lambda mu: np.tan(mu) - a / mu
            for n in range(self.Nsum):
                if (n == 0):
                    muinit = 0.1
                    muasym = 0
                else:
                    muinit = n * np.pi
                    muasym = n * np.pi + a / ((1 + a) * n * np.pi)  # mu_n for n >> 1
                mu = fsolve(func, muinit)[0]
                self.kn[n] = mu / self.L  # wave number
                Nn = (2 * mu + np.sin(2 * mu)) / (4 * self.kn[n])
                tmp = self.TL * np.sin(mu) / self.kn[n]
                tmp = tmp + ((self.TR - self.TL) / (self.L * self.kn[n]**2)) * \
                    (-1 + np.cos(mu) + mu * np.sin(mu))
                self.An[n] = tmp / Nn

    def __init__(self, **kwargs):
        super(Rod1D, self).__init__(**kwargs)

        self.kn = np.zeros(shape=self.Nsum)
        self.An = np.zeros(shape=self.Nsum)
        self.Bn = np.zeros(shape=self.Nsum)

        if self.alpha1 != 0 and self.beta1 == 0 and self.alpha2 != 0 and self.beta2 == 0:
            self.modes_BC1()
        elif self.alpha1 == 0 and self.beta1 != 0 and self.alpha2 == 0 and self.beta2 != 0:
            self.modes_BC2()
        elif self.alpha1 != 0 and self.beta1 == 0 and self.alpha2 == 0 and self.beta2 != 0:
            self.modes_BC3()
        elif self.alpha1 == 0 and self.beta1 != 0 and self.alpha2 != 0 and self.beta2 == 0:
            self.modes_BC4()
        else:
            self.modes_BCgen()

    def _run(self, x, t):
        temperature = np.zeros(shape=x.shape)
        tempnonhom = np.zeros(shape=x.shape)

        if self.alpha1 != 0 and self.beta1 == 0 and self.alpha2 != 0 and self.beta2 == 0:
            # BC1
            N = self.Nsum
            T1 = self.gamma1 / self.alpha1
            T2 = self.gamma2 / self.alpha2
            tempnonhom = T1 + (T2 - T1) * x / self.L
        elif self.alpha1 == 0 and self.beta1 != 0 and self.alpha2 == 0 and self.beta2 != 0:
            # BC2
            N = self.Nsum
            F1 = self.gamma1 / self.beta1
            F2 = self.gamma2 / self.beta2
            if F1 != F2:
                raise ValueError("The flux at either end of rod must be equal")
            tempnonhom = F1 * x
        elif self.alpha1 != 0 and self.beta1 == 0 and self.alpha2 == 0 and self.beta2 != 0:
            # BC3
            N = self.Nsum
            T1 = self.gamma1 / self.alpha1
            F2 = self.gamma2 / self.beta2
            tempnonhom = T1 + F2 * x  # = Ta + (Tb - Ta) * x / L
        elif self.alpha1 == 0 and self.beta1 != 0 and self.alpha2 != 0 and self.beta2 == 0:
            # BC4
            N = self.Nsum
            F1 = self.gamma1 / self.beta1
            T2 = self.gamma2 / self.alpha2
            tempnonhom = (T2 - self.L * F1) + F1 * x  # = Ta + (Tb - Ta) * x / L
        else:
            N = self.Nsum
            a1 = float(self.alpha1)
            b1 = float(self.beta1) / self.L
            a2 = float(self.alpha2)
            b2 = float(self.beta2) / self.L
            c1 = float(self.gamma1)
            c2 = float(self.gamma2)
            T1 = (b2 * c1 - b1 * c2 + self.L * a2 * c1) / (a1 * b2 - a2 * b1 + self.L * a1 * a2)
            T2 = (b2 * c1 - b1 * c2 + self.L * a1 * c2) / (a1 * b2 - a2 * b1 + self.L * a1 * a2)
            tempnonhom = T1 + (T2 - T1) * x / self.L

        # construct time dependent solution
        for n in range(self.Nsum):
            temperature += (self.An[n] * np.cos(self.kn[n] * x) + self.Bn[n] * np.sin(self.kn[n] * x)) * \
              np.exp(-self.kappa * self.kn[n]**2 * t)

        # add homogeneous and nonhomogeneous
        temperature = temperature + tempnonhom

        return ExactSolution([x, temperature],
                             names=['position',
                                    'temperature',
                                    ])
