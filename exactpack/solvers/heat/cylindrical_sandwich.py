r"""The Cylindrical Sandwich [Dawes2016]_ .

This test problem was proposed by Allan Dawes in Ref. [Dawes2015]_, and the exact
analytic solution was presented in Appendix B of Ref. [Dawes2016]_. This problems
considers heat conduction in a 2D annulus with inner radius :math:`r=a` and outer
radius :math:`r=b`, with :math:`0 \le \theta \le \pi/2` (the first quadrant of the
plane). The heat conduction equation on the annulus, expressed in cylindrical coordinates,
takes the form

.. math::
  \frac{1}{\kappa}\,
  \frac{\partial T}{\partial t}
  &=
  \nabla^2 T
  \\[5pt]
  &=
  \frac{1}{r}\,\frac{\partial}{\partial r}\!\left( r \frac{\partial T}{\partial r} \right)
  +
  \frac{1}{r^2} \, \frac{\partial^2 T}{\partial \theta^2}
  \ .

The differential equation holds in the open annular region :math:`a < r < b` and
:math:`0 < \theta < \pi/2` at time :math:`t > 0`, and we take the initial condition
in this region are taken as

.. math::
  T(r,\theta,t=0) = 0
  \ .

The boundary conditions are

.. math::
  T(r, \theta = 0,t) &= T_0
  \\
  T(r,\theta = \pi/2,t) &= T_1
  \\[5pt]
  \partial_r T(r = a,\theta,t) &= 0
  \\
  \partial_r T(r=b,\theta,t) &= 0
  \ .

The boundary condition at :math:`\theta=0, \pi/2` render the system non-homogeneous.
Therefore, the solution is a combination of a general homogeneous solution and a
specific time-independent solution to the non-homogeneous Laplace equation,

.. math::
  T(r,\theta,t) = \tilde T(r,\theta,t) + \bar T(r,\theta)
   \ .

The non-homogeneous Laplace equation is

.. math::
  \nabla^2 \bar T
  &=
  \frac{1}{r}\,\frac{\partial}{\partial r}\!\left( r \frac{\partial \bar T}{\partial r} \right)
  +
  \frac{1}{r^2} \, \frac{\partial^2 \bar T}{\partial \theta^2}
  = 0
  \\[5pt]
  \bar T(r, \theta=0) &= T_1  ~~~~~ \bar T(r, \theta=\pi/2) = T_1
  \\
  \partial_r \bar T(r=a, \theta) &= 0  ~~~~~~~ \partial_r \bar T(r=b, \theta=0) = 0
  \ ,

which has the solution

.. math::
  \bar T(r,\theta) = T_0 + \frac{2\theta}{\pi}\, T_1
  \ .


The homogeneous heat equation becomes

.. math::
  \frac{1}{\kappa}\,
  \frac{\partial \tilde T}{\partial t}
  &=
  \frac{1}{r}\,\frac{\partial}{\partial r}\!\left( r \frac{\partial \tilde T}{\partial r} \right)
  +
  \frac{1}{r^2} \, \frac{\partial^2 \tilde T}{\partial \theta^2}
  \\[5pt]
  \tilde T(r,\theta,t=0) &= - \bar T(r,\theta)
  \\[5pt]
  \tilde T(r, \theta=0) &= 0  ~~~~~ \tilde T(r, \theta=\pi/2) = 0
  \\
  \partial_r \tilde T(r=a, \theta) &= 0  ~~~~~ \partial_r \tilde T(r=b, \theta=0) = 0
  \ .

The initial condition is of the form of a constant plus linear term.

**NOTE** This solver is currently only accurate to :math:`10^{-3}`. This is
because the Bessel functions become highly oscillatory for large mode numbers,
and the quadrature method used to calculated the coefficients becomes unreliable.
For higher accuracy, one must use an asymptotic form of the high mode Bessel
functions should be used. This will be implemented in a future release.

"""

import numpy as np
from scipy import special as sp
from scipy.optimize import newton
from scipy.integrate import quad

from ...base import ExactSolver, ExactSolution


class CylindricalSandwich(ExactSolver):

    r"""Computes the solution to Cylindrical Sandwich [Dawes2016]_ .
    """

    parameters = {
        'kappa': "Thermal diffusivity",
        'a': "Inner radius of annulus",
        'b': "Outer radius of annulus",
        'T1': r"Temperature boundary condition along :math:`\theta=\pi/2` for :math:`a \le r \le b`",
        'T0': r"Temperature boundary condition along :math:`\theta=0` for :math:`a \le r \le b`",
        'Nsum': "Number of terms to include in the n-sum.",
        'Msum': "Number of terms to include in the m-sum.",
        'NonHomogeneousOnly': "If True, then compute only the static nonhomogeneous solution. "
        }

    kappa = 1.0
    a = 0.25
    b = 0.85
    T1 = 1.0  # BC at theta=pi/2
    T0 = 0.0  # BC at theta=0
    Nsum = 20
    Msum = 100
    NonHomogeneousOnly = False

    # boundary condition function
    @staticmethod
    def bc_solve(x, k, a, b):
        F = (sp.jn(k - 1, x * a) - sp.jn(k + 1, x * a)) * \
            (sp.yn(k - 1, x * b) - sp.yn(k + 1, x * b)) / 4.0 \
            - \
            (sp.jn(k - 1, x * b) - sp.jn(k + 1, x * b)) * \
            (sp.yn(k - 1, x * a) - sp.yn(k + 1, x * a)) / 4.0
        return F

    # mode numbers alpha[n,m]
    def alpha(self, N, M, a, b):
        alphax = np.zeros(shape=(N, M))
        betax = np.zeros(shape=(N, M))
        for n in xrange(N):
            k = 2 * (n + 1)
            x0 = 0.1
            for m in xrange(M):
                alphanm = newton(CylindricalSandwich.bc_solve, x0, args=(k, a, b))
                x0 = alphanm + 4
                alphax[n, m] = alphanm
                betax[n, m] = -(sp.jn(k + 1, alphanm * a) - sp.jn(k - 1, alphanm * a)) /\
                    (sp.yn(k + 1, alphanm * a) - sp.yn(k - 1, alphanm * a))
        return alphax, betax

    # R = R[n,m]
    def R(self, x, k, m, alphanm, betanm):
        Rnm = sp.jn(k, alphanm * x) + betanm * sp.yn(k, alphanm * x)
        return Rnm

    # rR2 = x * R[n,m]**2
    def rR2(self, x, k, m, alphanm, betanm):
        Rnm = sp.jn(k, alphanm * x) + betanm * sp.yn(k, alphanm * x)
        return x * Rnm**2

    # normalization Anm = quad(rR2, a, b,args=(k,m,alphanm,betanm))
    def Anm_numerical(self, a, b, k, m, alphanm, betanm):
        Anm = quad(CylindricalSandwich.rR2, a, b, args=(k, m, alphanm, betanm))[0]
        return Anm

    # normalization Anm analtyic
    def Anm_analytic(self, a, b, k, m, alphanm, betanm):
        Rnmb = CylindricalSandwich.R(self, self.b, k, m, alphanm, betanm)
        Rnma = CylindricalSandwich.R(self, self.a, k, m, alphanm, betanm)
        Anm = (1./2.) * (self.b**2 - m**2/alphanm**2) * Rnmb**2 - \
            (1./2.) * (self.a**2 - m**2/alphanm**2) * Rnma**2
        return Anm

    def __init__(self, **kwargs):

        super(CylindricalSandwich, self).__init__(**kwargs)

    def _run(self, rtheta_list, t):
        # unpack rtheta_list
        r = rtheta_list[0]
        theta = rtheta_list[1]
        #

        alphax = np.zeros((self.Nsum, self.Msum))

        # dT = r * R[n,m]
        def dTinRun(x, k, m, alphanm, betanm):
            Tnm = sp.jn(k, x * alphanm) + betanm * sp.yn(k, x * alphanm)
            return x * Tnm

        # specific nonhomogeneous contribution \bar T(x,y)
        tempnonhom = self.T0 + 2 * self.T1 * theta / np.pi
        # general homogeneous contribution \tilde T(x, y, t)
        temperature = 0
        if self.NonHomogeneousOnly == False:
            alphax, betax = CylindricalSandwich.alpha(self, self.Nsum, self.Msum, self.a, self.b)
            for n in xrange(self.Nsum):  # fragile: n <= 20
                k = 2 * (n + 1)
                for m in xrange(self.Msum):
                    alphanm = alphax[n, m]  # eigen values based on Bessel Functions
                    betanm = betax[n, m]  # J vs Y weighting
                    Rnm = CylindricalSandwich.R(self, r, k, m, alphanm, betanm)
                    #
                    Rnmb = sp.jn(k, alphanm * self.b) + betanm * sp.yn(k, alphanm * self.b)
                    Rnma = sp.jn(k, alphanm * self.a) + betanm * sp.yn(k, alphanm * self.a)
                    Anm = (1./2.) * (self.b**2 - m**2/alphanm**2) * Rnmb - \
                      (1./2.) * (self.a**2 - m**2/alphanm**2) * Rnma
                    # Anm = CylindricalSandwich.Anm_analytic(self, self.a, self.b, k, m, alphanm, betanm)
                    Tnm = (4 * self.T1 / np.pi) * ((-1)**(k/2) / float(k)) * (1 / Anm) * \
                        quad(dTinRun, self.a, self.b, args=(k, m, alphanm, betanm))[0]
                    tmp = Tnm * Rnm * np.sin(k * theta) * np.exp(-self.kappa * alphanm * t)  # combine Tnm and tmp
                    temperature += tmp  # this line is throwing a waring during the unit test

        # add homogeneous and nonhomogeneous
        temperature = temperature + tempnonhom

        return ExactSolution([r, theta, temperature],
                    names=['position_r', 'angle_theta',
                           'temperature'], jumps=[])
