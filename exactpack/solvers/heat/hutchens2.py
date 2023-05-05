r"""This is the second heat conduction problem of Hutchens in Ref. [Hutchens2009]_ .

This problem involves the heat conduction of a cylindrical mass of radius
:math:`r=b` and height :math:`z=L`.

The steady state heat conduction equation for the temperature profile
:math:`T(r,z,t)` takes the following form in cylindrical coordinates,

.. math::
  \frac{1}{r^2} \frac{\partial}{\partial r}
  \left( r^2 \frac{\partial T}{\partial r} \right)
  +
  \frac{\partial^2 T}{\partial z^2}
  +
  \frac{g_0}{k}
  &= 0
  \ .

The boundary conditions are

.. math::
  \frac{\partial T}{\partial r} (r=0,z) &= 0
  \\[3pt]
  T(r=b,z) &= T_b
  \\[3pt]
  T(r,z=0) &= T_0
  \\[3pt]
  T(r,z=L) &= T_L
  \ .


A Fourier analysis and the separation of variables technique yields the
exact solution,

.. math::
  T(r,t) &=
  T_b + \Big(T_L - T_0 \Big) \frac{z}{L} + \frac{1}{2}\frac{g_0}{k} z \Big(L - z \Big) +
  \\
  & \frac{2}{\pi}\, T_L \sum_{n=0}^\infty \frac{(-1)^{2n+1}}{2n+1}\,
  \frac{I_0(\lambda_n r)}{I_0(\lambda_n b)}\,\sin(\lambda_n z)
  -
  \frac{4 g_0 L^2}{\pi^3 k}\, \sum_{n=0}^\infty \frac{1}{(2n+1)^3}
  \frac{I_0(\lambda_n r)}{I_0(\lambda_n b)}\,\sin(\lambda_n z)
  \ ,

where :math:`I_0(z)` is a modified Bessel function of the first kind,
and

.. math::
  \lambda_n = \frac{(2n+1)\pi}{L}
  \ .

In practice, we must truncate the series at some maximum value
:math:`n_{\rm max}=N`. Care must be taken when choosing the value of
:math:`N`, especially since the series solution is alternating in sign.
For the iron sphere of problem 1 in Ref. [Hutchens2009]_ , the
choice :math:`N=100` is made.

"""

import numpy as np
from scipy.special import i0 as i0

from ...base import ExactSolver, ExactSolution


class Hutchens2(ExactSolver):

    r"""Computes the solution to the second heat conduction problem of Hutchens.

    This solver is given in unitis in which the temperature is measured in eV.
    Arbitrary units can be chosen, but then then the user must supply a value
    for the Boltzmann constant :math:`k_{\rm B}`, or alternatively, the radiation
    constant :math:`a`.

    """

    parameters = {
        'k': "thermal conductivity [erg/s-cm-eV]",
        'g0': "rate of heat generation [erg/s-cm^3]",
        'Tb': "Temperature of the radial boundary at :math:`r=b` [eV]",
        'T0': "Temperature of the cylinder at :math:`z=0` [eV]",
        'TL': "Temperature of the cylinder at :math:`z=L` [eV]",
        'Nsum': "Number of terms to include in the sum.",
        'b': "Radius of the cylinder [cm]",
        'L': "Height of cylinder [cm]"
        }

    # iron
    k = 8.4695e10
    g0 = 1.e13
    #
    b = 1.0
    L = 2.0
    Tb = 5.0
    T0 = 2.0
    TL = 1.0
    Nsum = 100

    def __init__(self, **kwargs):

        super(Hutchens2, self).__init__(**kwargs)

    def _run(self, rzlist, t):

        r = rzlist[0]
        z = rzlist[1]

        temperature = self.T0 + (self.TL - self.T0) * z / self.L + \
            (self.g0 / (2 * self.k)) * z * (self.L - z)
        sum = 0
        for n in range(0, self.Nsum):
            nodd = float(2 * n + 1)
            lam = nodd * np.pi / self.L
            iratio = (2. / np.pi) * i0(lam * r) / i0(lam * self.b)
            sum += (2 * self.Tb - self.T0) * iratio * np.sin(lam * z) / nodd
            sum += self.TL * iratio * (-1)**nodd * np.sin(lam * z) / nodd
            sum += -(2 * self.g0 * self.L**2 / (np.pi**2 * self.k)) * np.sin(lam * z) / nodd**2
            temperature += sum

        return ExactSolution([r, z, temperature],
                names=['position_r', 'position_z', 'temperature'],
                jumps=[]
                )
