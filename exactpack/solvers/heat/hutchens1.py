r"""This is the first heat conduction problem of Hutchens in
Ref. [Hutchens2009]_ .

This problem involves the heat conduction of a spherical mass of radius
:math:`r=b`, with the material priorities of mass density :math:`\rho`,
thermal conductivity :math:`k`, and specific heat at constant pressure
:math:`c_{\rm p}`. In terms of these properties, the thermal diffusivity
is given by

.. math::
  \alpha = \frac{k}{\rho c_{\rm p}}
  \ .

The heat conduction equation for the temperature profile :math:`T(r,t)`
takes the following form in spherical coordinates,

.. math::
  \frac{\partial T}{\partial t} = \alpha \, \frac{1}{r^2} \frac{\partial}{\partial r}
  \left( r^2 \frac{\partial T}{\partial r} \right)
  \ .

The initial temperature of the sphere is taken to be :math:`T=T_0`,
while the radial surface temperature of the sphere is held at a fixed
at temperature :math:`T=T_b`, in which case the
initial and boundary conditions are,

.. math::
  T(r,0) &= T_0
  \\[8pt]
  \frac{\partial T}{\partial r} (0,t) &= 0
  \\[3pt]
  T(b,t) &= T_b
  \ .

A Fourier analysis and the separation of variables technique yields the
exact solution,

.. math::
  T(r,t) = T_b + \frac{2b}{\pi r}\, \Big(T_b - T_0 \Big)
  \sum_{n=1}^\infty \frac{(-1)^n}{n}\, \sin\left(\frac{n \pi}{b}\,r \right)
  \exp\left(- \alpha \frac{n^2 \pi^2}{b^2}\,t \right)
  \ .

In practice, we must truncate the series at some maximum value
:math:`n_{\rm max}=N`. Care must be taken when choosing the value of
:math:`N`, especially since the series solution is alternating in sign.
For the iron sphere of problem 1 in Ref. [Hutchens2009]_ , the
choice :math:`N=100` is made.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution, Jump, JumpCondition


class Hutchens1(ExactSolver):

    r"""Computes the solution to the first heat conduction problem of Hutchens.

    This solver is given in unitis in which the temperature is measured in eV.
    Arbitrary units can be chosen, but then then the user must supply a value
    for the Boltzmann constant :math:`k_{\rm B}`, or alternatively, the radiation
    constant :math:`a`.

    """

    parameters = {
        'k': "thermal conductivity [erg/s-cm-eV]",
        'cp': "specific heat at constant pressure [erg/g-eV]",
        'rho': "density [g/:math:`{\\rm cm^3}`]",
        'Tb': "Temperature of the radial boundary at :math:`r=b` [eV]",
        'T0': "Initial uniform temperature of the sphere` [eV]",
        'Nsum': "Number of terms to include in the sum.",
        'b': "Radius of the sphere [cm]"
        }

    # iron
    k = 8.4695e10
    cp = 5.2441e10
    rho = 7.897
    #
    b = 1.0
    Tb = 5.0
    T0 = 1.0
    Nsum = 100

    def __init__(self, **kwargs):

        super(Hutchens1, self).__init__(**kwargs)

    def _run(self, r, t):

        # thermal diffusivity
        alpha = self.k / (self.rho * self.cp)

        temperature = np.zeros(shape=r.shape)
        for n in range(1, self.Nsum):
            nb = np.pi * n / self.b
            x = (-1)**n / float(n)
            x = x * np.where(r != 0, (2 * self.b / (np.pi * r)) * np.sin(nb * r), 0)  # bad at r=0
            x = x * np.exp(-alpha * nb**2 * t)
            temperature += x
        temperature = np.where(r != 0, temperature, -1)  # value at r=0
        temperature = (self.Tb - self.T0) * temperature
        temperature = self.Tb + temperature

        return ExactSolution([r, temperature],
                             names=['radius', 'temperature']
                            )
