r"""The elastic-plastic piston problem is described with exact solution
derivations in [Udaykumar2003]_ for the hypoelastic model and [Lieberman2019]_
for the hyperelastic models. The problem consists of a 1D solid of length
:math:`l` being impacted by a piston on the left side with a velocity of
:math:`u_p`. The impact of the piston results in two waves traveling from the
left to the right: a lower compression, faster wave speed elastic precursor wave
and a higher compression, slower wave speed plastic wave (see left Figure below). 

The volumetric behavior of the solid is described using an equation of state.
The problem as presented in [Lieberman2019]_ and in this python package uses the
Mie-Gruneisen equation of state. The deviatoric behavior is described using one
of three different elasticity models coupled to a J2 plasticity model
[Lieberman2019]_ with perfect plasticity (i.e. no hardening). The equations for
each of the three elasticity models are as follows,

Hypoelastic:

..  math::
  \dot{\boldsymbol{\sigma}}^{\prime}=2\mathrm{G}(\dot{\boldsymbol{\varepsilon}}^{\prime}-\dot{\boldsymbol{\varepsilon}}^{\prime}_p)-\boldsymbol{\sigma}^{\prime}\dot{\boldsymbol{\omega}}+\dot{\boldsymbol{\omega}}\boldsymbol{\sigma}^{\prime}

Infinitesimal Hyperelastic:

..  math::
  \boldsymbol{\sigma}^{\prime}=2\mathrm{G}(\boldsymbol{\varepsilon}^{\prime}-\boldsymbol{\varepsilon}^{\prime}_p)

Infinitesimal Hyperelastic:

..  math::
  \tilde{\bf S}=2\mathrm{G}{\bf E}^{\prime}_e

where G is the shear modulus of the material, :math:`(\dot{})` represents a
rate, :math:`\boldsymbol{\sigma}^{\prime}` is the deviatoric Cauchy stress,
:math:`\tilde{\bf S}` is the second Piola-Kirchoff stress (which can be
transformed to the Cauchy stress), :math:`\boldsymbol{\varepsilon}^{\prime}` and
:math:`\boldsymbol{\varepsilon}^{\prime}_p` are the total and plastic deviatoric
strain, respectively, and :math:`{\bf E}^{\prime}_e` is the elastic deviatoric
Green-Lagrange strain. The yield criterion is in the form of
:math:`\mathrm{Y}-\sqrt{\frac{3}{2}\boldsymbol{\sigma}^{\prime}:\boldsymbol{\sigma}^{\prime}}=0`,
where Y is the yield stress.

The solution for this problem involves solving the Rankine-Hugoniot jump
equations for both the elastic wave and the plastic wave, in that order. Due to
the perfect plasticity, the values of of particle velocity, density, internal
energy and pressure behind the elastic wave are the same as the values at the
yield point. For a 1D problem the various strains and strain rates can be
expressed in terms of density at the yield point. By solving for the density at
yield from the elastic constitutive model, the values of particle velocity, wave
speed, internal energy and pressure at the yield point can be found from the
Rankine-Hugoniot jump equations and the Gruneisen equation of state. However,
the differences between the elastic models lead to different values of density
at yield for the same yield stress and shear modulus. This difference then
propagates through to the solution of particle velocity, wave speed, density,
internal energy and pressure for the rest of the problem. The solutions for
pressure with each elastic model are shown in the Figure below using the
parameters from [Lieberman2019]_.

.. figure:: EPpiston_plot.png
   :alt: Plot of the Elastic-Plastic Piston Solution
   :align: center
   :figwidth: 100%

   Plot of the elastic-plastic piston pressure solution (left) with a zoomed
   view of the elastic shock front (right).

The Figure shows the full solution for pressure for each model on the left
figure. All three models show the elastic wave (with the lower pressure) moving
ahead of the plastic wave (with the higher pressure). The differences between
the solutions cannot be easily identified at this scale. A zoom in of the
elastic wave is given in the right figure, which corresponds to the box within
the left figure. For the set of parameters used, the differences in the solution
for each elastic model are primarily in the solution to the fields behind the
elastic wave front. The differences in elastic and plastic wave speeds and the
fields behind the plastic wave front are minuscule and as such the differences
are not visible in the Figure. 

The various fields (density, pressure, etc.) behind the plastic wave are solved
through the jump equations where there is a non-zero initial velocity (in this
case the state behind the elastic wave front) and stress and a known velocity
behind the shock (the piston velocity). The solution is derived by solving a
residual between a pressure behind the shock derived from the jump equations and
a pressure derived from the equation of state, with both being described in
terms of the plastic wave speed. Once the plastic wave speed is determined from
the residual the values of density, internal energy and pressure behind the
plastic wave can be determined.

"""

from .ep_piston import EPpiston
