r"""The Guderley Problem.

In 1942, G. Guderley [Guderley1942]_ found a semi-analytic solution for
a self-similar convergent shock wave in an invisid, non-heat conducting
polytropic gas. The Euler equations for this problem take the form [Guderley2012]_

.. math::

  \frac{\partial \rho}{\partial t} + \frac{\partial (u \rho) }{\partial r}
  +
  (k - 1)\, \frac{u \rho}{r}
  &= 0
  \\
  \frac{\partial u}{\partial t} + u\, \frac{\partial u}{\partial r}
  +
  \frac{1}{\rho}\frac{\partial p}{\partial r} &= 0
  \\
  \frac{\partial }{\partial t} \Big( p \rho^{-\gamma} \Big)
  + u\, \frac{\partial }{\partial r} \Big( p \rho^{-\gamma} \Big)  &= 0 \ ,

where :math:`k=1,2,3` for planar, cylindrical, and spherical coordinates
respectively. The independent fluid variables are (i) the gas density
:math:`\rho(r,t)`, (ii) the velocity of the gas :math:`u(r,t)`, and (iii)
the pressure :math:`p(r,t)`, where the physical quantities are specified at
at given radius :math:`r` and time :math:`t`. The specific internal energy
:math:`e(r,t)` is related to other fluid variables by an equation of state
(EOS).  For an ideal gas at constant entropy, the EOS takes the form

.. math::
  p = (\gamma-1) \rho e  \ ,

where :math:`\gamma \equiv c_p/c_v` is the ratio of specific heats at constant
pressure and volume. For a solution to exist, we must specify the initial
conditions in front of the shock wave, i.e. for :math:`r \le r_{\rm shock}(0)` .
For the Guderley problem, the state of the pre-shocked region is :
:math:`\rho(r,0) = \rho_0={\rm const}`, with vanishing initial velocity and
pressure, :math:`u(r,0)=0` and :math:`p(r,0)=0`. The default density is in cgs
units is :math:`\rho_0= 1\,{\rm g/cm^3}`.

Note:  :py:mod:`exactpack.solvers.guderley` loads :py:mod:`exactpack.solvers.guderley.ramsey`. 

"""
from ramsey import Guderley
