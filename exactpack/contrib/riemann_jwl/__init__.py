r"""The independent fluid variables are (i) the gas density
:math:`\rho(r,t)`, (ii) the velocity of the gas :math:`u(r,t)`, and
(iii) the pressure :math:`P(r,t)`, each at spatial location :math:`r`
and time :math:`t`. The specific internal energy :math:`e(r,t)` is
related to the other fluid variables by the equation of state (EOS)
for an ideal gas at constant entropy,

.. math::

  JWL

where :math:`\gamma \equiv c_p/c_v` is the ratio of specific heats at
constant pressure and volume. For a mono-atomic ideal gas we have
:math:`\gamma=5/3`. The Euler equations take the form

.. math::

  \frac{\partial \rho}{\partial t} + u \frac{\partial \rho}{\partial r}
  +
  \frac{\rho}{r^{k-1}} \frac{\partial}{\partial r}\Big( u r^{k-1} \Big)
  &= 0
  \\
  \frac{\partial u}{\partial t} + u\, \frac{\partial u}{\partial r}
  +
  \frac{1}{\rho}\frac{\partial P}{\partial r} &= 0
  \\
  \frac{\partial }{\partial t} \Big( P \rho^{-\gamma} \Big)
  + u\, \frac{\partial }{\partial r} \Big( P \rho^{-\gamma} \Big)  &= 0

where :math:`k=1,2,3` for planar, cylindrical, and spherical coordinates
respectively.
"""

from .kamm import RiemannJWL
