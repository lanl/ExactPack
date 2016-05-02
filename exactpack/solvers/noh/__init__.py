r""" The Noh problem [Noh1987]_ is a self-similar adiabatic compression
wave in an ideal gas, and it can be formulated in spherical,
cylindrical, or planar geometry. The independent fluid variables are
(i) the gas density :math:`\rho(r,t)`, (ii) the velocity of the gas
:math:`u(r,t)`, and (iii) the pressure :math:`P(r,t)`, each at spatial
location :math:`r` and time :math:`t`. Note that in spherical coordinates, 
:math:`u(r,t)` is the radial velocity of the gas, and a negative value 
indicates that gas is flowing in toward the origin. The specific
internal energy :math:`e(r,t)` is related to the other fluid variables
by the equation of state (EOS) for an ideal gas at constant entropy,

.. math::
  P = (\gamma-1) \rho e  \ ,

where :math:`\gamma \equiv c_p/c_v` is the ratio of specific heats at
constant pressure and volume. The Euler equations for the conservation of
mass, momentum, and entropy can be written as:

.. math::

  \frac{\partial \rho}{\partial t} + u\, \frac{\partial \rho}{\partial r}
  +
  \frac{\rho}{r^{k-1}} \frac{\partial}{\partial r}\Big( u\, r^{k-1} \Big)
  &= 0
  \\
  \frac{\partial u}{\partial t} + u\, \frac{\partial u}{\partial r}
  +
  \frac{1}{\rho}\frac{\partial P}{\partial r} &= 0
  \\
  \frac{\partial }{\partial t} \Big( P \rho^{-\gamma} \Big)
  + u\, \frac{\partial }{\partial r} \Big( P \rho^{-\gamma} \Big)  &= 0 \ ,

where :math:`k=1,2,3` is the geometry factor specifying planar,
cylindrical, or spherical symmetry, respectively.

This problem is often cast in dimensionless variables in which the gas is moving
radially inward with uniform constant velocity :math:`u(r,0)=-1\, {\rm cm/s}`
and constant density :math:`\rho(r,0)=1\,{\rm g/cm^3}.` An infinitesimal time
after the the gas accumulates at the origin :math:`r=0`, an infinitely strong
stagnation shock forms and starts propagating outwards, leaving behind a region
of non-zero pressure and internal energy in its wake.  This problem exercises
the code's ability to transform kinetic energy into internal energy, and the
fidelity with which supersonic flows are tracked [Timmes2005]_ .

The analytic solution is particularly simple, taking the closed form given by
[Gehmeyr1997]_ (typos in the paper corrected). Given a uniform inward radial 
velocity :math:`u(r,0)=-u_0` (with :math:`u_0` positive), and a constant density 
:math:`\rho(r,0)=\rho_0`, and keeping the gas constant :math:`\gamma` general, 
the solution takes the form:

.. math::

   r_\mathrm{shock} = \frac{1}{2}\,(\gamma - 1) \, u_0 \, t ~~~~ \mathrm{for}~~ u_0 > 0\ ,

with the post-shock state given by:

.. math::

   r < r_\mathrm{shock}(t): &    \\
   \rho(r,t) &= \left( \frac{\gamma+1}{\gamma-1} \right)^k \,\rho_0  \\
   e(r,t) &= \frac{1}{2}\, u_0^2  \\
   u(r,t) &= ~0  \\
   P(r,t) &= \frac{4^k}{3}\,\rho_0 u_0^2

and the pre-shock state as:

.. math::

   r > r_\mathrm{shock}(t): & \\
   \rho(r,t) &= \left( 1 + \frac{u_0 \, t}{x} \right)^{k-1} \, \rho_0 \\
   e(r,t) &= \phantom{-}0  \\
   u(r,t) &= -u_0 ~~~~ (u_0 > 0) \\
   P(r,t) &= \phantom{-}0 \\

The geometry factor :math:`k=1, 2, 3` corresponds to planar,
cylindrically, and spherical geometries, respectively. To cast the equations in
terms of dimensionless coordinates, set :math:`u_0=1` and :math:`\rho_0 = 1`.

By default, :py:mod:`exactpack.solvers.noh` loads
:py:mod:`exactpack.solvers.noh.noh1`.
"""

from noh1 import Noh, PlanarNoh, CylindricalNoh, SphericalNoh
