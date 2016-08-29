r"""The Sedov Problem.

The Sedov problem.

The Sedov blast wave problem [Sedov1959]_ [Kamm2000]_ models a rapid
point-source explosion in an
invisid, non-heat conducting polytropic gas, caused by a release of 
of energy :math:`E_0` at the origin :math:`r=0` at time :math:`t=0`.
The independent fluid variables are (i) the mas density :math:`\rho(r,t)`, 
(ii) the velocity of the gas :math:`u(r,t)`, and (iii) the pressure
:math:`P(r,t)`, each at spatial location :math:`r` and time :math:`t`. The 
specific internal energy :math:`e(r,t)` is related to the other fluid variables
by the equation of state (EOS) for an ideal gas at constant entropy,

.. math::
  P = (\gamma-1) \rho e  \ ,

where :math:`\gamma \equiv c_p/c_v` is the ratio of specific heats at
constant pressure and volume. The Euler equations take the form

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
  + u\, \frac{\partial }{\partial r} \Big( P \rho^{-\gamma} \Big)  &= 0 \ ,

where :math:`k=1,2,3` for planar, cylindrical, and spherical coordinates
respectively. The initial density :math:`\rho(r,0) = \rho_0` is taken to 
be constant in :math:`r` (with the default value :math:`\rho_0= 1\,{\rm
g/cm^3}`), and the initial velocity and pressure :math:`u(r,0)=0` and
:math:`P(r,0)=0` everywhere in space (for :math:`r \ne 0`). At
:math:`t=0^+`, an infinitely strong shock wave forms at the origin, moving
radially outward, with energy

.. math::

  E_0 = \int dV \Bigg[\frac{1}{2}\, \rho u^2 + \frac{P}{\gamma-1}\Bigg] \ ,

where the integration runs over the volume behind the shock at time :math:`t`,
i.e. the integration runs over :math:`0 \le r \le r_{\rm shock}(t)`. The volume
element takes the form :math:`dV = 4\pi\, r^2 dr` for :math:`k=3` (spherical),
:math:`dV =2\pi\, r dr` for :math:`k=2` (cylindrical), and :math:`dV = dr` for
:math:`k=1` (planar). We take the gas
constant to have the value :math:`\gamma=7/5=1.4`, and for the spherically
symmetric case :math:`k=3` we also take :math:`E_0=0.851072\,{\rm erg}`, in
which case
the shock arrives at :math:`r_{\rm shock} = 1\,{\rm cm}` at the final time
:math:`t = 1\, {\rm s}`. Similarly, for :math:`k=2, 1`, we take
:math:`E_0=0.311357, 0.0673185`, so that the shock arrives at :math:`r_{\rm
shock}=0.75, 0.5` at :math:`t=1`, respectively (these conventions allow all
three cases to be plotted on the same graph at the final time).

By default, :py:mod:`exactpack.solvers.sedov` loads
:py:mod:`exactpack.solvers.sedov.timmes`.

"""

from .timmes import Sedov

class PlanarSedov(Sedov):
    """The standard planar Sedov problem, with a default value of \
    :math:`\gamma=7/5` and :math:`E_0=0.0673185` erg.
    """

    parameters = {'gamma': Sedov.parameters['gamma']}
    geometry = 1
    eblast = 0.0673185
    gamma = 1.4

class CylindricalSedov(Sedov):
    """The standard cylindrical Sedov problem, with a default value of \
    :math:`\gamma=7/5` and :math:`E_0=0.3113572` erg.
    """

    parameters = {'gamma': Sedov.parameters['gamma']}
    geometry = 2
    eblast = 0.311357
    gamma = 1.4

class SphericalSedov(Sedov):
    """The standard spherical Sedov problem, with a default value of \
    :math:`\gamma=7/5` and :math:`E_0=0.851072` erg.
    """

    parameters = {'gamma': Sedov.parameters['gamma']}
    geometry = 3
    eblast = 0.851072
    gamma = 1.4
