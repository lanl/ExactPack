r"""The Su-Olson Problem

This is a one-dimensional, half-space, non-Equilibrium Marshak burn
wave. The radiative transfer model is a one-group diffusion
approximation with a Marshak radiation boundary condition. The
radiation temperature field (in energy units) and the matter energy
field are denoted by :math:`T = T(z,t)` and :math:`E=E(z,t)` for
:math:`0 \le z < \infty`. Fluid motion is neglected, and the equations
of motion become

.. math::

  \frac{\partial E}{\partial t} - \frac{\partial }{\partial z}\Big[
  \frac{c}{3 \kappa(T)}\, \frac{\partial E}{\partial z} \Big]
  &=
  c\,\kappa(T) \Big[ a\,T^4 - E \Big]
  \\
  c_v(T)\,\frac{\partial T}{\partial t}
  &=
  c\,\kappa(T) \Big[ E - a\,T^4\Big] ,

where :math:`\kappa(T)` is the opacity of the material, :math:`c` is
the speed of light, :math:`a` is the radiation constant, and
:math:`c_v(T)` is the specific heat at constant volume. The equations
can be solved for the (non-physical) model in which

.. math::
   c_v(T) &= \alpha\, T^3
   \\
   \kappa(T) &= constant .
   :label: cvkappaDef

When the matter radiates as a black body, :math:`E= a T^4`, and we see
that :math:`c_v = \partial E/\partial T = 4 a`. The Marshak boundary
condition at :math:`z=0` is

.. math::
   E(0,t) - \frac{2}{3 \kappa(0,t)}\,\frac{\partial E(0,t)}{\partial z}
   =
   \frac{4}{c} F_{\rm in}  ,

where :math:`F_{\rm in}` is incident radiation energy flux incident on
the :math:`z=0` surface, and the :math:`z \to \infty` boundary
condition is :math:`E(\infty,t)=0`.  The initial conditions are
:math:`E(z,0)=T(z,0)=0`.

.. [suolson96] Bingjing Su and Gordon L. Olson,
   Benchmark Results for the Non-equilibrium Marshak Diffusion Problem,
   J. Quant. Spectrosc. Radiat. Transfer **56** 337 (1996)

"""

from .timmes import SuOlson
