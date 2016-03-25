r"""The Reinicke Meyer-ter-Vehn (RMTV) problem


The energy source of the problem considered here is sufficiently large
that heat conduction dominates the fluid flow, and a thermal front
leads a hydrodynamic shock. The equations of motion are

.. math::
  \frac{\partial\rho}{\partial t} + u \frac{\partial\rho}{\partial r}
  +
  \frac{\rho}{r^{k-1}} \frac{\partial}{\partial r}\Big(
  r^{k-1} u  \Big) &= 0
  \\
  \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial r} +
  \frac{1}{\rho}   \frac{\partial P}{\partial r} &= 0
  \\
  \frac{\partial e}{\partial t} + u \frac{\partial e}{\partial r} -
  \frac{P}{\rho^2}\left(\frac{\partial \rho}{\partial t} +
  u \frac{\partial \rho}{\partial r}
  \right)
  &=
  \frac{1}{\rho r^{k-1}} \frac{\partial }{\partial r}\left(
  r^{k-1} \chi \frac{\partial T}{\partial r} \right)

where :math:`k=1,2,3` for planar, cylindrical, or spherical geometry,
and the thermal conductivity is parameterized by


.. math::
   \chi(\rho,T) = \chi_0\, \rho^a T^b ~~{\rm where}~ a \le 0 ~{\rm and}~
   b \ge 1
   :label: chidef

The :math:`\gamma`-law Equation of State (EOS) for the gas is
written

.. math::
   P = \Gamma \rho T ~~{\rm and}~~ e = \frac{\Gamma T}{\gamma - 1} \ ,
   :label: BigGamma

where :math:`\Gamma` is the Gruneisen gas coefficient. The initial
cold (:math:`T=0`) density profile is

.. math::

  \rho_0(r) = g_0 r^\kappa G(\xi) ~~{\rm where}~ \kappa < 0 ,

with :math:`G(\xi)` being a dimensionless profile function of the
dimensionless self-similarity position variable :math:`\xi`. NOTE:
Ask Jim about this.


.. [rmtv91] P. Reinicke and J. Meyer-ter-Vehn,
   The point explosion with heat conduction,
   Phys. Fluids A **3** (7) 1807 (1991).

.. [rmtvKamm] James R. Kamm,
   Investigation of the Reinicke & Meyer-ter-Vehn
   Equations: I. The Strong Conduction Case,
   LA-UR-00-4304



"""
from .timmes import Rmtv
