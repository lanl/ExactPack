r"""The Coggeshall problems.

The Coggeshall [Coggeshall1991]_ problems are a collection of exact
solutions to the one-dimensional Euler equations with heat conduction and
no viscosity.  The fluid field variables are the mass density :math:`\rho(r,t)`,
the fluid velocity
:math:`u(r,t)`, the fluid temperature :math:`T(r,t)`, and the specific
internal energy :math:`e(r,t)` of the fluid material, where :math:`r`
is the spatial coordinate in planar, cylindrical, or spherical
geometry. The :math:`\gamma`-law Equation of State (EOS) for the gas
is written

.. math::
  P &= \Gamma\, \rho T
  \\[3pt]
  e &= \frac{\Gamma\, T}{\gamma - 1} \ ,

where :math:`\Gamma` is the Gruneisen gas constant and :math:`\gamma
\equiv c_{\rm p}/c_{\rm v}` is the adiabatic exponent. Dividing these equations
gives the standard adiabatic :math:`\gamma`-law EOS,

.. math::
  P = (\gamma - 1) \, \rho e \ .

Upon taking :math:`\rho`, :math:`u`, and :math:`T` as the independent variables,
the conservation of mass, momentum, and energy imply the following balance equations:

.. math::

  \frac{\partial \rho}{\partial t}   +
  u \frac{\partial \rho}{\partial r} +
  \rho \frac{\partial u}{\partial r} +  \frac{k \rho u}{r} &= 0
  \\
  \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial r} +
  \frac{\Gamma T}{\rho}\frac{\partial \rho}{\partial r} +
  \Gamma\, \frac{\partial T}{\partial r} &= 0
  \\
  \frac{T}{\gamma-1}\left[ \frac{\partial T}{\partial t}   +
  u \frac{\partial T}{\partial r} \right]   +
  \Gamma T\frac{\partial u}{\partial r}   +
  \Gamma T \frac{k u}{r} + \frac{1}{\rho}\left[\frac{\partial F}{\partial r} +
  \frac{k F}{r}
  \right] &= 0 \ ,

where :math:`k=0,1,2` is the geometry factor specifying planar, cylindrical,
or spherical coordinates, respectively. Note that the geometry option
in ExactPack is different, with :math:`{\rm\bf geometry} = k + 1 = 1, 2, 3`.
The quantity :math:`F` in the energy equation is the magnitude of the heat
flux vector, which, in the diffusion approximation, takes the form

.. math::
 {\vec F}
  =
  - \frac{c\, \lambda}{3}\, {\vec \nabla}\, a T^4 \ ,

with :math:`c` being the speed of light, :math:`a` the radiation constant,
and :math:`\lambda` the radiation mean-free-path. To obtain a semi-analytic
solution, we parameter :math:`\lambda` by

.. math::
   \lambda(\rho,T) = \lambda_0\, \rho^\alpha T^\beta ,
   :label: lambdaDef


where :math:`\lambda_0` is a dimensionfull constant, and :math:`\alpha` and
:math:`\beta` are dimensionless parameters in the ranges  :math:`-1 \le \alpha \le 2`
and :math:`1 \le \beta \le 3`. The mean-free-path :math:`\lambda` is related to the
Rosseland mean opacity :math:`\kappa` by :math:`\kappa= 1/\lambda\rho`. The heat
flux can also be written in term of the heat conductivity :math:`K`, defined by

.. math::

  {\vec F}
  =
  \frac{4 c \lambda a T^3}{3} ~ {\vec \nabla} T \equiv K(\rho,T) {\vec \nabla} T .
"""

from .cog1  import Cog1
from .cog2  import Cog2
from .cog3  import Cog3
from .cog4  import Cog4
from .cog5  import Cog5
from .cog6  import Cog6
from .cog7  import Cog7
from .cog8  import Cog8
from .cog9  import Cog9
from .cog10 import Cog10
from .cog11 import Cog11
from .cog12 import Cog12
from .cog13 import Cog13
from .cog14 import Cog14
from .cog16 import Cog16
from .cog17 import Cog17
from .cog18 import Cog18
from .cog19 import Cog19
from .cog20 import Cog20
from .cog21 import Cog21


