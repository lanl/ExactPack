r""" Heat Conduction Test Problems

The Heat Conduction test problems are meant to explore interesting and
numerically relevant regimes of the classical diffusive heat flow problem
in 1, 2, and 3 spatial dimensions, usually in simple geometries such as
planar, cylindrical or spherical. The heat flow problem is "local", in
that it may be specified on a (possibly compact) connected spatial region
:math:`\Omega` with boundary :math:`\partial\Omega`. The temperature at
point :math:`{\bf x} \in \Omega` and time :math:`t > 0` is denoted by
:math:`T = T({\bf x},t)`, and satisfies the diffusion equation

.. math::
  \frac{\partial T}{\partial t}
  =
  {\vec\nabla} \cdot \Big[ \kappa({\bf x}) \, \vec\nabla  T\big]  \ ,
  :label: diffEqGen

for all :math:`{\bf x} \in \Omega` and :math:`t > 0`, with :math:`\kappa=
\kappa({\bf x})` being the heat diffusion parameter at position :math:`{\bf x}`.
For simplicity, we assume that :math:`\kappa` is independent of position,
in which case, the heat flow equation reduces to

.. math::
  \frac{1}{\kappa} \, \frac{\partial T}{\partial t}
  =
  \nabla^2 T \hskip1.0cm {\bf x} \in \Omega \,\,{\rm and}\,\, t > 0 \ .
  :label: diffEq

To solve this equation, we must provide an initial temperature profile
:math:`T_0({\bf x})` for all :math:`{\bf x} \in \Omega`, and a boundary
condition :math:`\gamma({\bf x})` that specifies the temperature on
:math:`{\bf x} \in \partial \Omega` for arbitrary time,

.. math::
  T({\bf x}, 0)
  &=
  T_0({\bf x})
  \hskip1.0cm {\bf x} \in \Omega  \ ,
  \\[5pt]
  T({\bf x}, t)
  &=
  \gamma({\bf x})
  \hskip1.2cm {\bf x} \in \partial\Omega \,,\, t>0  \ .
  :label: InitCondNon

In fact, we will often consider the more general boundary condition (BC)
in which a linear combination of the temperature and heat flux is specified,

.. math::
  \alpha \, T({\bf x}, t) + \beta \, {\vec{\bf n}}\cdot {\vec\nabla}T({\bf x}, t)
  = \gamma({\bf x})
  \hskip1.0cm {\bf x} \in \partial\Omega \, ,\, t>0  \ ,
  :label: InitCondNonB

where :math:`\vec {\bf n} = {\vec {\bf n}}({\bf x})` is the outward normal to
the boundary at  :math:`{\bf x} \in \partial\Omega`. The coefficients
:math:`\alpha=\alpha({\bf x})` and :math:`\beta=\beta({\bf x})` are in general
functions of the position on the boundary, although we shall usually take
them to be constants independent of position.

In finding the solution for this problem, there are two cases to consider:
homogeneous and nonhomogeneous BC's. In the homogeneous case, the right-hand
side of the BC is taken to vanish, :math:`\gamma=0`, and the problem becomes

.. math::
  \frac{\partial \tilde T}{\partial t}
  &=
  \kappa^2 \, \nabla^2 \tilde T
  \hskip1.0cm {\bf x} \in \Omega \,,\, t > 0
  \\
  \alpha \, \tilde T + \beta \, {\vec{\bf n}}\cdot {\vec\nabla} \tilde T
  &= 0
  \hskip2.0cm {\bf x}\in \partial\Omega  \,,\, t > 0 \ ,
  :label: TildeT

where we have used the notation :math:`\tilde T({\bf x},t)` to denote the homogeneous
solution. In this case, the sum of any number of solutions remains a solution (this is
not true for nonhomogeneous BC's, for which :math:`\gamma \ne 0`), and consequently, the
most general homogeneous solution can be found by separation of variables and a
normal mode analysis on the spatial solutions. This gives a general solution of the form

.. math::
  \tilde T({\bf x}, t)
  =
  \sum_n D_n \, X_n({\bf x}) \, e^{-\kappa \, k_n^2 t} \ ,
  :label: TidleTSol

where the modes :math:`X_n({\bf x})` satisfy the Helmholtz equation

.. math::
  \nabla^2 X_n + k_n^2\, X_n = 0 \ ,
  :label: XnPDE

and the mode numbers :math:`k_n` are determined from the homogeneous BC's.
The modes :math:`X_n` are orthogonal, and since the Helmholtz equation is linear,
we can scale the :math:`X_n` to be orthonormal,

.. math::
  \int_\Omega dx \, X_n({\bf x}) \, X_m({\bf x}) = \delta_{n m}  \ .
  :label: XnXm

The values of the expansion coefficients :math:`D_n` will be determined from the
initial conditions (IC's) and the nonhomogeneous solution, as given by
:eq:`DnSolve` below.

The next step is to find a specific nonhomogeneous solution, which we accomplish
by solving the static equation (Laplace's equation) with the appropriate nonhomogeneous
BC,

.. math::
  \nabla^2 \bar T({\bf x})
  &=0
  \hskip1.5cm {\bf x} \in \Omega
  \\
  \alpha \, \bar T({\bf x}) + \beta \, {\vec{\bf n}}\cdot {\vec\nabla}
  \bar T({\bf x})
  &= \gamma({\bf x})
  \hskip1.0cm {\bf x}\in \partial\Omega  \ .
  :label: BarT

We denote the nonhomogeneous solution by :math:`\bar T({\bf x})`.

Once we have found :math:`\tilde T` and :math:`\bar T`, the most general time
dependent solution is the sum of the homogeneous and nonhomogeneous solutions,

.. math::
  T({\bf x},t) = \tilde T({\bf x}, t) + \bar T({\bf x})  \ .
  :label: Tgen

The initial condition now becomes :math:`T_0({\bf x}) = \tilde T({\bf x},0)
+ \bar T({\bf x})`, or

.. math::
  \sum_n  D_n \, X_n({\bf x}) = T_0({\bf x}) - \bar T({\bf x})  \ ,
  :label: Tzero

which can be solved for :math:`D_n` using the orthogonality condition
:eq:`XnXm`,

.. math::
  D_n = \int_\Omega dx \, \Big[T_0({\bf x}) - \bar T({\bf x}) \Big] X_m({\bf x})  \ .
  :label: DnSolve

"""

from .rod1d import Rod1D
from .planar_sandwich import PlanarSandwich
from .planar_sandwich_hot import PlanarSandwichHot
from .planar_sandwich_half import PlanarSandwichHalf
from .planar_sandwich_dawes import PlanarSandwichDawes
from .cylindrical_sandwich import CylindricalSandwich
from .hutchens1 import Hutchens1
from .hutchens2 import Hutchens2
from .rectangle import Rectangle
