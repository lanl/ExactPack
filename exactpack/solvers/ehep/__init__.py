r"""The Escape of HE Products problem.

The escape of HE products (EHEP) problem was first published by Fickett
and Rivard in 1974 [Fickett1974]_. In 2002, [Dykema2002]_ published a derivation of
the characteristics of the exact solution in :math:`x`-:math:`t` space.
A complete description of the problem, the exact solution equations,
and the solution algorithm is presented in [Doebling2015]_.
The problem concerns a planar
one-dimensional rod of HE extending from the origin for a length
of :math:`\tilde{x}`, as shown in the Figure. 

.. figure:: ehep_drawing.png
   :alt: Diagram of the EHEP problem
   :align: center
   :figwidth: 100%

   Diagram of the EHEP problem

The HE is a polytropic
ideal gas with adiabatic index
:math:`\gamma=3` and C-J detonation velocity :math:`D_{\rm CJ}`.
(The value :math:`\gamma=3` is required to enable the derivation of
the exact solution.) The unreacted HE and the reacted HE are assumed
to have the same material properties.
To the left of the HE is a piston moving in the +x direction with
velocity :math:`u_p`. To the right of the HE is a void. At :math:`t=0`,
the detonation
wave departs from the origin in the +x direction, and the piston begins
to move and isentropically compresses the reaction products. When the
detonation wave reaches :math:`\tilde{x}` at time :math:`\tilde{t}`,
the HE has all
been consumed, and the material begins to expand isentropically into
the void region. At the same time :math:`\tilde{t}`, the arrival of the
detonation wave at an interface with lesser impedence to the right
causes a rarefaction wave to be propagated from :math:`\tilde{x}` in the
-x direction.
The independent fluid variables are
(i) the gas density :math:`\rho(x,t)`, (ii) the velocity of the gas
:math:`u(x,t)`, and (iii) the pressure :math:`p(x,t)`, each at spatial
location :math:`x` and time :math:`t`. The specific internal energy
:math:`e(r,t)` is related to the other fluid variables
by the equation of state (EOS) for an ideal gas at constant entropy,

.. math::
  p = (\gamma-1) \rho e  \ ,

where :math:`\gamma \equiv c_p/c_v` is the ratio of specific heats at constant
pressure and volume. The choice of :math:`\gamma=3` produces solution
characteristics that are straight lines in the :math:`x`-:math:`t` plane, and
the values for the physical variables are then derived for each region defined
by the characteristics.

To find the solution at a given value of :math:`(x,t)`, we must first determine
the region in which the point lies, and then use the corresponding Euler equations
for :math:`\rho(x,t), u(x,t), p(x,t), e(x,t)` in that region. The easiest way
to describe the regions is to define the coordinates of the vertices of the
polygon that bounds each region. To close the polygons, we must select a maximum
distance :math:`x_{\rm max}` and maximum time :math:`t_{\rm max}`. For each
region, the vertices that describe its polygon are listed here:

Region 0V (ahead of the shock in the void region):

..  math::

  (\tilde{x},0) \\
  (\tilde{x},\tilde{t}) \\
  (x_{\rm max},\frac{x_{\rm max}}{D}) \\
  (x_{\rm max},0) \\
  c_s = 0\\
  u = 0 \\
  p = 0 \\
  \rho = 0

Region 0H (ahead of the shock within the HE):

..  math::

  (0,0) \\
  (\tilde{x},\tilde{t}) \\
  (\tilde{x},0) \\
  c_s = 0\\
  u = 0 \\
  p = 0 \\
  \rho = \rho_0

For Regions I through V, :math:`p` and :math:`\rho` are given as functions
of :math:`c_s`:

..  math::

  p = \frac{16}{27} \rho_0 D^2  \left(\frac{c_s}{D}\right)^3 \\
  \rho = \frac{16}{9} \rho_0 \left(\frac{cs}{D}\right)

Region I (behind the detonation wave):

..  math::

  (0,0) \\
  \frac{3\tilde{x}}{2}\left(1-\frac{D}{4u_p+2D},
      \frac{1}{2u_p+D}\right) \\
  (\tilde{x},\tilde{t})\\
  c_s = \frac{1}{2} \left(\frac{x}{t} + \frac{D}{2}\right)\\
  u   = \frac{1}{2} \left(\frac{x}{t} - \frac{D}{2}\right)\\

Region II (isentropic expansion into the void)

..  math::

  (\tilde{x},\tilde{t}) \\
  \frac{3\tilde{x}}{2}\left(1-\frac{D}{4u_p+2D},
      \frac{1}{2u_p+D}\right) \\
  \left( (2u_p+D/2)t_{\rm max},t_{\rm max} \right)\\
  (x_{\rm max},\frac{x_{\rm max}}{D})	\\
  c_s = \frac{1}{2} \left(\frac{x}{t}-\frac{x-\tilde{x}}{t-\tilde{t}}\right)\\
  u = \frac{1}{2} \left(\frac{x}{t}+\frac{x-\tilde{x}}{t-\tilde{t}}\right)

Region III (fluid pushed by piston)

..  math::

  (0,0) \\
  \frac{3\tilde{x}}{2}\left(1-\frac{D}{4u_p+2D},
      \frac{1}{2u_p+D}\right) \\
  \frac{3\tilde{x}}{2u_p+D}(1,u_p) \\
  c_s = u_p + \frac{D}{2} \\
  u = u_p

Region IV (fluid expansion and rarefaction)

..  math::

  \frac{3\tilde{x}}{2}\left(1-\frac{D}{4u_p+2D},
      \frac{1}{2u_p+D}\right) \\
  \frac{3\tilde{x}}{2u_p+D}(1,u_p) \\
  \left(-\frac{3}{2}\tilde{x} + (2u_p+\frac{D}{2})t_{\rm max}, t_{\rm max}\right) \\
  \left(x_{\rm max}, \frac{x_{\rm max}}{2u_p+D/2}\right) \\
  c_s = u_p + \frac{D}{2}\left(\frac{1}{2} - \frac{x-\tilde{x}}
        {Dt-\tilde{x}}\right) \\
  u = u_p + \frac{D}{2}\left(\frac{1}{2} + \frac{x-\tilde{x}}
        {Dt-\tilde{x}}\right)

Region V (rarefaction interaction with piston)

..  math::

  \frac{3\tilde{x}}{2u_p+D}(1,u_p) \\
  (u_p t_{\rm max},t_{\rm max}) \\
  \left(-\frac{3}{2}\tilde{x} + (2u_p+\frac{D}{2})t_{\rm max}, t_{\rm max}\right) \\
  c_s = \frac{(D-u_p)\tilde{t}}{t-\tilde{t}} \\
  u = \frac{x-u_p\tilde{t}}{t-\tilde{t}}

Region 00 (behind the piston)

..  math::

  (0,0)\\
  (0,t_{\rm max})\\
  (u_p t_{\rm max},t_{\rm max})
"""

from ehep import EscapeOfHEProducts
