r"""The 1D Riemann problems assume an imaginary membrane, located at :math:`x_M`, that separates the left and right states of a shocktube.
Each state is semi-infinite in extent, but computationally only occupies enough space for the problem under consideration.
Mathematically, :math:`(x,t) \in \mathbf{R}^1 \times \mathbf{R}^{1+}`, and for :math:`t=0^-` the left state is defined over the domain :math:`x_L = \{x \, | \, x_{\textrm{min}} \leq x < x_M\}`, whereas the right state is defined over :math:`x_R = \{x \, | x_M < x \geq x_{\textrm{max}}\}`.
A state prescribes constant pressure, density, and velocity, :math:`\left( p, \rho, u \right)`, values for :math:`t=0^{-}` for either the whole left state or the whole right state.
Similarly, the adiabatic index, :math:`\gamma`, is specified over the whole state, and is not required to be the same for the left and right states, however, it is currently assumed to be constant in time.
The internal energy is assumed to be calorically perfect, :math:`e = e(p, \rho)`.

At :math:`t=0` the membrane is removed, allowing the left and right regions to interact.
The response is a combination of shock waves, rarefactions, and a contact discontinuity, all propagating from the membrane's initial location.
Shock waves differ from contact discontinuities in that shocks are discontinuous in density, velocity, and pressure, whereas contact discontinuities are continuous in velocity and pressure, but discontinuous in density.

.. _fig_riemann_1d_drawing_ref:
.. figure:: riemann_1d_drawing.png
   :align: center
   :scale: 50 %

   Diagram of the 1D Riemann shocktube (bottom), and the :math:`x-t` diagram (top) of a solution showing a rarefaction wave traveling to the left, the membrane contact discontinuity and a shock traveling to the right. The initial left and right states are colored blue. The left and right star-states adjacent the contact discontinuity are colored pink. The rarefaction fan is colored with a blue-to-pink gradient.

..
   The Figure. :numref:`fig_riemann_1d_drawing_ref` is the.

These problems exercise the 1D, planar Riemann solver of a hydrodynamics code for an inviscid, non-heat conducting fluid satisfying the conservation statements of mass, momentum and energy:

.. math::
   \partial_t \rho + \partial_x \left( \rho \, u \right) = 0 \, , \\
   \partial_t \left( \rho \, u \right) + \partial_x \left( \rho \, u^2 + p \right) = 0 \, , \\
   \partial_t \left( \frac{1}{2} \rho \, u^2 + \rho \, e \right) + \partial_x \left[ u \left( \frac{1}{2} \rho \, u^2 + \rho \, e + p \right) \right] = 0 \, .
   :label: conservation

===============================
Solution Method for General EOS
===============================

In this section we present the solution method for a general EOS.
As shown below, given the initial left and right state values and functions related to the speed of sound, self-similar solutions representing rarefaction and shock conditions are constructed for the left and right states.
Rarefaction solutions exist when :math:`p^{\star} < p_{L,R}`, the initial pressure in the left state or the right state.
Shock solutions exist when :math:`p^{\star} > p_{L,R}`.
The rarefaction solution requires simultaneously integrating a pair of ordinary differential equations from the initial state values to the :math:`^{\star}`-state.
The shock solution requires  simultaneously satisfying a nonlinear pair of algebraic equations.

Rarefaction solution for a general EOS
++++++++++++++++++++++++++++++++++++++

The rarefaction states are integrated along characteristics, which may be defined via the self-similar coordinate, :math:`\xi = x / t`.
The space and time derivatives written in this coordinate are

.. math::
   \partial_x = \left( \partial_x \xi \right) \partial_{\xi} = t^{-1} \, \partial_{\xi} \, , \\
   \partial_t = \left( \partial_t \xi \right) \, \partial_{\xi} = - \, \xi \, t^{-1} \, \partial_{\xi} \, ,
   :label: similarity_transformation

so that Equations :eq:`conservation` become

.. math::
   \left( u - \xi \right) \, \partial_{\xi} \rho + \rho \, \partial_{\xi} u = 0 \, , \\
   \left( u - \xi \right) \, \rho \, \partial_{\xi} u + \partial_{\xi} p = 0 \, , \\
   \left( u - \xi \right) \, \partial_{\xi} S = 0 \, .
   :label: similarity_conservation

In this form energy conservation reverts to the first law of thermodynamics.
We assume :math:`\left( u - \xi \right) \neq 0`, so entropy is conserved along characteristics in a rarefaction zone.
Combining the first two equations produces the relation :math:`\left( u - \xi \right)^2 \, \partial_{\xi} \rho + \partial_{\xi} p = 0`.
Since we already have a constant entropy condition, the pressure derivative may be passed to density via the chain-rule:

.. math::
   \left( u - \xi \right)^2 \partial_{\xi} \rho + \left( \partial_{\xi} \rho \right) \left( \partial_{\rho} p \right)_{S = \textrm{const}} = \left( u - \xi \right)^2 \partial_{\xi} \rho + a^2 \, \partial_{\xi} \rho = 0 \, ,

where :math:`a` is the sound speed.
It can now be seen that :math:`\left( u - \xi \right) = \pm \, a`.
The signs correspond to left (+) or right (-) traveling waves, respectively.
These relations hold along all characteristics, so terms may be rearranged and the derivatives rewritten as differentials.
In this way, the following integrals are produced from 1) momentum conservation and 2) the relation provided by the first two equations:

.. math::
   \int^{u^{\star}}_{u_0} du = \pm \int^{p^{\star}}_{p_0} \frac{dp}{\rho \, a} \, , \qquad
   \int^{\rho^{\star}}_{\rho_0} d\rho = \int^{p^{\star}}_{p_0} \frac{dp}{a^2} \, .

These equations may be written in a more suggestive form that also has the benefit of being more in-line with what the solver is actually doing:

.. math::
   \frac{du}{dp} = \frac{1}{\rho \, a(p, \rho)} \, , \qquad
   \frac{d\rho}{dp} = \frac{1}{a( p, \rho)^2} \, .

The left-hand sides represent the integrals but it is the expressions on the right-hand side that matter.
Given an analytic expression for :math:`a = a(p, \rho)`, the initial conditions :math:`\left( p_0, \rho_0, u_0 \right)`, and an integration array for :math:`p^{\star}`, these equations may be integrated from :math:`\left( p_0, \rho_0, u_0 \right) \rightarrow \left( p^{\star}, \rho^{\star}, u^{\star} \right)`.
Since a rarefaction wave is smoothly integrated from the quiescent state at :math:`\left( p_0, \rho_0, u_0 \right)` to the :math:`^{\star}`-state, the intermediate values for :math:`(p, \rho, u)` are strictly monotonic in this region.
Further, since rarefaction waves occur when :math:`p^{\star} < p_0`, and :math:`a^2 > 0`, we see that density must decrease monotonically when going from :math:`\rho_0` to :math:`\rho^{\star}`.

.. _fig_PUdiag_LeeJWL_zoomed_fig:
.. figure:: riemann_PUdiagram_LeeJWL_solution_zoomed.png
   :scale: 50 %
   :align: center

   P-U diagram for the Lee JWL problem, solved using the general EOS solver. The left states are the blue lines; the right states are red. Shocks are solid, rarefactions are dashed, and the initial :math:`(p_0, u_0)` state is dotted. Shocks exists for :math:`p > p_0`, and rarefactions below :math:`p_0`. The rarefaction pressure is bounded from below by 0. An imposed maximum shock pressure is :math:`10 \times \max \left( p_0 \right)`; in the figure, :math:`p_{\max} = 20 [dyn/cm^2]`.

Shock solution for a general EOS
++++++++++++++++++++++++++++++++

The shock states represent jump conditions relating the conservation fluxes on the left and right sides of the shock, so that the conservation equations are now

.. math::
   \rho \, u = \rho_0 \, u_0 \, , \\
   \rho \, u^2 + p = \rho_0 \, u_0^2 + p_0 \, , \\
   u \left( \frac{1}{2} \rho \, u^2 + \rho \, e + p \right) = u_0 \left( \frac{1}{2} \rho_0 \, u_0^2 + \rho_0 \, e_0 + p_0 \right) \, .
   :label: conservation_fluxes

In this we've translated to a reference frame in which the shock is at rest.
This shift only affects velocity, but none of the other physical fields, and is straight-forward to undo at the end of the analysis.
Now we rearrange the momentum jump condition, use mass conservation and solve for :math:`u` to produce

.. math::
   u = \pm \sqrt{\left( \frac{\rho_0}{\rho} \right) \left[ \frac{p - p_0}{\rho - \rho_0} \right]}
   :label: math_u

Notice that the momentum jump condition is symmetric when interchanging all three variables from the left and right sides, :math:`\left( \rho, u, p \right) \leftrightarrow \left( \rho_0, u_0, p_0 \right)`.
Using the above expression, eqn. (:eq:`math_u`), this allows us to quickly write

.. math::
   u_0 = \pm \sqrt{\left( \frac{\rho}{\rho_0} \right) \left[ \frac{p - p_0}{\rho - \rho_0} \right]}
   :label: math_u0

In these expressions the "+"-sign is for left-going waves and the "-"-sign is for right-going waves.

In the energy jump condition, use mass conservation on both sides and then the expressions for :math:`u` and :math:`u_0` just obtained to produce

.. math::
   e^{\star} \left( p^{\star}, \rho^{\star} \right) + \frac{p^{\star}}{\rho^{\star}} + \frac{1}{2} \frac{\rho_0}{\rho^{\star}} \frac{p^{\star} - p_0}{\rho^{\star} - \rho_0} - \left( e_0 \left( p_0, \rho_0 \right) + \frac{p_0}{\rho_0} + \frac{1}{2} \frac{\rho^{\star}}{\rho_0} \frac{p^{\star} - p_0}{\rho^{\star} - \rho_0} \right) = 0 \, .
   :label: shock_energy_condition

Moving now to a frame in which the shock is not at rest, we define the shock speed, :math:`U`, so that

.. math::
   u_0 = U \pm \sqrt{\left( \frac{\rho}{\rho_0} \right) \left[ \frac{p - p_0}{\rho - \rho_0} \right]} \, , \\
   u = U \pm \sqrt{\left( \frac{\rho_0}{\rho} \right) \left[ \frac{p - p_0}{\rho - \rho_0} \right]} \, ,

and the difference between these velocities is

.. math::
   u - u_0 = \pm \sqrt{\left( \frac{\rho_0}{\rho} \right) \left[ \frac{p - p_0}{\rho - \rho_0} \right]} \mp \sqrt{\left( \frac{\rho}{\rho_0} \right) \left[ \frac{p - p_0}{\rho - \rho_0} \right]} \, .
   :label: math_velocity_diff_equation

Assuming an array of "known" values for :math:`p`, and given :math:`\left( \rho_0, p_0, u_0 \right)`, the simultaneous solution of equations :eq:`shock_energy_condition` and :eq:`math_velocity_diff_equation` provides solutions for :math:`u` and :math:`\rho`.
..
   This is how the solid shock lines are produced in :numref:`fig_PUdiag_LeeJWL_zoomed_fig`.

Splicing solutions to determine the wave structure
++++++++++++++++++++++++++++++++++++++++++++++++++

..
   In :numref:`fig_PUdiag_LeeJWL_zoomed_fig` we have zoomed in on the region where the blue and red lines cross.
The solid (shock) portion of each line is a natural extension of the dashed (rarefaction) portion, so these curves can be spliced together.
Since these curves are smooth and monotonic it is straight-forward to take the difference between their velocity values and determine where the absolute value of this difference is a minimum.
This point defines the :math:`\left( p^{\star}, \rho^{\star}, u^{\star} \right)` state, which is the solution we've been seeking.
Last, we determine whether these :math:`^{\star}`-state values exist within the rarefaction or shock arrays for each state.
What remains is to propagate the waves along their self-similar trajectories and produce spatially-dependent values of the physical fields at a given instant of time.

==================================================================
Solution Method for an Ideal-Gas Polytropic :math:`\gamma`-law EOS
==================================================================

Transform to 

Rarefaction solution
++++++++++++++++++++

Transform to 

Shock solution
++++++++++++++

Transform to 

==================
Problem Statements
==================

We define six standard Riemann problems in the table below.
These all use the same :math:`\gamma` for both the left and right states.

===================== =========== =========== ============== ================= =========== ============== ================= =========== ==============
Test                  :math:`x_M` :math:`t_f` :math:`\rho_L` :math:`u_L`       :math:`p_L` :math:`\rho_R` :math:`u_R`       :math:`p_R` :math:`\gamma`  
===================== =========== =========== ============== ================= =========== ============== ================= =========== ==============
1 Sod                 0.5         0.25        1.0            0.0               1.0         0.125          0.0               0.1         7/5
2 Einfeldt            0.5         0.15        1.0            :math:`-2.0`      0.4         1.0            2.0               0.4         7/5
3 Stationary Contact  0.8         0.012       1.0            :math:`-19.59745` 1000        1.0            :math:`-19.59745` 0.02        7/5
4 Slow Shock          0.5         1.0         1.0            :math:`-0.810631` 31/3        1.0            :math:`-3.44`     1.0         7/5
5 Shock Contact Shock 0.5         0.3         1.0            0.5               1.0         1.25           :math:`-0.5`      1.0         7/5
6 LeBlanc             0.3         0.5         1.0            0.0               2/30        0.01           0.0               2/(3e10)    5/3
===================== =========== =========== ============== ================= =========== ============== ================= =========== ==============

.. table:: **P-U diagrams for the Lee JWL problem, solved using the general EOS solver.**

   +---------------------------------------------------+----------------------------------------------------------+
   | .. figure:: riemann_PUdiagram_LeeJWL_solution.png | .. figure:: riemann_PUdiagram_LeeJWL_solution_zoomed.png |
   +---------------------------------------------------+----------------------------------------------------------+

"""

from .riemann import *
