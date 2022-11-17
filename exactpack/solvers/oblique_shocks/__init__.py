r"""The 1D Riemann problems assume an imaginary membrane, located at :math:`x_M`, that separates the left and right states of a shocktube.
Each state is semi-infinite in extent, but computationally only occupies enough space for the problem under consideration.
Mathematically, :math:`(x,t) \in \mathbf{R}^1 \times \mathbf{R}^{1+}`, and for :math:`t=0^-` the left state is defined over the domain :math:`x_L = \{x \, | \, x_{\textrm{min}} \leq x < x_M\}`, whereas the right state is defined over :math:`x_R = \{x \, | x_M < x \geq x_{\textrm{max}}\}`.
A state prescribes constant pressure, density, and velocity, :math:`\left( p, \rho, u \right)`, values for :math:`t=0^{-}` for either the whole left state or the whole right state.
Similarly, the adiabatic index, :math:`\gamma`, is specified over the whole state, and is not required to be the same for the left and right states, however, it is currently assumed to be constant in time.
The internal energy is assumed to be calorically perfect, :math:`e = e(p, \rho)`.
# 
# At :math:`t=0` the membrane is removed, allowing the left and right regions to interact.
# The response is a combination of shock waves, rarefactions, and a contact discontinuity, all propagating from the membrane's initial location.
# Shock waves differ from contact discontinuities in that shocks are discontinuous in density, velocity, and pressure, whereas contact discontinuities are continuous in velocity and pressure, but discontinuous in density.
# 
# .. _fig_riemann_1d_drawing_ref:
# .. figure:: riemann_1d_drawing-modified.png
#    :align: center
#    :scale: 50 %
# 
#    Diagram of the 1D Riemann shocktube (bottom), and the :math:`x-t` diagram (top) of a solution showing a rarefaction wave traveling to the left, the membrane contact discontinuity and a shock traveling to the right. The initial left and right states are colored blue. The left and right star-states adjacent the contact discontinuity are colored pink. The rarefaction fan is colored with a blue-to-pink gradient.
# 
# These problems exercise the 1D, planar Riemann solver of a hydrodynamics code for an inviscid, non-heat conducting fluid satisfying the conservation statements of mass, momentum and energy:
# 
# .. math::
#    \partial_t \rho + \partial_x \left( \rho \, u \right) = 0 \, , \\
#    \partial_t \left( \rho \, u \right) + \partial_x \left( \rho \, u^2 + p \right) = 0 \, , \\
#    \partial_t \left( \frac{1}{2} \rho \, u^2 + \rho \, e \right) + \partial_x \left[ u \left( \frac{1}{2} \rho \, u^2 + \rho \, e + p \right) \right] = 0 \, .
#    :label: conservation
# 
# ===============================
# Solution Method for General EOS
# ===============================
# 
# In this section we present the solution method for a general EOS.
# As shown below, given the initial left- and right-state values, :math:`(p, u, \rho)_L` and :math:`(p, u, \rho)_R`, and the speed of sound as a function of density and pressure, self-similar solutions representing either rarefaction or shock conditions are constructed for the left and right states.
# When generically discussing either of the left- or right-states, they may be denoted :math:`(p, u, \rho)_0`.
# In the solution process, the contact discontinuity separating the left and right states yields equal pressures and velocities, :math:`p^{\star}_L = p^{\star}_R` and :math:`u^{\star}_L = u^{\star}_R`, but unequal densities, :math:`\rho^{\star}_L \neq \rho^{\star}_R`.
# The state values adjacent the contact discontinuity are denoted by the :math:`{ }^{\star}`-superscript.
# Rarefaction solutions exist when :math:`p^{\star} < p_0`.
# Shock solutions exist when :math:`p^{\star} > p_0`.
# The rarefaction solution requires simultaneously integrating a pair of ordinary differential equations (ODEs) from the initial state values to the :math:`^{\star}`-state.
# The shock solution requires  simultaneously satisfying a nonlinear pair of algebraic equations.
# 
# Rarefaction solution for a general EOS
# ++++++++++++++++++++++++++++++++++++++
# 
# The rarefaction states are integrated along characteristics, which may be defined via the self-similar coordinate, :math:`\xi = x / t`.
# The space and time derivatives written in this coordinate are
# 
# .. math::
#    \partial_x = \left( \partial_x \xi \right) \partial_{\xi} = t^{-1} \, \partial_{\xi} \, , \\
#    \partial_t = \left( \partial_t \xi \right) \, \partial_{\xi} = - \, \xi \, t^{-1} \, \partial_{\xi} \, ,
#    :label: similarity_transformation
# 
# so that Equations :eq:`conservation` become
# 
# .. math::
#    \left( u - \xi \right) \, \partial_{\xi} \rho + \rho \, \partial_{\xi} u = 0 \, , \\
#    \left( u - \xi \right) \, \rho \, \partial_{\xi} u + \partial_{\xi} p = 0 \, , \\
#    \left( u - \xi \right) \, \partial_{\xi} S = 0 \, .
#    :label: similarity_conservation
# 
# In this form energy conservation reverts to the first law of thermodynamics.
# We assume :math:`\left( u - \xi \right) \neq 0`, so entropy is conserved along characteristics in a rarefaction zone.
# Combining the first two equations produces the relation :math:`\left( u - \xi \right)^2 \, \partial_{\xi} \rho + \partial_{\xi} p = 0`.
# Since we already have a constant entropy condition, the pressure derivative may be passed to density via the chain-rule:
# 
# .. math::
#    \left( u - \xi \right)^2 \partial_{\xi} \rho + \left( \partial_{\xi} \rho \right) \left( \partial_{\rho} p \right)_{S = \textrm{const}} = \left( u - \xi \right)^2 \partial_{\xi} \rho + a^2 \, \partial_{\xi} \rho = 0 \, ,
# 
# where :math:`a` is the sound speed.
# It can now be seen that :math:`\left( u - \xi \right) = \pm \, a`.
# The signs correspond to left (+) or right (-) traveling waves, respectively.
# These relations hold along all characteristics, so terms may be rearranged and the derivatives rewritten as differentials.
# In this way, the following integrals are produced from 1) momentum conservation and 2) the relation provided by the first two equations:
# 
# .. math::
#    \int^{u^{\star}}_{u_0} du = \pm \int^{p^{\star}}_{p_0} \frac{dp}{\rho \, a} \, , \qquad
#    \int^{\rho^{\star}}_{\rho_0} d\rho = \int^{p^{\star}}_{p_0} \frac{dp}{a^2} \, .
# 
# Given an analytic expression for :math:`a = a(p, \rho)`, the initial conditions for either the left- or right-state :math:`\left( p_0, \rho_0, u_0 \right)`, and an integration array for :math:`p` between :math:`[p_0, p^{\star}]`, these equations may be integrated from :math:`\left( p_0, \rho_0, u_0 \right) \rightarrow \left( p^{\star}, \rho^{\star}, u^{\star} \right)`.
# Since :math:`(p, a, \rho)` are required to be positive, and a rarefaction wave is smooth and continuous between the quiescent state and the contact discontinuity, the intermediate values for :math:`(p, \rho, u)` are strictly monotonic in a rarefaction zone.
# Further, since rarefaction waves occur when :math:`p^{\star} < p_0`, we see that density must decrease monotonically when going from :math:`\rho_0` to :math:`\rho^{\star}`.
# 
# These equations may be written in a more suggestive form that also has the benefit of being more in-line with how the solver is numerically integrating the ODEs:
# 
# .. math::
#    \frac{du}{dp} = \frac{1}{\rho \, a(p, \rho)} \, , \qquad
#    \frac{d\rho}{dp} = \frac{1}{a( p, \rho)^2} \, .
# 
# The left-hand sides represent the integrals but it is the expressions on the right-hand side that are integrated numerically.
# 
# Shock solution for a general EOS
# ++++++++++++++++++++++++++++++++
# 
# The shock states represent jump conditions relating the conservation fluxes on the left and right sides of the shock, so that the conservation equations are now
# 
# .. math::
#    \rho \, u = \rho_0 \, u_0 \, , \\
#    \rho \, u^2 + p = \rho_0 \, u_0^2 + p_0 \, , \\
#    u \left( \frac{1}{2} \rho \, u^2 + \rho \, e + p \right) = u_0 \left( \frac{1}{2} \rho_0 \, u_0^2 + \rho_0 \, e_0 + p_0 \right) \, .
#    :label: conservation_fluxes
# 
# In this we've translated to a reference frame in which the shock is at rest.
# This shift only affects velocity, but none of the other physical fields, and is straight-forward to undo at the end of the analysis.
# Now we rearrange the momentum jump condition, use mass conservation and solve for :math:`u` to produce
# 
# .. math::
#    u = \pm \sqrt{\left( \frac{\rho_0}{\rho} \right) \left[ \frac{p - p_0}{\rho - \rho_0} \right]}
#    :label: math_u
# 
# Since the shock separates the contact discontinuity region from the initial state, it would be appropriate to relabel :math:`u` as :math:`u^{\star}`, which is done below.
# Notice that the momentum jump condition is symmetric when interchanging all three variables from the left and right sides, :math:`\left( \rho, u, p \right) \leftrightarrow \left( \rho_0, u_0, p_0 \right)`.
# Using the above expression, eqn. :eq:`math_u`, this allows us to quickly write
# 
# .. math::
#    u_0 = \pm \sqrt{\left( \frac{\rho}{\rho_0} \right) \left[ \frac{p - p_0}{\rho - \rho_0} \right]}
#    :label: math_u0
# 
# In these expressions the "+"-sign is for left-going waves and the "-"-sign is for right-going waves.
# 
# In the energy jump condition, use mass conservation on both sides and then the expressions for :math:`u` and :math:`u_0` just obtained to produce
# 
# .. math::
#    e^{\star} \left( p^{\star}, \rho^{\star} \right) + \frac{p^{\star}}{\rho^{\star}} + \frac{1}{2} \frac{\rho_0}{\rho^{\star}} \frac{p^{\star} - p_0}{\rho^{\star} - \rho_0} - \left( e_0 \left( p_0, \rho_0 \right) + \frac{p_0}{\rho_0} + \frac{1}{2} \frac{\rho^{\star}}{\rho_0} \frac{p^{\star} - p_0}{\rho^{\star} - \rho_0} \right) = 0 \, .
#    :label: shock_energy_condition
# 
# Moving now to a frame in which the shock is not at rest, we define the shock speed, :math:`U`, so that
# 
# .. math::
#    u_0 = U \pm \sqrt{\left( \frac{\rho^{\star}}{\rho_0} \right) \left[ \frac{p^{\star} - p_0}{\rho^{\star} - \rho_0} \right]} \, , \\
#    u^{\star} = U \pm \sqrt{\left( \frac{\rho_0}{\rho^{\star}} \right) \left[ \frac{p^{\star} - p_0}{\rho^{\star} - \rho_0} \right]} \, ,
# 
# and the difference between these velocities is
# 
# .. math::
#    u^{\star} - u_0 = \pm \sqrt{\left( \frac{\rho_0}{\rho^{\star}} \right) \left[ \frac{p^{\star} - p_0}{\rho^{\star} - \rho_0} \right]} \mp \sqrt{\left( \frac{\rho^{\star}}{\rho_0} \right) \left[ \frac{p^{\star} - p_0}{\rho^{\star} - \rho_0} \right]} \, .
#    :label: math_velocity_diff_equation
# 
# Assuming an array of "known" values for :math:`p^{\star}`, and given :math:`\left( \rho_0, p_0, u_0 \right)`, the simultaneous solution of equations :eq:`shock_energy_condition` and :eq:`math_velocity_diff_equation` provides solutions for :math:`u` and :math:`\rho`.
# 
# Splicing solutions to determine the wave structure
# ++++++++++++++++++++++++++++++++++++++++++++++++++
# 
# The solid (shock) portion of each line is a natural extension of the dashed (rarefaction) portion, so these curves can be spliced together.
# Since these curves are smooth and monotonic it is straight-forward to take the difference between their velocity values and determine where the absolute value of this difference is a minimum.
# This point defines the :math:`\left( p^{\star}, \rho^{\star}, u^{\star} \right)` state, which is the solution we've been seeking.
# Last, we determine whether these :math:`^{\star}`-state values exist within the rarefaction or shock arrays for each state.
# What remains is to propagate the waves along their self-similar trajectories and produce spatially-dependent values of the physical fields at a given instant of time.
# 
# .. _fig_PUdiag_LeeJWL_zoomed_fig:
# .. figure:: riemann_PUdiagram_LeeJWL_solution_zoomed.png
#    :scale: 50 %
#    :align: center
# 
#    P-U diagram for the Lee JWL problem, solved using the general EOS solver. The left states are the blue lines; the right states are red. Shocks are solid, rarefactions are dashed, and the initial :math:`(p_0, u_0)` state is dotted. Shocks exists for :math:`p > p_0`, and rarefactions below :math:`p_0`. The rarefaction pressure is bounded from below by 0. An imposed maximum shock pressure is :math:`10 \times \max \left( p_0 \right)`; in the figure, :math:`p_{\max} = 20 [dyn/cm^2]`.
# 
# ==================================================================
# Solution Method for an Ideal-Gas Polytropic :math:`\gamma`-law EOS
# ==================================================================
# 
# The 2013 paper by LoraClavijo [LoraClavijo2013]_ provides a useful explanation of the simultaneous equations to be solved for an ideal-gas EOS Riemann problem.
# Additionally, this ideal-gas EOS solver uses ideas from the 1988 paper by Gottlieb and Groth [GottliebGroth1988]_, specifically their Figure 3, to determine whether each left and right zone is occupied by a shock or rarefaction wave.
# 
# ==================
# Problem Statements
# ==================
# 
# In this section we present 7 problem statements for the ideal-gas EOS, and two problem statements for the Jones-Wilkins-Lee (JWL) EOS.
# The JWL EOS is of the Mie-Gruneisen form and typically used for high-explosive modeling.
# 
# Seven ideal-gas EOS problem statements
# ++++++++++++++++++++++++++++++++++++++
# 
# We define seven standard Riemann problems in the table below.
# The first six all use the same :math:`\gamma` for both the left and right states, while the seventh uses different gammas.
# For convenience, variable dimensions are: :math:`x_M \, [cm]`, :math:`t_f \, [s]`, :math:`\rho \, [g/cm^3]`, :math:`u \, [cm/s]`, :math:`p \, [dyn/cm^2]`, :math:`\gamma` is dimensionless.
# 
# ===================== =========== =========== ============== =========== =========== ============== =========== =========== ================ ================
# Test                  :math:`x_M` :math:`t_f` :math:`\rho_L` :math:`u_L` :math:`p_L` :math:`\rho_R` :math:`u_R` :math:`p_R` :math:`\gamma_L` :math:`\gamma_R`  
# ===================== =========== =========== ============== =========== =========== ============== =========== =========== ================ ================
# 1 Sod                 0.5         0.25        1.0            0.0         1.0         0.125          0.0         0.1         7/5              7/5
# 2 Einfeldt            0.5         0.15        1.0            -2.0        0.4         1.0            2.0         0.4         7/5              7/5
# 3 Stationary Contact  0.8         0.012       1.0            -19.59745   1000        1.0            -19.59745   0.02        7/5              7/5
# 4 Slow Shock          0.5         1.0         1.0            -0.810631   31/3        1.0            -3.44       1.0         7/5              7/5
# 5 Shock Contact Shock 0.5         0.3         1.0            0.5         1.0         1.25           -0.5        1.0         7/5              7/5
# 6 LeBlanc             0.3         0.5         1.0            0.0         1/15        0.01           0.0         2/(3e10)    5/3              5/3
# 7 Modified Sod        0.5         0.2         1.0            0.0         2.0         0.125          0.0         0.1         2                7/5
# ===================== =========== =========== ============== =========== =========== ============== =========== =========== ================ ================
# 
# Plots of each solution are given below.
# These seven problems, and their reversed states provide the basis of for most of the unittests for both the ideal-gas and general EOS solvers.
# 
# +-------------------------------------+------------------------------------------+
# | .. figure:: sod.png                 | .. figure:: einfeldt.png                 |
# +-------------------------------------+------------------------------------------+
# | .. figure:: stationary_contact.png  | .. figure:: slow_shock.png               |
# +-------------------------------------+------------------------------------------+
# | .. figure:: shock_contact_shock.png | .. figure:: leblanc.png                  |
# +-------------------------------------+------------------------------------------+
# | .. figure:: sod_modified.png        | this space intentionally left blank      |
# +-------------------------------------+------------------------------------------+
# 
# Two JWL EOS problem statements
# ++++++++++++++++++++++++++++++
# 
# We define two standard Riemann problems in the table below.
# In addition to defining values for density, velocity and pressure for the left- and right-states, we also have to define parameters for the JWL EOS.
# For convenience, variable dimensions are: :math:`x_M \, [cm]`, :math:`t_f \, [\mu s]`, :math:`\rho \, [g/cm^3]`, :math:`u \, [cm/\mu s]`, :math:`p \, [Mbar]`, :math:`\gamma` is dimensionless, :math:`e_0 \, [Mbar-cm^3/g]`, :math:`\Gamma_0 = \gamma - 1` is dimensionless, :math:`A \, [Mbar]`, :math:`B \, [Mbar]`, and :math:`R_1` and :math:`R_2` are dimensionless.
# 
# ===================== ============== =========== ================ =========== =========== ============== =========== ===========
# Shyue
# --------------------------------------------------------------------------------------------------------------------------------
# left and right states :math:`x_M`    :math:`t_f` :math:`\rho_L`   :math:`u_L` :math:`p_L` :math:`\rho_R` :math:`u_R` :math:`p_R`
# ===================== ============== =========== ================ =========== =========== ============== =========== ===========
# .                     50             12          1.7              0           10          1              0           0.5
# --------------------- -------------- ----------- ---------------- ----------- ----------- -------------- ----------- -----------
#                                                                                                                         
# --------------------- -------------- ----------- ---------------- ----------- ----------- -------------- ----------- -----------
# JWL parameters        :math:`\rho_0` :math:`e_0` :math:`\Gamma_0` :math:`A`   :math:`B`   :math:`R_1`    :math:`R_2`  
# --------------------- -------------- ----------- ---------------- ----------- ----------- -------------- ----------- -----------  
# .                     1.84           0           0.25             8.545       0.205       4.6            1.35         
# ===================== ============== =========== ================ =========== =========== ============== =========== ===========  
# 
# 
# 
# ===================== ============== =========== ================ =========== =========== ============== =========== ===========
# Lee
# --------------------------------------------------------------------------------------------------------------------------------
# left and right states :math:`x_M`    :math:`t_f` :math:`\rho_L`   :math:`u_L` :math:`p_L` :math:`\rho_R` :math:`u_R` :math:`p_R`
# ===================== ============== =========== ================ =========== =========== ============== =========== ===========
# .                     50             20          0.9525           0           1           3.81           0           2
# --------------------- -------------- ----------- ---------------- ----------- ----------- -------------- ----------- -----------
#                                                                                                                           
# --------------------- -------------- ----------- ---------------- ----------- ----------- -------------- ----------- -----------
# JWL parameters        :math:`\rho_0` :math:`e_0` :math:`\Gamma_0` :math:`A`   :math:`B`   :math:`R_1`    :math:`R_2`     
# --------------------- -------------- ----------- ---------------- ----------- ----------- -------------- ----------- -----------
# .                     1.905          0           0.8938           632.1       -0.04472    11.3           1.13               
# ===================== ============== =========== ================ =========== =========== ============== =========== ===========
# 
# Plots of each solution are given below.
# These two problems, and their reversed states, provide the basis for many of the unittests for the general EOS solver.
# 
# +-----------------------+------------------------+
# | .. figure:: shyue.png | .. figure:: lee.png    |
# +-----------------------+------------------------+
"""

from .oblique_shocks import *
