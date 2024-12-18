r"""The 2-section steady-state 2D Riemann problem was originally described by Glaz and Wardlaw [Glaz1985]_.
Descriptions of the solution method were given by Loh and Hui [Loh1990]_, in Appendix A, as well as [Loh1994]_, in Section 4.
Other problem statements and results for comparison are given in the 1999 paper by Hui, Li and Li [Hui1999]_, and the 2012 monograph by Hui and Xu [Hui2012]_.
This problem assumes an imaginary membrane separating the top and bottom states, and a left boundary condition along which separate uniform state and flow conditions are imposed above and below the membrane.
The solver is currently setup to only handle an ideal-gas equation of state, but does allow for different values of the adiabatic index in the left and right states.

The same set of simple waves are produced in these solutions as in the 1D Riemann problem: shock waves, contact discontinuities, and rarefaction fans.
Shock waves differ from contact discontinuities in that shocks are discontinuous in density, velocity, and pressure, whereas contact discontinuities are continuous in velocity and pressure, but discontinuous in density.
In further agreement with the 1D Riemann solution, these solutions can be grouped as: shock-contact-shock ('SCS'), rarefaction-contact-rarefaction ('RCR'), rarefaction-contact-shock ('RCS'), and shock-contact-rarefaction ('SCR').
Solutions producing a vacuum between two rarefaction waves, so called rarefaction-contact-vacuum-contact-rarefaction ('RCVCR') solutions, are not yet considered in this solver and will be ignored for the time being.
The contact discontinuity lies along angle, :math:`\delta`, measured with respect to the horizontal axis, and separates the top and bottom star-states.
In 1D, pressure, :math:`p`, and velocity are both constant across the contact discontinuity; in 2D, pressure is still constant across the contact discontinuity, and the flow direction (or flow angle) must be aligned with the contact discontinuity angle, and is the same for both the top and bottom star-states, so that :math:`p_{\textrm{T}}^{\star} = p_{\textrm{B}}^{\star}` and :math:`\delta_{\textrm{T}}^{\star} = \delta_{\textrm{B}}^{\star}`, whereas both the x- and y-velocity components, :math:`u` and :math:`v`, may jump when crossing the contact discontinuity.
As such, both density, :math:`\rho`, and Mach number, :math:`M`, jump when crossing the contact discontinuity, so that :math:`\left( \rho, M \right)^{\star}_{\textrm{T}} \neq \left( \rho, M \right)^{\star}_{\textrm{B}}`, generally.

+------------------------------------+------------------------------------+
| .. figure:: schematic_figure1.png  | .. figure:: schematic_figure2.png  | 
+------------------------------------+------------------------------------+ 

The above schematic diagrams show an enlarged boundary condition on the left of each diagram (or what may be taken as a freeze region in some codes), and an analytic solution calculated in the right region for a specific RCS morphology.
All lines are common between the left and right diagrams.
The left diagram annotations denote the separate regions consisting of 1) the initial top and bottom states, 2) the calculated top and bottom star-states, and 3) the simple waves separating these regions, which are the shock wave, contact discontinuity, and rarefaction fan.
The right figure annotations denote 1) the variables used to define the top and bottom states, which are the pressure, :math:`p`, density, :math:`\rho`, Mach number, :math:`M`, and the flow angle, :math:`\vartheta`; the top and bottom state values are subscripted 'T' or 'B', respectively, and 2) the top and bottom star-state values, for which pressure and the flow angle are required to be equal across the contact discontinuity, :math:`p_{\textrm{T}}^{\star} = p_{\textrm{B}}^{\star}` and :math:`\delta_{\textrm{T}}^{\star} = \delta_{\textrm{B}}^{\star}`, whereas density and Mach number are required to be discontinuous across the contact discontinuity, :math:`\left( \rho, M \right)_{\textrm{T}}^{\star} \neq \left( \rho, M \right)_{\textrm{B}}^{\star}`.

We outline here the solution procedure given by Loh and Liou [Loh1994]_ in their Section 4.
Imagine the contact discontinuity as a rigid wall and it then may be viewed as separating two supersonic steady-state problems in two dimensions.
One problem is that of a determining the angle of a shock interface for planar flow over a rigid wall, which is the the contact discontinuity in this scenario.
The other problem is that of Prandtl-Meyer flow turning through a rarefaction fan to flow down a rigid wall, which again is the contact discontinuity.
Solutions of these problems are well understood.
However, both solutions assume the angle, :math:`\delta`, of the rigid wall is known, whereas the angle of the contact discontinuity is not prescribed in the original problem statement.
This results in a root finding exercise to determine :math:`\delta`.
Of course, the same is true for a SCS or RCR solution, as well.

The Prandtl-Meyer function represents the angle, in radians, through which supersonic flow turns to arrive at a given Mach number:

.. math::
   \nu(M) = \sqrt{\frac{\gamma + 1}{\gamma - 1}} \arctan \sqrt{\frac{\gamma - 1}{\gamma + 1} \left( M^2 - 1 \right)} - \arctan \sqrt{M^2 - 1} \, .

The flow deflection angle through an oblique shock is similarly well known,

.. math::
   \theta_i = \arctan \left[ \frac{\alpha_i - 1}{\gamma \, M_i^2 - \alpha_i + 1} \sqrt{\left( \frac{2 \, \gamma \, M_i^2}{\left( \gamma + 1 \right) \, \alpha_i + \gamma - 1} - 1 \right)} \right] \, ,

where :math:`\alpha_i = p / p_i` and :math:`i = [T, B]` for the top and bottom state and flow values, respectively.

These two functions may be combined and written together

.. math::
   \Phi_i(p) =
   \begin{cases}
     \vartheta_i + \theta_i          & p \geq p_i \\
     \vartheta_i + \nu(M_i) - \nu(M) & p <    p_i \, ,
   \end{cases}

where, as a reminder, :math:`\vartheta_i` is the flow direction in the top and bottom states captured in the boundary conditions.

+-------------------------------------+--------------------------------------------+
| .. figure:: solution_procedure.png  | .. figure:: solution_procedure-zoomed.png  | 
+-------------------------------------+--------------------------------------------+ 

Given the initial state-value for pressure, these equations allow the creation of a pressure-deflection diagram relating all possible compression and expansion states that are accessible to the initial pressure state, as done in the figure above.
Regions for which pressure increases relative to the initial state value represent shock compression, given by a solid colored line above, while expansion regions represent decreasing pressure states and are represented by colored dashed lines.
The horizontal colored dotted lines represent the initial pressure state values, above which the state undergoes compression, and below which the state undergoes expansion.
This pressure-deflection diagram is created for the top and bottom states, and the point where the curves overlap, :math:`\Phi_T(p^{\star}) = \Phi_B(p^{\star})`, represents the star-state solution for the pressure, :math:`p^{\star}`, and contact discontinuity angle, :math:`\delta^{\star}`.
Additionally, whether pressure has increased or decreased is obvious in the diagram (see the zoomed figure on the right), and so compression or expansion is easily ascertained for each region, which determines the solution morphology.

Given the solution for the star-state pressure, the top and bottom star-state values for density are determined via

.. math::
   \rho_i^{\star} = 
   \begin{cases}
     \rho_i \left[ \frac{\left( \gamma + 1 \right) \alpha_i^{\star} + \gamma - 1}{\left( \gamma - 1 \right) \alpha_i^{\star} + \gamma + 1} \right] & \alpha_i^{\star} \geq 1 \\
     \rho_i \, \left( \alpha_i^{\star} \right)^{1 / \gamma}        & \alpha_i^{\star} <    1 \, .
   \end{cases}

where :math:`\alpha_i^{\star} = p^{\star} / p_i`.
For an ideal-gas equation of state, the specific internal energy can now be determined, as well as the sound speed so that the x- and y-velocity components obtain, and thus the flow speed throughout the whole problem domain.

+----------------------------------------+-----------------------------------------+
| .. figure:: lineout.png                | .. figure:: pressure-cool_colormap.png  | 
+----------------------------------------+-----------------------------------------+ 
| .. figure:: density-cool_colormap.png  | .. figure:: sie-cool_colormap.png       | 
+----------------------------------------+-----------------------------------------+ 
| .. figure:: Mach-cool_colormap.png     | .. figure:: velx-cool_colormap.png      |
+----------------------------------------+-----------------------------------------+ 
| .. figure:: vely-cool_colormap.png     | .. figure:: speed-cool_colormap.png     |
+----------------------------------------+-----------------------------------------+ 

The figures above, from left to right, top to bottom, show a lineout of the kind that can be used for verification comparisons against code results, followed by the 2D streak plot solutions for pressure, density, specific internal energy, Mach number, the x-component of velocity, the y-component of velocity, and speed.
The top and bottom states for this problem are: bottom_state = [1, 1, 2.4, 0, 1.4]; top_state = [0.25, 0.5, 7, 0, 1.4], as given by Hui, Li and Li [Hui1999]_.
As a reminder, the state variables in order are [pressure, density, Mach number, flow angle in degrees, adiabatic index].
In the lineout, the left region is associated with the bottom state in the streak plots, and the right region with the top state.
Moving from left-to-right across the lineout shows how the variables respond traversing the rarefaction fan, the bottom star-state, jumping across the contact discontinuity to the the top star-state, and finally jumping across the shock to reach the top state.
In the streak plots, following a vertical line from bottom to top moves variable values across simple waves and through regions in the same order as following the lineout from left to right.
"""

from .ep_riemann2D_2section_steadystate import IGEOS_Solver
from .plot_utils import plot_data
