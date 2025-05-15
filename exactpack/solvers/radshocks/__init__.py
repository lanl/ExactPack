r""" 
Semi-analytic, nonrelativistic, equilibrium-diffusion radiative-shock solutions were originally presented by Sen and Guess in 1957 [Guess1957]_, but they neglected the radiation energy density and the radiation pressure.
The nonequilibrium-diffusion solutions were originally presented by Heaslet and Baldwin in 1963 [Heaslet1963]_, but they also neglected the radiation energy density and the radiation pressure.
In 2007, Lowrie and Rauenzahn [Lowrie2007]_ revisited the equilibrium-diffusion problem and presented solutions in which the radiation energy density and radiation pressure are retained.
In 2008, Lowrie and Edwards [Lowrie2008]_ revisited the nonequilibrium-diffusion problem, and similarly presented solutions in which the radiation energy density and radiation pressure are retained.
Subsequently, Ferguson, Morel and Lowrie extended the nonequilibrium-diffusion solutions to incorporate :math:`\text{S}_{\text{n}}`-transport [Ferguson2017]_.
The solution method for the equilibrium-diffusion solver implemented in ExactPack is described in the paper by Lowrie and Rauenzahn [Lowrie2007]_.
The solution method for the nonequilibrium-diffusion solver and the :math:`\text{S}_{\text{n}}`-transport solver, as implemented in ExactPack, are described in the paper by Ferguson, Morel and Lowrie [Ferguson2017]_.
These solvers also incorporate an asymptotic parameter ensuring that they preserve the asymptotic equilibrium-diffusion limit [Ferguson2017a]_, which the user can vary.

The time-independent, 1D, planar, nondinemsionalized radiation-hydrodyanmic (RH) equations are:

.. math::
   \partial_x \left( \rho \, u \right) = 0 \, , \\

   \partial_x \left( \rho \, u^2 + p \right) = - P_0 \, S_{\text{rp}} \, , \\

   \partial_x \left[ u \left( \frac{1}{2} \, \rho \, u^2 + \rho \, e + p \right) \right] = - P_0 \, {\cal C}_0 \, S_{\text{re}} \, , \\

   S_{\text{rp}} = \partial_x {\cal P} = - \sigma_{\text{t}} \, {\cal F} + \beta \left( \sigma_{\text{t}} \, {\cal P} + \sigma_{\text{s}} \, {\cal E} + \sigma_{\text{a}} \, T^4 \right) \, , \\

   S_{\text{re}} = \partial_x {\cal F} = \sigma_{\text{a}} \left( T^4 - {\cal E} \right) + \beta \left( \sigma_{\text{a}} - \sigma_{\text{s}} \right) {\cal F} + 4 \left[ \beta^2 \left( \sigma_{\text{s}} - \sigma_{\text{a}} \right) {\cal P} \right]_{\text{eq}} \, , \\

   \mu \, \partial_x I = - \sigma_{\text{t}} \, I + \frac{\sigma_{\text{s}}}{4 \, \pi} {\cal E} + \frac{\sigma_{\text{a}}}{4 \, \pi} T^4 - 2 \frac{\sigma_{\text{s}}}{4 \, \pi} \beta \, {\cal F} + \beta \, \mu \left( \sigma_{\text{t}} \, I + \frac{3 \, \sigma_{\text{s}}}{4 \, \pi} {\cal E} + \frac{3 \, \sigma_{\text{a}}}{4 \, \pi} T^4 \right) + \frac{1}{\pi} \left[ \beta^2 \left( 2 \, \sigma_{\text{s}} - 3 \, \sigma_{\text{t}} \, \mu^2 \right) {\cal P} \right]_{\text{eq}} \, ,

which are accurate through :math:`{\cal O}(\beta \equiv u / c)` and contain an :math:`{\cal O}(\beta^2)` correction to the radiation-energy source, :math:`S_{\text{re}}`, and to the radiation-transport equation.
If the material and radiation are assumed to be out of equilibrium we couple the material internal energy to the radiation internal energy:

.. math::
   \rho \, u \, \partial_x e + p \, \partial_x u = - P_0 \, {\cal C}_0 \, S_{\text{rie}} \, , \\
   S_{\text{rie}} = S_{\text{re}} - \beta \, S_{\text{rp}} \, .

The total cross section is the sum of the absorption and scattering cross sections: :math:`\sigma_{\text{t}} = \sigma_{\text{a}} + \sigma_{\text{s}}`.
Additionally, it is assumed that both of these cross sections may depend on the material temperature or the density:

.. math::
   \sigma(T, \rho) = \sigma_o \, \rho^a \, T^b \, ,

such that, for a constant cross section, :math:`a = 0 = b`.

All of the equations above are used in the :math:`\text{S}_{\text{n}}`-transport solver.
For the nonequilibrium-diffusion solver the radiation-transport equation is dropped, and the Eddington approximation, :math:`{\cal P} = {\cal E} / 3`, is adopted.
For the equilibrium-diffusion solver the equation coupling the radiation and material internal energies is dropped, and the Eddington approximation is adopted, as is the equilibrium approximation, :math:`{\cal E} = T^4`.

.. figure:: radshock_drawing.png
   :alt: Generic radiative-shock solution showing the material and radiation temperatures, the direction of fluid flow, an embedded hydrodynamic shock, and the Zel'dovich spike, as well as the precursor and relaxation regions.
   :align: center
   :figwidth: 100%

   Generic radiative-shock solution showing the material and radiation temperatures, the direction of fluid flow, an embedded hydrodynamic shock, and the Zel'dovich spike, as well as the precursor and relaxation regions.

For an ideal-gas :math:`\gamma`-law EOS, the nondimensional material internal energy, pressure, and local Mach number can be written as:

.. math::
   e = \frac{T}{\gamma \left( \gamma - 1 \right)} \, , \\
   p = \frac{\rho \, T}{\gamma} \, , \\
   {\cal M} = \frac{{\cal M}_0}{\rho \, \sqrt{T}} \, .

The problem to be solved is the same, regardless of whether the radiation is described by equilibrium-diffusion, nonequilibrium-diffusion, or :math:`\text{S}_{\text{n}}`-transport: determine the structure of the time-independent, 1D, planar, radiative shock obeying the nondinemsionalized RH equations above.
The fluid is assumed to flow in the :math:`+x`-direction, while the shock is assumed to move in the :math:`-x`-direction.
The reference state with subscript-"0" refers to the pre-shock, upstream, equilibrium boundary condition, satisfied as :math:`x \rightarrow -\infty`, while the subscript-"1" refers to the post-shock, downstream, equilibrium boundary condition, satisfied as :math:`x \rightarrow +\infty`.
The nondimensional pre-shock equilibrium state is assumed to be known since reference dimensional values, e.g., :math:`\tilde{\rho}_0` and :math:`\tilde{T}_0`, are taken from this region, and we therefore choose to set :math:`\rho_0 = 1`, :math:`T_0 = 1`, :math:`{\cal E}_0 = 1`, and :math:`{\cal P}_0 = 1/3`.

The problem statement is:

* Assume: An ideal-gas :math:`\gamma`-law equation-of-state (EOS) such that :math:`p = \rho \, e \, (\gamma - 1)`, for a fluid obeying Eulerian hydrodynamics and interacting with radiation.

* Given: The values for :math:`\gamma`, :math:`{\cal M}_0`, :math:`\tilde{\rho}_0`, and :math:`\tilde{T}_0`, the functions :math:`\sigma_{\text{a}}(\rho, T)`, :math:`\sigma_{\text{s}}(\rho, T)`, and :math:`\sigma_{\text{t}}(\rho, T) = \sigma_{\text{a}}(\rho, T) + \sigma_{\text{s}}(\rho, T)`.  The reason for specifying the dimensional values of :math:`\tilde{\rho}_0` and :math:`\tilde{T}_0` is to obtain a value of :math:`P_0` that is consistent with the ideal-gas :math:`\gamma`-law EOS.

* Calculate: Values for the functions :math:`p(x)`, :math:`\rho(x)`, :math:`u(x)`, :math:`T(x)`, :math:`{\cal M}(x)`, :math:`{\cal E}(x)`, and :math:`{\cal F}(x)`.

After manipulating the ordinary differential equations (ODEs) above, and taking the spatial derivative of the local Mach number, the system can be reduced to two ODEs that are written as functions of the pair :math:`({\cal P}, {\cal M})` only:

.. math::
   \frac{d {\cal P}}{d x} = \frac{\sigma_{\text{t}} \, {\cal M}_0}{{\cal C} \, P_0} \left[ \frac{T - 1}{\gamma - 1} + \frac{{\cal M}_0^2}{2 \, \rho^2} \left( 1 - \rho^2 \right) + P_0 \left( \frac{\sigma_{\text{t}} \, {\cal P} + \sigma_{\text{s}} \, {\cal E} + \sigma_{\text{a}} \, T^4}{\rho \, \sigma_{\text{t}}} - \frac{4}{3} \right) \right] \, , \\
   \frac{d {\cal M}}{d x} = - \frac{P_0 \, {\cal M} \left( \gamma + 1 \right)}{2 \, \rho \, T \left( {\cal M}^2 - 1 \right)} \left[ \frac{d {\cal P}}{d x} - \frac{\left( \gamma - 1 \right) \left( \gamma \, {\cal M}^2 + 1 \right)}{\left( \gamma + 1 \right) {\cal M}_0} {\cal C} \, \rho \, S_{\text{rie}} \right] \, .

These two ODEs do not need to be modified to obtain the nonequilibrium-diffusion solution, nor the :math:`\text{S}_{\text{n}}`-transport solution.
For the equilibrium-diffusion solutions, the second ODE is unnecessary, and the Eddington approximation and the equilibrium approximation are applied to the first ODE, which is inverted and integrated over the temperature domain, :math:`[T_0, T_1]`.

It is an assumption of the nonequilibrium-diffusion solver and the :math:`\text{S}_{\text{n}}`-transport solver, as currently written, that the spatial derivative of the local Mach number is a monotonic function and therefore invertible.
The reason for inverting this equation is to integrate over Mach-space since in the precursor region the local Mach number is strictly greater than one, and in the relaxation region the local Mach number is strictly less than one.
Therefore, the two ODEs to be integrated are:

.. math::
   \frac{d x}{d {\cal M}} = \frac{d x}{d {\cal M}} \left( {\cal P}, {\cal M} \right) = \left( \frac{d {\cal M}}{d x} \right)^{-1} \, , \\
   \frac{d {\cal P}}{d {\cal M}} = \frac{d {\cal P}}{d {\cal M}} \left( {\cal P}, {\cal M} \right) = \frac{d {\cal P}}{d x} \frac{d x}{d {\cal M}} \, .
"""

from .nED_radshocks import ED_Solver, nED_Solver, Sn_Solver, ie_Solver
