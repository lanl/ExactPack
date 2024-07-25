r"""The Noh Problem with Black Box Equation of State.

The Noh problem [Noh1987]_ is a self-similar adiabatic compression
wave in an ideal gas, and it can be formulated in spherical,
cylindrical, or planar geometry. The independent fluid variables are
(i) the mass density :math:`\rho(r,t)`, (ii) the velocity of the gas
:math:`u(r,t)`, and (iii) the pressure :math:`P(r,t)`, each at spatial
location :math:`r` and time :math:`t`. We denote the shock speed by :math:`D`. Note that in spherical coordinates,
:math:`u(r,t)` is the radial velocity of the gas, and a negative value 
indicates that gas is flowing in toward the origin. 

This solver rests on the fact that the solution to the Noh Problem is completely determined 
by the Rankine-Hugoinot conditions, also called the jump conditions. Indeed, if shocked density (:math:`\rho_L`), pressure (:math:`P_L`)
and shock speed (:math:`D`) values are known, then the complete solution for the Noh Problem is given by: 

.. math::
            \rho(t,x) = \begin{cases}
                \rho_L & \frac{x}{t}< D \\
            \rho_0 \left(1 - \frac{u_0 t}{x} \right)^m & \frac{x}{t} > D
            \end{cases} 

.. math::
            P(t,x) = \begin{cases}
                P_L & \frac{x}{t}< D \\
                P_0 & \frac{x}{t}> D 
            \end{cases} 

.. math::
            u(t,x) = \begin{cases}
                0 & \frac{x}{t}< D \\
                u_0 & \frac{x}{t}> D 
            \end{cases} 


The goal, therefore, is to find these values. The solver does so by solving the jump conditions given by: 

.. math:: 
         \rho_0 \left( 1 - \frac{u_0}{D}\right)^{m+1}  = \rho_L 

.. math::
        P_L = P_R  - \rho_L D u_0 

.. math::
        e_L  =  e_R + \frac{1}{2} u_0^2 - \frac{u_0 P_0}{\rho_L D}


The solver consists of a residual function based on the above equations, its Jacobian, and the Jacobian's inverse, and uses
a Newton Solver to find the root of the residual function.  

Within `solution_tools/residual_functions`, there are four residual functions. The first is `noh_residual`, the second is 
`simplified_noh_residual`, the third is `pressure_noh_residual`, and `simplified_pressure_noh_residual`. This warrants some explanation. 

The `noh_residual` is a :math:`\mathbb{R}^3 \to \mathbb{R}^3` function that solves for the shocked density, pressure, and shock 
speed values. This is the work-horse function: it is meant to solve the Noh problem in any geomety (1,2,3) with any initial conditions 
and any equation of state (assuming that they are theoretically admissible for the Noh problem; see [Ramsey17]_ for restrictions). Note that 
this class solves the jumps conditions by closing the system using :math:`e=e(\rho, P)`. 

The `simplified_noh_residual`, by constrast, is a :math:`\mathbb{R}^2 \to \mathbb{R}^2` function that solves for the shocked density and pressure. 
That is, it does not solve for the shock speed. However, this is possible because of a simplifying assumption: :math:`m=0` and :math:`P_0 = 0`. 
Therefore, `simplified_noh_residual` should only be used if the Noh problem is being posed in planar geometry and the initial pressure is zero. Note that 
this class solves the jumps conditions by closing the system using :math:`e=e(\rho, P)`. 

The 'pressure_noh_residual` is similar to `noh_residual` in structure and purpose. However, instead of closing the jump conditions with :math:`e = e(\rho, P)`, it does so instead 
with :math:`P = P(\rho, e)`, and so instead solves for the shocked density, energy, and shock speed. It has the same level of general-pupose intent as `noh_residual`. 

The 'simplified_pressure_noh_residual` is to `pressure_noh_residual` as `simplified_noh_residual` is to `noh_residual`. It uses the simplifying assumptions to 
reduce the jump conditions to two equations of three variables---it's just that instead of closing them with :math:`e = e(\rho, P)` (as in `simplified_noh_residual`), 
it closes them with :math:`P = P(\rho,e)`. 

We include all four for two reasons. First, we include the `pressure` versions because certain equation of state libraries do not provide a call for 
:math:`e = e(\rho, P)`. While it is possible to perform an energy inversion--given :math:`\rho, P` compute :math:`e` using :math:`P = P(\rho,e)`-- 
it can be a serious problems when dealing with complicated equations of state. The Newton iteration now has to find the energy for its computations in addition to its own iteration. 
The `pressure` versions avoid this problem by directly accessing :math:`P(\rho,e)` from 
the library. It is still necessary to compute initial energy, which would require an inversion, but it is only one inversion outside of the Newton iteration. 

We include the `simplified` versions for a simple reason: if the `noh_residual` or `pressure_noh_residual` is struggling to find a solution, then perhaps the `simplified_noh_residual` or `simplfied_pressure_noh_residual`
 will have better
luck since they only have to solve for two variables instead of three. Furthermore, since they are is a 2D systems, computing the determinants 
and inverse of the Jacobians can easily be done by hand (and has been done) and hardcoded. Thus, all computations done in `simplified_noh_residual` and `simplified_pressure_noh_residual` are done directly, 
whereas `noh_residual` and `pressure_noh_residual` involve a numerical inversion to find the Jacobian's inverse. (Of course, this is not difficult as it is a 3D system, but the point stands, especially
if many iterations are being done.) In short: users should default to using `noh_residual` or `pressure_noh_residual` and use `simplified_noh_residual` or `simplified_pressure_noh_residual` if they are encountering 
difficulties finding a solution. 

All residual functions require the initial conditions of the problem (fluid variables and the geometry of the problem) and an EoS object. 
This object must have a set of member functions. (Observe that it is precisely this location that makes the solver "Black Box": the EoS object does not need to 
be algebraic or analytic sense. For example, a wrapper that accesses tabulated values is an acceptable EoS object!) Furthermore, 
as with any good Newton Solver, the user can select the initial guess, convergence tolerance, and maximum number of iterations. 

The solver is found in `blackboxnoh.py`; it couples the Newton solver with the residual function to solve the jump conditions. 
Once that is done, it constructs the solution found above using the base classes. 

This problem is often cast in dimensionless variables in which the gas is moving
radially inward with uniform constant velocity :math:`u(r,0)=-1\, {\rm cm/s}`
and constant density :math:`\rho(r,0)=1\,{\rm g/cm^3}.` An infinitesimal time
after the the gas accumulates at the origin :math:`r=0`, an infinitely strong
stagnation shock forms and starts propagating outwards, leaving behind a region
of non-zero pressure and internal energy in its wake.  This problem exercises
the code's ability to transform kinetic energy into internal energy, and the
fidelity with which supersonic flows are tracked [Timmes2005]_ . Furthermore, it 
may construct semi-analytic solutions when arbitrary equations of state are used, 
allowing for tests not just of hydrodynamics, but also of EoS implementations. 

The geometry factor :math:`k=1, 2, 3` corresponds to planar,
cylindrically, and spherical geometries, respectively. To cast the equations in
terms of dimensionless coordinates, set :math:`u_0=1` and :math:`\rho_0 = 1`.

"""











from .blackboxnoh import NohBlackBoxEos, PlanarNohBlackBox, CylindricalNohBlackBox, SphericalNohBlackBox