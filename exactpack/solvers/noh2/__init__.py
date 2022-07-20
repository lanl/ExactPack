r"""The Noh2 Problem.

========================================
The Uniform Collapse problem, a.k.a Noh2
========================================

Problem statement
-----------------

The "uniform collapse" problem, from Noh's JCP (1987), page 92. The fluid is
initially distributed uniformly with initial density, internal
energy, and velocity prescribed as:

..  math::
    :label: nohic

    \rho(r,0) &= \rho_0\\
    e(r,0) &= e_0\\
    u(r,0) &= -r \\

The EOS gives the temperature profile in terms of the kinematic variables.    
    

Problem solution
----------------

We note that this problem is identical to a special case of Coggeshall #1 (1991),
where the general solution is:

..  math::
    :label: cogsol

    \rho(r,t) &= \rho_0 \, r^b t^{-b-k-1}\\
    u(r,t) &= r/t\\
    T(r,t) &= T_0 \, r^{-b}t^{b-(\gamma-1)(k+1)} \\

where :math:`b, k, \rho_0, T_0` are free parameters, and :math:`\gamma` is the
adiabatic constant for the gas. 

Let us multiply the temperature equation by the ideal gas constant :math:`R` 
to get specific internal energy :math:`e=RT`, choose the values :math:`b=0` and
:math:`k=\delta-1` (where :math:`\delta` is a geometry factor, equal to 1 for
slab, 2 for cylindrical, 3 for spherical), make a variable substitution in time
of :math:`t = 1-\tau`, and make a variable substitution in space of :math:`R=-r`.

Now Eq. :eq:`cogsol` reduces to:

..  math::
    :label: solRTau

    \rho(R,\tau) &= \rho_0 (1-\tau)^{-\delta}\\
    u(R,\tau) &= -R/(1-\tau)\\
    e(R,\tau) &= e_0\, (1-\tau)^{-(\gamma-1)\delta} \\

For the sake of simplicity we substitute :math:`t` for :math:`\tau` and :math:`r`
for :math:`R`.  Compute
pressure using the polytropic EOS :math:`p=\rho e (\gamma-1)`. We also drop the 
function call to :math:`r` on those variables that are independent of it. 
We now have the complete solution to the Uniform Collapse problem:

..  math::
    :label: nohsol

    \rho(t) &= \rho_0 \,  (1-t)^{-\delta}\\
    u(r,t) &= -r(1-t)^{-1}\\    
    e(t) &= e_0 \, (1-t)^{-\delta(\gamma-1)} \\
    p(t) &= (\gamma-1)\rho_0 e_0 \, (1-t)^{-\delta\gamma} \\

It should be noted that the thermodynamic quantities :math:`\rho, p, e` are
uniform over the problem domain. 
Also, the time domain of the problem is :math:`0\leq t < 1`. Note that at :math:`t=1`
the solution becomes singular.

Conservation Equations
----------------------

We will now endeavor to demonstrate that the above equations satisfy the 
conservation equations.

Euler equations, from Fickett & Davis, page 124

Conservation of mass:
*********************

..  math::
    :label: mass
    
    \dot{\rho} + \rho \frac{\partial{u}}{\partial{r}} = -\frac{\alpha \rho u}{r}

    
where :math:`\alpha=0, 1, 2` for slab, cylinder, sphere

First we note that :math:`\alpha = \delta -1`. From Eq. :eq:`nohsol` we get:

..  math::

    \dot{\rho} = \delta \rho_0 (1-t)^{-\delta-1}

    \frac{\partial{u}}{\partial{R}} = -(1-t)^{-1}

thus, the left hand side of Eq. :eq:`mass` becomes:

..  math::

     \delta \rho_0 (1-t)^{-\delta-1} + \rho_0(1-t)^{-\delta}(-1)/(1-t)\\

     = \delta \rho_0 (1-t)^{-\delta-1} - \rho_0(1-t)^{-\delta-1}\\

     = \rho_0 (1-t)^{-\delta-1} (\delta - 1)\\

     = \rho_0 (\delta-1) (1-t)^{-\delta-1}


and the right hand side of Eq. :eq:`mass` becomes

..  math::

        - \frac{(\delta-1)\rho_0(1-t)^{-\delta} (-r)/(1-t)}{r} \\
          = \frac{(\delta-1)\rho_0(1-t)^{-\delta} }{(1-t)}\\

          = \rho_0 (\delta-1) (1-t)^{-\delta-1}


QED: Mass is conserved.

Conservation of momentum:
*************************

..  math::
    :label: mom
    
    \dot{u} + \frac{1}{\rho}\frac{\partial{p}}{\partial{r}} = 0
    

First we compute :math:`\dot{u}`:

..  math::

    \dot{u} = \frac{\partial{u}}{\partial{t}} + 
        \frac{\partial{u}}{\partial{r}} \dot{r}\\


    \frac{\partial{u}}{\partial{t}} &= -r(1-t)^{-2}\\
    \frac{\partial{u}}{\partial{r}} &= -(1-t)^{-1}\\
    \dot{r} &= u

and thus

..  math::

    \dot{u} &=  -r(1-t)^{-2} - (1-t)^{-1}(-r)(1-t)^{-1}\\
            &=  -r(1-t)^{-2} + r(1-t)^{-2}\\
            &= 0


We note that :math:`p` is not a function of :math:`r`, and therefore
:math:`\frac{\partial{p}}{\partial{r}}=0`.  Thus, momentum is conserved.

Conservation of energy:
***********************

..  math::
    :label: ener

    \dot{e} + p\dot{v}= 0

where :math:`v = 1/\rho` (specific volume).
First we compute :math:`\dot{e}` and :math:`\dot{v}`:

..  math::

    \dot{e} = \delta(\gamma-1)e_0(1-t)^{-\delta(\gamma-1)-1}

    \dot{v} = \frac{d(1/\rho)}{dt} = 1/\rho_0 \delta(1-t)^{\delta-1}*(-1)
        =   -\delta/\rho_0 (1-t)^{\delta-1}


Substituting into the left hand side of Eq. :eq:`ener` yields:

..  math::

    \delta(\gamma-1)e_0(1-t)^{-\delta(\gamma-1)-1} + 
        [(\gamma-1)\rho_0 e_0 (1-t)^{-\delta\gamma}]*
        [-\delta/\rho_0 (1-t)^{\delta-1}]

which reduces to

..  math::

    \delta(\gamma-1)e_0(1-t)^{-\delta(\gamma-1)-1} -
        \delta(\gamma-1)e_0 (1-t)^{-\delta\gamma+\delta-1}

    =0


QED: Energy is conserved.


Derivation of solution from equations of motion
-----------------------------------------------

Assume that the entire fluid collapses to the origin in one time step, with
velocity proportional to radius. The velocity solution thus has the 
form :math:`u=-r(1-t)^{-1}`.

Start with conservation of mass:

..  math::
    :label: mass2
    
    \dot{\rho} + \rho \frac{\partial{u}}{\partial{r}} = -\frac{\alpha \rho u}{r}

Using the assumed velocity solution, and also :math:`\alpha=\delta-1`, the components
of Eq. :eq:`mass2` can be computed as:

..  math::
    :label: mass2comp

    \frac{\partial{u}}{\partial{r}} = -(1-t)^{-1}

     -\frac{\alpha \rho u}{r} = (\delta-1)(1-t)^{-1} \rho

Substituting Eq. :eq:`mass2comp` into Eq. :eq:`mass2` yields:

..  math::
    :label: mass2sub

    \dot{\rho} - (1-t)^{-1} \rho   = (\delta-1)(1-t)^{-1} \rho

    \dot{\rho} = \delta (1-t)^{-1} \rho

    \frac{d\rho}{\rho} = \frac{\delta dt}{(1-t)}

    \ln \rho = -\delta \ln(1-t) + C_1

    \rho(t) = C_2 (1-t)^{-\delta}

Applying the inital condition :math:`\rho(0) = \rho_0` yields

..  math::
    :label: mass2sub2

    \rho_0 = C_2

Thus the solution for density is:

..  math::
    :label: rhosol

    \rho(t) = \rho_0 (1-t)^{-\delta}


Now, solve conservation of energy for the specific internal energy, :math:`e`:
    
..  math::
    :label: ener2

    \dot{e} + p\dot{v}= 0

Compute the components of Eq. :eq:`ener2`:

..  math::
    :label: ener2comp

    p = (\gamma-1) \rho e = (\gamma-1) \rho_0 (1-t)^{-\delta} e

    v = \rho^{-1} = \rho_0^{-1}(1-t)^{\delta}

    \dot{v} = -\rho_0^{-1} \delta (1-t)^{\delta-1} 

Substituting :eq:`ener2comp` into :eq:`ener2` yields:

..  math::
    :label: ener2sub

    \dot{e} + [(\gamma-1) \rho_0 (1-t)^{-\delta} e] * [-\rho_0^{-1} \delta (1-t)^{\delta-1}] = 0

    \dot{e} = (\gamma-1) \delta (1-t)^{-1} e

    \frac{de}{e} = \frac{\delta(\gamma-1)dt}{(1-t)}

    \ln e = -\delta(\gamma-1) \ln (1-t) + C_3

    e(t) = C_4 (1-t)^{-\delta(\gamma-1)}

Applying the inital condition :math:`e(0) = e_0` yields

..  math::

    e_0 = C_4

Thus the solution for specific internal energy is:

..  math::

    e(t) = e_0 (1-t)^{-\delta(\gamma-1)}


And thus by the EOS, pressure is:

..  math::

    p &= (\gamma-1) \rho e\\
     &= (\gamma-1) \rho_0 (1-t)^{-\delta} e_0 (1-t)^{-\delta(\gamma-1)}\\
    &= (\gamma-1) \rho_0  e_0 (1-t)^{-\delta \gamma}


We now have the complete solution to the Uniform Collapse problem:

..  math::

    \rho(t) &= \rho_0 (1-t)^{-\delta}\\
    p(t) &= (\gamma-1)\rho_0 e_0 (1-t)^{-\delta\gamma} \\
    e(t) &= e_0(1-t)^{-\delta(\gamma-1)} \\
    u(r,t) &= -r(1-t)^{-1}\\

We have already proven that this solution satisfies conservation of momentum.


By default, :py:mod:`exactpack.solvers.noh2` loads
:py:mod:`exactpack.solvers.noh2.noh2`.
"""

from .noh2 import Noh2, PlanarNoh2, CylindricalNoh2, SphericalNoh2

