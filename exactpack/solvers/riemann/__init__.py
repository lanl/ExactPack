r"""The Riemann problem.

This problem exercises the one-dimensional Cartesian Riemann solver for
an inviscid, non-heat conducting polytropic gas. The independent fluid variables
are (i) the mass density :math:`\rho(r,t)`, (ii) the velocity of the gas
:math:`u(r,t)`, and (iii) the pressure :math:`P(r,t)`, each at spatial location
:math:`r` and time :math:`t`. The specific internal energy :math:`e(r,t)` is
related to the other fluid variables by the equation of state (EOS)
for an ideal gas at constant entropy,

.. math::
  P = (\gamma-1) \rho e  \ ,

where :math:`\gamma \equiv c_p/c_v` is the ratio of specific heats at
constant pressure and volume. The Euler equations take the form

.. math::

  \frac{\partial \rho}{\partial t} + u \frac{\partial \rho}{\partial r}
  +
  \frac{\rho}{r^{k-1}} \frac{\partial}{\partial r}\Big( u r^{k-1} \Big)
  &= 0
  \\
  \frac{\partial u}{\partial t} + u\, \frac{\partial u}{\partial r}
  +
  \frac{1}{\rho}\frac{\partial P}{\partial r} &= 0
  \\
  \frac{\partial }{\partial t} \Big( P \rho^{-\gamma} \Big)
  + u\, \frac{\partial }{\partial r} \Big( P \rho^{-\gamma} \Big)  &= 0 \ ,

where :math:`k=1,2,3` for planar, cylindrical, and spherical coordinates
respectively.

The gas starts with constant values of density, velocity, and pressure in
two contiguous regions, the boundary between which is located at :math:`x=x_0`:

.. math::
   x < x_0: &
   \\
   \rho(x,0) &= \rho_{\scriptscriptstyle L}
   \\
   u(x,0) &= u_{\scriptscriptstyle L}
   \\
   P(x,0) &= P_{\scriptscriptstyle L} \ ,
   :label: riemannL

and   
   
.. math::
   x > x_0: &
   \\
   \rho(x,0) &= \rho_{\scriptscriptstyle R}
   \\
   u(x,0) &= u_{\scriptscriptstyle R}
   \\
   P(x,0) &= P_{\scriptscriptstyle R} \ .
   :label: riemannR

We can imagine a membrane at location :math:`r_0` separating the left and right
regions. At :math:`t=0` we remove the membrane, allowing the left and right regions
to interact. The resulting wave is a combination of shock waves, rarefactions,
and contact discontinuities. Shock waves differ from contact discontinuities in
that shocks are discontinuous in density, velocity, and pressure, while contacts
are continuous in velocity, but discontinuous in density and pressure. 

We shall choose six variants of the Riemann problem to test code algorithms
in various physics regimes, as defined in the Tables below. The solution can be
computed at any time :math:`t_{\rm fin}` with adiabatic coefficient :math:`\gamma`.

=========== =================== =================== =================== ==============
Test Number Problem Name        :math:`r_0`         :math:`t_{\rm fin}` :math:`\gamma`
=========== =================== =================== =================== ==============
1           Sod                 0.5                 0.25                7/5           
2           Einfeldt            0.5                 0.15                7/5           
3           Stationary-Contact  0.8                 0.012               7/5           
4           Slow-Shock          0.5                 1.0                 7/5           
5           Shock-Contact-Shock 0.5                 0.3                 7/5           
6           LeBlanc             0.3                 0.5                 5/3           
=========== =================== =================== =================== ==============

For each test case, we choose the initial conditions on either side of the membrane
at :math:`r_0` to be

==== ============== =========== ============================ =============== =========== ============================
Test :math:`\rho_L` :math:`u_L` :math:`P_L`                  :math:`\rho_R`  :math:`u_R` :math:`P_R`                 
==== ============== =========== ============================ =============== =========== ============================
1    1.0            0.0         1.0                          0.125           0.0         0.1
2    1.0            -2.0        0.4                          1.0             2.0         0.4
3    1.0            -19.59745   :math:`10^3`                 1.0             -19.59745   :math:`10^{-2}`
4    3.857143       -0.810631    10.33333                    1.0             -3.44       1.0
5    1.0            0.5          1.0                         1.25            -0.5        1.0
6    1.0            0.0          :math:`(2/3)\times 10^{-1}` :math:`10^{-2}` 0.0         :math:`(2/3)\times 10^{-10}`
==== ============== =========== ============================ =============== =========== ============================

These problems are implemented in the following helper classes.
   
"""

from .kamm import Riemann


class Sod(Riemann):  # test1
    """Test1: This is the canonical Sod shock tube with a
    rarefaction-contact-shock structure. While not a challenging problem,
    it quickly identifies algorithmic problems resolving basic wave structure.
    """

    parameters = {'gammal': Riemann.parameters['gammal'],
                  'gammar': Riemann.parameters['gammar'],
                  'interface_loc': Riemann.parameters['interface_loc'],
                  'rhol': Riemann.parameters['rhol'],
                  'pl': Riemann.parameters['pl'],
                  'ul': Riemann.parameters['ul'],
                  'rhor': Riemann.parameters['rhor'],
                  'pr': Riemann.parameters['pr'],
                  'ur': Riemann.parameters['ur']
    }

    gammal = 1.4
    gammar = 1.4
    interface_loc = 0.5
    rhol = 1.0
    pl = 1.0
    ul = 0.0
    rhor = 0.125
    pr = 0.1
    ur = 0.0


class Einfeldt(Riemann):  # test2
    """Test2: This is the Einfeldt (or 1-2-3) problem, and
    consists of two strong rarefaction waves, with a near-vacuum between
    them. Computational methods that conserve total energy might show
    internal energy errors for this problem.
    
    """

    parameters = {'gammal': Riemann.parameters['gammal'],
                  'gammar': Riemann.parameters['gammar'],
                  'interface_loc': Riemann.parameters['interface_loc'],
                  'rhol': Riemann.parameters['rhol'],
                  'pl': Riemann.parameters['pl'],
                  'ul': Riemann.parameters['ul'],
                  'rhor': Riemann.parameters['rhor'],
                  'pr': Riemann.parameters['pr'],
                  'ur': Riemann.parameters['ur']
    }

    gammal = 1.4
    gammar = 1.4
    interface_loc = 0.5
    rhol = 1.0
    pl = 0.4
    ul = -2.0
    rhor = 1.0
    pr = 0.4
    ur = 2.0


class StationaryContact(Riemann):  # test3
    """Test3: This is the stationary contact problem, and
    consists of a strong shock wave moving to the right, a stationary
    contact, and a strong rarefaction moving to the left. This problem
    is based on the left part of the well-known Woodward-Colella problem,
    but with the velocity shifted to make the contact stationary. This
    problem tests an algorithm's dissipation by how much the contact is
    smeared.
    
    """

    parameters = {'gammal': Riemann.parameters['gammal'],
                  'gammar': Riemann.parameters['gammar'],
                  'interface_loc': Riemann.parameters['interface_loc'],
                  'rhol': Riemann.parameters['rhol'],
                  'pl': Riemann.parameters['pl'],
                  'ul': Riemann.parameters['ul'],
                  'rhor': Riemann.parameters['rhor'],
                  'pr': Riemann.parameters['pr'],
                  'ur': Riemann.parameters['ur']
    }

    gammal = 1.4
    gammar = 1.4
    interface_loc = 0.8
    rhol = 1.
    pl = 1000.
    ul = -19.59745
    rhor = 1.0
    pr = 0.01
    ur = -19.59745


class SlowShock(Riemann):  # test4
    """Test4: This is the slow shock problem, and consists of
    a Mach 3 shock wave moving slowly to the right. Some numerical methods
    exhibit unphysical oscillations behind the shock.
    
    """

    parameters = {'gammal': Riemann.parameters['gammal'],
                  'gammar': Riemann.parameters['gammar'],
                  'interface_loc': Riemann.parameters['interface_loc'],
                  'rhol': Riemann.parameters['rhol'],
                  'pl': Riemann.parameters['pl'],
                  'ul': Riemann.parameters['ul'],
                  'rhor': Riemann.parameters['rhor'],
                  'pr': Riemann.parameters['pr'],
                  'ur': Riemann.parameters['ur']
    }

    gammal = 1.4
    gammar = 1.4
    interface_loc = 0.5
    rhol = 3.857143
    pl = 10.33333
    ul = -0.810631
    rhor = 1.0
    pr = 1.0
    ur = -3.44


class ShockContactShock(Riemann):  # test5
    """Test5: This is the shock-contact-shock problem. When
    two shocks separate from the initial state, with a contact between
    them, errors are produced in all fields, and this problem tests how
    well an algorithm handles those errors. This problem is similar to
    the planar Noh problem but with weaker shocks.
    
    """

    parameters = {'gammal': Riemann.parameters['gammal'],
                  'gammar': Riemann.parameters['gammar'],
                  'interface_loc': Riemann.parameters['interface_loc'],
                  'rhol': Riemann.parameters['rhol'],
                  'pl': Riemann.parameters['pl'],
                  'ul': Riemann.parameters['ul'],
                  'rhor': Riemann.parameters['rhor'],
                  'pr': Riemann.parameters['pr'],
                  'ur': Riemann.parameters['ur']
    }

    gammal = 1.4
    gammar = 1.4
    interface_loc = 0.5
    rhol = 1.0
    pl = 1.0
    ul = 0.5
    rhor = 1.25
    pr = 1.0
    ur = -0.5


class LeBlanc(Riemann):  # test6
    """Test6: This is the LeBlanc problem, which is a strong
    shock, strong rarefaction version of the basic rarefaction-contact-shock
    problem. This is a good test of a method's robustness.

    """

    parameters = {'gammal': Riemann.parameters['gammal'],
                  'gammar': Riemann.parameters['gammar'],
                  'interface_loc': Riemann.parameters['interface_loc'],
                  'rhol': Riemann.parameters['rhol'],
                  'pl': Riemann.parameters['pl'],
                  'ul': Riemann.parameters['ul'],
                  'rhor': Riemann.parameters['rhor'],
                  'pr': Riemann.parameters['pr'],
                  'ur': Riemann.parameters['ur']
    }

    gammal = 5.0 / 3.0
    gammar = 5.0 / 3.0
    interface_loc = 0.3
    rhol = 1.0
    pl = 0.666667e-1
    ul = 0.0
    rhor = 1.e-2
    pr = 0.666667e-10
    ur = 0.0
