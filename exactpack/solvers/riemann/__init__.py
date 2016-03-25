r"""The independent fluid variables are (i) the gas density
:math:`\rho(r,t)`, (ii) the velocity of the gas :math:`u(r,t)`, and
(iii) the pressure :math:`P(r,t)`, each at spatial location :math:`r`
and time :math:`t`. The specific internal energy :math:`e(r,t)` is
related to the other fluid variables by the equation of state (EOS)
for an ideal gas at constant entropy,

.. math::
  P = (\gamma-1) \rho e  \ ,

where :math:`\gamma \equiv c_p/c_v` is the ratio of specific heats at
constant pressure and volume. For a mono-atomic ideal gas we have
:math:`\gamma=5/3`. The Euler equations take the form

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
  + u\, \frac{\partial }{\partial r} \Big( P \rho^{-\gamma} \Big)  &= 0

where :math:`k=1,2,3` for planar, cylindrical, and spherical coordinates
respectively.

The gas starts out with constant values of density, velocity, and
pressure in two contiguous regions delimited by a distance
:math:`r_0`:

.. math::
   r < r_0: &
   \\
   \rho(r,0) &= \rho_{\scriptscriptstyle L}
   \\
   u(r,0) &= u_{\scriptscriptstyle L}
   \\
   P(r,0) &= P_{\scriptscriptstyle L}
   :label: riemannL

.. math::
   r > r_0: &
   \\
   \rho(r,0) &= \rho_{\scriptscriptstyle R}
   \\
   u(r,0) &= u_{\scriptscriptstyle R}
   \\
   P(r,0) &= P_{\scriptscriptstyle R} \ .
   :label: riemannR
"""

from kamm import Riemann


class Sod(Riemann):  # test1
    """Test1 for Riemann.
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
    """Test2 for Riemann.
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
    """Test3 for Riemann.
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
    """Test4 for Riemann.
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
    """Test5 for Riemann.
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
    """Test6 for Riemann.
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
