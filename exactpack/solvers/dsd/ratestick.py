r"""A DSD Rate Stick solver in Python.

This is a pure Python implementation of the Rate Stick solution using Numpy.

The DSD Rate Stick problem is used to test how well a level set solver
applies boundary conditions to an HE burn time table calculation, as the
solution to this problem is dominated by the applied boundary conditions.
In this problem, the boundaries of the explosive align with the DSD
computational mesh.

In a cylindrical configuration, a rate stick consists of a right circular
cylinder of high explosive (HE) confined by a tube of inert material. A
solution can also be defined for a planar inert-HE-inert sandwich
configuration.

The cylindrical (**geometry** :math:`=2`) configuration is defined by the
radius of the HE cylinder, :math:`R`, and the edge angle, :math:`\omega_c`,
between the HE and the tube material. The system is assumed to be centered
on the :math:`z`-axis, so that :math:`0 \le r \le R`. In the following
equations, :math:`x` is used to represent the :math:`r` direction and
:math:`y` is used to represent the :math:`z` direction.

The planar (**geometry** :math:`=1`) sandwich configuration is assumed to be
centered on the :math:`y`-axis. Thus, the problem is defined by the half
thickness of the HE slab, :math:`R`, and the edge angle, :math:`\omega_c`,
between the HE and the inert material on either side. It is assumed that the
same material is used on both sides of the HE. While the HE exists in the
region :math:`-R \le x \le R`, only the right half of this domain will be
modeled (:math:`0 \le x \le R`).

At time :math:`t_d = 0.0`, the shape of the HE burn front shock wave is
prescribed. The shape will be determined by the choice of initial condition,
as described below. Any HE behind the initial burn front is assumed to ignite
at time :math:`t_d`, as if it were part of the detonation system.

The detonation trajectory for the above conditions is given by

.. math:: y = f(x,t)

where the function :math:`f` satisfies the PDE

.. math:: f_t = D_n \sqrt{1+(f_x)^2}

where :math:`D_n` is the velocity of the HE detonation wave in the
shock-normal direction, which is described by a deviation from the nominal
constant Chapman-Jouguet detonation shock speed :math:`D_{CJ}` of the HE.

In [Bdzil]_, the deviation in :math:`D_n` is linear with respect to the
curvature, :math:`\kappa`, of the detonation shock front:

.. math:: D_n = D_{CJ} - \alpha \kappa

With this form of :math:`D_n` and the geometrically appropriate curvatures,
the function :math:`f` satisfies the PDE

.. math::
    f_t - D_{CJ} \sqrt{1+(f_x)^2} = \alpha \frac{f_{xx}}{1+(f_x)^2} +
    \alpha n \frac{f_x}{x}

where :math:`n =` **geometry** :math:`- 1`. As currently implemented,
:math:`\kappa` will be calculated separately, then :math:`D_n` will be
calculated from this linear form. The function :math:`f` will then be
calculated from the more general form above. In this way, any other form
for :math:`D_n` could be added at a later date.

In the cylindrical case, the boundary conditions to be applied are

.. math::
    f_x (0,t) = 0

    f_x (R,t) = \cot (\omega_c)

In the planar case, the nominal boundary conditions are

.. math::
    f_x (-R,t) = \cot (\omega_c)

    f_x (R,t) = - \cot (\omega_c)

Because only the right half of the planar case is modeled, the actual boundary
conditions to be applied are the same as the cylindrical case.

Three initial conditions have been implemented in the solver. In all
three cases, the outer edge (:math:`x = R`) of the burn front shock wave is
assumed to be located at :math:`y = 0` at time :math:`t_d`. The desired
initial condition can be indicated through the :math:`IC` parameter.

* Case :math:`IC = 1`

  The HE is initiated on a circular detonation arc with radius :math:`r_d`,
  which is centered on the :math:`y`-axis. The center of the detonation is
  located at :math:`(0,y_d)`, where

  .. math:: y_d = - \sqrt{{r_d}^2 - R^2}

  The location of the burn front at :math:`t_d = 0.0`, which provides the
  initial condition for the PDE, is thus found to be:

  .. math:: f(x,0) = y_d + \sqrt{{r_d}^2 - x^2}

  This case represents the physical situation that occurs when an expanding
  spherical detonation front reaches the bottom of the containment for the
  HE rate stick.

  The initial angle between the normal to the shock wave and the normal to
  the confinement boundary cannot exceed the applied edge angle. If this
  condition is violated, application of the edge angle results in a negative
  curvature, which is not allowed in level set theory. Thus, the radius of the
  detonation front, :math:`r_d`, and the radius/thickness of the HE, :math:`R`,
  must satisfy the condition

  .. math:: r_d >= \frac{R}{\omega_c}.

  Note: This initial condition is slightly different than as described in
  [Bdzil]_. The location of the front of the detonation (therefore, the center
  of the detonation) has been moved by 1 unit in the negative
  :math:`y`-direction.

* Case :math:`IC = 2`

  A circular arc for detonation initiation that is compatible with the
  boundary conditions above has a detonation radius of

  .. math:: r_d = \frac{R}{\cos (\omega_c)}

  which clearly satisfies the initial angle condition stated in the previous
  case. The center of the detonation is then located at :math:`(0,y_d)`, where

  .. math:: y_d = - \sqrt{{r_d}^2 - R^2}

  Any value input for :math:`r_d` will be overridden by the calculated
  value. Using this calculated value, the initial condition for the PDE
  is given by

  .. math:: f(x,0) = y_d + \sqrt{{r_d}^2 - x^2}


* Case :math:`IC = 3`

  Under experimental conditions, the HE can also be initiated by a planar
  detonation wave, which has an initial condition

  .. math:: f(x,0) = 0.0

  This condition is equivalent to an infinite detonation radius and satisfies
  the initial angle condition stated in the first case. Any value input for
  :math:`r_d` will be ignored in the calculation.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class RateStick(ExactSolver):
    r"""Computes the numerical solution to the Rate Stick Problem.

    For the planar case (**geometry** :math:`= 1`), the HE slab is assumed
    to be centered on the :math:`y`-axis. The right half is modeled in the
    :math:`xy`-plane. For the cylindrical case (**geometry** :math:`= 2`),
    the HE cylinder is assumed to be centered on the :math:`z`-axis. It is
    modeled in the :math:`rz`-plane. Note that the [Bdzil]_ paper uses
    :math:`n` to define the coordinate system, where :math:`n =`
    **geometry** :math:`- 1`. The edge angle, :math:`\omega_c` is assumed
    to satisfy :math:`0 < \omega_c < \frac{\pi}{2}`.

    The nominal detonation velocity of the HE, :math:`D_{CJ}`, must
    be positive. The linear coefficient, :math:`\alpha`, of detonation
    velocity deviance must also be positive.

    Detonation time is assumed to be :math:`t_d = 0.0`. The burn front
    shock wave is assumed to have just reached the confinement material at
    detonation time, with the shock wave located at :math:`y=0` at the outer
    edge of the HE (:math:`x = R`) before application of the boundary
    condition.

    The initial angle between the normal to the shock wave and the normal
    to the confinement boundary is assumed to be no more than the applied
    edge angle. Thus, if :math:`IC = 1`, the radius of the detonation front,
    :math:`r_d`, and the radius/thickness of the HE, :math:`R`, must satisfy
    the condition :math:`r_d >= \frac{R}{\cos (\omega_c)}`. This condition is
    automatically satisfied by the other two initial conditions.

    Default values are selected to reflect the description in [Bdzil]_, with
    the caveat listed in the first initial condition above. Default values
    are **geometry** :math:`= 1`, :math:`R = 1.0`,
    :math:`\omega_c = \frac{\pi}{4}`, :math:`D_{CJ} = 1.0`,
    :math:`\alpha = 0.1`, :math:`IC = 1` and :math:`r_d = \sqrt{626}`,
    which produces a detonator location of :math:`(0.0,-25.0)`.

    The solution for :math:`y` of the burn front is calculated on a fine
    :math:`(x, t)` mesh to obtain :math:`y = y(x, t)`. The solution is then
    interpolated to the requested :math:`xy` mesh and inverted to obtain
    the burn time as :math:`t = t(x, y)`. The requested mesh must have a
    larger mesh spacing than the mesh used for the calculation. For efficiency
    in inverting the solution, the user must input the number of nodes in
    the :math:`x`-direction, :math:`xnodes`, and the number of nodes in the
    :math:`y`-direction, :math:`ynodes` in the requested mesh.

    """

    parameters = {
        'geometry': "1=planar, 2=cylindrical",
        'R': "radius of HE cylinder or half-thickness of HE slab",
        'omega_c': "DSD edge angle between HE and inert",
        'D_CJ': "nominal detonation velocity of the HE",
        'alpha': "coefficient of linear detonation velocity deviance",
        'IC': "initial condition (see descriptions)",
        'r_d': "initial detonation front radius",
        't_f': "final time",
        'xnodes': "number of nodes in x-direction",
        'ynodes': "number of nodes in y-direction"
        }

    # Default values

    geometry = 1       # geometry = n + 1 (n is used in Bdzil document)
    R = 1.0
    omega_c = np.pi / 4.0
    D_CJ = 1.0
    alpha = 0.1
    IC = 1
    r_d = np.sqrt(626.0)
    t_f = 6.0
    xnodes = 0          # must be changed by user
    ynodes = 0          # must be changed by user

    def __init__(self, **kwargs):

        """Input evaluation points as an :math:`N` x :math:`2` 2D array of
        positions:
        [ [:math:`x_0`, :math:`y_0`],
        [:math:`x_1`, :math:`y_1`], ...,
        [:math:`x_N`, :math:`y_N`] ].

        A time value must be input in order to use the ExactPack
        machinery. The time value is ignored in the calculation.

        """

        super(RateStick, self).__init__(**kwargs)

        if self.geometry not in [1, 2]:
            raise ValueError("geometry must be 1 or 2")

        if self.R <= 0:
            raise ValueError('Radius/thickness must be > 0')

        if self.omega_c <= 0:
            raise ValueError('DSD edge angle must be > 0')

        if self.omega_c >= (np.pi / 2.0):
            raise ValueError('DSD edge angle must be < pi/2')

        if self.D_CJ <= 0:
            raise ValueError('Detonation velocity must be > 0')

        if self.alpha < 0:
            raise ValueError('Alpha must be >= 0')

        if self.IC not in [1, 2, 3]:
            raise ValueError('IC must be 1, 2 or 3')

        if (self.IC == 1) and (self.r_d < self.R / np.cos(self.omega_c)):
            raise ValueError('Detonation radius must satisfy edge angle '
                             + 'condition')

        if self.t_f <= 0:
            raise ValueError('Final time must be positive')

        if self.xnodes <= 0:
            raise ValueError('Number of x-nodes must be specified')

        if self.ynodes <= 0:
            raise ValueError('Number of y-nodes must be specified')

    def _run(self, xylist, t):

        if self.xnodes * self.ynodes != xylist.shape[0]:
            raise ValueError('xnodes and ynodes do not match xylist')

        if max(xylist[:, 0]) != self.R:
            raise ValueError('xylist does not match R')

        if min(xylist[:, 0]) != 0.0:
            raise ValueError('xylist must reach x=0')

        btime = -10.0 * np.ones(len(xylist))

        # calculate on a fine, prescribed mesh
        dx = 0.002                       # mesh size wanted
        # time step size must satisfy stability criterion - 80% CFL given
        dt = 0.8 * (0.5 * dx**2.0 / self.alpha)

        # number of cells = R/dx + 1 ghost cell at x=0
        nx = int(self.R / dx) + 1
        dx = self.R / float(nx - 1)           # actual mesh size

        # x array needs nodes from -dx to R, numbered 0 to nx
        # node 0 is a ghost node (nodes = cells + 1)
        xpts = np.linspace(-dx, self.R, nx + 1)

        # t array, step by dt
        tpts = np.arange(0.0, self.t_f + dt, dt)

        # y array is x in columns, t in rows, keep 2 sets of y
        ypts = np.zeros((2, len(xpts)))

        # calculate values for BCs
        cosomega = np.cos(self.omega_c)
        bcslope = cosomega / np.sqrt(1.0 - cosomega**2.0)

        # set up initial condition -- IC=3 is already set
        if self.IC == 1:
            y_d = -1.0 * np.sqrt(self.r_d**2.0 - self.R**2.0)  # det center y
            ypts[0, :] = np.sqrt(self.r_d**2.0 - xpts**2.0) + y_d
        elif self.IC == 2:
            r_d = self.R / cosomega
            y_d = -1.0 * np.sqrt(r_d**2.0 - self.R**2.0)      # det center y
            ypts[0, :] = np.sqrt(r_d**2.0 - xpts**2.0) + y_d

        # apply boundary conditions to nodes 0 and nx
        ypts[0, 0] = ypts[0, 2]                               # reflective
        ypts[0, nx] = ypts[0, nx - 1] - dx * bcslope          # edge angle

        # interpolate y for x in xylist
        yinterp0 = np.interp(xylist[:, 0], xpts, ypts[0, :])

        # check if points in xylist are behind detonation front
        for ndx, vec in enumerate(xylist):
            if vec[1] <= yinterp0[ndx]:
                btime[ndx] = tpts[0]                        # tpts[0] = 0.0

        # loop over time steps - step=0 at tpts[1]
        for tstep, time in enumerate(tpts[1:]):
            # calculate derivatives - index is 1 less than matching x value
            # and goes from 0 to nx - 2
            yx = (ypts[0, 2:nx+1] - ypts[0, 0:nx-1]) / (2.0 * dx)
            yxx = (ypts[0, 2:nx+1] - 2.0 * ypts[0, 1:nx]
                   + ypts[0, 0:nx-1]) / dx**2.0
            factor = np.sqrt(1.0 + yx**2.0)

            # calculate curvature - index is 1 less than matching x value
            # and goes from 0 to nx - 2
            if self.geometry == 1:
                kappa = -yxx / factor**3.0
            else:
                kappa = np.zeros(len(yx))
                kappa[0] = -2.0 * yxx[0]
                kappa[1:] = - yxx[1:] / factor[1:]**3.0                \
                            - yx[1:] / (xpts[2:nx] * factor[1:])

            # calculate detonation velocity D_n
            detvel = self.D_CJ - self.alpha * kappa

            # calculate new position for time = tpts[step+1]
            ypts[1, 1:nx] = ypts[0, 1:nx] + dt * detvel * factor

            # apply boundary conditions to nodes 0 and nx
            ypts[1, 0] = ypts[1, 2]                            # reflective
            ypts[1, nx] = ypts[1, nx - 1] - dx * bcslope       # edge angle

            # interpolate y for x in xylist
            yinterp1 = np.interp(xylist[:, 0], xpts, ypts[1, :])
            dy = yinterp1 - yinterp0

            # check for points in xylist behind new detonation front
            # and interpolate detonation time (on tpts[step:step+1])
            # find starting and ending rows to speed up process
            startrow = -10
            stoprow = -10

            for row in range(self.ynodes):
                if startrow < 0:
                    lowest = min(btime[row*self.xnodes:(row+1)*self.xnodes])
                    if lowest < 0.0:
                        startrow = row
                if startrow >= 0:
                    highest = max(btime[row*self.xnodes:(row+1)*self.xnodes])
                    if highest < 0.0:
                        stoprow = row
                        break
            if startrow == -10:
                startrow = self.ynodes
            if stoprow == -10:
                stoprow = self.ynodes

            for count, vec in enumerate(xylist[startrow*self.xnodes:
                                               (stoprow+1)*self.xnodes]):
                index = count + startrow * self.xnodes
                if btime[index] == -10.0:
                    if vec[1] <= yinterp1[index]:
                        tinc = (vec[1] - yinterp0[index]) / dy[index] * dt
                        btime[index] = tpts[tstep] + tinc

            # move ynew to yold and yinterp1 to yinterp0
            ypts[0, :] = ypts[1, :]
            yinterp0 = yinterp1

        return ExactSolution([xylist[:, 0], xylist[:, 1], btime],
                             names=['position_x',
                                    'position_y',
                                    'burntime'],
                             jumps=[])
