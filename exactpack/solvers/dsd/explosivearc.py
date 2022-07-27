r"""A DSD Explosive Arc solver in Python.

This is a pure Python implementation of the Explosive Arc solution using Numpy.

The DSD Explosive Arc problem is used to test how well a level set solver
applies boundary conditions to an HE burn time table calculation, as the
solution to this problem is dominated by the applied boundary conditions.
In this problem, the boundaries of the explosive do not align with the DSD
computational mesh.

An explosive arc consists of a semi-annulus of high explosive (HE) which
occupies the region
:math:`r_1 <= r <= r_2, - \frac{\pi}{2} <= \theta <= \frac{\pi}{2}`. The
problem is solved in an :math:`r \theta`-coordinate system. An edge angle
is defined at each of the radial boundaries. The free boundary found at the
inner radius, :math:`r_1`, is defined by the HE sonic angle, :math:`\omega_s`.
The boundary at the outer radius, :math:`r_2`, can be defined by either the
HE confinement angle, :math:`\omega_c`, if a confinement material is present
or by a fixed or reflective angle, :math:`\omega_e = \frac{\pi}{2}`, if HE
is present outside the modeled region.

At time :math:`t_d = 0.0`, the shape of the HE burn front shock wave is
prescribed. Any HE behind the initial burn front is assumed to ignite
at time :math:`t_d`, as if it were part of the detonation system. The
detonator is located at :math:`(x_d, y_d)` where :math:`y_d` is at the
center of the lower end of the annulus, i.e.,

.. math:: y_d = - \frac{r_1 + r_2}{2}`

The detonator is located outside of the annulus, so the :math:`x`-coordinate
of the detonator position is negative. The inital burn front is then
located at

.. math:: {(x - x_d)}^2 + {(y - y_d)}^2 = {r_d}^2

where :math:`r_d` is the radius of the initial detonation front, found by

.. math:: r_d = \sqrt{{|x_d|}^2 + {\left( \frac{r_2 - r_1}{2} \right)}^2}


The detonation trajectory for the above conditions is given by

.. math:: \theta = f(r,t)

where the function :math:`f` satisfies the PDE

.. math:: f_t = D_n \sqrt{1+(rf_r)^2}

where :math:`D_n` is the velocity of the HE detonation wave in the
shock-normal direction, which is described by a deviation from the nominal
constant Chapman-Jouguet detonation shock speed :math:`D_{CJ}` of the HE.

In [Bdzil]_, the deviation in :math:`D_n` is linear with respect to the
curvature, :math:`\kappa`, of the detonation shock front:

.. math:: D_n = D_{CJ} - \alpha \kappa

With this form of :math:`D_n` and the geometrically appropriate curvatures,
the function :math:`f` satisfies the PDE

.. math::
    f_t = - \frac{D_{CJ}}{r} \sqrt{1+(rf_r)^2} + \frac{\alpha}{r}
    \frac{rf_{rr}+r^2{f_r}^3+2f_r}{1+(rf_r)^2}

As currently implemented, :math:`\kappa` will be calculated separately,
then :math:`D_n` will be calculated from this linear form. The function
:math:`f` will then be calculated from the more general form above. In
this way, any other form for :math:`D_n` could be added at a later date.

The boundary conditions to be applied are

.. math::
    r_1 f_r (r_1, t) = \cot (\omega_s)
    
    r_2 f_r (r_2, t) = - \cot (\omega_e)


"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class ExplosiveArc(ExactSolver):
    r"""Computes the numerical solution to the Explosive Arc Problem.

    The HE semi-annulus is assumed to lie in the region
    :math:`r_1 <= r <= r_2, - \frac{\pi}{2} <= \theta <= \frac{\pi}{2}`.
    The half-thickness of the annulus is then :math:`R = \frac{r_2 - r_1}{2}`.
    The solution is obtained in the :math:`r \theta`-plane.

    The boundary at the inner radius, :math:`r = r_1` is assumed to be free,
    thus the DSD edge angle there is the HE sonic angle and
    :math:`\omega_{in} = \omega_s`, where the sonic angle is assumed to
    satisfy :math:`0 < \omega_s < \frac{\pi}{2}`. The boundary at the outer
    radius, :math:`r = r_2` is either confined by a material, in which case
    the DSD edge angle is the confinement angle and
    :math:`\omega_{out} = \omega_c`, or a fixed boundary, in which case the
    DSD edge angle is :math:`\omega_{out} = \frac{\pi}{2}`. For the confined
    case, :math:`\omega_c` is assumed to satisfy
    :math:`\omega_s < \omega_c < \frac{\pi}{2}`.

    The detonator is located at :math:`(x_d, y_d)` where :math:`y_d` is
    assumed to be in the center of the lower end of the annulus, i.e.,
    :math:`y_d = - \frac{r_1 + r_2}{2}`. The detonator is assumed to be
    located outside of the annulus, so the :math:`x`-coordinate of the
    detonator position, :math:`x_d`, is assumed to be negative.
    The nominal detonation velocity of the HE, :math:`D_{CJ}`, must
    be positive. The linear coefficient, :math:`\alpha`, of detonation
    velocity deviance must also be positive.

    Detonation time is assumed to be :math:`t_d = 0.0`. The burn front
    shock wave is assumed to have just reached the confinement material at
    detonation time, with the shock wave located at
    :math:`\theta = - \frac{pi}{2}` at the edges of the annulus
    (:math:`r = r_1` and :math:`r = r_2`) before application of the
    boundary conditions.

    Default values are selected to reflect the description in [Bdzil]_.
    Default values are **geometry** :math:`= 1`, :math:`r_1 = 2.0`,
    :math:`r_2 = 4.0`, :math:`\omega_{in} = \frac{\pi}{4}`,
    :math:`\omega_{out} = \frac{\pi}{2}`, :math:`x_d = -4.0`,
    :math:`D_{CJ} = 1.0` and :math:`\alpha = 0.1`. This produces a detonator
    location of :math:`(-4.0,-3.0)` and an initial detonation radius of
    :math:`r_d = \sqrt{17.0}`.

    The incoming Cartesian (:math:`xy`) mesh is converted to a polar
    (:math:`r \theta`) mesh. The solution for :math:`\theta` of the burn front
    is calculated on a fine :math:`(r, t)` mesh to obtain
    :math:`\theta = \theta(r, t)`. The solution is then interpolated to the
    requested :math:`r \theta` mesh and inverted to obtain the burn time as
    :math:`t = t(r, \theta)`. The requested mesh must have a larger mesh
    spacing than the mesh used for the calculation. For efficiency
    in inverting the solution, the user must input the number of nodes in
    the :math:`x`-direction, :math:`xnodes`, and the number of nodes in the
    :math:`y`-direction, :math:`ynodes` in the requested mesh.
    """

    parameters = {
        'geometry': "1=planar",
        'r_1': "inner radius of HE arc",
        'r_2': "outer radius of HE arc",
        'omega_in': "inner radius DSD free-surface angle",
        'omega_out': "outer radius DSD edge angle",
        'x_d': ":math:`x`-coordinate of detonator location",
        'D_CJ': "nominal detonation velocity of the HE",
        'alpha': "coefficient of linear detonation velocity deviance",
        't_f': "final time",
        'xnodes': "number of nodes in x-direction",
        'ynodes': "number of nodes in y-direction"
        }

    # Default values

    geometry = 1       
    r_1 = 2.0
    r_2 = 4.0
    omega_in = np.pi / 4.0
    omega_out = np.pi / 2.0
    x_d = -4.0
    D_CJ = 1.0
    alpha = 0.1
    t_f = 14.0
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

        super(ExplosiveArc, self).__init__(**kwargs)

        if self.geometry not in [1]:
            raise ValueError("geometry must be 1")

        if self.r_1 <= 0:
            raise ValueError('Inner radius must be > 0')

        if self.r_2 <= 0:
            raise ValueError('Outer radius must be > 0')

        if self.r_2 <= self.r_1:
            raise ValueError('Outer radius must be larger than inner radius')

        if self.omega_in <= 0:
            raise ValueError('Inner DSD edge angle must be > 0')

        if self.omega_in >= (np.pi / 2.0):
            raise ValueError('Inner DSD edge angle must be < pi/2')

        if self.omega_out < self.omega_in:
            raise ValueError('Outer DSD edge angle must be >= ' +
                             'inner DSD edge angle')

        if self.omega_out > (np.pi / 2.0):
            raise ValueError('Outer DSD edge angle must be <= pi/2')

        if self.x_d >= 0:
            raise ValueError('Detonator position must be < 0')

        if self.D_CJ <= 0:
            raise ValueError('Detonation velocity must be > 0')

        if self.alpha < 0:
            raise ValueError('Alpha must be >= 0')

        if self.t_f <= 0:
            raise ValueError('Final time must be positive')

        if self.xnodes <= 0:
            raise ValueError('Number of x-nodes must be specified')

        if self.ynodes <= 0:
            raise ValueError('Number of y-nodes must be specified')

    def _run(self, xylist, t):

        if self.xnodes * self.ynodes != xylist.shape[0]:
            raise ValueError('xnodes and ynodes do not match xylist')

        # convert Cartesian to polar
        rtmesh = np.zeros(xylist.shape)

        for index, vec in enumerate(xylist):
            if vec[0] < 0.0:
                raise ValueError('HE must have positive x-position')
            rtmesh[index, 0] = np.sqrt(vec[0]**2.0 + vec[1]**2.0)
            if rtmesh[index, 0] == 0.0:
                raise ValueError('HE must have positive inner radius')
            if vec[0] == 0.0:
                if vec[1] > 0.0:
                    rtmesh[index, 1] = np.pi / 2.0
                elif vec[1] < 0.0:
                    rtmesh[index, 1] = - np.pi / 2.0
            else:
                rtmesh[index, 1] = np.arctan(vec[1] / vec[0])

        # check for coverage of annulus region
        if (max(rtmesh[:, 0]) - self.r_2) > 0.00001:
            raise ValueError('xylist does not match r_2')

        if (min(rtmesh[:, 0]) - self.r_1) > 0.00001:
            raise ValueError('xylist does not match r_1')

        if (max(rtmesh[:, 1]) - np.pi / 2.0) > 0.00001:
            raise ValueError('xylist does not reach pi/2')

        if (min(rtmesh[:, 1]) - -np.pi / 2.0) > 0.00001:
            raise ValueError('xylist does not reach -pi/2')

        btime = -10.0 * np.ones(len(xylist))

        # calculate on a fine, prescribed mesh
        dr = 0.004                       # mesh size wanted
        # time step size must satisfy stability criterion - 80% CFL given
        dt = 0.8 * (0.5 * dr**2.0 / self.alpha)

        # number of cells = (r2 - r1)/dr
        nr = int((self.r_2 - self.r_1) / dr)
        dr = (self.r_2 - self.r_1) / float(nr)           # actual mesh size

        # r array needs nodes from r1 to r2, numbered 0 to nr
        # nodes = cells + 1
        rpts = np.linspace(self.r_1, self.r_2, nr + 1)

        # t array, step by dt
        tpts = np.arange(0.0, self.t_f + dt, dt)

        # theta array is r in columns, t in rows, keep 2 sets of theta
        thetapts = np.zeros((2, len(rpts)))

        # calculate values for BCs
        cosomega1 = np.cos(self.omega_in)
        inbcslope = cosomega1 / np.sqrt(1.0 - cosomega1**2.0)
        cosomega2 = np.cos(self.omega_out)
        outbcslope = cosomega2 / np.sqrt(1.0 - cosomega2**2.0)

        # set up initial condition
        # calculate distance from det to annulus, det location
        xd0 = abs(self.x_d)
        yd0 = (self.r_1 + self.r_2) / 2.0
        rd0 = np.sqrt(xd0**2.0 + ((self.r_2 - self.r_1) / 2.0)**2.0)
        q1 = xd0**2.0 + yd0**2.0
        q2 = q1 - rd0**2.0

        # calculate theta of shock front for each r
        for idx, rpt in enumerate(rpts):
            a4y = 4.0 * q1
            b4y = 4.0 * yd0 * (rpt**2.0 + q2)
            c4y = (rpt**2.0 + q2)**2.0 - 4.0 * rpt**2.0 * xd0**2.0
            yint = (-b4y - np.sqrt(b4y**2.0 - 4.0 * a4y * c4y)) / (2.0 * a4y)
            xint = np.sqrt(rpt**2.0 - yint**2.0)
            if xint == 0.0:
                thetapts[0, idx] = - np.pi / 2.0
            else:
                thetapts[0, idx] = np.arctan(yint / xint)

        # apply boundary conditions to nodes 0 and nx (edge angles)
        thetapts[0, 0] = thetapts[0, 1] - dr * inbcslope / self.r_1
        thetapts[0, nr] = thetapts[0, nr-1] - dr * outbcslope / self.r_2

        # interpolate theta for r in rtmesh
        thinterp0 = np.interp(rtmesh[:, 0], rpts, thetapts[0, :])

        # check if points in rtmesh are behind detonation front
        for ndx, vec in enumerate(rtmesh):
            if vec[1] <= thinterp0[ndx]:
                btime[ndx] = tpts[0]                        # tpts[0] = 0.0

        # loop over time steps - step=0 at tpts[1]
        for tstep, time in enumerate(tpts[1:]):
            # calculate derivatives - index is 1 less than matching r value
            # and goes from 0 to nr - 2
            fr = (thetapts[0, 2:nr+1] - thetapts[0, 0:nr-1]) / (2.0 * dr)
            frr = (thetapts[0, 2:nr+1] - 2.0 * thetapts[0, 1:nr]
                   + thetapts[0, 0:nr-1]) / dr**2.0
            factor = np.sqrt(1.0 + rpts[1:nr]**2.0 * fr**2.0)

            # calculate curvature - index is 1 less than matching r value
            # and goes from 0 to nr - 2
            kappa = (rpts[1:nr] * frr + rpts[1:nr]**2.0 * fr**3.0
                     + 2.0 * fr) / factor**3.0

            # calculate detonation velocity D_n
            detvel = self.D_CJ - self.alpha * kappa

            # calculate new position for time = tpts[step+1]
            thetapts[1, 1:nr] = thetapts[0, 1:nr] + dt * detvel * factor

            # apply boundary conditions to nodes 0 and nx
            thetapts[1, 0] = thetapts[1, 1] - dr * inbcslope / self.r_1
            thetapts[1, nr] = thetapts[1, nr-1] - dr * outbcslope / self.r_2

            # interpolate theta for r in rtmesh
            thinterp1 = np.interp(rtmesh[:, 0], rpts, thetapts[1, :])
            dth = thinterp1 - thinterp0

            # check for points in rtmesh behind new detonation front
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

            for count, vec in enumerate(rtmesh[startrow*self.xnodes:
                                               (stoprow+1)*self.xnodes]):
                index = count + startrow * self.xnodes
                if btime[index] == -10.0:
                    if vec[1] <= thinterp1[index]:
                        tinc = (vec[1] - thinterp0[index]) / dth[index] * dt
                        btime[index] = tpts[tstep] + tinc

            # move thetanew to thetaold and thinterp1 to thinterp0
            thetapts[0, :] = thetapts[1, :]
            thinterp0 = thinterp1

        return ExactSolution([xylist[:, 0], xylist[:, 1], btime],
                             names=['position_x',
                                    'position_y',
                                    'burntime'],
                             jumps=[])
