r"""Exact solution for the Escape of HE Products problem

"""
from ...base import ExactSolver, ExactSolution, Jump, JumpCondition
import warnings
import math

#  Load the path functions to enable point-in-polygon analysis
import matplotlib.path as path
import numpy as np


class EscapeOfHEProducts(ExactSolver):
    r""" Computes the solution to the Escape of HE Products problem.
    For the complete problem definition and exact solution, see
    Doebling, 2015. The problem default values are selected to be consistent
    with the original problem definition in Fickett and Rivard, 1974.

    Default values are: :math:`D=0.85 cm/\mu sec, \rho=1.6,
    u_p=0.5, \tilde{x}=1.0, x_{max}=10., t_{max}=10.`

    .. [Doebling] S. Doebling, "The Escape of High Explosive Products:
       An Exact-Solution Problem for Verification of Hydrodynamics Codes,"
       LA-UR-15-22547
    .. [Fickett] W. Fickett and C. Rivard, Test Problems for Hydrocodes.
       LASL Report, LA-5479, Los Alamos Scientific Laboratory
       (1974, Rev 1981)

    """

    parameters = {
        'geometry': '1=axial',
        'gamma': 'adiabatic index, must be 3.0',
        'D': "Detonation velocity of the HE (cm/us)",
        'rho_0': "Initial density of the HE (g/cc)",
        'up': "Velocity of the piston (cm/us)",
        'xtilde': "width of the HE material (cm)",
        'xmax': "maximum value of x allowed for exact solution",
        'tmax': "maximum value of t allowed for exact solution"
        }

    #
    #  Define default values
    #

    geometry = 1
    gamma = 3.
    D = 0.85
    rho_0 = 1.6
    up = 0.05
    xtilde = 1.0
    xmax = 10.
    tmax = 10.

    def __init__(self, **kwargs):
        """Set default values if necessary, check for valid inputs,
        and compute polygon vertices for the parameter values.

        """

        super(EscapeOfHEProducts, self).__init__(**kwargs)

        # check for illegal input values

        if self.D <= 0:
            raise ValueError('Detonation velocity must be > 0')

        if self.rho_0 <= 0:
            raise ValueError('Initial density must be > 0')

        if self.up < 0:
            raise ValueError('Piston velocity must be >= 0')
        if self.up >= self.D/(self.gamma+1):
            raise ValueError(('Piston velocity must be less than ',
                              'C-J particle velocity'))

        if self.xtilde <= 0 or self.xtilde > self.xmax:
            raise ValueError('xtilde must be between zero and xmax')

        if self.tmax <= 0:
            raise ValueError('tmax must be >0 ')

        # Calculate the polygon corners that describe the x-t diagram

        corners = dict()
        endpts = dict()

        #  Create "points at boundary" to close the open-ended polygons,
        #   at far values for curves, A, B, C, and E and along each axis

        #   Use the formulas for each line to do this

        # arrival time of detonation wave at outer boundary of HE

        self.ttilde = self.xtilde / self.D

        bdypointA = (self.xmax, self.xmax / self.D)
        bdypointB = (self.up * self.tmax, self.tmax)
        bdypointC = (self.xmax, self.xmax / (2. * self.up + self.D / 2.))
        bdypointC = ((2. * self.up + self.D / 2.) * self.tmax, self.tmax)
        bdypointE = (-3. / 2. * self.xtilde +
                     (2. * self.up + self.D / 2.) * self.tmax, self.tmax)
        bdypoint_x0 = (0., self.tmax)
        bdypoint_t0 = (self.xmax, 0.)

        #   Compute key locations

        #      Intersection point of curves C and D

        tcornerCD = (3. / 2.) * self.xtilde / (2. * self.up + self.D)
        xcornerCD = (3. / 2.) * self.xtilde *\
            (1. - self.D / (4. * self.up + 2. * self.D))

        #      Intersection point of curves B and D

        tcornerBD = 3. * self.xtilde / (2. * self.up + self.D)
        xcornerBD = self.up * tcornerBD

        #   Region 0V
        #    This region is above t=0, to the right of
        #    the HE-vacuum boundary, and below Boundary A

        regcorn = []
        # HE-vacuum boundary at t=0
        regcorn.append((self.xtilde, 0.))
        # HE-vacuum boundary at t=ttilde
        regcorn.append((self.xtilde, self.ttilde))
        regcorn.append(bdypointA)          # boundary point along Boundary A
        regcorn.append(bdypoint_t0)        # boundary point along t=0

        corners['0V'] = regcorn

        #   Region 0H
        #    This region is above t=0, to the left of the HE-vacuum boundary,
        #    and below Boundary A

        regcorn = []
        regcorn.append((0., 0.))              # origin, lower left corner
        # HE-vacuum boundary at t=ttilde
        regcorn.append((self.xtilde, self.ttilde))
        regcorn.append((self.xtilde, 0.))         # HE-vacuum boundary at t=0

        corners['0H'] = regcorn

        #   Region I
        #    This region is above Boundary A, below Boundary C,
        #    and below Boundary D

        regcorn = []
        regcorn.append((0., 0.))                 # origin, lower left corner
        regcorn.append((self.xtilde, self.ttilde))      # upper right corner
        # upper left corner, intersection of boundaries C and D
        regcorn.append((xcornerCD, tcornerCD))

        corners['I'] = regcorn

        #   Region II
        #    This region is above Boundary D, below Boundary C,
        #    and above Boundary A

        regcorn = []
        regcorn.append((self.xtilde, self.ttilde))        # bottom corner
        regcorn.append((xcornerCD, tcornerCD))  # left corner
        regcorn.append(bdypointC)               # boundary point along bdy C
        regcorn.append(bdypointA)               # boundary point along bdy A

        corners['II'] = regcorn

        #   Region III
        #    This region is above Boundary C, below Boundary D,
        #    and below Boundary B

        regcorn = []
        regcorn.append((0., 0.))                  # origin, lower left corner
        regcorn.append((xcornerCD, tcornerCD))   # upper right corner
        regcorn.append((xcornerBD, tcornerBD))   # upper left corner

        corners['III'] = regcorn

        #   Region IV
        #    This region is above Boundary D,
        #    above Boundary C, and below Boundary E

        regcorn = []
        regcorn.append((xcornerCD, tcornerCD))       # bottom corner
        regcorn.append((xcornerBD, tcornerBD))       # upper left corner
        # inifinity point along Boundary E
        regcorn.append(bdypointE)
        # boundary point along Boundary C
        regcorn.append(bdypointC)

        corners['IV'] = regcorn

        #   Region V
        #    This region is above Boundary E, and below Boundary B

        regcorn = []
        regcorn.append((xcornerBD, tcornerBD))       # bottom corner
        # boundary point along Boundary B
        regcorn.append(bdypointB)
        # boundary point along Boundary E
        regcorn.append(bdypointE)

        corners['V'] = regcorn

        #   Region 00
        #    This region is above Boundary B and below the x=0 axis
        #    (behind the piston)

        regcorn = []
        regcorn.append((0., 0.))              # origin, bottom corner
        regcorn.append(bdypoint_x0)        # boundary point along x=0
        regcorn.append(bdypointB)          # boundary point along Boundary B

        corners['00'] = regcorn

        #
        # Define the characteristic boundaries using the pre-defined points
        #

        pts = []
        pts.append((0., 0.))
        pts.append(bdypointA)
        endpts['A'] = pts

        pts = []
        pts.append((0., 0.))
        pts.append(bdypointB)
        endpts['B'] = pts

        pts = []
        pts.append((0., 0.))
        pts.append(bdypointC)
        endpts['C'] = pts

        pts = []
        pts.append((self.xtilde, self.ttilde))
        pts.append((xcornerBD, tcornerBD))
        endpts['D'] = pts

        pts = []
        pts.append((xcornerBD, tcornerBD))
        pts.append(bdypointE)
        endpts['E'] = pts

        self.endpts = endpts
        self.corners = corners

        return

    def _run(self, xvec, t):
        r''' Evaluate the physical variables at (x,t)
             Determine the Region for this combination of x and t
        '''

        corners = self.corners
        D = self.D
        rho_0 = self.rho_0
        up = self.up
        xtilde = self.xtilde
        ttilde = self.ttilde
        gamma = self.gamma

        # Initialize the physical variables

        csvec = np.empty_like(xvec)
        uvec = np.empty_like(xvec)
        pvec = np.empty_like(xvec)
        evec = np.empty_like(xvec)
        rhovec = np.empty_like(xvec)
        regvec = []  # For string variables, need to use a regular python list

        # Loop over x

        for i, x in enumerate(xvec):
            if path.Path(corners['I']).contains_point((x, t)) or \
                    self.point_on_boundary(corners['I'], (x, t)):
                # For this region, include boundary
                cs = 0.5 * (x / t + D / 2.)
                u = 0.5 * (x / t - D / 2.)
                p, rho = self.p_rho(rho_0, cs, D)
                reg = 'I'
            elif path.Path(corners['II']).contains_point((x, t)) or \
                    self.point_on_boundary(corners['II'], (x, t)):
                # For this region, include boundary
                cs = max(0.5 * (x / t - (x - xtilde) / (t - ttilde)), 0.)
                # need to ensure that cs is not negative (due to numerical
                # issues with the polygon boundary
                u = 0.5 * (x / t + (x - xtilde) / (t - ttilde))
                p, rho = self.p_rho(rho_0, cs, D)
                reg = 'II'
            elif path.Path(corners['III']).contains_point((x, t)) or \
                    self.point_on_boundary(corners['III'], (x, t)):
                # For this region, include boundary
                cs = up + D / 2.
                u = up
                p, rho = self.p_rho(rho_0, cs, D)
                reg = 'III'
            elif path.Path(corners['IV']).contains_point((x, t)) or \
                    self.point_on_boundary(corners['IV'], (x, t)):
                # For this region, include boundary
                cs = up + 0.5 * D * (0.5 - (x - xtilde) / (D * t - xtilde))
                u = up + 0.5 * D * (0.5 + (x - xtilde) / (D * t - xtilde))
                p, rho = self.p_rho(rho_0, cs, D)
                reg = 'IV'
            elif path.Path(corners['V']).contains_point((x, t)) or \
                    self.point_on_boundary(corners['V'], (x, t)):
                # For this region, include boundary
                cs = (D - up) * ttilde / (t - ttilde)
                u = (x - up * ttilde) / (t - ttilde)
                p, rho = self.p_rho(rho_0, cs, D)
                reg = 'V'
            elif path.Path(corners['00']).contains_point((x, t)):
                # For this region,  do not include boundary
                cs = 0.
                p = 0.
                rho = 0.
                u = 0.
                reg = '00'
            elif path.Path(corners['0V']).contains_point((x, t)):
                # For this region, do not include boundary
                cs = 0.
                p = 0.
                rho = 0.
                u = 0.
                reg = '0V'
            elif path.Path(corners['0H']).contains_point((x, t)):
                # For this region, do not include boundary
                cs = 0.
                u = 0.
                p = 0.
                rho = rho_0
                reg = '0H'
            else:
                p = 0.
                rho = 0.
                u = 0.
                cs = 0.
                reg = None

            #  Compute internal energy using EOS

            if rho != 0:
                e = p / rho / (gamma - 1.0)
            else:
                e = 0.

            csvec[i] = cs
            uvec[i] = u
            pvec[i] = p
            evec[i] = e
            rhovec[i] = rho
            regvec.append(reg)

        return ExactSolution([xvec, rhovec, pvec, evec, csvec, uvec, regvec],
                             names=['position',
                                    'density',
                                    'pressure',
                                    'specific_internal_energy',
                                    'sound_speed',
                                    'velocity',
                                    'region'
                                    ],
                             jumps=None)

        #  In future, add shock values into the jumps attribute

    def p_rho(self, rho_0, cs, D):
        r''' Compute shocked pressure and Density
             given initial density, sound speed and shock speed
        '''

        p = 16. / 27. * rho_0 * D ** 2 * (cs / D) ** 3
        rho = 16. / 9. * rho_0 * cs / D

        return p, rho

    def point_on_boundary(self, corners, point, tol=1e-12):
        r''' Determine whether a given point lies on the boundary
          of the polygon defined by "corners"
        
          Algorithm:
            on_boundary = False
            Loop over each successive pair of corners
             (including last to first)
            For each pair of corners:
                   Call point_on_line
                   If True, assign to on_boundary
        '''

        on_bound = False

        for i in range(len(corners) - 1):
            if self.point_on_line(corners[i:i + 2], point, tol):
                on_bound = True

        #  Check from last point back to first point
        if self.point_on_line((corners[len(corners) - 1], corners[0]),
                              point, tol):
            on_bound = True

        return on_bound

    def point_on_line(self, corners, point, tol=1e-5):
        r'''   Corners is a list of 2 points
               Compute distance between point and first corner
               Compute distance between point and second corner
               Compute distance between the two corners
               If magnitude of the sum of the first two distances minus the
                   third distance is less than a tolerance, then the point is
                   on the line
        '''

        corner0 = corners[0]
        corner1 = corners[1]

        dist01 = math.hypot(corner0[0] - corner1[0], corner0[1] - corner1[1])
        distp0 = math.hypot(corner0[0] - point[0], corner0[1] - point[1])
        distp1 = math.hypot(corner1[0] - point[0], corner1[1] - point[1])

        return abs(distp0 + distp1 - dist01) < tol
