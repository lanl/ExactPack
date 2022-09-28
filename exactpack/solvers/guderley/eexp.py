"""This function calculates numerical values of the similarity exponent "alpha"
for the converging shock problem in cylindrical or spherical geometry using the
"ode" and "zeroin_a" routines.

The values of "lambda" can be found in Tables 6.4 (cyl) and 6.5 (sph) of
[Lazarus1981]_

While the output of this function is the "standard lambda" appearing in the
report above, the function itself is actually solves for alpha = 1/lambda
through the terminology found in: [Chisnell1998]_

Chisnell's formulation of the problem is more lucid and useful for the
evaluation of the similarity exponent than the exposition appearing in the
Lazarus paper, though the ultimate output is identical to a large number of
significant digits.

2007.07.17 S. Ramsey: initial development -- seems to work except for
                       gamma < 1.01 ... I can't seem to get more than 2 of
                       Lazarus' significant figures.

2007.07.23 S. Ramsey: Cleaned up code

2022.09.26 J. Thrussell: Code translated from Fortran to Python.
"""
import numpy as np
from math import sqrt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp


def eexp(nnn, gamm):
    global g
    global n
    # Here we read in (or set) the space index (n) and specific heat
    # ratio (g) from the namelist file.
    n = nnn
    g = gamm

    if n not in (2, 3):
        raise ValueError('Invalid Geometry input.')
    # Data does not exist for gamma > 9999 or gamma < 1.00001 (in these
    # cases a polytropic gas probably doesn't make sense anyway), so
    # if we try to find the similarity exponent for one of these
    # cases, we punt.
    if not (1.00001 < g < 9999.0):
        raise ValueError('Invalid polytropic index.')

    # tol = sys.float_info.epsilon
    tol = 1.0e-10
    # Next we set the range in which we expect alpha to lie for a given
    # geometry and specific heat ratio. See the README file for a more
    # detailed discussion of the origin of these approximations.
    a0num = -2.0 - g - sqrt(2.0) * g * sqrt(g / (g - 1.0))
    a0dem = -2.0 - sqrt(2.0) * g * sqrt(g / (g - 1.0)) - g * n
    a0 = a0num / a0dem

    if g > 3.732050808:
        amin = a0
    else:
        amin = (4.0 + 2.0 * sqrt(2.0) * sqrt((g**3) * (-1.0 + n)**2)
                + g * (-6.0 + (2.0 + g) * n)) \
                / (4.0 + g * (-8.0 + n * (4.0 + g * n))) + 0.000001

    amax = 1.05 * a0

    if amax >= 1.0:
        err_str = 'Maximum alpha exceeds unity. Adjust "amax" premultiplier'
        raise ValueError(err_str)
    # The "exact" value of alpha is found through the "zeroin_a" routine.
    # We are attempting to find the value of alpha that zeros the
    # "Cdiff" function defined below.
    alpha = brentq(Cdiff, amin, amax, xtol=tol, args=(n, g))

    return 1.0 / alpha


def Cdiff(alpha, en, gamma):
    """In principle, the solution curve C(V) of Chisnell Eq. (3.1) must pass
    through a critical point at a prescribed location in (V, C) phase space.
    This location is determined through a singular analysis of Eq. (3.1) itself.

    The function Cdiff, starting from known initial conditions, numerically
    integrates Eq. (3.1) to the prescribed location in V and returns the
    corresponding value of C. This value of C is then compared to the prescribed
    value of C. This difference is ostensibly zero for a single, correct choice
    of alpha. It is made to be zero by the "zeroin" routine.

    Input:

        en = a dummy variable that simulates the geometry index (n)
        gamma = a dummy variable that simulates the specific heat
        ratio (gamma)
        (these variables are passed to the function from the driver
        program)

    Output:

        Cdiff = The phase space difference evaluated at the critical
        point. The smallness of this difference is a good
        measure of the correctness of alpha.
    """
    global a
    abserr = 1.0e-9
    relerr = 1.0e-8
    neqn = 1
    y = np.zeros(neqn)

    a = alpha
    # It was determined by Lazarus that the V-coordinate of the critical
    # point through which the solution curve must pass is algebraically
    # distinct for different ranges of the specific heat ratio. The
    # "critical" value at which this distinction is realized is taken as
    # given for each geometry type.
    if n == 3:
        gammacrit = 1.8697680
    elif n == 2:
        gammacrit = 1.9092084

    # The calculation of the critical (V0,C0) pair follows.
    V0dem = 2.0 * g * (n - 1.0)
    factor = g * n - 2.0
    disc = 8.0 * (a - 1.0) * a * g * (n - 1.0) + (2.0 - g + a * factor)**2

    if disc < 0.0:
        raise ValueError('Analytic result for V0 is non-real.')

    if g >= gammacrit:
        V0num1 = 2.0 - 2.0 * a - g + a * g * n - sqrt(disc)
        V0 = V0num1 / V0dem
    else:
        V0num2 = 2.0 - 2.0 * a - g + a * g * n + sqrt(disc)
        V0 = V0num2 / V0dem

    C0 = (V0 - a)**2

    # The coordinate (V0,C0) just calculated analytically must be
    # matched by a numerical integration of Chisnell Eq. (3.1) starting
    # from the shock point (Vs,Cs) = (t, y(1)) below. This numerical
    # integration is carried through by the "ode" subroutine.
    t = (2.0 * a) / (g + 1.0)
    y[0] = (2.0 * g * (g - 1.0) * a**2) / (g + 1.0)**2
    tout = V0

    soln = solve_ivp(fe, (t, tout), y, rtol=relerr, atol=abserr,
                     method='DOP853')
    y = soln.y[:, -1]

    # The difference function between the analytically and numerically
    # obtained values of the coordinate C0 is returned by the function.
    return C0 - y[0]


def fe(t, y):
    """fe is the RHS of Chisnell Eq. (3.1), and is used in the numerical
    integration of Eq. (3.1) through the call to "ode."
    """
    # Establishment of the various factors appearing in Eq. (3.1)
    delta = (t - a)**2 - y[0]
    Q = n * t * (t - a) + (2.0 / g) * (1.0 - a) * (a - t) - t * (t - 1.0)
    numer = y[0] * (2.0 * delta * (a - t + (1.0 - a)
                    * (1.0 / g)) + (g - 1.0) * (a - t) * Q)
    denom = delta * (n * t - 2.0 * (1.0 - a) * (1.0 / g)) * (a - t) \
        + ((a - t)**2.0) * Q

    # Computation of the RHS of Eq. (3.1)
    return numer/denom
