"""Python RMTV solver.

This is a Python re-implementation Frank Timmes RMTV Fortran solver.
"""
import numpy as np
from math import exp, log
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq


def rmtv(r, aval_in, bval_in, chi0, gamma,
         bigamma, rf, xif_in, xis, beta0_in, g0):
    """Solves the rmtv in one-dimension, spherical coordinates
    this a highly simplified version of kamm's code that solves for 
    an array of values for a specific tri-lab verification test problem.

    Args: 
        r (array): Desired radial positions for the solution in cm
        aval_in (float): Power in thermal conductivity chi0 * rho**a * T**b
        bval_in (float): Power in thermal conductivity chi0 * rho**a * T**b
        chi0 (float): Coefficient in thermal conductivity chi0 * rho**a * T**b
        gamma (float): Ratio of specific heats
        bigamma (float): Gruneisen coefficient (gamma-1)*ener = pres/den = G*temp
        rf (float): Position of the heat front in cm
        xif_in (float): Dimensionless position of the heat front
        xis (float): Dimensionless position of the shock front
        beta0 (float): Eigenvalue of the problem
        g0 (float): Heat front scaling parameter

    Returns:
        tuple: A 5-tuple containing:
            - *ndarray*: den, density  g/cm**3
            - *ndarray*: tev, temperature ev
            - *ndarray*: ener, specific internal energy erg/g
            - *ndarray*: pres, presssure erg/cm**3
            - *ndarray*: vel, velocity cm/sh
    """
    nstep = len(r)
    den = np.zeros(nstep)
    tev = np.zeros(nstep)
    ener = np.zeros(nstep)
    pres = np.zeros(nstep)
    vel = np.zeros(nstep)

    for i in range(nstep):
        d, t, e, p, v = rmtv_1d(r[i], aval_in, bval_in, chi0, gamma,
                                bigamma, rf, xif_in, xis, beta0_in, g0)
        den[i] = d
        tev[i] = t
        ener[i] = e
        pres[i] = p
        vel[i] = v

    return den, tev, ener, pres, vel


def rmtv_1d(rpos, aval_in, bval_in, chi0, gamma,
            bigamma, rf, xif_in, xis, beta0_in, g0):
    """Solves the rmtv in one-dimension, spherical coordinates
    this a highly simplified version of kamm's code that solves for 
    a specific value for a specific tri-lab verification test problem.

    Args: 
        rpos (float): Desired radial position for the solution in cm
        aval_in (float): Power in thermal conductivity chi0 * rho**a * T**b
        bval_in (float): Power in thermal conductivity chi0 * rho**a * T**b
        chi0 (float): Coefficient in thermal conductivity chi0 * rho**a * T**b
        gamma (float): Ratio of specific heats
        bigamma (float): Gruneisen coefficient (gamma-1)*ener = pres/den = G*temp
        rf (float): Position of the heat front in cm
        xif_in (float): Dimensionless position of the heat front
        xis (float): Dimensionless position of the shock front
        beta0 (float): Eigenvalue of the problem
        g0 (float): Heat front scaling parameter

    Returns:
        tuple: A 5-tuple containing:
            - *float*: den, density  g/cm**3
            - *float*: tev, temperature ev
            - *float*: ener, specific internal energy erg/g
            - *float*: pres, presssure erg/cm**3
            - *float*: vel, velocity cm/sh
    """
    # Local parameters
    tol = 1.0e-16
    zero = 0.0
    abserr = 1.0e-14
    relerr = 1.0e-12
    
    # Used for the ODE integration
    nvar = 4
    jwork = 100 + 21 * nvar
    epsr = 4.0e-10
    epsa = 4.0e-10
    xi_small = 1.0e-4

    # transfer passed values to common
    # xgeom is for spherical coordinates
    global aval
    global bval
    global xif
    global beta0
    global xgeom
    aval = aval_in
    bval = bval_in
    xif = xif_in
    beta0 = beta0_in
    xgeom = 3.0

    # Initialize work arrays
    iwork = np.zeros(5, dtype=np.int32)
    work = np.zeros(jwork)

    # frequent factors
    global alpha
    global amu
    global kappa
    global sigma
    twoa = 2.0 * aval
    twob = 2.0 * bval
    alpha = (twob - twoa + 1.0)/(twob - (xgeom + 2.0)*aval +xgeom)
    amu = 2.0 / (gamma - 1.0)
    kappa = -((twob - 1.0)*xgeom + 2.0)/(twob - twoa + 1.0)
    sigma = (twob - 1.0)/(alpha*(1.0 - aval))

    # equations 28, 30, 33, 29
    # for the scale factor, the phyiscal time, and the shock front position
    zeta = (((0.5 * beta0 * bigamma**(bval + 1.0) * g0**(1.0 - aval) / chi0)**\
             (1.0 / (twob - 1.0))) / alpha)**alpha
    time = (rf / zeta / xif)**(1.0 / alpha)
    rs = zeta * 1.0 * abs(time)**alpha

    # this section does a root find to obtain the initial conditions
    # bracket the initial zero-value of u
    ustar = brentq(rmtvfun, 0, 0.5, xtol=tol)
    
    # form the converged value of the integral
    ans = quad(fun, zero, ustar, epsabs=abserr, epsrel=relerr)[0]
    
    # equation 11 for the position to start the integration from
    xistar = xif * exp(-(beta0 * (xif**((twob - 1.0) / alpha)) * ans))
    rstar = xistar * zeta * time**alpha

    # equation 11, 13 for the initial values of the other functions
    gstar = 1.0 / (1.0 - ustar )
    hstar = xistar**(-sigma) * gstar
    wstar = 0.5 * (amu - (( amu + 1.0) * ustar))
    tstar = ustar * (1.0 - ustar)

    # now integrate
    # beyond the heat front
    if (rpos > rstar):
        den = g0 * rpos**kappa
        vel = 0.0
        ener = 0.0
        pres = 0.0
        tev = 0.0
    
    # integrate from the heat front to perhaps the shock front
    else:
        ystart = np.array([ustar, hstar, wstar, tstar])

        xiwant = rpos / zeta / time**alpha
        xi_end = max(xis,xiwant)
        eta1 = log(xistar)
        eta2 = log(xi_end)
        soln = solve_ivp(derivs, (eta1, eta2), ystart, rtol=epsr, atol=epsa)
        ystart = soln.y[:, -1]
        # apply equation 15 of kamm 2000 for the post-shock values if we must
        # integrate farther
        if (rpos <= rs):
            usub2 = ystart[0]
            hsub2 = ystart[1]
            wsub2 = ystart[2]
            tsub2 = ystart[3]

            ystart[0] = 1.0 - (tsub2 / (1.0 - usub2))
            ystart[1] = (1.0 - usub2)**2 / tsub2 * hsub2
            ystart[2] = (tsub2 * wsub2 - 0.5 * ((1.0-usub2)**4 - tsub2**2)\
                         / (1.0 - usub2)) / (1.0 - usub2)**2
            ystart[3] = tsub2
            # and integrate to near the origin if nned be
            eta1 = eta2
            xi_end = max(xi_small,xiwant)
            eta2 = log(xi_end)
            soln = solve_ivp(derivs, (eta1, eta2), ystart,
                             rtol=epsr, atol=epsa)
            ystart = soln.y[:, -1]
        # convert the integration variables to physical quantities
        # equations 5, 2 of kamm 2000
        vel = alpha * rpos * ystart[0] / time
        den = g0 * rpos**kappa * xi_end**sigma * ystart[1]
        ener = (alpha * rpos / time)**2 * ystart[3] / (gamma - 1.0)
        pres = (gamma - 1.0) * den * ener
        tev = (alpha * rpos / time)**2 * ystart[3] / bigamma
        # convert from jerk = 1e16 erg,  kev = 1e3 ev,  sh = 10e-8 s to cgs
        # units
        vel  = vel  * 1.0e8
        ener = ener * 1.0e16
        pres = pres * 1.0e16
        tev  = tev  * 1.0e3

    return den, tev, ener, pres, vel


def rmtvfun(u):
    """evaluates the expression for the initial integral for a root find

    Args:
        u (float): Upper bound of integral

    Returns:
        float: The integrated value.
    """
    zero = 0.0
    abserr = 1.0e-14
    relerr = 1.0e-12
    smallval = 1.0e-12
    ans = quad(fun, zero, u, epsabs=abserr, epsrel=relerr)[0]
    return log(1.0 - smallval) + (beta0 *\
               (xif**(((2.0 * bval) - 1.0) / alpha)) * ans)


def fun(y):
    """evaluates the integrand of the initial integral"""
    return ((1.0 - (2.0 * y)) / (amu - ((amu + 1.0)*y))) \
             * (y**(bval - 1.0)) \
             * ((1.0 - y)**(bval - aval))


def derivs(t, y):
    """evaluates the rhs of the system of odes """
    eps16 = 1.0e-16
    eps12 = 1.0e-12
    # some factors
    y1m1 = y[0] - 1.0

    if (alpha == 0):
        raise ValueError('alpha = 0 in routine derivs')

    alphainv = 1.0 / alpha

    if (abs(y[1]) <= eps16 or abs(y[3]) <= eps16):
        print('derivs:  y[1] or y[3] <  eps16')
        omega = 1.0 / eps12 * np.sign(1.0, y[2]) * np.sign(1.0, beta0) \
                * np.sign(1.0, y[1]) * np.sign(1.0, y[3])
    else:
        if (abs(aval - 1.0) <= eps16):
            raise ValueError('aval=1 in routine derivs')
        omega = y[2] * y[1]**(1.0 - aval) * y[3]**(-bval) / beta0

    # rhs of original (coupled) ode system
    # equation 26
    g1 = sigma - (xgeom + kappa + sigma) * y[0]
    g2 = y[0] * (alphainv - y[0]) + y[3] * (2.0 * omega - kappa - sigma)
    g3 = omega * (amu * y1m1 + 2.0 * y[2]) + amu * (alphainv - 1.0) \
         - xgeom * y[0] - (xgeom + kappa + sigma) * y[2]
    g4 = -2.0 * (1.0 + omega)

    # rhs of uncoupled ode system
    # equations 24, 25
    denom = y[3] - y1m1**2
    if (abs(denom) <= eps16):
        raise ValueError('denom=0 in routine derivs')
    temp = g2 - (y1m1 * g1)
    yp = np.zeros(4)
    yp[0] = g1 - (y1m1 * temp) / denom
    yp[1] = y[1] * temp / denom
    yp[2] = g3 - (yp[0] + y[2] * yp[1] / y[1] )
    yp[3] = y[3] * g4

    return yp
